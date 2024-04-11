import torch
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from sklearn.linear_model import Ridge


def generate_pred_repr_with_label(repr, raw_data, pred_len):
    """
    Generate prediction representation and labels for time series prediction.

    Parameters:
        repr (torch.Tensor): The representation tensor of shape [bs, length, hidden_dim],
                             where bs is batch size, length is the sequence length,
                             and hidden_dim is the dimensionality of the representation.
        raw_data (torch.Tensor): The raw time series data tensor of shape [bs, length, channels].
        pred_len (int): The prediction length.

    Returns:
        tuple: A tuple containing two numpy arrays:
               - The reshaped representation of shape [(bs * length) - pred_len, hidden_dim].
               - The labels of shape [(bs*length)-pred_len, pred_len * channels].
    """
    # Ensure the tensors are on CPU and convert them to numpy arrays
    repr_np = repr.cpu().numpy()
    raw_data_np = raw_data.cpu().numpy()

    # Flatten the batch and time dimensions
    flat_repr = repr_np.reshape(-1, repr_np.shape[2])
    flat_raw_data = raw_data_np.reshape(-1, raw_data_np.shape[2])

    # Slice the representation to adjust for prediction length
    repr_for_pred = flat_repr[:-pred_len]

    n_samples, n_channels = flat_raw_data.shape
    n_examples = n_samples - pred_len + 1  # 切片之后的样本数

    labels = np.lib.stride_tricks.as_strided(
        flat_raw_data,
        shape=(n_examples, pred_len, n_channels),
        strides=(
            flat_raw_data.strides[0], flat_raw_data.strides[0], flat_raw_data.strides[1])
    )
    labels = labels[1:]  # Remove the first element

    return repr_for_pred, labels


def fit_ridge(train_X, train_y, val_X, val_y):
    alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    val_results = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha).fit(train_X, train_y)
        valid_pred = ridge.predict(val_X)
        loss = np.sqrt(((valid_pred - val_y) ** 2).mean()) + \
            np.abs(valid_pred - val_y).mean()
        val_results.append(loss)
    best_alpha = alphas[np.argmin(val_results)]

    ridge = Ridge(alpha=best_alpha)
    ridge.fit(train_X, train_y)
    return ridge


class TimeMAEForecasting:
    def __init__(self, args, model, train_loader, val_loader, test_loader, pred_lens):
        self.args = args
        self.device = args.device
        self.num_epoch = args.num_epoch
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.pred_lens = pred_lens

    def forecasting(self):
        print('forecasting...')
        self.model = self.model.to(self.device)
        self.model.linear_proba = "forecasting"
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.scheduler = LambdaLR(
            self.optimizer, lr_lambda=lambda step: self.args.lr_decay ** step)

        tqdm_train_loader = tqdm(self.train_loader)
        self.model.train()
        for pred_len in self.pred_lens:
            for epoch in range(self.num_epoch):
                loss_sum = 0
                for idx, batch in enumerate(tqdm_train_loader):
                    batch = [x.to(self.device) for x in batch]
                    raw_data, _ = batch

                    self.optimizer.zero_grad()
                    repr = self.model(raw_data)
                    train_repr, train_labels = generate_pred_repr_with_label(
                        repr, raw_data, pred_len)
                    val_repr, val_labels = generate_pred_repr_with_label(
                        repr, raw_data, pred_len)
                    ridge = fit_ridge(train_repr, train_labels,
                                      val_repr, val_labels)