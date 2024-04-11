import torch
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from sklearn.linear_model import Ridge
import time
from args import Test_data, Train_data
try:
    from args import VAL_data
except:
    print("No validation data")

def generate_pred_repr_with_label(repr, raw_data, pred_len):
    """
    Generate prediction representation and labels for time series prediction.

    Parameters:
        repr (torch.Tensor): The representation tensor of shape [n_instances, length, hidden_dim],
                             where m_instances is numbers of instances, length is the sequence length,
                             and hidden_dim is the dimensionality of the representation.
        raw_data (torch.Tensor): The raw time series data tensor of shape [n_instances, length, channels].
        pred_len (int): The prediction length.

    Returns:
        tuple: A tuple containing two numpy arrays:
               - The reshaped representation of shape [(n_instances * length) - pred_len, hidden_dim].
               - The labels of shape [(n_instances * length)-pred_len, pred_len * channels].
    """

    # Flatten the batch and time dimensions
    flat_repr = repr.reshape(-1, repr.shape[2])
    flat_raw_data = raw_data.reshape(-1, raw_data.shape[2])

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

    print(f"repr_for_pred shape: {repr_for_pred.shape}, labels shape: {labels.shape}")
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

def call_loss(self, pred, target):
        loss = np.abs(pred - target).mean() + np.sqrt(((pred - target) ** 2).mean())
        return loss


class TimeMAEForecasting:
    def __init__(self, args, model, train_loader, val_loader, test_loader, pred_lens):
        self.args = args
        self.device = args.device
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_raw_data, _ = Train_data
        self.val_raw_data, _ = VAL_data
        self.test_raw_data, _ = Test_data
        self.pred_lens = pred_lens



    def forecasting(self):
        print('forecasting...')
        self.model.to(self.device)
        self.model.linear_proba = "forecasting"

        self.model.eval()

        train_time = {}
        pred_time = {}
        _, length, _ = self.train_raw_data.shape
        train_repr = np.empty([0, length, self.args.d_model])
        val_repr = np.empty([0, length, self.args.d_model])
        test_repr = np.empty([0, length, self.args.d_model])
        with torch.no_grad():
            for _, batch in enumerate(self.train_loader):
                batch = [x.to(self.device) for x in batch]
                raw_data, _ = batch
                repr = self.model(raw_data).cpu().numpy()
                train_repr = np.concatenate((train_repr, repr), axis=0)
                
            for _, batch in enumerate(self.val_loader):
                batch = [x.to(self.device) for x in batch]
                raw_data, _ = batch
                repr = self.model(raw_data).cpu().numpy()
                val_repr = np.concatenate((val_repr, repr), axis=0)

            for _, batch in enumerate(self.test_loader):
                batch = [x.to(self.device) for x in batch]
                raw_data, _ = batch
                repr = self.model(raw_data).cpu().numpy()
                test_repr = np.concatenate((test_repr, repr), axis=0)

        for pred_len in self.pred_lens:
            train_repr, train_labels = generate_pred_repr_with_label(train_repr, self.train_raw_data, pred_len)
            val_repr, val_labels = generate_pred_repr_with_label(val_repr, self.val_raw_data, pred_len)
            t = time.time()
            test_repr, test_labels = generate_pred_repr_with_label(test_repr, self.test_raw_data, pred_len)
            ridge = fit_ridge(train_repr, train_labels, val_repr, val_labels)
            train_time[pred_len] = time.time() - t

            t = time.time()
            test_pred = ridge.predict(test_repr)
            pred_time[pred_len] = time.time() - t
            loss = call_loss(test_pred, test_labels)
            print(f"Prediction length: {pred_len}, loss: {loss / len(self.test_loader)}")
        
        # print time cost
        for k, v in train_time.items():
            print(f"Training time for prediction length {k}: {v}")
        
        for k, v in pred_time.items():
            print(f"Prediction time for prediction length {k}: {v}")
