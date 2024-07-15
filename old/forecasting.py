import torch
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from sklearn.linear_model import Ridge
import time


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
            flat_raw_data.strides[0],
            flat_raw_data.strides[0],
            flat_raw_data.strides[1],
        ),
    )
    labels = labels[1:]  # Remove the first element

    labels = labels.reshape(-1, pred_len * n_channels)
    print(f"repr_for_pred shape: {repr_for_pred.shape}, labels shape: {labels.shape}")
    return repr_for_pred, labels


def fit_ridge(train_X, train_y, val_X, val_y):
    alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    val_results = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha).fit(train_X, train_y)
        valid_pred = ridge.predict(val_X)
        loss = (
            np.sqrt(((valid_pred - val_y) ** 2).mean())
            + np.abs(valid_pred - val_y).mean()
        )
        val_results.append(loss)
    best_alpha = alphas[np.argmin(val_results)]
    print(f"Best alpha: {best_alpha}")
    print(f"Best loss: {val_results[np.argmin(val_results)]}")

    ridge = Ridge(alpha=best_alpha)
    ridge.fit(train_X, train_y)
    return ridge


def call_loss(pred, target, mode="MSE"):
    if mode == "MSE":
        loss = ((pred - target) ** 2).mean()
    elif mode == "MAE":
        loss = np.abs(pred - target).mean()
    elif mode == "MAE+MSE":
        loss = np.abs(pred - target).mean() + np.sqrt(
            ((pred - target) ** 2).mean()
        )  # 加 sqrt 为了保证度量相同
    else:
        raise ValueError(f"Unknown loss mode: {mode}")
    return loss


class TimeMAEForecasting:
    def __init__(self, args, model, data_loader):
        self.args = args
        self.device = args.device
        self.model = model
        self.train_loader, self.val_loader, self.test_loader = data_loader
        self.pred_len = args.pred_len
        self.verbose = True
        self.lr_decay = 0.98
        self.num_epoch = args.num_epoch
        self.channels = args.data_shape[1]
        self.samples = self.pred_len * self.channels
        self.save_path = args.save_path_each_pred

    def forecasting_ridge(self):
        print("forecasting...")
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
            train_repr_forecast, train_labels_forecast = generate_pred_repr_with_label(
                train_repr, self.train_raw_data, pred_len
            )
            val_repr_forecast, val_labels_forecast = generate_pred_repr_with_label(
                val_repr, self.val_raw_data, pred_len
            )
            t = time.time()
            test_repr_forecast, test_labels_forecast = generate_pred_repr_with_label(
                test_repr, self.test_raw_data, pred_len
            )
            ridge = fit_ridge(
                train_repr_forecast,
                train_labels_forecast,
                val_repr_forecast,
                val_labels_forecast,
            )
            train_time[pred_len] = time.time() - t

            t = time.time()
            test_pred = ridge.predict(test_repr_forecast)
            pred_time[pred_len] = time.time() - t
            loss = call_loss(test_pred, test_labels_forecast)
            print(test_pred[0][0:10])
            print(test_labels_forecast[0][0:10])
            print(f"Prediction length: {pred_len}, loss: {loss}")

        # print time cost
        for k, v in train_time.items():
            print(f"Training time for prediction length {k}: {v}")

        for k, v in pred_time.items():
            print(f"Prediction time for prediction length {k}: {v}")

    def forecasting_finetune(self):
        self.result_file = open(self.save_path + "/train_result.txt", "w")
        self.result_file.close()
        self.result_file = open(self.save_path + "/test_result.txt", "w")
        self.result_file.close()

        self.model.to(self.device)
        self.model.linear_proba = "forecasting"

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: self.lr_decay**step,
            verbose=self.verbose,
        )
        self.criterion = torch.nn.MSELoss()

        self.best_val_loss = 10000

        for epoch in range(self.num_epoch // 10):
            train_loss_epoch, train_time_cost = self._train_single_epoch()
            val_loss_epoch, val_time_cost = self._eval_single_epoch()
            curr_lr = self.scheduler.get_last_lr()[0]
            print(f"Current learning rate: {curr_lr}")
            self.scheduler.step()

            self.result_file = open(self.save_path + "train_result.txt", "a+")
            self.print_process(
                "Finetune epoch:{0},loss:{1},training_time:{2}".format(
                    epoch + 1, train_loss_epoch, train_time_cost
                )
            )
            print(
                "Finetune train epoch:{0},loss:{1},training_time:{2}".format(
                    epoch + 1, train_loss_epoch, train_time_cost
                ),
                file=self.result_file,
            )
            self.print_process(
                "Finetune epoch:{0},loss:{1},validation_time:{2}".format(
                    epoch + 1, val_loss_epoch, val_time_cost
                )
            )
            print(
                "Finetune epoch:{0},loss:{1},validation_time:{2}".format(
                    epoch + 1, val_loss_epoch, val_time_cost
                ),
                file=self.result_file,
            )
            self.result_file.close()

            if val_loss_epoch < self.best_val_loss:
                self.best_val_loss = val_loss_epoch
                torch.save(
                    self.model.state_dict(), self.save_path + "finetune_model.pkl"
                )

        test_loss = self._eval_model()
        self.result_file = open(self.save_path + "test_result.txt", "a+")
        print(f"Test loss: {test_loss}", file=self.result_file)
        self.result_file.close()

        self.print_process("Test loss: {0}".format(test_loss))
        return test_loss

    def _train_single_epoch(self):
        t0 = time.perf_counter()
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader) if self.verbose else self.train_loader

        loss_sum = 0
        for idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(self.device) for x in batch]
            data, pred_label = batch

            self.optimizer.zero_grad()
            pred_output = self.model(data, self.pred_len)
            loss = self.criterion(pred_output, pred_label)
            loss_sum += loss.item()
            loss.backward()
            self.optimizer.step()
        t1 = time.perf_counter()

        return loss_sum / len(self.train_loader), t1 - t0

    def _eval_single_epoch(self):
        t0 = time.perf_counter()
        self.model.eval()
        tqdm_dataloader = tqdm(self.val_loader) if self.verbose else self.val_loader

        loss_sum = 0

        with torch.no_grad():
            for idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                data, pred_label = batch

                pred_output = self.model(data, self.pred_len)
                loss = self.criterion(pred_output, pred_label)
                loss_sum += loss.item()

        t1 = time.perf_counter()
        return loss_sum / len(self.val_loader), t1 - t0

    def _eval_model(self):
        self.model.load_state_dict(torch.load(self.save_path + "finetune_model.pkl"))
        self.model.eval()
        tqdm_dataloader = tqdm(self.test_loader) if self.verbose else self.test_loader

        loss_sum = 0
        with torch.no_grad():
            for idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                data, pred_label = batch

                pred_output = self.model(data, self.pred_len)
                loss = self.criterion(pred_output, pred_label)
                loss_sum += loss.item()

        return loss_sum / len(self.test_loader)

    def print_process(self, *x):
        if self.verbose:
            print(*x)
