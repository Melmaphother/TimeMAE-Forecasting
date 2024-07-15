import numpy as np
import torch
import torch.utils.data as Data


class Dataset(Data.Dataset):
    def __init__(self, device, mode, data, wave_len):
        self.device = device
        self.datas, self.label = data
        self.mode = mode
        self.wave_len = wave_len
        self.__padding__()

    def __padding__(self):
        origin_len = self.datas[0].shape[0]
        if origin_len % self.wave_len:
            padding_len = self.wave_len - (origin_len % self.wave_len)
            padding = np.zeros(
                (len(self.datas), padding_len, self.datas[0].shape[1]), dtype=np.float32
            )
            self.datas = np.concatenate([self.datas, padding], axis=-2)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        data = torch.tensor(self.datas[item]).to(self.device)
        label = self.label[item]
        return data, torch.tensor(label).to(self.device)

    def shape(self):
        return self.datas[0].shape


class ETTForecastingDataset(Data.Dataset):
    def __init__(self, args, mode="train"):
        self.args = args
        self.device = args.device
        self.pred_len = args.pred_len
        self.mode = mode
        self.slicing_size = 128
        self.__read_data__()

    def __read_data__(self):
        self.raw_data = torch.load(self.args.data_path + self.mode + "_forecasting.pt")
        self.raw_data = np.array(self.raw_data)

        self.raw_data_drop_last_pred = self.raw_data[: -self.pred_len]

        n_samples, n_features = self.raw_data_drop_last_pred.shape
        n_examples = n_samples - self.slicing_size + 1
        self.sliced_data = np.lib.stride_tricks.as_strided(
            self.raw_data_drop_last_pred,
            shape=(n_examples, self.slicing_size, n_features),
            strides=(
                self.raw_data_drop_last_pred.strides[0],
                self.raw_data_drop_last_pred.strides[0],
                self.raw_data_drop_last_pred.strides[1],
            ),
        )
        # print(self.sliced_data.shape)
        """
        pred_len = 720
        etth1        train val test
        raw_data:    (8640, 7) (2880, 7) (2880, 7)
        raw_data_drop_last_pred: (7920, 7) (2160, 7) (2160, 7)
        sliced_data: (7793, 128, 7) (2033, 128, 7) (2033, 128, 7)
        """

    def __len__(self):
        return len(self.sliced_data)

    def __getitem__(self, idx):
        data = self.sliced_data[idx]
        label_l = idx + self.slicing_size
        label_r = label_l + self.pred_len
        label = self.raw_data[label_l:label_r]
        data = torch.tensor(data).to(self.device)
        pred_label = torch.tensor(label).to(self.device)
        return data, pred_label

    def shape(self):
        return self.sliced_data[0].shape
