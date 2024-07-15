import numpy as np
import pandas as pd
import torch
from pathlib import Path
from argparse import Namespace
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class ETTDataset(Dataset):
    def __init__(
            self,
            args: Namespace,
            flag: str = 'train',
            file_name: str = 'ETTh1',
            freq: str = 'h',
            scale: bool = True,
    ):
        """
        Constructor of the ETTDataset class, including ETTh1, ETTh2, ETTm1, ETTm2.
        :param args: args of whole project
        :param flag: 'train', 'val' or 'test'
        :param file_name: 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'
        :param freq: 'h' represents hourly, 'm' represents 15 minutes (a quarter), must be compatible with file_name
        :param scale: True or False, whether to normalize the data, using 'StandardScaler' default
        """
        assert flag in ['train', 'val', 'test'], f'{flag} must be in [train, val, test]'
        assert file_name in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'], f'{file_name} must be in [ETTh1, ETTh2, ETTm1, ETTm2]'
        assert freq in ['h', 'm'], f'{freq} must be in [h, m]'
        assert freq in file_name, f'{freq} is not compatible with {file_name}'
        self.args = args
        self.flag = flag
        self.file_name = file_name
        self.freq = freq
        self.scale = scale
        self.scaler = StandardScaler()
        self.slice_h_map = {
            'train': slice(0, 12 * 30 * 24),
            'val': slice(12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24),
            'test': slice(12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24),
        }
        self.slice_m_map = {
            'train': slice(0, 12 * 30 * 24 * 4),
            'val': slice(12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4),
            'test': slice(12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4),
        }
        self.__read_data__()

    def __read_data__(self):
        file_path = Path(self.args.data_dir) / f'{self.file_name}.csv'
        ett_data = pd.read_csv(file_path, index_col='date', parse_dates=True)
        ett_data = ett_data.to_numpy(dtype=np.float32)

        if self.scale:
            train_slice = self.slice_h_map['train'] if self.freq == 'h' else self.slice_m_map['train']
            self.scaler.fit(ett_data[train_slice])
            ett_data = self.scaler.transform(ett_data)

        _slice = self.slice_h_map[self.flag] if self.freq == 'h' else self.slice_m_map[self.flag]
        self.data = ett_data[_slice]

    def __len__(self):
        return len(self.data) - self.args.seq_len - self.args.pred_len + 1

    def __getitem__(self, idx):
        input_begin = idx
        input_end = idx + self.args.seq_len
        label_begin = input_end - self.args.label_len
        label_end = label_begin + self.args.label_len + self.args.pred_len

        x = self.data[input_begin:input_end]
        y = self.data[label_begin:label_end]

        x = torch.tensor(x, dtype=torch.float32).to(self.args.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.args.device)

        return x, y

    def inverse_transform(self, data):
        if self.scale:
            data = self.scaler.inverse_transform(data)
        return data
