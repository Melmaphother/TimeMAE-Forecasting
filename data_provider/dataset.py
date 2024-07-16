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
        Args:
            args: args of whole project
            flag: 'train', 'val' or 'test'
            file_name: 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'
            freq: 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'
            scale: True or False, whether to normalize the data, using 'StandardScaler' default
        """
        match (flag, file_name, freq):
            case ('train' | 'val' | 'test', 'ETTh1' | 'ETTh2', 'h'):
                pass
            case ('train' | 'val' | 'test', 'ETTm1' | 'ETTm2', 'm'):
                pass
            case _:
                raise ValueError(f'Invalid combination of flag, file_name and freq: {flag=}, {file_name=}, {freq=}')

        match scale:
            case True | False:
                pass
            case _:
                raise ValueError(f'Invalid scale: {scale=}')

        self.args = args
        self.flag = flag
        self.file_name = file_name
        self.freq = freq
        self.scale = scale
        self.scaler = StandardScaler()
        self.slice_hour_map = {
            'train': slice(0, 12 * 30 * 24),
            'val': slice(12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24),
            'test': slice(12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24),
            # sampling interval = 1 hour
            # train: 12 months = [0, 8639) samples = 8640 samples
            # val: 4 months = [8640, 11520) samples = 2880 samples
            # test: 4 months = [11520, 14400) samples = 2880 samples
        }
        self.slice_15_min_map = {
            'train': slice(0, 12 * 30 * 24 * 4),
            'val': slice(12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4),
            'test': slice(12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4),
            # sampling interval = 15 minutes
            # train: 12 months = [0, 34559) samples = 34560 samples
            # val: 4 months = [34560, 46080) samples = 11520 samples
            # test: 4 months = [46080, 57600) samples = 11520 samples
        }
        self.__read_data__()

    def __read_data__(self):
        file_path = Path(self.args.data_dir) / 'ETT-small' / f'{self.file_name}.csv'
        ett_data = pd.read_csv(file_path, index_col='date', parse_dates=True)
        ett_data = ett_data.to_numpy(dtype=np.float32)

        if self.scale:
            train_slice = self.slice_hour_map['train'] if self.freq == 'h' else self.slice_15_min_map['train']
            self.scaler.fit(ett_data[train_slice])
            ett_data = self.scaler.transform(ett_data)

        _slice = self.slice_hour_map[self.flag] if self.freq == 'h' else self.slice_15_min_map[self.flag]
        self.data = ett_data[_slice]

    def __len__(self):
        return len(self.data) - self.args.seq_len - self.args.pred_len + 1

    def __getitem__(self, idx):
        input_begin = idx
        input_end = idx + self.args.seq_len
        label_begin = input_end
        label_end = input_end + self.args.pred_len

        x = self.data[input_begin:input_end]
        y = self.data[label_begin:label_end]

        x = torch.tensor(x, dtype=torch.float32).to(self.args.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.args.device)

        return x, y

    def inverse_transform(self, data):
        if self.scale:
            data = self.scaler.inverse_transform(data)
        return data


class HARDataset(Dataset):
    def __init__(
            self,
            args: Namespace,
            flag: str = 'train',
            scale: bool = True,
    ):
        """
        Constructor of the HARDataset class.
        Args:
            args: args of whole project
            flag: 'train', 'val' or 'test'
            scale: True or False, whether to normalize the data, using 'StandardScaler' default
        """
        match flag:
            case 'train' | 'val' | 'test':
                pass
            case _:
                raise ValueError(f'Invalid flag: {flag=}')

        match scale:
            case True | False:
                pass
            case _:
                raise ValueError(f'Invalid scale: {scale=}')

        self.args = args
        self.flag = flag
        self.scale = scale
        self.scaler = StandardScaler()
        self.__read_data__()

    def __read_data__(self):
        train_file_path = Path(self.args.data_dir) / 'HAR' / 'train.pt'
        val_file_path = Path(self.args.data_dir) / 'HAR' / 'val.pt'
        test_file_path = Path(self.args.data_dir) / 'HAR' / 'test.pt'

        train_data = torch.load(train_file_path)
        val_data = torch.load(val_file_path)
        test_data = torch.load(test_file_path)

        if self.scale:
            self.scaler.fit(train_data)
            train_data = self.scaler.transform(train_data)
            val_data = self.scaler.transform(val_data)
            test_data = self.scaler.transform(test_data)

        match self.flag:
            case 'train':
                _data = train_data
            case 'val':
                _data = val_data
            case 'test':
                _data = test_data
            case _:
                raise ValueError(f'Invalid flag: {self.flag=}')

        self.data = _data['samples'].transpose(1, 2)
        self.labels = _data['labels']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32).to(self.args.device)
        y = torch.tensor(self.labels[idx], dtype=torch.long).to(self.args.device)

        return x, y
