import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import os


def window_slicing(data, slicing_size=128):
    """
    [n_samples, n_features] -> [n_examples, slicing_size, n_features]
    """
    n_samples, n_features = data.shape
    n_examples = n_samples - slicing_size + 1
    reshaped_data = np.lib.stride_tricks.as_strided(
        data, 
        shape=(n_examples, slicing_size, n_features), 
        strides=(data.strides[0], data.strides[0], data.strides[1])
    )
    return reshaped_data


def preprocess_ett(file_name):
    data = pd.read_csv(f'ETT-small/{file_name}.csv', index_col='date', parse_dates=True)

    data = data.to_numpy()
    if file_name == 'ETTh1' or file_name == 'ETTh2':
        train_slice = slice(None, 12*30*24)
        valid_slice = slice(12*30*24, 16*30*24)
        test_slice = slice(16*30*24, 20*30*24)
    elif file_name == 'ETTm1' or file_name == 'ETTm2':
        train_slice = slice(None, 12*30*24*4)
        valid_slice = slice(12*30*24*4, 16*30*24*4)
        test_slice = slice(16*30*24*4, 20*30*24*4)
    else:
        raise ValueError('Unknown dataset')

    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)

    train_data = window_slicing(data[train_slice])
    valid_data = window_slicing(data[valid_slice])
    test_data = window_slicing(data[test_slice])

    train_data, valid_data, test_data = torch.from_numpy(train_data), torch.from_numpy(valid_data), torch.from_numpy(test_data)

    if not os.path.exists(f'ETT-small/{file_name}'):
        os.makedirs(f'ETT-small/{file_name}')
    
    torch.save(train_data, f'ETT-small/{file_name}/train.pt')
    torch.save(valid_data, f'ETT-small/{file_name}/val.pt')
    torch.save(test_data, f'ETT-small/{file_name}/test.pt')

if __name__ == '__main__':
    preprocess_ett('ETTh1')
    preprocess_ett('ETTh2')
    preprocess_ett('ETTm1')
    preprocess_ett('ETTm2')