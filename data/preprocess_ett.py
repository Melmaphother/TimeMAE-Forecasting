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

def window_slicing_forecasting(data, slicing_size=128):
    n_samples, n_features = data.shape
    # 无需填充，直接计算可以完整切片的样本数量
    n_samples_sliced = n_samples - (n_samples % slicing_size)
    # 只取可以整除slicing_size的部分
    sliced_data = data[:n_samples_sliced]
    n_examples = n_samples_sliced // slicing_size
    # 计算新的strides
    new_strides = (slicing_size * data.strides[0],) + data.strides
    # 使用as_strided创建分片视图
    sliced_data = np.lib.stride_tricks.as_strided(
        sliced_data, 
        shape=(n_examples, slicing_size, n_features), 
        strides=new_strides
    )
    return sliced_data


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

    # train_data_forecasting = window_slicing_forecasting(data[train_slice])
    # valid_data_forecasting = window_slicing_forecasting(data[valid_slice])
    # test_data_forecasting = window_slicing_forecasting(data[test_slice])
    # print(data[train_slice].shape, data[valid_slice].shape, data[test_slice].shape)
    # print(train_data.shape, valid_data.shape, test_data.shape)
    # print(train_data_forecasting.shape, valid_data_forecasting.shape, test_data_forecasting.shape)
    """
    etth1
    (8640, 7) (2880, 7) (2880, 7)
    (8513, 128, 7) (2753, 128, 7) (2753, 128, 7)
    (68, 128, 7) (23, 128, 7) (23, 128, 7)
    """

    train_data, valid_data, test_data = torch.from_numpy(train_data), torch.from_numpy(valid_data), torch.from_numpy(test_data)
    train_data_forecasting, valid_data_forecasting, test_data_forecasting = torch.from_numpy(data[train_slice]), torch.from_numpy(data[valid_slice]), torch.from_numpy(data[test_slice])
    # train_data_forecasting, valid_data_forecasting, test_data_forecasting = torch.from_numpy(train_data_forecasting), torch.from_numpy(valid_data_forecasting), torch.from_numpy(test_data_forecasting)

    if not os.path.exists(f'ETT-small/{file_name}'):
        os.makedirs(f'ETT-small/{file_name}')
    
    torch.save(train_data, f'ETT-small/{file_name}/train.pt')
    torch.save(valid_data, f'ETT-small/{file_name}/val.pt')
    torch.save(test_data, f'ETT-small/{file_name}/test.pt')

    torch.save(train_data_forecasting, f'ETT-small/{file_name}/train_forecasting.pt')
    torch.save(valid_data_forecasting, f'ETT-small/{file_name}/val_forecasting.pt')
    torch.save(test_data_forecasting, f'ETT-small/{file_name}/test_forecasting.pt')

if __name__ == '__main__':
    preprocess_ett('ETTh1')
    preprocess_ett('ETTh2')
    preprocess_ett('ETTm1')
    preprocess_ett('ETTm2')