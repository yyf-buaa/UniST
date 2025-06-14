import numpy as np
import torch as th
import json
import torch
import datetime
import copy
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
class Dataset_Train(Dataset):
    def __init__(self, data, norm_channel_0, norm_channel_1, start_date='2019-01-01'):
        self.data = data  # 稀疏张量，形状 (B, T, N, M, C)
        self.T = data.shape[1]
        self.N = data.shape[2]
        self.M = data.shape[3]
        self.C = data.shape[4]
        self.my_scaler_channel_0 = norm_channel_0
        self.my_scaler_channel_1 = norm_channel_1
        # 预先生成所有可能的时间特征（按最大长度）
        aligned_T = (self.T // 8) * 8
        dates = pd.date_range(start=start_date, periods=aligned_T, freq='D')
        self.time_features = torch.tensor([
            (
                dt.weekday(),
                dt.month-1,
                dt.day-1,
                dt.year,
                (dt.month - 1) // 3  # 季度
            ) for dt in dates
        ])  # shape: (T, 5)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        T = self.data.shape[1]
        aligned_T = (T // 8) * 8
        batch_data = self.data[idx].to_dense().unsqueeze(dim=0)[:, :aligned_T].float()
        batch_data = batch_data.reshape(-1, 8, self.N, self.M, self.C)  # (B, 8, N, M, C)
        batch_data[...,0] = self.my_scaler_channel_0.transform(batch_data[...,0].reshape(-1,1)).reshape(batch_data[...,0].shape)
        batch_data[...,1] = self.my_scaler_channel_1.transform(batch_data[...,1].reshape(-1,1)).reshape(batch_data[...,1].shape)
        batch_data = batch_data.permute(0, 4, 1, 2, 3)  # (B, C, 8, N, M)
        time_features = self.time_features[:aligned_T].unsqueeze(dim=0).reshape(-1, 8, 5)  # ( 8, 5)
        return batch_data, time_features
    
class Dataset_test(Dataset):
    def __init__(self, data, norm_channel_0, norm_channel_1, start_date='2020-07-01'):
        self.data = data  # 稀疏张量，形状 (B, T, N, M, C)
        self.T = data.shape[1]
        self.N = data.shape[2]
        self.M = data.shape[3]
        self.C = data.shape[4]
        self.my_scaler_channel_0 = norm_channel_0
        self.my_scaler_channel_1 = norm_channel_1
        # 预先生成所有可能的时间特征（按最大长度）
        aligned_T = (self.T // 8) * 8
        dates = pd.date_range(start=start_date, periods=aligned_T, freq='D')
        self.time_features = torch.tensor([
            (
                dt.weekday(),
                dt.month-1,
                dt.day-1,
                dt.year,
                (dt.month - 1) // 3  # 季度
            ) for dt in dates
        ]) # shape: (T, 5)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        T = self.data.shape[1]
        aligned_T = (T // 8) * 8
        batch_data = self.data[idx].to_dense().unsqueeze(dim=0)[:, :aligned_T].float() # Convert sparse tensor to dense
        batch_data = sliding_window(batch_data,8) 
        batch_data[...,0] = self.my_scaler_channel_0.transform(batch_data[...,0].reshape(-1,1)).reshape(batch_data[...,0].shape)
        batch_data[...,1] = self.my_scaler_channel_1.transform(batch_data[...,1].reshape(-1,1)).reshape(batch_data[...,1].shape)
        batch_data = batch_data.permute(0, 4, 1, 2, 3)  # (B, C, 8, N, M)
        time_features = self.time_features[:aligned_T].unsqueeze(dim=0)
        time_features = sliding_window_ts(time_features,8) 
        return batch_data, time_features

class MinMaxNormalization(object):
    """
        MinMax Normalization --> [-1, 1]
        x = (x - min) / (max - min).
        x = x * 2 - 1
    """

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X


import torch

def sliding_window(tensor, seq_len):
    """
    Apply sliding window on the time dimension (dim=1) of the tensor.
    
    Parameters:
    - tensor: Input tensor of shape (B, T, N, M, 2)
    - seq_len: Length of each sliding window sequence
    
    Returns:
    - Tensor of shape (B * num_windows, seq_len, N, M, 2)
    """
    B, T, N, M, C = tensor.shape
    if seq_len > T:
        raise ValueError("seq_len must be less than or equal to T.")
    # 使用 unfold 在时间维度上滑动
    windows = tensor.unfold(dimension=1, size=seq_len, step=1)  #torch.Size([1, 177, 10, 10, 2, 8])

    # 调整维度顺序，把 seq_len 放到合适的位置
    # 最终形状为 (B, num_windows, seq_len, N, M, 2)
    windows = windows.permute(0, 1, 5, 2, 3, 4)  # shape: (B, num_windows, seq_len, N, M, 2)

    # 合并 batch 和 window 维度
    num_windows = T - seq_len + 1
    output = windows.reshape(-1, seq_len, N, M, C)

    return output

def sliding_window_ts(tensor, seq_len):
    """
    Apply sliding window on the time dimension (dim=1) of the tensor.
    
    Parameters:
    - tensor: Input tensor of shape (B, T, C), e.g., (B, T, 10)
    - seq_len: Length of each sliding window sequence
    
    Returns:
    - Tensor of shape (B * num_windows, seq_len, C)
    """
    B, T, C = tensor.shape

    if seq_len > T:
        raise ValueError("seq_len must be less than or equal to T.")

    # 使用 unfold 在时间维度上展开滑动窗口
    # shape: (B, num_windows, seq_len, C)
    windows = tensor.unfold(dimension=1, size=seq_len, step=1).permute(0, 1, 3, 2)  # shape: (B, num_windows, seq_len, C)

    output = windows.reshape(-1, windows.shape[-2], windows.shape[-1])

    return output
# def sliding_window(tensor, seq_len):
#     """
#     Apply sliding window on the first dimension (T) of the tensor.
    
#     Parameters:
#     - tensor: Input tensor of shape (B, T, N, M, 2)
#     - seq_len: The length of each sliding window sequence
    
#     Returns:
#     - Tensor of shape (B', seq_len, N, M, 2), where B is the number of windows
#     """
#     T, N, M, C = tensor.shape
#     # Calculate the number of windows
#     B = T - seq_len + 1
    
#     # Check if there are enough elements for at least one window
#     if B < 1:
#         raise ValueError("The sequence length is too long for the given tensor.")
    
#     # Prepare indices for gathering slices
#     indices = torch.arange(seq_len).reshape(1, seq_len).repeat(B, 1)
#     offsets = torch.arange(B).reshape(B, 1)
#     indices = indices + offsets
    
#     # Use advanced indexing to gather the sliding windows
#     output = tensor[indices]
    
#     return output.reshape(B, seq_len, N, M, C)
def get_nonzero_channel_values(sparse_tensor, channel_idx):
    """
    获取稀疏张量中指定 channel 的所有非零值
    
    参数:
    - sparse_tensor: 稀疏张量，形状 (B, T, N, M, C)
    - channel_idx: 要提取的通道索引
    
    返回:
    - 指定通道的所有非零值组成的一维张量
    """
    indices = sparse_tensor.coalesce().indices()  # 形状: (5, nnz)
    values = sparse_tensor.coalesce().values()    # 形状: (nnz, C)
    # 获取所有非零元素的通道索引
    channel_indices = indices[-1, :]  # 最后一维是通道维度
    
    # 找到对应 channel_idx 的非零元素索引
    mask = (channel_indices == channel_idx)
    
    # 提取对应通道的非零值
    channel_values = values[mask]
    
    return channel_values

def data_load_single(args, dataset): 
    train_data = torch.load('/home/yyf/private/fire_data/label_train_sparse_tensor.pt')
    test_data = torch.load('/home/yyf/private/fire_data/label_test_sparse_tensor.pt')
    my_scaler_channel_0 = MinMaxNormalization()
    channel_0_values_train = get_nonzero_channel_values(train_data, 0)
    channel_0_values_test = get_nonzero_channel_values(test_data, 0)
    MAX_0 = max(channel_0_values_train.max().item(), channel_0_values_test.max().item())
    MIN_0 = 0
    my_scaler_channel_0.fit(np.array([MIN_0, MAX_0]))
    my_scaler_channel_1 = MinMaxNormalization()
    channel_1_values_train = get_nonzero_channel_values(train_data, 1)
    channel_1_values_test = get_nonzero_channel_values(test_data, 1)
    MAX_1 = max(channel_1_values_train.max().item(), channel_1_values_test.max().item())
    MIN_1 = 0
    my_scaler_channel_1.fit(np.array([MIN_1, MAX_1]))
    train_dataset = Dataset_Train(train_data, my_scaler_channel_0, my_scaler_channel_1)
    test_dataset = Dataset_test(test_data, my_scaler_channel_0, my_scaler_channel_1)
    val_dataaset = test_dataset
    data = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True) 
    test_data = DataLoader(test_dataset , num_workers=8, batch_size = args.batch_size, shuffle=False)
    val_data = DataLoader(val_dataaset, num_workers=8, batch_size = args.batch_size, shuffle=False)
    my_scaler=[my_scaler_channel_0,my_scaler_channel_1]
    return  data, test_data, val_data, my_scaler

def data_load(args):

    data_all = []
    test_data_all = []
    val_data_all = []
    my_scaler_all = []
    my_scaler_all = {}

    for dataset_name in args.dataset.split('*'):
        data, test_data, val_data, my_scaler = data_load_single(args,dataset_name)
        data_all.append([dataset_name, data])
        test_data_all.append(test_data)
        val_data_all.append(val_data)
        my_scaler_all[dataset_name] = my_scaler

    data_all = [(name,i) for name, data in data_all for i in data]
    random.seed(1111)
    random.shuffle(data_all)
    
    return data_all, test_data_all, val_data_all, my_scaler_all


def data_load_main(args):

    data, test_data, val_data, scaler = data_load(args)

    return data, test_data, val_data, scaler

