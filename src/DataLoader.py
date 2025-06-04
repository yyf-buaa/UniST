import numpy as np
import torch as th
import json
import torch
import datetime
import copy
import random
import pandas as pd
import numpy as np
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
    Apply sliding window on the first dimension (T) of the tensor.
    
    Parameters:
    - tensor: Input tensor of shape (T, N, M, 2)
    - seq_len: The length of each sliding window sequence
    
    Returns:
    - Tensor of shape (B, seq_len, N, M, 2), where B is the number of windows
    """
    T, N, M, C = tensor.shape
    # Calculate the number of windows
    B = T - seq_len + 1
    
    # Check if there are enough elements for at least one window
    if B < 1:
        raise ValueError("The sequence length is too long for the given tensor.")
    
    # Prepare indices for gathering slices
    indices = torch.arange(seq_len).reshape(1, seq_len).repeat(B, 1)
    offsets = torch.arange(B).reshape(B, 1)
    indices = indices + offsets
    
    # Use advanced indexing to gather the sliding windows
    output = tensor[indices]
    
    return output.reshape(B, seq_len, N, M, C)

def data_load_single(args, dataset): 
    X_train = torch.tensor(np.load('/home/yyf/private/Global Fire Prediction/dataset/label_non_scale_train.npy').astype(np.float32)).reshape(-1,12,20,20,2)
    X_test = torch.tensor(np.load('/home/yyf/private/Global Fire Prediction/dataset/label_non_scale_test.npy').astype(np.float32)).reshape(-1,20,20,2)
    X_val = torch.tensor(np.load('/home/yyf/private/Global Fire Prediction/dataset/label_non_scale_test.npy').astype(np.float32)).reshape(-1,20,20,2)
    args.seq_len = X_train.shape[1]
    H, W = X_train.shape[3], X_train.shape[4]  
    X_test = sliding_window(X_test,args.seq_len) 
    X_val = sliding_window(X_val,args.seq_len) 
    X_train_ts = pd.concat([pd.read_csv('/home/yyf/private/Global Fire Prediction/dataset/trainmax.csv')['date'],pd.read_csv('/home/yyf/private/Global Fire Prediction/dataset/valmax.csv')['date']])
    X_train_ts=torch.tensor([(datetime.datetime.strptime(t,'%Y-%m-%d').weekday(),datetime.datetime.strptime(t,'%Y-%m-%d').month) for t in X_train_ts.values]).reshape(-1,12,20,20,2)
    X_val_ts = pd.read_csv('/home/yyf/private/Global Fire Prediction/dataset/testmax.csv')['date']
    X_val_ts = torch.tensor([(datetime.datetime.strptime(t,'%Y-%m-%d').weekday(),datetime.datetime.strptime(t,'%Y-%m-%d').month) for t in X_val_ts.values])
    X_val_ts = X_val_ts.reshape(-1,20,20,2)
    X_test_ts = X_val_ts.clone()
    X_test_ts = sliding_window(X_test_ts,args.seq_len) 
    X_val_ts = sliding_window(X_val_ts,args.seq_len) 
    X_train_ts = X_train_ts[:,:,0,0,:].squeeze()
    X_val_ts = X_val_ts[:,:,0,0,:].squeeze()
    X_test_ts = X_test_ts[:,:,0,0,:].squeeze()

    my_scaler_channel_0 = MinMaxNormalization()
    MAX_0 = max(torch.max(X_train[...,0]).item(), torch.max(X_test[...,0]).item(), torch.max(X_val[...,0]).item())
    MIN_0 = min(torch.min(X_train[...,0]).item(), torch.min(X_test[...,0]).item(), torch.min(X_val[...,0]).item())
    my_scaler_channel_0.fit(np.array([MIN_0, MAX_0]))
    my_scaler_channel_1 = MinMaxNormalization()
    MAX_1 = max(torch.max(X_train[...,1]).item(), torch.max(X_test[...,1]).item(), torch.max(X_val[...,1]).item())
    MIN_1 = min(torch.min(X_train[...,1]).item(), torch.min(X_test[...,1]).item(), torch.min(X_val[...,1]).item())
    my_scaler_channel_1.fit(np.array([MIN_1, MAX_1]))
    X_train[...,0] = my_scaler_channel_0.transform(X_train[...,0].reshape(-1,1)).reshape(X_train[...,0].shape)
    X_test[...,0] = my_scaler_channel_0.transform(X_test[...,0].reshape(-1,1)).reshape(X_test[...,0].shape)
    X_val[...,0] = my_scaler_channel_0.transform(X_val[...,0].reshape(-1,1)).reshape(X_val[...,0].shape)
    X_train[...,1] = my_scaler_channel_1.transform(X_train[...,1].reshape(-1,1)).reshape(X_train[...,1].shape)
    X_test[...,1] = my_scaler_channel_1.transform(X_test[...,1].reshape(-1,1)).reshape(X_test[...,1].shape)
    X_val[...,1] = my_scaler_channel_1.transform(X_val[...,1].reshape(-1,1)).reshape(X_val[...,1].shape)
    X_train_period = torch.tensor(np.load("/home/yyf/private/Global Fire Prediction/dataset/static_feat_train.npy").astype(np.float32)).reshape(-1,12,20,20,26)
    X_test_period = torch.tensor(np.load("/home/yyf/private/Global Fire Prediction/dataset/static_feat_test.npy").astype(np.float32)).reshape(-1,20,20,26)
    X_val_period = torch.tensor(np.load("/home/yyf/private/Global Fire Prediction/dataset/static_feat_test.npy").astype(np.float32)).reshape(-1,20,20,26)
    X_test_period = sliding_window(X_test_period,args.seq_len) 
    X_val_period = sliding_window(X_val_period,args.seq_len) 
    X_train = X_train.permute(0,4,1,2,3)
    X_val = X_val.permute(0,4,1,2,3)
    X_test = X_test.permute(0,4,1,2,3)
    X_train_period = X_train_period.permute(0,4,1,2,3)
    X_val_period = X_val_period.permute(0,4,1,2,3)
    X_test_period = X_test_period.permute(0,4,1,2,3)
    data = [[X_train[i], X_train_ts[i], X_train_period[i]] for i in range(X_train.shape[0])]
    test_data = [[X_test[i], X_test_ts[i], X_test_period[i]] for i in range(X_test.shape[0])]
    val_data = [[X_val[i], X_val_ts[i], X_val_period[i]] for i in range(X_val.shape[0])]

    if args.mode == 'few-shot':
        data = data[:int(len(data)*args.few_ratio)]

    if H + W < 32:
        batch_size = args.batch_size_1
    elif H + W < 48:
        batch_size = args.batch_size_2
    elif H + W < 64:
        batch_size = args.batch_size_3
    data = th.utils.data.DataLoader(data, num_workers=4, batch_size=batch_size, shuffle=True) 
    test_data = th.utils.data.DataLoader(test_data, num_workers=4, batch_size = 4 * batch_size, shuffle=False)
    val_data = th.utils.data.DataLoader(val_data, num_workers=4, batch_size = 4 * batch_size, shuffle=False)
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

