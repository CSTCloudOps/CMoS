from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
import os


def read_sets(sets):
    array_list = []
    for one_set in sets:
        datas = np.load("../data/{}.npz".format(one_set))
        array_names = datas.files
        for array_name in array_names:
            array_list.append(datas[array_name])
        
    return array_list
        
class MTSDataset(Dataset):
    def __init__(self, train_sets, seq_len, pred_len, phase="train", train_prop=0.7, valid_prop=0.1):
        super().__init__()
        self.data = read_sets(train_sets)
        self.data = np.stack(self.data, axis=1)

        self.train_data = self.data[:int(train_prop * len(self.data))]
        
        self.scaler = StandardScaler()
        self.scaler.fit(self.train_data)
        self.data = self.scaler.transform(self.data)
        
        if phase == "train":
            self.data = self.data[:int(train_prop * len(self.data))]
        elif phase == "valid":
            self.data = self.data[int(train_prop * len(self.data)) - seq_len: int((train_prop + valid_prop) * len(self.data))]
        elif phase == "test":
            self.data = self.data[int((train_prop + valid_prop) * len(self.data)) - seq_len:]
        else:
            raise NotImplementedError("Wrong phase.")
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index):
        return self.data[index: index + self.seq_len], self.data[index + self.seq_len: index + self.seq_len + self.pred_len]
