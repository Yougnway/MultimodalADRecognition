import os
import csv
from random import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class smile_data(Dataset):
    def __init__(self, file, type):
        super().__init__()
        self.type = type
        with open(file, 'r') as f:
            csvreader = csv.reader(f)
            self.data = list(csvreader)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        path, label = self.data[index]
        gemaps_path = path.replace('Scripts', 'open_smile/'+self.type)
        egemaps_path = gemaps_path.replace('.log', '.'+self.type+'.npy')
        egemaps = np.load(egemaps_path)  # (88,)
        egemaps = self.standard(egemaps)
        # egemaps = np.expand_dims(egemaps, axis=0)
        egemaps = torch.from_numpy(egemaps).float()

        label = torch.tensor(int(label))

        return egemaps, label
    
    def standard(self, data):
        eps = 1e-06
        mean = np.mean(data)
        var = np.var(data)
        return (data - mean) / (var + eps)


def get_loader(csv_file, type, bs=16, nw=4, shuffle=True):
    dataset = smile_data(file=csv_file, type=type)
    loader = DataLoader(dataset, batch_size=bs, num_workers=nw, shuffle=shuffle)
    return loader

if __name__ == '__main__':
    dataset = get_loader("CSV_Files/train_cv.csv", 'IS10_paraling')
    for i, (inputs, labels) in enumerate(dataset):
        print(i, inputs.shape, labels)
    