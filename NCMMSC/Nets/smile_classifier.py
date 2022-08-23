import imp
import torch
import torch.nn as nn


class eGeMAPS_classfier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(88, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Linear(32, num_classes),
        )
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class ComParE_2016_classfier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base_channels = 16
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, base_channels, 3, 2, 1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2, 1)
        )
        # self.conv2 = nn.Sequential(
        #     nn.Conv1d(base_channels, base_channels, 3, 2, 1),
        #     nn.BatchNorm1d(base_channels),
        #     nn.ReLU(inplace=True)
        # )
        self.conv3 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels*2, 3, 2, 1),
            nn.BatchNorm1d(base_channels*2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2, 1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(base_channels*2, base_channels*4, 3, 2, 1),
            nn.BatchNorm1d(base_channels*4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2, 1),
            nn.AdaptiveAvgPool1d((1))
        )
        self.fc = nn.Conv1d(base_channels*4, num_classes, 1)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        # x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        x = x.view(x.shape[0], -1)
        return x


class IS10_paraling(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base_channels = 32
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, base_channels, 3, 2, 1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, 3, 2, 1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels*2, 3, 2, 1),
            nn.BatchNorm1d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(base_channels*2, base_channels*2, 3, 2, 1),
            nn.BatchNorm1d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(base_channels*2, base_channels*4, 3, 2, 1),
            nn.BatchNorm1d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        # self.conv6 = nn.Sequential(
        #     nn.Conv1d(base_channels*4, base_channels*4, 3, 1, 1),
        #     nn.BatchNorm1d(base_channels*4),
        #     nn.ReLU(inplace=True)
        # )
        self.fc = nn.Sequential(
            nn.Linear(base_channels*4*50, 256),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.conv6(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
