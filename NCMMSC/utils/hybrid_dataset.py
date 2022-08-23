import csv
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

FUSIONS = ['add', 'cat']

class hybrid_dataset(Dataset):
    def __init__(self, csv_file, features, feature_tails, fusion, norms, old_dir, feature_dirs):
        super().__init__()
        self.features = features
        self.feature_tails = feature_tails
        self.old_dir = old_dir
        self.feature_dirs = feature_dirs
        self.fusion = fusion
        self.norms = norms
        assert self.fusion in FUSIONS, "Unknown fusion method!"
        with open(csv_file, 'r') as f:
            self.items = list(csv.reader(f))
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        path, label = self.items[idx]
        # traverse all the pathes
        feats = []
        for ind in range(len(self.features)):
            feat_path = path.replace(self.old_dir, self.feature_dirs[ind])
            feat_path  = feat_path.replace(".log", self.feature_tails[ind])
            feat_type = self.feature_tails[ind].split(".")[-2]
            feat = np.load(feat_path)
            if feat_type in self.norms.keys():
                norm = np.load(self.norms[feat_type])
                feat = (feat - norm[0]) / (norm[1] + 1e-9)
            # feat = torch.from_numpy(feat).float()
            feats.append(feat)
        if self.fusion == 'add':
            feats = np.stack(feats, axis=0)  # (n, 192)
            feats = np.mean(feats, axis=0)   # 192
            data = torch.from_numpy(feats).float()
        elif self.fusion == 'cat':
            feats = np.concatenate(feats, axis=0) # nx192
            data = torch.from_numpy(feats).float()

        label = torch.tensor(int(label))
        
        return data, label


def build_hybrid_dataloader(cfg):
    num_workers = cfg.DATA.NUM_WORKERS
    batch_size = cfg.DATA.BATCH_SIZES
    train_csv = cfg.DATA.CSV_TRAIN_FILE
    val_csv = cfg.DATA.CSV_VAL_FILE
    test_csv = cfg.DATA.CSV_TEST_FILE
    features = cfg.DATA.FEATURES
    old_dir = cfg.DATA.OLD_DIR
    feature_dirs = cfg.DATA.FEAT_DIRS
    feature_tails = cfg.DATA.TAILS
    fusion = cfg.DATA.FUSION

    is10 = cfg.DATA.IS10_NORM
    egemap = cfg.DATA.EGEMAPS_NORM
    compare = cfg.DATA.COMPARE_NORM
    norms = {'IS10_paraling': is10, 'eGeMAPS': egemap, 'ComParE_2016': compare}
    # train val test
    dataloader = {}
    if train_csv != '':
        train_set = hybrid_dataset(train_csv, features, feature_tails, fusion, norms, old_dir, feature_dirs)
        train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
        dataloader['train_loader'] = train_loader
    if val_csv != '':
        val_set = hybrid_dataset(val_csv, features, feature_tails, fusion, norms, old_dir, feature_dirs)
        val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=num_workers)
        dataloader['val_loader'] = val_loader
    if test_csv != '':
        test_set = hybrid_dataset(test_csv, features, feature_tails, fusion, norms, old_dir, feature_dirs)
        test_loader = DataLoader(test_set, batch_size, shuffle=False, num_workers=num_workers)
        dataloader['test_loader'] = test_loader
    return dataloader
