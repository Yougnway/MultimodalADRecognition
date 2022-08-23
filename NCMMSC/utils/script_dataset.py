import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


class cls_data(Dataset):
    def __init__(self, csv_file):
        super().__init__()
        with open(csv_file, 'r') as f:
            self.items = list(csv.reader(f))
        # bert-base-chinese
        # hfl/chinese-roberta-wwm-ext
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        path, label = self.items[idx]
        f = open(path, 'r', encoding='utf-8')
        # print(f.readlines(), len(f.readlines()))
        txt = f.readlines()[0]  # get txt information
        encoded_dict = self.tokenizer.encode_plus(
            txt,
            add_special_tokens=True,
            max_length=256,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoded_dict['input_ids'].squeeze()
        attention_mask = encoded_dict['attention_mask'].squeeze()
        label = torch.tensor(int(label))
        
        return input_ids, attention_mask, label


class cls_data_short(Dataset):
    def __init__(self, csv_file):
        super().__init__()
        with open(csv_file, 'r') as f:
            self.items = list(csv.reader(f))
        # bert-base-chinese
        # hfl/chinese-roberta-wwm-ext
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        path, label = self.items[idx]
        with open(path, 'r', encoding='utf-8') as f:
        # print(f.readlines(), len(f.readlines()))
            txt = f.readlines()[0]  # get txt information
        encoded_dict = self.tokenizer.encode_plus(
            txt,
            add_special_tokens=True,
            max_length=48,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoded_dict['input_ids'].squeeze()
        attention_mask = encoded_dict['attention_mask'].squeeze()
        label = torch.tensor(int(label))
        
        return input_ids, attention_mask, label

def get_dataloader(csv_file, bs=4, nw=2, shuffle=True):
    dataset = cls_data_short(csv_file=csv_file)
    loader = DataLoader(dataset, batch_size=bs, num_workers=nw, shuffle=shuffle)
    return loader


if __name__ == '__main__':
    dataloader = get_dataloader('CSV_Files/train.csv')
    for i, input in enumerate(dataloader):
        print(input[0])
        print(input[1])
        print(input[2])
        break
