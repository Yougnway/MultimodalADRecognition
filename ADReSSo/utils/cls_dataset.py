import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from transformers import AlbertTokenizer


class cls_data(Dataset):
    def __init__(self, csv_file):
        super().__init__()
        with open(csv_file, 'r') as f:
            self.items = list(csv.reader(f))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        path, label = self.items[idx]
        path_txt = path.replace("Scripts_Continuous", "Scripts_Continuous")
        with open(path_txt, 'r', encoding='utf-8') as f:
            txt = f.readlines()[0]  # get txt information and remove \n
        encoded_dict = self.tokenizer.encode_plus(
            txt,
            add_special_tokens=True,
            max_length=256,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        label = int(label)
        input_ids = encoded_dict['input_ids'].squeeze()
        attention_mask = encoded_dict['attention_mask'].squeeze()
        label = torch.tensor(label)
        
        return txt, input_ids, attention_mask, label


def get_dataloader(csv_file, bs=8, nw=4, shuffle=True):
    dataset = cls_data(csv_file=csv_file)
    loader = DataLoader(dataset, batch_size=bs, num_workers=nw, shuffle=shuffle)
    return loader


if __name__ == '__main__':
    dataloader = get_dataloader('nfoldsplits/train_0.csv')
    for i, input in enumerate(dataloader):
        print(input[0])
        print(input[1])
        print(input[2])
        print(input[3])
        break
