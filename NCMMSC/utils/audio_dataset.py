import csv
import torch
import librosa
import random
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Processor


class audio_dataset(Dataset):
    def __init__(self, csv_file, train):
        super().__init__()
        self.train = train
        with open(csv_file) as f:
            self.items = list(csv.reader(f))
            # facebook/wav2vec2-base
            # -960h
            # jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn
            # ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt
        self.processor = Wav2Vec2Processor.from_pretrained(
            "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        path = path.replace('.log', '.wav')
        path = path.replace('long_scripts', 'long_audio')
        data, sr = librosa.load(path, sr=16000)
        input_values = self.process(data, sr=sr, max_len=320000)
        label = torch.tensor(int(label))
        return input_values, label

    def test_collate(self, batch):
        assert len(batch) == 1, "Please, set test set batch to 1!"
        return batch[0][0], batch[0][1].unsqueeze(0).repeat(3)

    def process(self, data, sr, max_len):
        # 分为前中后三段，训练时随机取段，测试时三段综合
        if self.train:
            return self.get_train(data, sr, max_len)
        else:
            return self.get_test(data, sr, max_len)
    
    def get_train(self, data, sr, max_len):
        # random select
        N = data.shape[0]
        idx = random.randint(0, N-max_len-1)
        ret = self.func(data[idx:idx+max_len], sr, max_len)
        return ret
    
    def get_test(self, data, sr, max_len):
        N = data.shape[0]
        M = 3
        gap = (N-max_len) // M
        ret = torch.zeros((M, max_len))
        for i in range(M):
            ret[i] = self.func(data[i*gap:i*gap+max_len], sr, max_len)
        return ret

    def func(self, data, sr, max_len):
        ret = self.processor(data,
                             sampling_rate=sr,
                             max_length=max_len,
                             padding='max_length',
                             truncation=True,
                             return_tensors="pt")
        ret = ret.input_values.squeeze()
        return ret


class audio_dataset_short(Dataset):
    def __init__(self, csv_file, train):
        super().__init__()
        self.train = train
        with open(csv_file) as f:
            self.items = list(csv.reader(f))
            # facebook/wav2vec2-base
            # -960h
            # jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn
            # ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt
        self.processor = Wav2Vec2Processor.from_pretrained(
            "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        path = path.replace('.log', '.wav')
        path = path.replace('short_scripts', 'short_audio')
        data, sr = librosa.load(path, sr=16000)
        input_values = self.process(data, sr=sr, max_len=96000)
        label = torch.tensor(int(label))
        return input_values, label

    def process(self, data, sr, max_len):
        # random select
        N = data.shape[0]
        idx = random.randint(0, N-max_len)
        ret = self.func(data[idx:idx+max_len], sr, max_len)
        return ret

    def func(self, data, sr, max_len):
        ret = self.processor(data,
                             sampling_rate=sr,
                             max_length=max_len,
                             padding='max_length',
                             truncation=True,
                             return_tensors="pt")
        ret = ret.input_values.squeeze()
        return ret


def get_audioloader(csv_file, bs=4, nw=2, shuffle=True, train=True):
    dataset = audio_dataset_short(csv_file=csv_file, train=train)
    # if train:
    loader = DataLoader(dataset,
                        batch_size=bs,
                        num_workers=nw,
                        shuffle=shuffle)
    # else:
    #     loader = DataLoader(dataset,
    #                         batch_size=1,
    #                         num_workers=0,
    #                         shuffle=shuffle,
    #                         collate_fn=dataset.test_collate)
    return loader


if __name__ == '__main__':
    dataloader = get_audioloader('CSV_Files/train.csv', train=False)
    for idx, (inputs, label) in enumerate(dataloader):
        print(inputs.shape, label)
        break