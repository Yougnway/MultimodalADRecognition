import csv
import torch
import librosa
import random
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Processor


class audio_dataset(Dataset):
    def __init__(self, csv_file, train=True):
        super().__init__()
        with open(csv_file) as f:
            self.items = list(csv.reader(f))
        self.train = train
        # facebook/wav2vec2-large-xlsr-53
        # facebook/wav2vec2-base-960h
        # patrickvonplaten/wav2vec2-base-v2
        # facebook/wav2vec2-large-lv60
        # jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        path = path.replace('.log', '.wav')
        path = path.replace('Scripts_Continuous', 'audio')
        data, sr = librosa.load(path, sr=16000)
        input_values = self.process(data, sr, max_len=160000)
        label = torch.tensor(int(label))
        return input_values, label

    def process(self, data, sr, max_len):
        N = data.shape[0]
        K = 5
        gap = (N-max_len)//K
        audio_data = torch.zeros([K, 160000])
        for i in range(K):
            audio_data[i] = self.func(data[gap*i:gap*i+max_len], sr, max_len)
        return audio_data

    def func(self, data, sr, max_len):
        ret = self.processor(data,
                             sampling_rate=sr,
                             max_length=max_len,
                             padding='max_length',
                             truncation=True,
                             return_tensors="pt")
        ret = ret.input_values.squeeze()
        return ret


def get_audioloader(csv_file, bs=8, nw=2, shuffle=True, train=True):
    dataset = audio_dataset(csv_file=csv_file, train=train)
    # if train:
    loader = DataLoader(dataset,
                        batch_size=bs,
                        num_workers=nw,
                        shuffle=shuffle)
    # else:
    #     loader = DataLoader(dataset,
    #                         batch_size=1,
    #                         num_workers=1,
    #                         shuffle=shuffle,
    #                         collate_fn=dataset.test_collate)
    return loader


if __name__ == '__main__':
    dataloader = get_audioloader('nfoldsplits/train_0.csv', train=False)
    for idx, (inputs, label) in enumerate(dataloader):
        print(inputs.shape, label)
        break