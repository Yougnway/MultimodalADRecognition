import os
import torch
import librosa
import numpy as np
import random
from Nets.WavClassifier import WavClassifier
from transformers import Wav2Vec2Processor
from transformers import BertTokenizer
from Nets.BertClassifier import BertClassifier
from Nets.global_clip_model import Globa_Clip


class WavFeatureExtracter(object):
    def __init__(self, long=True):
        super().__init__()
        self.long = long
        self.processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
        self.model = WavClassifier(3)
        self.init_model('save_models/short_task/random3/wav2vec.pth')
        self.model.eval()
    
    def init_model(self, param_path):
        self.model.load_state_dict(torch.load(param_path))
    
    def draw_feature(self, file_path, save_path):
        max_len = 320000 if self.long else 96000
        wav_data = self.audio_func(file_path, max_len=max_len)
        with torch.no_grad():
            features = self.model.extract_embeding(wav_data)
            features = features.mean(dim=0)
        features = features.detach().cpu().numpy()
        np.save(save_path, features)
    
    def audio_func(self, file_path, max_len):
        data, sr = librosa.load(file_path, sr=16000)
        N = data.shape[0]
        if self.long:
            M = 3
            gap = (N-max_len) // M
            ret = torch.zeros((M, max_len))
            for i in range(M):
                ret[i] = self.wav_process(data[i*gap:i*gap+max_len], sr, max_len).squeeze()
        else:
            ret = self.wav_process(data, sr, max_len)
        return ret
    
    def wav_process(self, data, sr, max_len):
        ret = self.processor(data,
                             sampling_rate=sr,
                             max_length=max_len,
                             padding='max_length',
                             truncation=True,
                             return_tensors="pt")
        return ret.input_values


class ScriptFeatureExtracter(object):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.model = BertClassifier(3)
        self.init_model('save_models/short_task/random3/robert.pth')
        self.model.eval()
    
    def init_model(self, param_path):
        self.model.load_state_dict(torch.load(param_path))
    
    def draw_feature(self, file_path, save_path):
        ids, mask = self.text_func(file_path)
        with torch.no_grad():
            features = self.model.extract_embeding(ids, mask)
            features = features.mean(dim=0)
        features = features.detach().cpu().numpy()
        np.save(save_path, features)
        
    
    def text_func(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
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
        input_ids = encoded_dict.input_ids
        attention_mask = encoded_dict.attention_mask
        return input_ids, attention_mask


textExtracter = ScriptFeatureExtracter()
transcript_root = "Data/short_scripts"
transcript_save_root = "Data/short_pretrain/random3/script"
for root, dirs, files in os.walk(transcript_root):
    if files != []:
        # isTest = root.split("/")[-1] == 'test'
        # if isTest:
        #     continue
        for fn in files:
            fn_path = os.path.join(root, fn)
            print(fn_path)
            save_path = fn_path.replace(transcript_root, transcript_save_root)
            save_path = save_path.replace('.log', '.text.npy')
            # if root.split("/")[-1] == 'test_short':
            #     save_path = save_path.replace('test_short', 'test')
            textExtracter.draw_feature(fn_path, save_path)

wavExtracter = WavFeatureExtracter(long=False)
wav_root = "Data/short_audio"
wav_save_root = "Data/short_pretrain/random3/wav"
for root, dirs, files in os.walk(wav_root):
    if files != []:
        # isTest = root.split("/")[-1] == 'test'
        # if isTest:
        #     continue
        for fn in files:
            fn_path = os.path.join(root, fn)
            print(fn_path)
            save_path = fn_path.replace(wav_root, wav_save_root)
            save_path = save_path.replace('.wav', '.wav.npy')
            # if root.split("/")[-1] == 'test_short':
            #     save_path = save_path.replace('test_short', 'test')
            wavExtracter.draw_feature(fn_path, save_path)
