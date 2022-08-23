import os
import torch
import librosa
import numpy as np
from Nets.WavLMClassfier import WavClassifier
from transformers import Wav2Vec2Processor
from transformers import BertTokenizer
from Nets.BertClassifier import BertClassifier
from Nets.global_clip_model import Globa_Clip


class WavFeatureExtracter(object):
    def __init__(self, fold=0):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = WavClassifier()
        self.init_model('save_models/audio/audio-'+str(fold)+'.pth')
        self.model.eval()
    
    def init_model(self, param_path):
        self.model.load_state_dict(torch.load(param_path))
    
    def draw_feature(self, file_path, save_path):
        wav_data = self.audio_func(file_path, max_len=160000)
        wav_data = wav_data.unsqueeze(dim=0)
        with torch.no_grad():
            features = self.model.extract_embeding(wav_data)
            features = features.mean(dim=0)
        features = features.detach().cpu().numpy()
        np.save(save_path, features)
    
    def audio_func(self, file_path, max_len):
        data, sr = librosa.load(file_path, sr=16000)
        N = data.shape[0]
        M = 5
        gap = (N-max_len) // M
        # M can be regarded as batch size
        ret = torch.zeros((M, max_len))
        for i in range(M):
            ret[i] = self.wav_process(data[i*gap:i*gap+max_len], sr, max_len)
        return ret
    
    def wav_process(self, data, sr, max_len):
        ret = self.processor(data,
                             sampling_rate=sr,
                             max_length=max_len,
                             padding='max_length',
                             truncation=True,
                             return_tensors="pt")
        return ret.input_values.squeeze()


class ScriptFeatureExtracter(object):
    def __init__(self, fold=0):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.model = BertClassifier()
        self.init_model('save_models/bert/bert-'+str(fold)+'.pth')
        self.model.eval()
    
    def init_model(self, param_path):
        self.model.load_state_dict(torch.load(param_path))
    
    def draw_feature(self, file_path, save_path):
        ids, mask = self.text_func(file_path)
        features = self.model.extract_embeding(ids, mask)
        features = features.mean(dim=0)
        features = features.detach().cpu().numpy()
        np.save(save_path, features)
    
    def text_func(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            txt = f.readlines()[1][:-1]  # get txt information
        encoded_dict = self.tokenizer.encode_plus(
            txt,
            add_special_tokens=True,
            max_length=256,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoded_dict.input_ids
        attention_mask = encoded_dict.attention_mask
        return input_ids, attention_mask

fold_n = 0
for i in range(10):
    fold_n = i
    textExtracter = ScriptFeatureExtracter(fold=fold_n)
    transcript_root = "Data/Scripts_Continuous"
    transcript_save_root = "Data/pretrain/script/fold_"+str(fold_n)
    os.makedirs(transcript_save_root)
    for root, dirs, files in os.walk(transcript_root):
        if files != []:
            for fn in files:
                fn_path = os.path.join(root, fn)
                print(fn_path)
                save_path = fn_path.replace(transcript_root, transcript_save_root)
                save_path = save_path.replace('.log', '.text.npy')
                textExtracter.draw_feature(fn_path, save_path)

    wavExtracter = WavFeatureExtracter(fold=fold_n)
    wav_root = "Data/Audios"
    wav_save_root = "Data/pretrain/wav/fold_"+str(fold_n)
    os.makedirs(wav_save_root)
    for root, dirs, files in os.walk(wav_root):
        if files != []:
            for fn in files:
                fn_path = os.path.join(root, fn)
                print(fn_path)
                save_path = fn_path.replace(wav_root, wav_save_root)
                save_path = save_path.replace('.wav', '.wav.npy')
                wavExtracter.draw_feature(fn_path, save_path)
