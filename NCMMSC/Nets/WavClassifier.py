import torch
from transformers import Wav2Vec2Model, Wav2Vec2ForSequenceClassification
import torch.nn as nn


class WavClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # samitizerxu/wav2vec2-xls-r-300m-zh-CN
        # jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn
        # facebook/wav2vec2-base
        # -960h
        # ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt
        self.wav2vec = Wav2Vec2Model.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
        self.embedding = nn.Linear(1024, 192)
        self.activation = nn.GELU()
        self.classifier = nn.Linear(192, num_classes)

    def forward(self, audio):
        feature = self.wav2vec(audio)[0].mean(dim=1)  # B, 768
        out = self.activation(self.embedding(feature))
        out = self.classifier(out)
        return out
    
    def extract_embeding(self, audio):
        feature = self.wav2vec(audio)[0].mean(dim=1)
        out = self.embedding(feature)
        return out


if __name__ == "__main__":
    bert = WavClassifier(3)
    audios = torch.rand((2, 4800))
    # ids = torch.randint(0, 1000, (2, 256))
    # mask = torch.ones((2, 256))
    print(bert(audios))