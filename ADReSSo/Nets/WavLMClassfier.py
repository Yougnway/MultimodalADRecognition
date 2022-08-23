import torch
from transformers import Wav2Vec2Model, WavLMModel
import torch.nn as nn


class WavClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # facebook/wav2vec2-large-xlsr-53
        # facebook/wav2vec2-base-960h
        # facebook/wav2vec2-large-lv60
        # jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn
        self.model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base-960h")
        # print(self.model)
        # for p in self.model.parameters():
        #     p.requires_grad = False
        self.embedding = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Linear(256, 192),
        )
        self.activation = nn.GELU()
        self.classifier = nn.Sequential(
            nn.Linear(192, 32),
            nn.GELU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, audio):
        B, N, _ = audio.shape
        feature = []
        for i in range(N):
            out = self.model(audio[:, i])[0].mean(dim=1)
            out = self.activation(self.embedding(out))
            feature.append(out)
        feature = torch.stack(feature, dim=1)
        feature = feature.mean(dim=1)
        out = self.classifier(feature)

        return out
    
    def extract_embeding(self, audio):
        B, N, _ = audio.shape
        feature = []
        for i in range(N):
            out = self.model(audio[:, i])[0].mean(dim=1)
            out = self.activation(self.embedding(out))
            feature.append(out)
        feature = torch.stack(feature, dim=1)
        feature = feature.mean(dim=1)
        return feature




if __name__ == "__main__":
    bert = WavClassifier(2)
    audios = torch.rand((2, 4800))
    # ids = torch.randint(0, 1000, (2, 256))
    # mask = torch.ones((2, 256))
    print(bert(audios))