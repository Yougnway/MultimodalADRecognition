import torch
from transformers import BertModel
from transformers import AlbertModel
import torch.nn as nn


class BertClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # roberta-base
        # bert-base-uncased
        self.model = BertModel.from_pretrained("bert-base-uncased")
        # self.model = AlbertModel.from_pretrained("albert-base-v2")
        self.embedding = nn.Sequential(
            nn.Linear(768, 192),
        )
        self.activation = nn.GELU()
        self.classifier = nn.Linear(in_features=192, out_features=num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids, attention_mask=attention_mask)
        out = out.pooler_output
        out = self.activation(self.embedding(out))
        out = self.classifier(out)
        return out
    
    def extract_embeding(self, input_ids, attention_mask):
        out = self.model(input_ids, attention_mask=attention_mask)
        out = out.pooler_output
        out = self.activation(self.embedding(out))
        return out



if __name__ == "__main__":
    bert = BertClassifier(3)
    ids = torch.randint(0, 1000, (2, 256))
    mask = torch.ones((2, 256))
    print(bert(ids, mask))