import torch
from transformers import BertModel, BertForSequenceClassification
import torch.nn as nn


# class BertClassifier(nn.Module):
#     def __init__(self, num_classes=3):
#         super().__init__()
#         self.model_bert = BertForSequenceClassification.from_pretrained(
#             "hfl/chinese-roberta-wwm-ext-large", # Use the 12-layer BERT model, with an uncased vocab.
#             num_labels = num_classes      # The number of output labels--2 for binary classification.  
#         )

#     def forward(self, input_ids, attention_mask):
#         # out = self.model(input_ids, attention_mask=attention_mask)
#         # out = out.pooler_output
#         # out = self.classifier(out)
#         out = self.model_bert(input_ids, attention_mask=attention_mask).logits
#         return out


class BertClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # hfl/chinese-bert-wwm-ext
        # hfl/chinese-roberta-wwm-ext
        # bert-base-chinese
        self.model = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
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