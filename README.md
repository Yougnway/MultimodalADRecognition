# Introduction

steps:

- fine-tune BERT `python Bert.py`
- fine-tune Wav2Vec2 `python Wav2vec.py`
- extract pretrain features `python pretrain_feature.py`
- train and test svm classifier `python ml_classifier.py`

Beacuse we have extacted pretrain features, you can only run `python ml_classifier.py` to evaluate the model.