import csv
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# from sklearn.
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, recall_score
from sklearn.svm import SVC


def make_dataset(csv_file, type):
    with open(csv_file, 'r') as f:
        csvReader = csv.reader(f)
        csvList = list(csvReader)
    
    X, Y = [], []
    for fn, label in csvList:
        # another
        IS_path = fn.replace('long_scripts', 'long_opensmile/IS10_paraling')
        IS_path = IS_path.replace('.log', '.IS10_paraling.npy')
        IS10 = np.load(IS_path)   # N x 64
        # # # # text
        text_path = fn.replace('long_scripts', 'long_pretrain/random1/script')
        text_path = text_path.replace('.log', '.text.npy')
        text_feature = np.load(text_path)
        # # # wav
        wav_path = fn.replace('long_scripts', 'long_pretrain/random1/wav')
        wav_path = wav_path.replace('.log', '.wav.npy')
        wav_feature = np.load(wav_path)

        features = np.concatenate([text_feature, wav_feature, IS10])
        X.append(features)
        Y.append(int(label))
    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)
    return X, Y


def evaluate(targets, predictions):
    print("evaluate model...")
    performance = {
        'acc': accuracy_score(targets, predictions),
        'precision': precision_score(targets, predictions, average='macro'),
        'recall': recall_score(targets, predictions, average='macro'),
        'f1': f1_score(targets, predictions, average='macro'),
        'matrix': confusion_matrix(targets, predictions).tolist()}
    return performance


f_type = 'IS10_paraling'
# type: eGeMAPS IS10_paraling ComParE_2016
train_X, train_Y = make_dataset('long_csvs/train.csv', f_type)
train_X = StandardScaler().fit_transform(train_X)
# classifier = RandomForestClassifier()
classifier = SVC(kernel='rbf')
classifier.fit(train_X, train_Y)
train_score = classifier.score(train_X, train_Y)
print('train Accuracy: ',train_score)
test_X, test_Y = make_dataset('long_csvs/test.csv', f_type)
test_X = StandardScaler().fit_transform(test_X)
predict_validation = classifier.predict(test_X)
performance = evaluate(test_Y, predict_validation)
for (key, value) in performance.items():
    print("{}: {}".format(key, value))