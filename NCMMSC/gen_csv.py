import os
import csv


# 多折存在 pandas 多方便啊！
root_path = ["Data/Scripts/train/", "Data/Scripts/test/"]
save_fn = ["CSV_Files_v2/train.csv", "CSV_Files_v2/test.csv"]


# relabel
label2num = {'HC': 0, 'MCI': 1, 'AD': 2}

for idx in range(2):
    root = root_path[idx]
    fns = os.listdir(root)
    fns.sort()
    f = open(save_fn[idx], 'w', newline='', encoding='utf-8')
    writer = csv.writer(f)
    for fn in fns:
        label = label2num[fn.split('_')[0]]
        writer.writerow([root+fn, str(label)])
