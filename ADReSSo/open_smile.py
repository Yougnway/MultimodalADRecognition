import os
import csv
import numpy as np
import argparse


parser = argparse.ArgumentParser(description="opensmile feature extractor")
parser.add_argument("--features", default="ComParE_2016", type=str,
                    help='types: ComParE_2016, eGeMAPS, IS10_paraling')

parser.add_argument("--split_type", default="train", type=str,
                    help='types: train, test')

args = parser.parse_args()

OPENSMILE_PATH = "/home/yangtao/Datasets/NCMMSC2021/ncmmsc2021-lab410/opensmile-3.0.0/build/progsrc/smilextract/SMILExtract"
'''
特征配置文件:
    config/compare16/ComParE_2016.conf
    config/egemaps/v01b/eGeMAPSv01b.conf
    config/is09-13/IS10_paraling.conf

'''
if args.features == 'ComParE_2016':
    fn = 'compare16/ComParE_2016.conf'
    sfn = "ComParE_2016/"
elif args.features == 'eGeMAPS':
    fn = 'egemaps/v01b/eGeMAPSv01b.conf'
    sfn = "eGeMAPS/"
elif args.features == 'IS10_paraling':
    fn = 'is09-13/IS10_paraling.conf'
    sfn = "IS10_paraling/"
elif args.features == 'IS09_emotion':
    fn = 'is09-13/IS09_emotion.conf'
    sfn = "IS09_emotion/"

CONFIG_PATH = "/home/yangtao/Datasets/NCMMSC2021/ncmmsc2021-lab410/opensmile-3.0.0/config/"+fn
SAVE_ROOT = "Data/open_smile/" + sfn + '/'

SMILE_CMD_HEAD = OPENSMILE_PATH + " -C " + CONFIG_PATH
if args.split_type == 'train':
    WAVE_ROOT = "Data/Audios/"
else:  # 测试集没有进行去噪处理！没有标签
    WAVE_ROOT = "Data/Noisy/test/"


# cmd format: SMILE_CMD_HEAD + " -I " + "example.wav" + " -O " + "filename.csv"
def Extract_From_Wav(wavfile, savepath, instname):
    cmd = SMILE_CMD_HEAD + " -I " + wavfile + " -O " + savepath + " -instname " + instname
    print(cmd)
    os.system(cmd)


def Walk_Folders(wave_root, save_root, feature_type):
    mfiles = os.listdir(wave_root)
    for mfile in mfiles:
        wave_file = wave_root + mfile
        save_path = save_root + mfile.replace('wav', feature_type) + '.npy'
        # 每次追加在最后一行
        Extract_From_Wav(wave_file, "Data/open_smile/temp_v1.csv", 'temp_v1.csv')
        f = open("Data/open_smile/temp_v1.csv", 'r')
        df = list(csv.reader(f))[-1]
        feature_vec = np.array(df[1:-1], np.double)
        np.save(save_path, feature_vec)
        print(feature_vec.shape, ' ', feature_vec.dtype)



def Print_SCV(csv_path):
    f = open(csv_path, 'r')
    df = list(csv.reader(f))[-1]
    feature_vec = np.array(df[1:-1])
    print(feature_vec.shape)


if os.path.exists("Data/open_smile/temp_v1.csv"):
    os.remove("Data/open_smile/temp_v1.csv")
Walk_Folders(WAVE_ROOT, SAVE_ROOT, args.features)
# Print_SCV("../../data/opensmile-features/AD_F_030807_001.csv")
# os.remove("ADReSSo_3D/hello.csv")