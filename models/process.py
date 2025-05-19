from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split
from collections import Counter
import torch
import numpy as np
import os
from torch.autograd import Variable
import pandas as pd
import csv
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 创建成功")
def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask==0))
    return np_mask


def read_file(src_path):
    text_list = []
    with open(src_path, 'r', encoding='utf-8') as reader:
        for line in reader:
            text_list.append(line.strip().replace('\n',''))
    return text_list
class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        waveform, dB, label = self.data[index]
        return {
                "waveform": waveform,
                "dB" : dB, 
                "label": label
            }

def data_standard(path):
    df = pd.read_csv(path).T
    dB = df.index
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(df)
    data_standardized_df = pd.DataFrame(data_standardized, columns=df.columns)
    data = torch.tensor(data_standardized_df.values.astype(np.float32))
    dB = torch.tensor([float(str_val) for str_val in dB])
    return data, dB

def create_dataloader(root, batch_size=16):
    normal_path = Path(root) / "0.csv"
    abnormal_path = Path(root) / "1.csv"
    normal_data,normal_dB = data_standard(normal_path)
    abnormal_data, abnormal_dB = data_standard(abnormal_path)
    
    raw_list_normal, raw_list_abnormal = [], []
    for data, dB in zip(normal_data, normal_dB):
        raw_list_normal.append((data.unsqueeze(-2), dB, 0))
    for data, dB in zip(abnormal_data, abnormal_dB):
        raw_list_abnormal.append((data.unsqueeze(-2), dB, 1))

    num_abnormal, num_normal = len(raw_list_abnormal), len(raw_list_normal)
    num_train_abnormal, num_train_normal = int(num_abnormal*0.9), int(num_normal*0.9)
    num_valid_abnormal, num_valid_normal = num_abnormal-num_train_abnormal, num_normal-num_train_normal

    train_dataset = CustomDataset(raw_list_normal[:num_train_normal]+raw_list_abnormal[:num_train_abnormal])
    val_dataset = CustomDataset(raw_list_normal[-num_valid_normal:]+raw_list_abnormal[-num_valid_abnormal:])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle = True)

    print("-"*20+"Dataset report"+"-"*20)
    print(f"train_normal: {num_train_normal}, train_abnormal: {num_train_abnormal}, train_tot: {len(train_dataset)}")
    assert num_train_abnormal+num_train_normal == len(train_dataset)
    print(f"valid_normal: {num_valid_normal}, valid_abnormal: {num_valid_abnormal}, valid_tot: {len(val_dataset)}")
    assert num_valid_abnormal+num_valid_normal == len(val_dataset)
    print(f"tot_normal: {num_train_normal+num_valid_normal}, tot_abnormal: {num_train_abnormal+num_valid_abnormal}, total: {len(train_dataset)+len(val_dataset)}")
    assert num_train_abnormal+num_valid_abnormal == num_abnormal and num_train_normal + num_valid_normal == num_normal
    print('-'*54)
    return train_dataloader, valid_dataloader
