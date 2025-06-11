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
import random

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
    def __init__(self, srcs, tgts):
        self.srcs = srcs
        self.tgts = tgts

    def __len__(self):
        assert len(self.srcs) == len(self.tgts)
        return len(self.srcs)

    def __getitem__(self, index):
        src = self.srcs[index]
        tgt = self.tgts[index]
        return {
                "src": src,
                "tgt": tgt
            }

def create_dataloader(root=None, batch_size=16):
    
    total_points = 300
    N_tr = 16000
    x_in_len = int(0.8 * total_points)
    y_out_len = int(total_points - x_in_len)
    srcs = np.empty((N_tr,x_in_len,2))
    tgts = np.empty((N_tr,y_out_len,2))

    for ind in range(N_tr):
        total_loc = []
        current_lat = random.randint(25,50)
        current_lon = random.randint(25,50)
        delta_lat = random.uniform(0.1,0.2)
        delta_lon = random.uniform(0.1,0.2)

        while len(total_loc) < total_points:
            total_loc.append([current_lon, current_lat])

            current_lat += delta_lat

            if current_lon > 179:
                current_lon = -179
            current_lon += delta_lon

        
        total_loc = np.array(total_loc)
        srcs[ind:] = total_loc[0:x_in_len]
        tgts[ind,:] = total_loc[x_in_len:total_points]

    train_dataset = CustomDataset(srcs, tgts)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, train_dataloader
