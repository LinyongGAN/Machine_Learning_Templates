from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
                "src": torch.tensor(src, dtype=torch.float32),
                "tgt": torch.tensor(tgt, dtype=torch.float32)
            }

def create_dataloader(file_path, test_size=0.2, random_state=42, batch_size=32):
    """
    加载并预处理数据
    """
    # 读取数据
    data = pd.read_csv(file_path)
    
    # 分离特征和标签
    X = data[['feature1', 'feature2']].values
    y = data['label'].values
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataloaders = DataLoader(CustomDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_dataloaders = DataLoader(CustomDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    return train_dataloaders, test_dataloaders, scaler
