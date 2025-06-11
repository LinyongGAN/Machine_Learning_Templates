import torch
from tqdm import tqdm
import os
import sys
from models.model import PoTSTransformer
sys.path.append(os.path.abspath(os.path.dirname('__file__')))
from models.process import create_dataloader, create_folder_if_not_exists
from sklearn.metrics import roc_curve, accuracy_score, f1_score
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import os
import numpy as np
from pathlib import Path
import pandas as pd
import random
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(data_path, prev_path, batch_size):

    train_dataloader, valid_dataloader = create_dataloader(root = data_path, batch_size = batch_size)
    print('model initialization...')
    model = PoTSTransformer(d_input = 2, d_model = 8, nhead = 1, num_encoder_layers = 1,
                            num_decoder_layers = 1, dim_feedforward = 8, dropout_p = 0.1,
                            layer_norm_eps = 1e-05).to(device)

    checkpoint = torch.load(prev_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    total_points = 300
    x_in_len = int(0.8 * total_points)
    y_out_len = int(total_points - x_in_len)
    srcs = np.empty((1,x_in_len,2))
    tgts = np.empty((1,y_out_len,2))

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
    
    srcs[0,:] = total_loc[0:x_in_len]
    srcs = torch.from_numpy(srcs)
    tgts[0,:] = total_loc[x_in_len:total_points]
    tgts = torch.from_numpy(tgts)

    X_in = srcs[0,:].float().to(device)
    Y_in = tgts[0,0].float().to(device)
    Y_out = tgts[0,:].float().to(device)

    # Add dimension for feature
    X_in = torch.unsqueeze(X_in,1)
    #X_in = torch.permute(X_in,(1,0))
    Y_in = torch.unsqueeze(Y_in,0)
    Y_in = torch.unsqueeze(Y_in,1)
    Y_out = torch.unsqueeze(Y_out,1)

    y_pred = torch.empty(Y_out.shape)
    y_pred[0] = Y_in # or SOS token

    for te_ind in range(1,Y_out.shape[0]):
        #当前已预测的部分
        target_in = y_pred[:te_ind].to(device)

        sequence_length = target_in.size(0)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(sequence_length)
        print(X_in.shape, target_in.shape)
        pred = model(X_in, target_in, tgt_mask = tgt_mask).detach()

        y_pred[1:te_ind+1] = pred

    #随机选一个测试样本可视化
    X_start = 0
    X_end = srcs.size(1) 
    Y_start = 0
    Y_end = Y_out.size(0)
    t_X = range(X_start, X_end)
    t_Y = range(Y_start,Y_end)

    print(Y_out)
    print(y_pred)
    Y_out=Y_out.cpu()
    y_pred=y_pred.cpu()

    plt.plot(srcs[0,t_X,0], srcs[0,t_X,1],'b-',label='X')
    plt.plot(Y_out[t_Y,0,0], Y_out[t_Y,0,1],'g-',label='Real Y')
    plt.plot(y_pred[t_Y,0,0], y_pred[t_Y,0,1],'r--',label='Pred Y')
    plt.plot(y_pred[t_Y,0,0], y_pred[t_Y,0,1],'rx')
    plt.title("training set")
    plt.xlabel('lon')
    plt.ylabel('lat')
    plt.legend()
    plt.show()
    plt.savefig("fig.png")
    

if __name__ == '__main__':
    evaluate(None, 
          "/home/ganlinyong/CapacityLargeModel/our implementation/weights/1000_statedict_227.30150546133518.pth",
          16)