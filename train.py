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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(data_path, prev_path, batch_size, learning_rate, epoch_nums, weights_path):
    best_metrics = [1 for i in range(3)]
    records = pd.DataFrame([], columns=["epoch num", "training loss", "validate loss"])

    train_dataloader, valid_dataloader = create_dataloader(root = data_path, batch_size = batch_size)
    print('model initialization...')
    model = PoTSTransformer(d_input = 2, d_model = 8, nhead = 1, num_encoder_layers = 1,
                            num_decoder_layers = 1, dim_feedforward = 8, dropout_p = 0.1,
                            layer_norm_eps = 1e-05).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10000], gamma=0.1)
    
    if prev_path != 0:
        checkpoint = torch.load(prev_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    total_params = sum(p.numel() for p in tqdm(model.parameters(), desc='caculating trainable parameters...') if p.requires_grad)
    print(f'###### total trainable parameter: {total_params}({(total_params/1000):.3f}K) ######')

    for epoch_cur in tqdm(range(epoch_nums)):

        running_loss = 0.0
        for batch in train_dataloader:
            X_in = batch["src"].float().to(device)
            #将目标数据的前n-1个时间步作为输入
            Y_in = batch["tgt"][:,:-1].float().to(device)
            #将目标数据的后n-1个时间步作为输出
            Y_out = batch["tgt"][:,1:].float().to(device)


            X_in = torch.permute(X_in,(1,0,2))
            Y_in = torch.permute(Y_in,(1,0,2))


            sequence_length = Y_in.size(0)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(sequence_length)

            
            Y_pred = model(X_in,Y_in, tgt_mask = tgt_mask)


            Y_out = Y_out.permute(1,0,2)
            Y_pred = torch.squeeze(Y_pred)

            loss = loss_fn(Y_pred,Y_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            scheduler.step()

        if epoch_cur % 50 == 0:
            print(f'   Epoch {epoch_cur} training loss {running_loss} (lr={optimizer.param_groups[0]["lr"]})')

        if epoch_cur % 100 == 0 or epoch_cur == epoch_nums-1:
            create_folder_if_not_exists(weights_path)
            torch.save({'model_state_dict': model.state_dict(),}, Path(weights_path) / f'{epoch_cur+1}_statedict_{running_loss}.pth')
        # best_metrics = [epoch_cur+1, running_loss, 0]
        # if epoch_cur % 10 == 9:
        #     records.to_csv("./records.csv", index=False)
    

if __name__ == '__main__':
    train(None, 0, 16, 0.023, 1000, './weights')
