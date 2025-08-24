import torch
from tqdm import tqdm
import os
import sys
from models.model import SimpleClassifier
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

    train_dataloader, valid_dataloader, scaler = create_dataloader(file_path = data_path, batch_size=batch_size)
    print('model initialization...')
    model = SimpleClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10000], gamma=0.1)
    
    if prev_path != 0:
        checkpoint = torch.load(prev_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    total_params = sum(p.numel() for p in tqdm(model.parameters(), desc='caculating trainable parameters...') if p.requires_grad)
    print(f'###### total trainable parameter: {total_params}({(total_params/1000):.3f}K) ######')

    best_accuracy = 0.0
    for epoch in range(epoch_nums):
        model.train()
        running_loss = 0.0
        
        for batch in train_dataloader:
            batch_x = batch['src'].to(device)
            batch_y = batch['tgt'].to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # 每个epoch结束后在测试集上评估
        model.eval()
        with torch.no_grad():
            accuracy = 0.0
            count = 0
            for batch in valid_dataloader:
                X_test_tensor = batch['src'].to(device)
                y_test_tensor = batch['tgt'].to(device).unsqueeze(1)
                test_outputs = model(X_test_tensor)
                predicted = (test_outputs > 0.5).float()
                accuracy += (predicted == y_test_tensor).sum()
                count += len(predicted)
            accuracy /= count

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # 保存最佳模型
                torch.save(model.state_dict(), 'best_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epoch_nums}], Loss: {running_loss/len(train_dataloader):.4f}, Accuracy: {accuracy:.4f}')
    
    print(f'Training completed. Best accuracy: {best_accuracy:.4f}')
    
    # 保存预处理器和模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler
    }, 'model_and_scaler.pth')
    

if __name__ == '__main__':
    train("./data/sample_data.csv", 0, 16, 0.023, 1000, './weights')
