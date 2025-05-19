import torch
from tqdm import tqdm
import os
import sys
from models.model import Model
sys.path.append(os.path.abspath(os.path.dirname('__file__')))
from models.process import create_dataloader, create_folder_if_not_exists
from sklearn.metrics import roc_curve, accuracy_score, f1_score
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import numpy as np
from pathlib import Path
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_eer(labels, scores):
    # 添加输入校验
    if len(labels) == 0 or len(scores) == 0:
        return float('nan'), None
    
    # 检查标签多样性
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        return float('nan'), None
    
    # 添加分数有效性检查
    if np.all(np.isnan(scores)) or np.all(scores == scores[0]):
        return float('nan'), None
    
    try:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1 - tpr
        min_index = np.nanargmin(np.absolute((fnr - fpr)))
        eer_threshold = thresholds[min_index]
        eer = (fpr[min_index] + fnr[min_index]) / 2
        return float(eer), float(eer_threshold)
    except Exception as e:
        print(f"EER calculation failed: {str(e)}")
        return float('nan'), None

def train(data_path, prev_epoch, batch_size, learning_rate, epoch_nums, weights_path):
    best_metrics = [1 for i in range(9)]
    records = pd.DataFrame([], columns=["epoch num", "training loss", "EER", "threshold", "Accuracy", "FAR", "FRR", "F1", "AUROC", "FAR@TAR95%", "validate loss"])

    train_dataloader, valid_dataloader = create_dataloader(root = data_path, batch_size = batch_size)
    print('model initialization...')
    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.98, 0.9))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=True)
    if prev_epoch != 0:
        checkpoint = torch.load(f"/root/2025_BME/weights/{prev_epoch}_statedict.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    total_params = sum(p.numel() for p in tqdm(model.parameters(), desc='caculating trainable parameters...') if p.requires_grad)
    print(f'###### total trainable parameter: {total_params}({(total_params/1000000000):.3f}B) ######')
    for epoch_cur in range(prev_epoch, epoch_nums):
        model.train()
        epoch_loss = 0
        step = 0
        for batch in tqdm(train_dataloader, desc=f'[{epoch_cur+1}/{epoch_nums}] training...'):
            src = batch['waveform'].to(device)
            tgt = batch['label'].to(device)
            pred = model(src)
            #print(tgt.shape, pred.shape)
            reshape_tgt = tgt.view(-1)
            reshape_pred = pred.view(len(reshape_tgt), -1)
            #print(reshape_pred.shape)
            loss = F.cross_entropy(reshape_pred, reshape_tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
        epoch_loss /= step
        print(f'[{epoch_cur+1}/{epoch_nums}], train loss: {epoch_loss}, saving {epoch_cur+1} state dict...')

        all_deepfake_labels = []
        all_deepfake_scores = []

        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            valid_step = 0
            for batch in tqdm(valid_dataloader):
                waveform = batch["waveform"].to(device)
                label = batch["label"].to(device)
                pred = model(waveform)

                reshape_label = label.view(-1)
                reshape_pred = pred.view(len(reshape_label), -1)
                v_loss = F.cross_entropy(reshape_pred, reshape_label)
                valid_loss += v_loss.item()
                valid_step += 1

                pred = pred.float().softmax(dim=-1)[:, 1].contiguous()
                all_deepfake_scores.extend(pred.detach().cpu().numpy().astype(np.float64).tolist())
                all_deepfake_labels.extend(label.detach().cpu().numpy().astype(np.int64).tolist())

            #print(all_deepfake_labels[:10], all_deepfake_scores[:10])

            final_eer, final_eer_threshold = compute_eer(np.array(all_deepfake_labels), np.array(all_deepfake_scores))
            final_accuracy = accuracy_score(all_deepfake_labels, (np.array(all_deepfake_scores) > 0.5).astype(int))
            final_f1 = f1_score(all_deepfake_labels, (np.array(all_deepfake_scores) > 0.5).astype(int))

            # 计算 AUROC 和 TAR@FAR=95%
            fpr, tpr, thresholds = roc_curve(all_deepfake_labels, all_deepfake_scores)
            auroc = np.trapz(tpr, fpr)

            # 计算TAR@FAR=95%的FAR
            target_tar = 0.95
            idx = np.argmin(np.abs(tpr - target_tar))
            far_at_tar95 = fpr[idx]

            # Calculate FAR and FRR even if we only have one class
            preds = (np.array(all_deepfake_scores) > 0.5).astype(int)
            false_accepts = np.sum((preds == 1) & (np.array(all_deepfake_labels) == 0))  # False positives (negative samples predicted as positive)
            false_rejects = np.sum((preds == 0) & (np.array(all_deepfake_labels) == 1))  # False negatives (positive samples predicted as negative)
            total_neg = np.sum(np.array(all_deepfake_labels) == 0)  # Total negative samples
            total_pos = np.sum(np.array(all_deepfake_labels) == 1)  # Total positive samples

            far = false_accepts / total_neg if total_neg > 0 else float('nan')
            frr = false_rejects / total_pos if total_pos > 0 else float('nan')
            print('-'*50)
            if not np.isnan(final_eer):
                print(f'Final Equal Error Rate (EER): {final_eer:.4f} at threshold {final_eer_threshold:.4f}')
            else:
                print('Final EER could not be computed due to insufficient classes.')
            print(f'Final Accuracy: {final_accuracy:.4f}')
            print(f'Final FAR: {far:.4f}' if total_neg > 0 else 'Final FAR could not be computed due to lack of negative samples.')
            print(f'Final FRR: {frr:.4f}' if total_pos > 0 else 'Final FRR could not be computed due to lack of positive samples.')
            print(f'Final F1: {final_f1:.4f}')
            print(f'AUROC: {auroc:.4f}')
            print(f'FAR@TAR=95%: {far_at_tar95:.4f}')
            print('-' * 50)
            print(f"{epoch_cur+1}|{final_eer}|{final_eer_threshold}|{final_accuracy}|{far}|{frr}|{final_f1}|{auroc}|{far_at_tar95}|")
            records.loc[len(records)]=[epoch_cur+1, epoch_loss, final_eer, final_eer_threshold, final_accuracy, far, frr, final_f1, auroc, far_at_tar95, valid_loss/valid_step]
        
        scheduler.step(valid_loss/valid_step)
        print(f"validate loss: {valid_loss/valid_step}, lr: {optimizer.param_groups[0]['lr']}")

        if final_eer < best_metrics[1]:
            create_folder_if_not_exists(weights_path)
            torch.save({
                'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
            }, Path(weights_path) / f'{epoch_cur+1}_statedict_{final_eer}.pth')
            best_metrics = [epoch_cur+1, final_eer, final_eer_threshold, final_accuracy, far, frr, final_f1, auroc, far_at_tar95]
        if epoch_cur % 10 == 9:
            records.to_csv("./records.csv", index=False)
    

if __name__ == '__main__':
    train(Path("your_data_path"), 0, 16, 1e-6, 200, '/weights') # [TODO]
