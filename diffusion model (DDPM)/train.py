import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from process import get_mnist_dataloader, create_folder_if_not_exists
from model import SimpleUnet, precompute_diffusion_vars, q_sample

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
batch_size = 128
learning_rate = 1e-3
num_epochs = 20
timesteps = 1000  # 扩散时间步数

def train():
    # 获取数据加载器
    train_loader = get_mnist_dataloader(batch_size)
    
    # 预先计算扩散变量
    diffusion_vars = precompute_diffusion_vars(timesteps, device)
    
    # 初始化模型
    model = SimpleUnet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            
            # 随机选择时间步
            t = torch.randint(0, timesteps, (images.shape[0],), device=device).long()
            
            # 生成噪声
            noise = torch.randn_like(images)
            
            # 前向扩散过程
            noisy_images = q_sample(images, t, diffusion_vars, noise)
            
            # 预测噪声 - 直接传递时间步t，让模型内部处理时间嵌入
            predicted_noise = model(noisy_images, t)
            
            # 计算损失
            loss = F.mse_loss(predicted_noise, noise)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # 保存模型
    create_folder_if_not_exists('./weight')
    torch.save(model.state_dict(), './weight/diffusion_model.pth')
    print("Model saved as diffusion_model.pth")
    
    return model, diffusion_vars

if __name__ == "__main__":
    train()