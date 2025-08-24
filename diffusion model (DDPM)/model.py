import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 定义beta调度
def linear_beta_schedule(timesteps, device):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps).to(device)

# 预先计算扩散过程所需的变量
def precompute_diffusion_vars(timesteps, device):
    betas = linear_beta_schedule(timesteps, device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0) # alpha bar
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # alpha bar t-1
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas) # squard root of 1/alpha
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod) # squard root of alpha bar
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod) # squard root of (1 - alpha bar)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'posterior_variance': posterior_variance
    }

# 将数据扩散到第t步
def q_sample(x_start, t, diffusion_vars, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = diffusion_vars['sqrt_alphas_cumprod'][t].reshape(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = diffusion_vars['sqrt_one_minus_alphas_cumprod'][t].reshape(-1, 1, 1, 1)
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# 简化U-Net模型，确保尺寸匹配
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 确保x1和x2的尺寸相同
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# 简化的时间嵌入方法
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim * 4)
        self.act = nn.SiLU()

    def forward(self, t):
        # 将时间步转换为正弦位置编码
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        
        # 重塑t以便广播
        t = t.float().reshape(-1, 1)
        emb = emb.reshape(1, -1)
        
        emb = t * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
            
        # 通过MLP处理
        emb = self.linear1(emb)
        emb = self.act(emb)
        emb = self.linear2(emb)
        return emb

class SimpleUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(32)
        
        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # 时间嵌入投影
        self.time_proj1 = nn.Linear(32 * 4, 128)
        self.time_proj2 = nn.Linear(32 * 4, 256)
        self.time_proj3 = nn.Linear(32 * 4, 512)
        self.time_proj4 = nn.Linear(32 * 4, 512)
        
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x, t):
        """
        x: 输入图像 (B, 1, 28, 28)
        t: 时间步 (B,)
        output: 预测的噪声 (B, 1, 28, 28)
        """
        # 时间嵌入
        t_emb = self.time_embedding(t)
        
        # 下采样路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # 添加时间信息到每个特征图
        t_emb1 = self.time_proj1(t_emb).view(-1, 128, 1, 1)
        t_emb2 = self.time_proj2(t_emb).view(-1, 256, 1, 1)
        t_emb3 = self.time_proj3(t_emb).view(-1, 512, 1, 1)
        t_emb4 = self.time_proj4(t_emb).view(-1, 512, 1, 1)
        
        x2 = x2 + t_emb1
        x3 = x3 + t_emb2
        x4 = x4 + t_emb3 + t_emb4
        
        # 上采样路径
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        return self.outc(x)