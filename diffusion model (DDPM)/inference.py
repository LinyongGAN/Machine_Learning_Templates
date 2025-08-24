import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import SimpleUnet, precompute_diffusion_vars

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
timesteps = 1000  # 扩散时间步数

def p_sample(model, x, t, t_index, diffusion_vars):
    betas_t = diffusion_vars['betas'][t].reshape(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = diffusion_vars['sqrt_one_minus_alphas_cumprod'][t].reshape(-1, 1, 1, 1)
    sqrt_recip_alphas_t = diffusion_vars['sqrt_recip_alphas'][t].reshape(-1, 1, 1, 1)
    
    # 使用模型预测噪声 - 直接传递时间步t
    predicted_noise = model(x, t)
    
    # 计算均值
    model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
    
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = diffusion_vars['posterior_variance'][t].reshape(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

def p_sample_loop(model, shape, diffusion_vars):
    # 从纯噪声开始
    img = torch.randn(shape, device=device)
    imgs = []
    
    for i in tqdm(reversed(range(0, timesteps)), desc='Sampling', total=timesteps):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i, diffusion_vars)
        # 使用detach()将张量从计算图中分离，然后转换为numpy数组
        imgs.append(img.cpu().detach().numpy())
    return imgs

def generate_images(model_path=None, num_images=16):
    # 预先计算扩散变量
    diffusion_vars = precompute_diffusion_vars(timesteps, device)
    
    # 加载模型
    model = SimpleUnet().to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 生成图像
    sample_shape = (num_images, 1, 28, 28)
    
    # 在推理时使用torch.no_grad()上下文管理器
    with torch.no_grad():
        generated_images = p_sample_loop(model, sample_shape, diffusion_vars)
    
    # 显示生成的图像
    fig, axes = plt.subplots(1, num_images, figsize=(20, 2))
    for i, ax in enumerate(axes):
        ax.imshow(generated_images[-1][i, 0], cmap='gray')
        ax.axis('off')
    plt.savefig('generated_images.png')
    plt.show()
    
    return generated_images

if __name__ == "__main__":
    generate_images('./weight/diffusion_model.pth')