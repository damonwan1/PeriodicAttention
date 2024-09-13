import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks


# 可学习的小波变换类
def morlet_wavelet_filter(center_freq, bandwidth, size):
    t = np.linspace(-size // 2, size // 2, size)
    wavelet = np.exp(1j * 2 * np.pi * center_freq * t) * np.exp(-t**2 / (2 * bandwidth**2))
    wavelet_real = np.real(wavelet)
    wavelet_imag = np.imag(wavelet)
    return wavelet_real, wavelet_imag

class LearnableWaveletTransform(nn.Module):
    def __init__(self, seq_len, num_scales=96, init_center_freq=1.0, init_bandwidth=1.5):
        super(LearnableWaveletTransform, self).__init__()
        self.seq_len = seq_len
        self.center_freq = nn.Parameter(torch.tensor(init_center_freq))
        self.bandwidth = nn.Parameter(torch.tensor(init_bandwidth))
        
        # 用Morlet滤波来初始化学习
        wavelet_real, wavelet_imag = morlet_wavelet_filter(1.0, 1.5, seq_len)
        self.h_filter = nn.Parameter(torch.tensor(wavelet_real[:num_scales], dtype=torch.float32))
        self.l_filter = nn.Parameter(torch.tensor(wavelet_imag[:num_scales], dtype=torch.float32))

        self.h_fn = nn.Conv1d(1, 1, num_scales, bias=False, padding=num_scales//2)
        self.l_fn = nn.Conv1d(1, 1, num_scales, bias=False, padding=num_scales//2)

        self.h_fn.weight = nn.Parameter(self.create_W(num_scales, False))
        self.l_fn.weight = nn.Parameter(self.create_W(num_scales, True))

    def forward(self, x):
        B, C, T = x.shape
        h_outs = []
        l_outs = []
        for c in range(C):
            h_out = self.h_fn(x[:, c:c+1, :])
            l_out = self.l_fn(x[:, c:c+1, :])
            h_outs.append(h_out)
            l_outs.append(l_out)
        h_outs = torch.cat(h_outs, dim=1)
        l_outs = torch.cat(l_outs, dim=1)
        return l_outs, h_outs

    def create_W(self, P, is_l):
        if is_l:
            filter_list = self.l_filter
        else:
            filter_list = self.h_filter

        max_epsilon = torch.min(torch.abs(filter_list))
        weight_np = torch.randn(1, 1, P) * 0.1 * max_epsilon

        for i in range(P):
            filter_index = 0
            for j in range(i - len(filter_list) // 2, P):
                if j < 0:
                    filter_index += 1
                elif filter_index < len(filter_list):
                    weight_np[0, 0, j] = filter_list[filter_index]
                    filter_index += 1
        return nn.Parameter(weight_np)

# 解决无梯度的问题，并且使用高低频中最大能量当做主导周期
class PeriodicityExtractor(nn.Module):
    def __init__(self, seq_len=100, num_scales=48):
        super(PeriodicityExtractor, self).__init__()
        self.scales = torch.arange(1, num_scales + 1, dtype=torch.float32)  # 示例尺度
        self.wavelet_transform = LearnableWaveletTransform(seq_len)
        self.fc = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        l_out, h_out = self.wavelet_transform(x)
        low_freq_power = l_out**2
        high_freq_power = h_out**2
        
#         # 合并低频和高频的功率信息
#         combined_power = torch.cat((low_freq_power, high_freq_power), dim=2)  # 形状为 [B, C, 2*T]
        
#         # 使用加权平均的方法来计算主导周期
#         B, C, T = combined_power.shape
#         scales = torch.arange(1, T + 1, device=combined_power.device, dtype=torch.float32)
#         weights = combined_power / torch.sum(combined_power, dim=2, keepdim=True)
#         dominant_periods = torch.sum(weights * scales, dim=2)
        # 确保 combined_power 的形状为 [B, C, 2 * output_length]
        combined_power = torch.cat((low_freq_power, high_freq_power), dim=2)
        output_length = combined_power.size(2)

        # 动态计算尺度的范围
        scales = torch.arange(1, output_length // 2 + 1, dtype=torch.float32).to(x.device)

        # 查找功率最大的三个点及其索引
        topk_values, topk_indices = torch.topk(combined_power, k=2, dim=2)

        # 打印 topk_indices 检查问题
        # print(f"topk_indices: {topk_indices.shape}")

        # 限制索引在尺度范围内
        topk_indices = topk_indices % scales.size(0)
        # print(f"topk_indices: {topk_indices.shape}")

        dominant_scales = topk_indices.float()
        # print(f"dominant_scales: {dominant_scales.shape}")
        
        # 使用网络将尺度转换为周期
        dominant_periods = self.fc(dominant_scales.unsqueeze(-1)).squeeze(-1)
        # print(f"dominant_periods: {dominant_periods.shape}")

        
        # 使用加权平均的方法来计算主导周期
        B, C, T = combined_power.shape
        weights = combined_power / torch.sum(combined_power, dim=2, keepdim=True)
         # 扩展 scales 一倍
        scales = torch.cat((scales, scales), dim=0)
        # print(scales.shape)
        # 查找功率最大的三个点及其索引
        avg_periods = torch.sum(weights * scales, dim=2)
        avg_period = avg_periods.unsqueeze(2)  # 形状为 [B, C, 1]
        
        # return avg_period
        
        dominant_periods = torch.cat((dominant_periods, avg_period), dim=2)
        return dominant_periods # 形状为 [B, C, 1]