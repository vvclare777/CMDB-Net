"""
局部感知路径
    核心任务：捕捉高频细节、边缘纹理、处理高不确定性（高方差）区域
    主导特征：CNN特征（擅长局部归纳偏置）
    关键算子：高通滤波、局部方差计算、梯度检测
全局结构路径
    核心任务：建模低频背景、语义一致性、处理低不确定性（平坦/稳定）区域
    主导特征：Mamba特征（擅长长序列建模）
    关键算子：低通滤波、全局平均池化、通道注意力
最后通过动态门控生成器，根据两条路径产出的“置信度图”来决定最终融合权重
"""
import sys
sys.path.append('/mnt/e/Github/CMDB-Net')
import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalPath(nn.Module):
    """局部细节感知路径，关注高频信息(High-Freq)、高不确定性区域(High-Variance/Edge)、CNN特征"""
    def __init__(self, in_channels):
        super(LocalPath, self).__init__()
        
        # 梯度感知 (Sobel算子是经典的边缘检测算子)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)  # 水平方向梯度检测核，对垂直边缘敏感
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)  # 垂直方向梯度检测核，对水平边缘敏感
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # 局部特征编码器
        self.local_encoder = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 1, in_channels // 2, 3, padding=1),  # 输入: 原始特征 + 高频特征 + 梯度图
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),  # 输出局部重要性图
            nn.Sigmoid()
        )
        
    def get_high_freq(self, x):  # 增强边缘和纹理信息，抑制平滑区域
        """拉普拉斯高频提取"""
        # 使用高斯模糊的残差作为高频
        blurred = F.avg_pool2d(x, 3, 1, 1)  # 3×3平均池化，相当于一个简单的低通滤波器
        return x - blurred  # 原始信号 - 低频信号 = 高频信号

    def get_gradient_mag(self, x):
        """计算特征图梯度的平均幅值"""
        x_mean = torch.mean(x, dim=1, keepdim=True)  # 在通道维度取平均，将多通道特征压缩为单通道
        g_x = F.conv2d(x_mean, self.sobel_x, padding=1)  # 计算水平梯度
        g_y = F.conv2d(x_mean, self.sobel_y, padding=1)  # 计算垂直梯度
        return torch.sqrt(g_x**2 + g_y**2 + 1e-8)  # 欧几里得距离公式

    def forward(self, x):
        # 提取高频特征 (纹理细节)
        high_freq = self.get_high_freq(x)
        
        # 提取梯度信息 (显式边缘)
        grad_map = self.get_gradient_mag(x)
        
        # 编码生成局部重要性权重
        # 拼接: 原始特征 + 高频部分 + 梯度图
        combined = torch.cat([x, high_freq, grad_map], dim=1)
        local_guidance = self.local_encoder(combined)
        
        return local_guidance, high_freq
    
class GlobalPath(nn.Module):
    """全局结构建模路径，关注低频信息(Low-Freq)、低不确定性区域(Stable/Flat)、语义一致性、Mamba特征"""
    def __init__(self, in_channels):
        super(GlobalPath, self).__init__()
        
        # 全局通道上下文 (SE-Block思想)
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )
        
        # 低频特征编码器
        self.global_encoder = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, 3, padding=1), # 输入: 原始特征 + 低频特征
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 3, padding=1), # 使用较大感受野
            nn.Sigmoid()
        )
        
    def get_low_freq(self, x):
        """自适应平均池化模拟低通滤波"""
        # 使用大核 (5x5) 的平均池化，平滑掉细节，只保留大体结构
        return F.avg_pool2d(x, 5, 1, 2)

    def forward(self, x):
        # 提取低频特征 (整体结构)
        low_freq = self.get_low_freq(x)
        
        # 计算通道注意力 (语义重要性)
        channel_weight = self.global_context(x).unsqueeze(-1).unsqueeze(-1)
        x_weighted = x * channel_weight
        
        # 编码生成全局重要性权重
        # 拼接: 通道加权特征 + 低频部分
        combined = torch.cat([x_weighted, low_freq], dim=1)
        global_guidance = self.global_encoder(combined)
        
        return global_guidance, low_freq
    
class DualPathGatedFusion(nn.Module):
    """双路径门控融合模块"""
    def __init__(self, cnn_channels, mamba_channels, out_channels):
        super(DualPathGatedFusion, self).__init__()
        
        # 特征对齐
        self.align_cnn = nn.Conv2d(cnn_channels, out_channels, 1)
        self.align_mamba = nn.Conv2d(mamba_channels, out_channels, 1)
        self.bn_cnn = nn.BatchNorm2d(out_channels)
        self.bn_mamba = nn.BatchNorm2d(out_channels)
        
        # 双路径定义
        self.local_path = LocalPath(out_channels)
        self.global_path = GlobalPath(out_channels)
        
        # 接收来自两条路径的Guidance Map，生成最终融合门控
        self.gate_fusion = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), # 输入: Local Guidance + Global Guidance
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 1), # 输出: [CNN_Gate, Mamba_Gate]
            nn.Softmax(dim=1)    # 竞争机制
        )
        
        # 特征融合后处理
        self.feature_mix = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 卷积用于调节残差
        self.shortcut = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, cnn_feat, mamba_feat):
        # 对齐与归一化
        cnn_feat = self.bn_cnn(self.align_cnn(cnn_feat))
        mamba_feat = self.bn_mamba(self.align_mamba(mamba_feat))
        
        # 双路径处理
        # local_guidance指示哪里是边缘/细节，cnn_high_freq为CNN的高频分量
        local_guidance, cnn_high_freq = self.local_path(cnn_feat)
        # global_guidance指示哪里是平坦区/背景，mamba_low_freq为Mamba的低频分量
        global_guidance, mamba_low_freq = self.global_path(mamba_feat)
        
        # 生成竞争门控，将局部和全局的guidance拼接，让网络学习如何分配权重
        guidance_stack = torch.cat([local_guidance, global_guidance], dim=1)
        gates = self.gate_fusion(guidance_stack)
        
        gate_cnn = gates[:, 0:1, :, :]   # 偏向局部/边缘的权重
        gate_mamba = gates[:, 1:2, :, :] # 偏向全局/结构的权重
        
        # 基础融合: 根据门控加权，软选择
        fused_base = gate_cnn * cnn_feat + gate_mamba * mamba_feat
        
        # 在基础融合之上，显式地注入CNN的高频细节和Mamba的低频结构，频域补偿
        fused_enhanced = fused_base + 0.1 * cnn_high_freq + 0.1 * mamba_low_freq
        
        # 后处理，特征精炼
        out = self.feature_mix(fused_enhanced)
        
        return out, {
            'gate_cnn': gate_cnn,
            'gate_mamba': gate_mamba,
            'local_guidance': local_guidance,
            'global_guidance': global_guidance
        }