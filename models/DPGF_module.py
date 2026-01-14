import sys
sys.path.append('/mnt/e/Github/CMDB-Net')
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearCrossModalAttention(nn.Module):
    """基于通道协方差的线性注意力"""
    def __init__(self, channels):
        super().__init__()
        self.scale = channels ** -0.5
        
        # 保持通道数不变，进行特征变换
        self.q_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)
        
        # 简单的层归一化，用于稳定训练
        self.ln = nn.LayerNorm(channels)

    def forward(self, query_feat, key_feat):
        B, C, H, W = query_feat.shape
        
        # [B, C, H, W] -> [B, C, HW]
        q = self.q_proj(query_feat).flatten(2)
        k = self.k_proj(key_feat).flatten(2)
        v = self.v_proj(key_feat).flatten(2)
        
        # 计算通道相关性图（Query的第i个通道与Key的第j个通道的相关性） [B, C, N] @ [B, N, C] -> [B, C, C]
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1) # 在通道维度归一化
        
        # 根据通道相关性，将Key对应的Value加权融合 [B, C, C] @ [B, C, N] -> [B, C, N]
        out = torch.bmm(attn, v)
        
        # 恢复形状并残差连接
        out = out.view(B, C, H, W)
        out = self.out_proj(out)
        
        # 残差连接 (原特征 + 跨模态补充特征)
        return query_feat + out

class LearnableMultiScaleFreqDecompose(nn.Module):
    """可学习的多尺度频域分解"""
    def __init__(self, channels):
        super().__init__()
        
        # 多尺度可学习低通滤波器
        # 模拟不同尺度的低通滤波器，大卷积核捕捉更大范围的平滑信息
        self.low_pass = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, k, padding=k//2, groups=channels),  # 深度可分离卷积，每个通道独立滤波
                nn.Conv2d(channels, channels, 1)  # 1×1卷积，用于通道混合和信息整合
            ) for k in [3, 5, 7]  # 分别使用3×3、5×5、7×7的卷积核
        ])
        
        # 尺度选择器，根据输入特征自动选择不同尺度的重要性
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，压缩空间维度 [B, C, H, W] -> [B, C, 1, 1]
            nn.Flatten(),  # 展平为向量形状 [B, C]
            nn.Linear(channels, 3),  # 全连接层，将通道特征映射到3个尺度权重上 [B, 3]
            nn.Softmax(dim=1)  # 归一化权重，使得和为1
        )

        # 可学习的高通滤波器
        # 可以自适应地提取图像的高频信息（边缘、纹理等）
        self.high_pass = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),  # 3x3深度可分离卷积
            nn.BatchNorm2d(channels),  # 归一化
            nn.ReLU(inplace=True),  # 激活，抑制随机噪声
            nn.Conv2d(channels, channels, 1)  # 1×1逐点卷积，将不同通道提取到的边缘信息进行线性组合
        )
        
    def forward(self, x):
        # 多尺度低频提取，对输入x分别应用三个尺度的低通滤波器
        low_freqs = [lp(x) for lp in self.low_pass]  # 列表low_freqs包含三个不同尺度的低频特征图
        
        # 自适应尺度选择
        scale_weights = self.scale_attention(x)  # [B, 3]
        
        # 将三个尺度的低频特征图按权重加权融合，得到最终的自适应低频成分
        low_freq = sum(w.view(-1,1,1,1) * lf for w, lf in zip(scale_weights.unbind(1), low_freqs))
        
        # 可学习高频 = 原始 - 自适应低频
        # x - low_freq是数学意义上的残差，混杂了有用的细节和无用的噪声
        high_freq = self.high_pass(x - low_freq)  # 增强边缘抑制噪声，将数学残差转换成语义高频特征
        
        return low_freq, high_freq, scale_weights  # 返回低频、高频及尺度权重
    
class UncertaintyAwareGating(nn.Module):
    """显式不确定性感知门控"""
    def __init__(self):
        super().__init__()
        
        # 不确定性校准，将计算出的物理方差校准为 0~1 的门控概率
        self.uncertainty_calibration = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),  # 输入通道为2（分别来自CNN和Mamba的方差图），输出16通道
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),  # 映射回1通道，生成最终的不确定性图
            nn.Sigmoid()  # 归一化到 [0, 1] 区间
        )
        
    def compute_local_variance(self, x, kernel_size=5):
        """计算局部方差作为不确定性度量 - 无参数的数学统计过程"""
        padding = kernel_size // 2
        
        # 局部均值 E[X]：利用平均池化模拟局部窗口内的均值
        mean = F.avg_pool2d(x, kernel_size, stride=1, padding=padding)
        
        # 局部方差 Var(X) = E[X^2] - (E[X])^2
        var = F.avg_pool2d(x**2, kernel_size, stride=1, padding=padding) - mean**2
        
        # 将所有通道的方差取平均，得到一张空间方差图 [B, 1, H, W]
        # 方差越大，代表该区域纹理越复杂或变化越剧烈（即不确定性高）
        return var.mean(dim=1, keepdim=True)
        
    def forward(self, cnn_feat, mamba_feat):
        # 空间不确定性，调用无参数的统计方法
        cnn_var = self.compute_local_variance(cnn_feat)      # [B, 1, H, W]
        mamba_var = self.compute_local_variance(mamba_feat)  # [B, 1, H, W]
        
        # 编码不确定性差异
        # 将两者的方差拼接，作为校准网络的输入
        var_diff = torch.cat([cnn_var, mamba_var], dim=1)  # [B, 2, H, W]
        
        # 通过极小的卷积网络融合两者信息，生成最终的不确定性图
        uncertainty_map = self.uncertainty_calibration(var_diff)
        
        # 返回映射图及原始方差
        return uncertainty_map, cnn_var, mamba_var
    
class DualPathGatedFusion(nn.Module):
    """双路径门控融合模块"""
    def __init__(self, cnn_channels, mamba_channels, out_channels):
        super(DualPathGatedFusion, self).__init__()
        
        # 特征对齐
        self.align_cnn = nn.Sequential(
            nn.Conv2d(cnn_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.align_mamba = nn.Sequential(
            nn.Conv2d(mamba_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        # 融合前线性跨模态交互
        self.cnn_ask_mamba = LinearCrossModalAttention(out_channels)  # CNN向Mamba查询全局上下文
        self.mamba_ask_cnn = LinearCrossModalAttention(out_channels)  # Mamba向CNN查询局部纹理细节

        # 可学习频域分解
        self.freq_decompose_cnn = LearnableMultiScaleFreqDecompose(out_channels)
        self.freq_decompose_mamba = LearnableMultiScaleFreqDecompose(out_channels)

        # 不确定性门控
        self.uncertainty_gate = UncertaintyAwareGating(out_channels)
        
        # 局部路径
        self.local_path = nn.Sequential(
            nn.Conv2d(out_channels * 2 + 1, out_channels // 2, 3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, 1, 1),
            nn.Sigmoid()
        )

        # 全局路径
        self.global_path = nn.Sequential(
            nn.Conv2d(out_channels * 2 + 1, out_channels // 2, 3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, 1, 1),
            nn.Sigmoid()
        )
        
        # 多级门控融合
        self.gate_fusion = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # local + global + uncertainty
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1),
            nn.Softmax(dim=1)
        )
        
        # 自适应频域补偿系数
        self.adaptive_freq_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels * 2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )

    def forward(self, cnn_feat, mamba_feat):
        # 特征对齐
        cnn_feat = self.align_cnn(cnn_feat)
        mamba_feat = self.align_mamba(mamba_feat)

        # 线性跨模态交互
        cnn_enhanced = self.cnn_ask_mamba(query_feat=cnn_feat, key_feat=mamba_feat)  # CNN特征增强
        mamba_enhanced = self.mamba_ask_cnn(query_feat=mamba_feat, key_feat=cnn_feat)  # Mamba特征增强

        # 可学习的频域分解
        _, cnn_high, cnn_scales = self.freq_decompose_cnn(cnn_enhanced)
        mamba_low, _, mamba_scales = self.freq_decompose_mamba(mamba_enhanced)

        # 不确定性估计（使用原始对齐特征，保留物理意义）
        uncertainty_map, _, _ = self.uncertainty_gate(cnn_feat, mamba_feat)
        
        # Local Path关注CNN增强特征 + CNN高频 + 不确定性区域
        local_input = torch.cat([cnn_enhanced, cnn_high, uncertainty_map], dim=1)
        local_guidance = self.local_path(local_input)

        # Global Path关注Mamba增强特征 + Mamba低频 + 确定性区域
        global_input = torch.cat([mamba_enhanced, mamba_low, 1 - uncertainty_map], dim=1)
        global_guidance = self.global_path(global_input)
        
        # 多级门控融合
        guidance_stack = torch.cat([
            local_guidance, 
            global_guidance, 
            uncertainty_map,
        ], dim=1)
        gates = self.gate_fusion(guidance_stack)
        
        gate_cnn = gates[:, 0:1, :, :]   # 偏向局部/边缘的权重
        gate_mamba = gates[:, 1:2, :, :] # 偏向全局/结构的权重
        
        # 基础加权融合
        fused_base = gate_cnn * cnn_enhanced + gate_mamba * mamba_enhanced

        # 自适应频域补偿
        freq_cat = torch.cat([cnn_high.mean(dim=[2,3]), mamba_low.mean(dim=[2,3])], dim=1)
        freq_weights = self.adaptive_freq_weight(freq_cat)  # [B, 2]
        w_high = freq_weights[:, 0:1].unsqueeze(-1).unsqueeze(-1)
        w_low = freq_weights[:, 1:2].unsqueeze(-1).unsqueeze(-1)
        
        fused_enhanced = fused_base + w_high * cnn_high + w_low * mamba_low
        
        return fused_enhanced, {
            'gate_cnn': gate_cnn,
            'gate_mamba': gate_mamba,
            'local_guidance': local_guidance,
            'global_guidance': global_guidance,
            'uncertainty_map': uncertainty_map,
            'freq_weights': freq_weights,
            'scale_weights_cnn': cnn_scales,
            'scale_weights_mamba': mamba_scales
        }