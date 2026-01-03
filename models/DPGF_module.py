import sys
sys.path.append('/mnt/e/Github/CMDB-Net')
import torch.nn as nn
import torch.nn.functional as F

class DualPathGatedFusion(nn.Module):
    """双路径门控融合模块"""
    def __init__(self, channels):
        super(DualPathGatedFusion, self).__init__()
        self.channels = channels
        
        # 局部纹理复杂度路径：通道方差 -> 单通道纹理图 -> 归一化权重
        self.local_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, bias=False),  # 将方差图归一化
            nn.Sigmoid()  # 输出[0,1]的局部权重
        )
        
        # 全局场景类别路径：GAP -> MLP -> 空间权重模板
        self.global_mlp = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()  # 输出[0,1]的通道权重
        )
        
        # 通道权重 -> 空间权重的投影
        self.channel_to_spatial = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
    def compute_local_weight(self, F_local):
        """计算局部纹理复杂度权重"""
        # 计算通道维度方差，方差大的位置表示通道间差异大，即纹理复杂
        mean = F_local.mean(dim=1, keepdim=True)  # [B, 1, H, W]，计算每个空间位置在通道维度上的均值
        variance = ((F_local - mean) ** 2).mean(dim=1, keepdim=True)  # [B, 1, H, W]，计算每个通道值与均值的平方差，再求平均得到方差
        
        # 将方差图通过1×1卷积和Sigmoid激活，输出值在[0,1]之间，高值对应纹理复杂区域
        W_local = self.local_conv(variance)  # [B, 1, H, W]
        
        return W_local
    
    def compute_global_weight(self, F_global):
        """计算全局场景类别权重"""
        B, C, H, W = F_global.shape
        
        # 全局平均池化，将特征图压缩为通道向量 [B, C]，保留全局场景信息，丢失空间信息
        gap = F.adaptive_avg_pool2d(F_global, 1).view(B, C)  # [B, C]
        
        # MLP，通道降维 → ReLU激活 → 通道升维，学习不同通道的重要性权重，Sigmoid确保权重在[0,1]
        channel_weight = self.global_mlp(gap).view(B, C, 1, 1)  # [B, C, 1, 1]
        
        # 应用通道权重
        weighted_feature = F_global * channel_weight  # [B, C, H, W]
        
        # 通过1×1卷积将通道特征投影到单通道，生成空间权重图，识别语义关键区域
        W_global = self.channel_to_spatial(weighted_feature)  # [B, 1, H, W]
        
        return W_global
    
    def forward(self, F_local, F_global):
        """前向传播"""
        # W_local: 基于纹理复杂度的局部权重
        # W_global: 基于场景重要性的全局权重
        W_local = self.compute_local_weight(F_local)     # [B, 1, H, W]
        W_global = self.compute_global_weight(F_global)  # [B, 1, H, W]
        
        # 生成门控图，同时满足"纹理复杂"和"场景关键"的位置获得高值
        Gate_map = W_local * W_global  # [B, 1, H, W]
        
        # 自适应融合，门控值高 → 更多使用局部特征 F_local，门控值低 → 更多使用全局特征 F_global
        F_fused = Gate_map * F_local + (1 - Gate_map) * F_global  # [B, C, H, W]
        
        return F_fused, Gate_map