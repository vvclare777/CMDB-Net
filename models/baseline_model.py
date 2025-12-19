import sys
sys.path.append('/mnt/e/Github/CMDB-Net')
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn_branch import CNNBranch
from models.mamba_branch import MambaBranch

class BaselineModel(nn.Module):
    """CNN-Mamba双分支基线模型 (简单相加策略) """
    def __init__(self, num_classes, in_channels=3, pretrained=True):
        super(BaselineModel, self).__init__()
        self.num_classes = num_classes
        
        # CNN分支 - ResNet34
        self.cnn_branch = CNNBranch(
            in_channels=in_channels, 
            weights=pretrained,
            align_channels=True  # 对齐通道数
        )
        
        # Mamba分支 - VMamba-tiny
        self.mamba_branch = MambaBranch(
            pretrained=pretrained,
            in_channels=in_channels,
            use_cuda=torch.cuda.is_available()
        )
        
        # 获取融合后的通道数 [128, 256, 512, 512]
        self.fused_channels = self.cnn_branch.get_output_channels()
        
        # 解码器 - 逐步上采样
        self.decoder = Decoder(
            in_channels_list=self.fused_channels,
            num_classes=num_classes
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            output: [B, num_classes, H, W]
        """
        # 提取CNN特征
        cnn_features = self.cnn_branch(x)
        
        # 提取Mamba特征
        mamba_features = self.mamba_branch(x)
        
        # 简单相加融合
        fused_features = {}
        for stage in ['stage1', 'stage2', 'stage3', 'stage4']:
            fused_features[stage] = cnn_features[stage] + mamba_features[stage]
        
        # 解码
        output = self.decoder(fused_features, target_size=x.shape[-2:])
        
        return output

class Decoder(nn.Module):
    """解码器 - 多尺度特征融合 + 上采样"""
    def __init__(self, in_channels_list, num_classes):
        super(Decoder, self).__init__()
        self.in_channels_list = in_channels_list  # [128, 256, 512, 512]
        self.num_classes = num_classes
        
        # 上采样 + 特征融合模块
        # Stage4 -> Stage3
        self.up4 = nn.Sequential(
            nn.Conv2d(in_channels_list[3], in_channels_list[2], 1),
            nn.BatchNorm2d(in_channels_list[2]),
            nn.ReLU(inplace=True)
        )
        
        # Stage3 融合
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels_list[2] * 2, in_channels_list[2], 3, padding=1),
            nn.BatchNorm2d(in_channels_list[2]),
            nn.ReLU(inplace=True)
        )
        
        # Stage3 -> Stage2
        self.up3 = nn.Sequential(
            nn.Conv2d(in_channels_list[2], in_channels_list[1], 1),
            nn.BatchNorm2d(in_channels_list[1]),
            nn.ReLU(inplace=True)
        )
        
        # Stage2 融合
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels_list[1] * 2, in_channels_list[1], 3, padding=1),
            nn.BatchNorm2d(in_channels_list[1]),
            nn.ReLU(inplace=True)
        )
        
        # Stage2 -> Stage1
        self.up2 = nn.Sequential(
            nn.Conv2d(in_channels_list[1], in_channels_list[0], 1),
            nn.BatchNorm2d(in_channels_list[0]),
            nn.ReLU(inplace=True)
        )
        
        # Stage1 融合
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels_list[0] * 2, in_channels_list[0], 3, padding=1),
            nn.BatchNorm2d(in_channels_list[0]),
            nn.ReLU(inplace=True)
        )
        
        # 最终分类头
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels_list[0], 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )
    
    def forward(self, features, target_size):
        """
        Args:
            features: dict with keys ['stage1', 'stage2', 'stage3', 'stage4']
            target_size: (H, W) 目标输出尺寸
        Returns:
            output: [B, num_classes, H, W]
        """
        x4 = features['stage4']  # [B, 512, H/32, W/32]
        x3 = features['stage3']  # [B, 512, H/16, W/16]
        x2 = features['stage2']  # [B, 256, H/8, W/8]
        x1 = features['stage1']  # [B, 128, H/4, W/4]
        
        # Stage4 -> Stage3
        x = self.up4(x4)
        x = F.interpolate(x, size=x3.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x3], dim=1)
        x = self.conv3(x)  # [B, 512, H/16, W/16]
        
        # Stage3 -> Stage2
        x = self.up3(x)
        x = F.interpolate(x, size=x2.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)  # [B, 256, H/8, W/8]
        
        # Stage2 -> Stage1
        x = self.up2(x)
        x = F.interpolate(x, size=x1.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, x1], dim=1)
        x = self.conv1(x)  # [B, 128, H/4, W/4]
        
        # 最终上采样到原图尺寸
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        x = self.final_conv(x)  # [B, num_classes, H, W]
        
        return x

# if __name__ == "__main__":
#     print("=" * 60)
#     print("测试 BaselineModel")
#     print("=" * 60)
    
#     # 创建模型
#     model = BaselineModel(num_classes=6, in_channels=3, pretrained=True)
    
#     # 测试前向传播
#     x = torch.randn(2, 3, 512, 512)
    
#     if torch.cuda.is_available():
#         model = model.cuda()
#         x = x.cuda()
    
#     with torch.no_grad():
#         output = model(x)
    
#     print(f"\n输入形状: {x.shape}")
#     print(f"输出形状: {output.shape}")
    
#     # 计算参数量
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     print(f"\n总参数量: {total_params/1e6:.2f}M")
#     print(f"可训练参数: {trainable_params/1e6:.2f}M")
    
#     print("\n" + "=" * 60)