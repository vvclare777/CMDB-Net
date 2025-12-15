import sys
sys.path.append('/mnt/e/Github/demo')
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
from thop import profile

class CNNBranch(nn.Module):
    """
    CNN分支 - 基于ResNet34
    使用官方预训练权重,提取多尺度特征

    输出4个stage的特征:
    - Stage 1: [B, 128, H/4, W/4]   (ResNet34原始64通道)
    - Stage 2: [B, 256, H/8, W/8]   (ResNet34原始128通道)
    - Stage 3: [B, 512, H/16, W/16] (ResNet34原始256通道)
    - Stage 4: [B, 512, H/32, W/32] (ResNet34原始512通道)
    """
    def __init__(self, in_channels=3, weights=True, align_channels=True):
        super(CNNBranch, self).__init__()
        self.align_channels = align_channels

        # 适配torchvision新版本weights参数
        if weights is True:
            weights = ResNet34_Weights.IMAGENET1K_V1
        elif weights is False:
            weights = None

        # 使用ResNet34作为backbone
        backbone = resnet34(weights=weights)

        # 修改第一层以适配任意通道数
        if in_channels != 3:  # 不是3通道，创建新的卷积层：in_channels，64通道，7x7卷积核，步长2，填充3
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:  # RGB默认3通道，直接使用ResNet的第一层
            self.conv1 = backbone.conv1

        # 继承ResNet的其他基础层
        self.bn1 = backbone.bn1          # 批归一化层
        self.relu = backbone.relu        # ReLU激活函数
        self.maxpool = backbone.maxpool  # 最大池化层，3x3窗口，步长2
        
        # ResNet的4个stage，每个stage包含多个残差块
        self.layer1 = backbone.layer1  # 64 channels, 下采样到1/4分辨率（stride=4）
        self.layer2 = backbone.layer2  # 128 channels, 下采样到1/8分辨率
        self.layer3 = backbone.layer3  # 256 channels, 下采样到1/16分辨率
        self.layer4 = backbone.layer4  # 512 channels, 下采样到1/32分辨率

        # ResNet34原始输出通道
        self.stage_channels = [64, 128, 256, 512]
        
        # 通道对齐层 - 对齐到与Mamba相同的通道数 [128, 256, 512, 512]
        if self.align_channels:
            self.channel_aligners = nn.ModuleList([
                self._make_aligner(64, 128),   # Stage 1: 64 -> 128
                self._make_aligner(128, 256),  # Stage 2: 128 -> 256
                self._make_aligner(256, 512),  # Stage 3: 256 -> 512
                nn.Identity(),                 # Stage 4: 512 -> 512 (不变)
            ])
            self.output_channels = [128, 256, 512, 512]
        else:
            self.output_channels = self.stage_channels

    def _make_aligner(self, in_channels, out_channels):
        """通道对齐层"""
        if in_channels == out_channels:
            return nn.Identity()
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入形状：（批次大小, 通道数, 高度, 宽度）
        Returns:
            dict: {
                'stage1': [B, 128, H/4, W/4],
                'stage2': [B, 256, H/8, W/8],
                'stage3': [B, 512, H/16, W/16],
                'stage4': [B, 512, H/32, W/32]
            }
        """
        # Stem 主干
        x = self.conv1(x)    # 第一层卷积：H/2, W/2
        x = self.bn1(x)      # 批归一化
        x = self.relu(x)     # ReLU激活
        x = self.maxpool(x)  # 最大池化：H/4, W/4

        # Stage 1-4
        x1 = self.layer1(x)   # [B, 64, H/4, W/4]
        x2 = self.layer2(x1)  # [B, 128, H/8, W/8]
        x3 = self.layer3(x2)  # [B, 256, H/16, W/16]
        x4 = self.layer4(x3)  # [B, 512, H/32, W/32]
        
        # 通道对齐
        if self.align_channels:
            x1 = self.channel_aligners[0](x1)  # [B, 128, H/4, W/4]
            x2 = self.channel_aligners[1](x2)  # [B, 256, H/8, W/8]
            x3 = self.channel_aligners[2](x3)  # [B, 512, H/16, W/16]
            x4 = self.channel_aligners[3](x4)  # [B, 512, H/32, W/32]

        # 返回dict格式，与MambaBranch一致
        return {
            'stage1': x1,
            'stage2': x2,
            'stage3': x3,
            'stage4': x4,
        }
    
    def get_output_channels(self):
        """返回输出特征的通道数"""
        return self.output_channels
    
    def get_stage_channels(self):
        """返回每个stage的原始通道数"""
        return self.stage_channels
    
# if __name__ == "__main__":
#     # 测试
#     model = CNNBranch(in_channels=3, align_channels=True)
#     x = torch.randn(2, 3, 256, 256)  # 创建随机输入：2个样本，3通道，256x256

#     # 前向传播
#     with torch.no_grad():
#         features = model(x)

#     print(f"输入形状: {x.shape}")
#     print(f"输出类型: {type(features)}")
    
#     print("各stage特征形状:")
#     for stage_name, feat in features.items():
#         print(f"  {stage_name}: {feat.shape}")
    
#     # 检查设备
#     print(f"\n模型设备: {next(model.parameters()).device}")

#     print(f"\n输出通道配置: {model.get_output_channels()}")
#     print(f"原始stage通道: {model.get_stage_channels()}")
    
#     # 计算参数量和FLOPs
#     try:
#         flops, params = profile(model, inputs=(x,), verbose=False)
#         print(f"\nParams: {params/1e6:.2f}M")
#         print(f"FLOPs: {flops/1e9:.2f}G")
#     except Exception as e:
#         print(f"\n无法计算FLOPs: {e}")