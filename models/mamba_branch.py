import sys
sys.path.append('/mnt/e/Github/CMDB-Net')
import torch
import torch.nn as nn
from thop import profile
from VMamba.vmamba import Backbone_VSSM

class MambaBranch(nn.Module):
    """
    Mamba分支 - 基于VMamba-tiny
    使用官方预训练权重,处理长距离依赖

    VMamba-tiny的4个stage输出:
    - Stage 1: [B, 96, H/4, W/4]
    - Stage 2: [B, 192, H/8, W/8]
    - Stage 3: [B, 384, H/16, W/16]
    - Stage 4: [B, 768, H/32, W/32]
    """
    def __init__(self, pretrained=True, in_channels=3, use_cuda=True):
        super(MambaBranch, self).__init__()
        self.in_channels = in_channels
        self.use_cuda = use_cuda
        
        # VMamba-tiny官方配置，不可修改
        self.backbone = Backbone_VSSM(
            out_indices=(0, 1, 2, 3),  # 返回所有4个stage的特征
            pretrained=None,  # 我们手动加载权重
            norm_layer="ln2d",  # 使用channel_first
            # 以下参数与vmamba_tiny_s1l8一致
            patch_size=4,
            in_chans=in_channels,
            depths=[2, 2, 8, 2],
            dims=[96, 192, 384, 768],
            ssm_d_state=1,
            ssm_ratio=1.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v05_noz",
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            drop_path_rate=0.2,
            patch_norm=True,
            downsample_version="v3",
            patchembed_version="v2",
            gmlp=False,
            use_checkpoint=False,
            channel_first=True,  # 添加这个参数
        )

        # 加载预训练权重
        if pretrained:
            self._load_pretrained_weights()
        
        # 移除分类头
        if hasattr(self.backbone, 'classifier'):
            self.backbone.classifier = nn.Identity()

        # VMamba-tiny四个stage的输出通道数
        self.stage_channels = [96, 192, 384, 768]

        # 特征投影层 - 将不同stage特征投影到统一维度
        self.feature_projections = nn.ModuleList([
            self._make_projection(96, 128),   # Stage 1: 96 -> 128
            self._make_projection(192, 256),  # Stage 2: 192 -> 256
            self._make_projection(384, 512),  # Stage 3: 384 -> 512
            self._make_projection(768, 512),  # Stage 4: 768 -> 512
        ])

        # 多尺度输出: 返回所有4个stage的特征
        self.output_channels = [128, 256, 512, 512]

        # 将模型移到CUDA
        if self.use_cuda and torch.cuda.is_available():
            self.backbone = self.backbone.cuda()
            self.feature_projections = self.feature_projections.cuda()
    
    def _make_projection(self, in_channels, out_channels):
        """特征投影层"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _load_pretrained_weights(self):
        """加载VMamba官方预训练权重"""
        checkpoint_path = "pretrained/vssm1_tiny_0230s_ckpt_epoch_264.pth"

        try:
            print(f"正在加载VMamba预训练权重: {checkpoint_path}")
            self.backbone.load_pretrained(checkpoint_path)
            print("预训练权重加载成功!")
        except FileNotFoundError:
            print(f"警告: 未找到预训练权重文件 {checkpoint_path}")
            print("将使用随机初始化的权重")
        except Exception as e:
            print(f"加载预训练权重时出错: {e}")
            print("将使用随机初始化的权重")
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: [B, C, H, W]
        Returns:
            dict: {
                'stage1': [B, 128, H/4, W/4],
                'stage2': [B, 256, H/8, W/8],
                'stage3': [B, 512, H/16, W/16],
                'stage4': [B, 512, H/32, W/32]
            }
        """
        # 确保输入与模型在同一设备
        if self.use_cuda and torch.cuda.is_available():
            x = x.cuda()

        # Backbone_VSSM返回的是列表，每个元素是一个stage的特征图
        features_list = self.backbone(x)  # 返回的是列表，每个元素是[B, C, H, W]

        # 对每个stage的特征进行增强和投影
        processed_features = {}
        for i, (feat, proj) in enumerate(zip(features_list, self.feature_projections)):
            feat_proj = proj(feat)
            processed_features[f'stage{i+1}'] = feat_proj
        
        return processed_features
    
    def get_output_channels(self):
        """返回输出特征的通道数"""
        return self.output_channels
    
    def get_stage_channels(self):
        """返回每个stage的原始通道数"""
        return self.stage_channels

# if __name__ == "__main__":
#     model = MambaBranch(pretrained=True, in_channels=3, use_cuda=True)
#     x = torch.randn(2, 3, 256, 256)

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