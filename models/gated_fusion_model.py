import sys
sys.path.append('/mnt/e/Github/CMDB-Net')
import torch
import torch.nn as nn
from models.cnn_branch import CNNBranch
from models.mamba_branch import MambaBranch
from models.baseline_model import Decoder
from models.DPGF_module import DualPathGatedFusion

class GatedFusionModel(nn.Module):
    """带门控融合的CNN-Mamba模型 (用于消融实验)"""
    def __init__(self, num_classes, in_channels=3, pretrained=True):
        super(GatedFusionModel, self).__init__()
        self.num_classes = num_classes
    
        self.cnn_branch = CNNBranch(
            in_channels=in_channels,
            weights=pretrained,
            align_channels=True
        )
        
        self.mamba_branch = MambaBranch(
            pretrained=pretrained,
            in_channels=in_channels,
            use_cuda=torch.cuda.is_available()
        )
        
        # 门控融合模块（4个stage各一个）
        self.fusion_modules = nn.ModuleDict({
            'stage1': DualPathGatedFusion(channels=128),
            'stage2': DualPathGatedFusion(channels=256),
            'stage3': DualPathGatedFusion(channels=512),
            'stage4': DualPathGatedFusion(channels=512),
        })
        
        # 解码器（输入为融合后的特征）
        self.decoder = Decoder(
            in_channels_list=[128, 256, 512, 512],
            num_classes=num_classes
        )
        
        # 存储gate_map用于可视化
        self.gate_maps = {}
    
    def forward(self, x):
        """前向传播"""
        # 提取CNN和Mamba特征
        cnn_features = self.cnn_branch(x)
        mamba_features = self.mamba_branch(x)
        
        # 门控融合各个stage
        fused_features = {}
        gate_stats = {}  # 收集门控统计信息
        self.gate_maps = {}  # 清空上次的gate_maps
        
        for stage in ['stage1', 'stage2', 'stage3', 'stage4']:
            F_local = cnn_features[stage]
            F_global = mamba_features[stage]
            
            # 门控融合
            F_fused, Gate_map = self.fusion_modules[stage](F_local, F_global)

            # 收集统计信息
            gate_stats[f'{stage}_mean'] = Gate_map.mean().item()
            gate_stats[f'{stage}_std'] = Gate_map.std().item()
            gate_stats[f'{stage}_min'] = Gate_map.min().item()
            gate_stats[f'{stage}_max'] = Gate_map.max().item()
            
            fused_features[stage] = F_fused
            self.gate_maps[stage] = Gate_map  # 保存用于可视化
        
        # 解码
        output = self.decoder(fused_features, target_size=x.shape[-2:])
        
        return output, gate_stats
    
    def get_gate_maps(self):
        """获取最近一次前向传播的gate_maps"""
        return self.gate_maps