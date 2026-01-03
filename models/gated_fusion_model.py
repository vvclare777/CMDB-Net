import sys
sys.path.append('/mnt/e/Github/CMDB-Net')
import torch
import torch.nn as nn
from models.cnn_branch import CNNBranch
from models.mamba_branch import MambaBranch
from models.baseline_model import Decoder
from models.improved_DPGF_module import DualPathGatedFusion

class GatedFusionModel(nn.Module):
    """加入双门控融合模块的模型"""
    def __init__(self, num_classes, in_channels=3, pretrained=True):
        super(GatedFusionModel, self).__init__()
        self.num_classes = num_classes
    
        # CNN分支 - 擅长局部特征/高频信息
        self.cnn_branch = CNNBranch(
            in_channels=in_channels,
            weights=pretrained,
            align_channels=True
        )
        
        # Mamba分支 - 擅长全局特征/长程依赖
        self.mamba_branch = MambaBranch(
            pretrained=pretrained,
            in_channels=in_channels,
            use_cuda=torch.cuda.is_available()
        )

        # 各阶段通道数配置
        self.stage_channels = {
            'stage1': 128,
            'stage2': 256,
            'stage3': 512,
            'stage4': 512,
        }

        # 双路径门控融合模块
        self.fusion_modules = nn.ModuleDict({
            stage: DualPathGatedFusion(
                cnn_channels=ch,
                mamba_channels=ch,
                out_channels=ch
            )
            for stage, ch in self.stage_channels.items()
        })
        
        # 解码器
        self.decoder = Decoder(
            in_channels_list=list(self.stage_channels.values()),
            num_classes=num_classes
        )
        
         # 存储中间结果用于可视化和分析
        self.gate_maps = {}
        self.debug_info = {}
    
    def forward(self, x):
        # 提取CNN和Mamba特征
        cnn_features = self.cnn_branch(x)
        mamba_features = self.mamba_branch(x)
        
        # 门控融合各个stage
        fused_features = {}
        gate_stats = {}  # 收集门控统计信息
        self.gate_maps = {}
        self.debug_info = {}
        
        for stage in ['stage1', 'stage2', 'stage3', 'stage4']:
            F_local = cnn_features[stage]
            F_global = mamba_features[stage]

            # 双路径门控融合
            F_fused, gate_info = self.fusion_modules[stage](F_local, F_global)

            # 提取门控信息
            gate_cnn = gate_info['gate_cnn']                # CNN权重图
            gate_mamba = gate_info['gate_mamba']            # Mamba权重图
            local_guidance = gate_info['local_guidance']    # 局部引导图
            global_guidance = gate_info['global_guidance']  # 全局引导图
            
            # 收集详细的门控统计信息
            gate_stats.update({
                # CNN门控统计
                f'{stage}_cnn_gate_mean': gate_cnn.mean().item(),
                f'{stage}_cnn_gate_std': gate_cnn.std().item(),
                f'{stage}_cnn_gate_min': gate_cnn.min().item(),
                f'{stage}_cnn_gate_max': gate_cnn.max().item(),
                
                # Mamba门控统计
                f'{stage}_mamba_gate_mean': gate_mamba.mean().item(),
                f'{stage}_mamba_gate_std': gate_mamba.std().item(),
                f'{stage}_mamba_gate_min': gate_mamba.min().item(),
                f'{stage}_mamba_gate_max': gate_mamba.max().item(),
                
                # 引导图统计
                f'{stage}_local_guidance_mean': local_guidance.mean().item(),
                f'{stage}_global_guidance_mean': global_guidance.mean().item(),
                
                # 门控差异（用于分析互补性）
                f'{stage}_gate_diff_mean': (gate_cnn - gate_mamba).abs().mean().item(),
            })
            
            # 存储用于可视化
            self.gate_maps[stage] = {
                'gate_cnn': gate_cnn.detach(),
                'gate_mamba': gate_mamba.detach(),
                'local_guidance': local_guidance.detach(),
                'global_guidance': global_guidance.detach(),
            }
            
            fused_features[stage] = F_fused
        
        # 解码
        output = self.decoder(fused_features, target_size=x.shape[-2:])
        
        return output, gate_stats
    
    def get_gate_maps(self):
        """获取最近一次前向传播的gate_maps"""
        return self.gate_maps
    
    # def analyze_gate_distribution(self):
    #     """分析门控分布, 验证CNN和Mamba的互补性假设"""
    #     analysis = {}
        
    #     for stage, gate_data in self.gate_maps.items():
    #         gate_cnn = gate_data['gate_cnn']
    #         gate_mamba = gate_data['gate_mamba']
            
    #         # 计算空间相关性（如果高度负相关，说明互补性好）
    #         gate_cnn_flat = gate_cnn.view(-1)
    #         gate_mamba_flat = gate_mamba.view(-1)
            
    #         # 皮尔逊相关系数
    #         cnn_centered = gate_cnn_flat - gate_cnn_flat.mean()
    #         mamba_centered = gate_mamba_flat - gate_mamba_flat.mean()
    #         correlation = (cnn_centered * mamba_centered).sum() / (
    #             cnn_centered.norm() * mamba_centered.norm() + 1e-8
    #         )
            
    #         # 计算各区域的主导情况
    #         cnn_dominant = (gate_cnn > gate_mamba).float().mean()
    #         mamba_dominant = (gate_mamba > gate_cnn).float().mean()
            
    #         analysis[stage] = {
    #             'correlation': correlation.item(),
    #             'cnn_dominant_ratio': cnn_dominant.item(),
    #             'mamba_dominant_ratio': mamba_dominant.item(),
    #             'balance_score': 1 - abs(cnn_dominant.item() - 0.5) * 2,  # 越接近0.5越平衡
    #         }
        
    #     return analysis