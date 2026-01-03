import sys
sys.path.append('/mnt/e/Github/CMDB-Net')
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
from models.gated_fusion_model import GatedFusionModel
from utils.dataset import create_dataloaders
from configs.potsdam_config import PotsdamConfig

class GateMapVisualizer:
    """门控图可视化工具"""
    def __init__(self, model, device):
        """
        Args:
            model: GatedFusionModel实例
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def visualize_single_image(self, image_path, save_dir):
        """
        可视化单张图像的gate_map
        Args:
            image_path: 图像路径
            save_dir: 保存目录
        """
        from utils.dataset import get_val_transform
        from configs.potsdam_config import PotsdamConfig
        
        config = PotsdamConfig()
        
        # 读取并预处理图像
        image = np.array(Image.open(image_path).convert('RGB'))
        transform = get_val_transform(config.IMG_SIZE)
        augmented = transform(image=image)
        img_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        # 前向传播
        with torch.no_grad():
            output, _ = self.model(img_tensor) # output is (logits, gate_stats)
            gate_maps = self.model.get_gate_maps()
        
        # 反归一化图像
        img = img_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # 获取预测结果
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 可视化所有stage的gate_map概览
        self._visualize_all_stages_overview(img, gate_maps, pred, 
                                   save_path=os.path.join(save_dir, f'{base_name}_overview.png'))
        
        # 可视化单个stage的详细信息
        for stage_name, gate_data in gate_maps.items():
            self._visualize_single_stage_details(img, gate_data, pred, stage_name,
                                        save_path=os.path.join(save_dir, f'{base_name}_{stage_name}_details.png'))
        
        print(f"可视化已保存到: {save_dir}")
    
    def _visualize_all_stages_overview(self, img, gate_maps, pred, save_path):
        """可视化所有stage的gate_cnn概览"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 原图
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 预测结果
        axes[0, 1].imshow(pred, cmap='tab10')
        axes[0, 1].set_title('Prediction', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Stage4的gate_cnn（最粗粒度）
        gate4 = gate_maps['stage4']['gate_cnn'].squeeze().cpu().numpy()
        gate4_resized = cv2.resize(gate4, (img.shape[1], img.shape[0]))
        im = axes[0, 2].imshow(gate4_resized, cmap='jet', vmin=0, vmax=1)
        axes[0, 2].set_title('CNN Gate (Stage4, 1/32)', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
        
        # Stage3的gate_cnn
        gate3 = gate_maps['stage3']['gate_cnn'].squeeze().cpu().numpy()
        gate3_resized = cv2.resize(gate3, (img.shape[1], img.shape[0]))
        im = axes[1, 0].imshow(gate3_resized, cmap='jet', vmin=0, vmax=1)
        axes[1, 0].set_title('CNN Gate (Stage3, 1/16)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
        
        # Stage2的gate_cnn
        gate2 = gate_maps['stage2']['gate_cnn'].squeeze().cpu().numpy()
        gate2_resized = cv2.resize(gate2, (img.shape[1], img.shape[0]))
        im = axes[1, 1].imshow(gate2_resized, cmap='jet', vmin=0, vmax=1)
        axes[1, 1].set_title('CNN Gate (Stage2, 1/8)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
        
        # Stage1的gate_cnn（最细粒度）
        gate1 = gate_maps['stage1']['gate_cnn'].squeeze().cpu().numpy()
        gate1_resized = cv2.resize(gate1, (img.shape[1], img.shape[0]))
        im = axes[1, 2].imshow(gate1_resized, cmap='jet', vmin=0, vmax=1)
        axes[1, 2].set_title('CNN Gate (Stage1, 1/4)', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    def _visualize_single_stage_details(self, img, gate_data, pred, stage_name, save_path):
        """详细可视化单个stage的所有组件"""
        # 提取各组件
        gate_cnn = gate_data['gate_cnn'].squeeze().cpu().numpy()
        gate_mamba = gate_data['gate_mamba'].squeeze().cpu().numpy()
        local_guidance = gate_data['local_guidance'].squeeze().cpu().numpy()
        global_guidance = gate_data['global_guidance'].squeeze().cpu().numpy()
        
        # Resize到原图大小
        h, w = img.shape[:2]
        gate_cnn = cv2.resize(gate_cnn, (w, h))
        gate_mamba = cv2.resize(gate_mamba, (w, h))
        local_guidance = cv2.resize(local_guidance, (w, h))
        global_guidance = cv2.resize(global_guidance, (w, h))
        
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        
        # 1. 原图
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Image', fontsize=12)
        axes[0, 0].axis('off')
        
        # 2. Local Guidance (CNN Input)
        im = axes[0, 1].imshow(local_guidance, cmap='viridis', vmin=0, vmax=1)
        axes[0, 1].set_title(f'Local Guidance (Edges/Texture)', fontsize=12)
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
        
        # 3. Global Guidance (Mamba Input)
        im = axes[0, 2].imshow(global_guidance, cmap='viridis', vmin=0, vmax=1)
        axes[0, 2].set_title(f'Global Guidance (Context/LowFreq)', fontsize=12)
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
        
        # 4. Prediction
        axes[0, 3].imshow(pred, cmap='tab10')
        axes[0, 3].set_title('Prediction', fontsize=12)
        axes[0, 3].axis('off')
        
        # 5. CNN Gate (Output)
        im = axes[1, 0].imshow(gate_cnn, cmap='jet', vmin=0, vmax=1)
        axes[1, 0].set_title(f'CNN Gate (Local Weight)', fontsize=12)
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
        
        # 6. Mamba Gate (Output)
        im = axes[1, 1].imshow(gate_mamba, cmap='jet', vmin=0, vmax=1)
        axes[1, 1].set_title(f'Mamba Gate (Global Weight)', fontsize=12)
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
        
        # 7. Gate Difference (CNN - Mamba)
        diff = gate_cnn - gate_mamba
        im = axes[1, 2].imshow(diff, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 2].set_title(f'Gate Difference (Red=CNN, Blue=Mamba)', fontsize=12)
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
        
        # 8. Gate Distribution
        axes[1, 3].hist(gate_cnn.flatten(), bins=50, alpha=0.5, label='CNN Gate', color='red')
        axes[1, 3].hist(gate_mamba.flatten(), bins=50, alpha=0.5, label='Mamba Gate', color='blue')
        axes[1, 3].set_title('Gate Value Distribution', fontsize=12)
        axes[1, 3].legend()
        
        plt.suptitle(f'Stage: {stage_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    def visualize_batch(self, dataloader, save_dir, num_samples=8):
        """
        批量可视化gate_map
        Args:
            dataloader: 数据加载器
            save_dir: 保存目录
            num_samples: 可视化样本数
        """
        os.makedirs(save_dir, exist_ok=True)
        
        count = 0
        for batch in dataloader:
            if count >= num_samples:
                break
            
            images = batch['image'].to(self.device)
            filenames = batch['filename']
            
            with torch.no_grad():
                outputs, _ = self.model(images)
                gate_maps = self.model.get_gate_maps()
            
            for i in range(images.shape[0]):
                if count >= num_samples:
                    break
                
                # 反归一化
                img = images[i].cpu().permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                pred = torch.argmax(outputs[i], dim=0).cpu().numpy()
                
                # 提取该样本的gate_maps
                sample_gates = {}
                for stage in ['stage1', 'stage2', 'stage3', 'stage4']:
                    sample_gates[stage] = {
                        k: v[i] for k, v in gate_maps[stage].items()
                    }
                
                base_name = os.path.splitext(filenames[i])[0]
                
                # 1. 概览图
                self._visualize_all_stages_overview(
                    img, sample_gates, pred,
                    save_path=os.path.join(save_dir, f'{base_name}_overview.png')
                )
                
                # 2. 详细图 (只画Stage1和Stage4作为代表)
                for stage in ['stage1', 'stage4']:
                    self._visualize_single_stage_details(
                        img, sample_gates[stage], pred, stage,
                        save_path=os.path.join(save_dir, f'{base_name}_{stage}_details.png')
                    )
                
                count += 1
                print(f"已处理: {count}/{num_samples}")
        
        print(f"批量可视化完成! 保存至: {save_dir}")

if __name__ == "__main__":    
    config = PotsdamConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载门控融合模型
    model = GatedFusionModel(
        num_classes=config.NUM_CLASSES,
        in_channels=3,
        pretrained=True
    ).to(device)
    
    # 加载训练好的权重
    checkpoint_path = 'checkpoints/best_model_exp3_1_2.pth' # 使用最新的权重
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # 处理可能的key不匹配问题 (如果保存时加了module.前缀)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        print(f"已加载模型权重: {checkpoint_path}")
    else:
        print(f"未找到权重文件: {checkpoint_path}, 使用随机初始化权重")
    
    # 创建可视化器
    visualizer = GateMapVisualizer(model, device)
    
    # 创建数据加载器
    _, val_loader, _ = create_dataloaders(config)
    
    # 批量可视化
    visualizer.visualize_batch(val_loader, save_dir='results/gate_maps_visualization', num_samples=5)
