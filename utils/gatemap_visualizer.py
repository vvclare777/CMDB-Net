import sys
sys.path.append('/mnt/e/Github/CMDB-Net')
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
from models.DPGF_module import GatedFusionModel
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
            output = self.model(img_tensor)
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
        
        # 可视化所有stage的gate_map
        self._visualize_all_stages(img, gate_maps, pred, 
                                   save_path=os.path.join(save_dir, f'{base_name}_all_stages.png'))
        
        # 可视化单个stage的详细信息
        for stage_name, gate_map in gate_maps.items():
            self._visualize_single_stage(img, gate_map, pred, stage_name,
                                        save_path=os.path.join(save_dir, f'{base_name}_{stage_name}.png'))
        
        print(f"可视化已保存到: {save_dir}")
    
    def _visualize_all_stages(self, img, gate_maps, pred, save_path):
        """可视化所有stage的gate_map"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 原图
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 预测结果
        axes[0, 1].imshow(pred, cmap='tab10')
        axes[0, 1].set_title('Prediction', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Stage4的gate_map（最粗粒度）
        gate4 = gate_maps['stage4'].squeeze().cpu().numpy()
        gate4_resized = cv2.resize(gate4, (img.shape[1], img.shape[0]))
        im = axes[0, 2].imshow(gate4_resized, cmap='jet', vmin=0, vmax=1)
        axes[0, 2].set_title('Gate Map (Stage4, 1/32)', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
        
        # Stage3的gate_map
        gate3 = gate_maps['stage3'].squeeze().cpu().numpy()
        gate3_resized = cv2.resize(gate3, (img.shape[1], img.shape[0]))
        im = axes[1, 0].imshow(gate3_resized, cmap='jet', vmin=0, vmax=1)
        axes[1, 0].set_title('Gate Map (Stage3, 1/16)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
        
        # Stage2的gate_map
        gate2 = gate_maps['stage2'].squeeze().cpu().numpy()
        gate2_resized = cv2.resize(gate2, (img.shape[1], img.shape[0]))
        im = axes[1, 1].imshow(gate2_resized, cmap='jet', vmin=0, vmax=1)
        axes[1, 1].set_title('Gate Map (Stage2, 1/8)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046)
        
        # Stage1的gate_map（最细粒度）
        gate1 = gate_maps['stage1'].squeeze().cpu().numpy()
        gate1_resized = cv2.resize(gate1, (img.shape[1], img.shape[0]))
        im = axes[1, 2].imshow(gate1_resized, cmap='jet', vmin=0, vmax=1)
        axes[1, 2].set_title('Gate Map (Stage1, 1/4)', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    def _visualize_single_stage(self, img, gate_map, pred, stage_name, save_path):
        """详细可视化单个stage的gate_map"""
        gate = gate_map.squeeze().cpu().numpy()
        gate_resized = cv2.resize(gate, (img.shape[1], img.shape[0]))
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 原图
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Image', fontsize=12)
        axes[0, 0].axis('off')
        
        # Gate map热力图
        im = axes[0, 1].imshow(gate_resized, cmap='jet', vmin=0, vmax=1)
        axes[0, 1].set_title(f'Gate Map ({stage_name})', fontsize=12)
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
        
        # 原图+Gate map叠加
        overlay = img.copy()
        gate_colored = cm.jet(gate_resized)[:, :, :3]
        overlay = overlay * 0.5 + gate_colored * 0.5
        axes[0, 2].imshow(overlay)
        axes[0, 2].set_title('Image + Gate Overlay', fontsize=12)
        axes[0, 2].axis('off')
        
        # 高权重区域mask（gate > 0.5）
        high_weight_mask = gate_resized > 0.5
        axes[1, 0].imshow(high_weight_mask, cmap='gray')
        axes[1, 0].set_title('High Weight Regions (>0.5)', fontsize=12)
        axes[1, 0].axis('off')
        
        # 预测结果
        axes[1, 1].imshow(pred, cmap='tab10')
        axes[1, 1].set_title('Prediction', fontsize=12)
        axes[1, 1].axis('off')
        
        # Gate分布直方图
        axes[1, 2].hist(gate_resized.flatten(), bins=50, color='skyblue', edgecolor='black')
        axes[1, 2].set_title('Gate Value Distribution', fontsize=12)
        axes[1, 2].set_xlabel('Gate Value')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].axvline(x=0.5, color='r', linestyle='--', label='Threshold=0.5')
        axes[1, 2].legend()
        
        # 添加统计信息
        stats_text = f'Mean: {gate_resized.mean():.3f}\n'
        stats_text += f'Std: {gate_resized.std():.3f}\n'
        stats_text += f'Min: {gate_resized.min():.3f}\n'
        stats_text += f'Max: {gate_resized.max():.3f}\n'
        stats_text += f'High weight ratio: {(gate_resized > 0.5).sum() / gate_resized.size:.2%}'
        
        axes[1, 2].text(1.05, 0.5, stats_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
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
                outputs = self.model(images)
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
                sample_gates = {
                    stage: gate_maps[stage][i] 
                    for stage in ['stage1', 'stage2', 'stage3', 'stage4']
                }
                
                base_name = os.path.splitext(filenames[i])[0]
                self._visualize_all_stages(
                    img, sample_gates, pred,
                    save_path=os.path.join(save_dir, f'{base_name}_gates.png')
                )
                
                count += 1
                print(f"已处理: {count}/{num_samples}")
        
        print(f"批量可视化完成! 保存至: {save_dir}")


def compare_models(baseline_model, gated_model, dataloader, save_dir, device):
    """
    对比基线模型和门控模型的预测结果
    Args:
        baseline_model: 基线模型
        gated_model: 门控融合模型
        dataloader: 数据加载器
        save_dir: 保存目录
        device: 计算设备
    """
    os.makedirs(save_dir, exist_ok=True)
    
    baseline_model.eval()
    gated_model.eval()
    
    for idx, batch in enumerate(dataloader):
        if idx >= 5:  # 只对比前5个batch
            break
        
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        filenames = batch['filename']
        
        with torch.no_grad():
            # 基线预测
            baseline_output = baseline_model(images)
            baseline_pred = torch.argmax(baseline_output, dim=1)
            
            # 门控预测
            gated_output = gated_model(images)
            gated_pred = torch.argmax(gated_output, dim=1)
            gate_maps = gated_model.get_gate_maps()
        
        for i in range(images.shape[0]):
            # 反归一化图像
            img = images[i].cpu().permute(1, 2, 0).numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            gt = masks[i].cpu().numpy()
            baseline_p = baseline_pred[i].cpu().numpy()
            gated_p = gated_pred[i].cpu().numpy()
            
            # 可视化对比
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            axes[0, 0].imshow(img)
            axes[0, 0].set_title('Original Image', fontsize=14)
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(gt, cmap='tab10')
            axes[0, 1].set_title('Ground Truth', fontsize=14)
            axes[0, 1].axis('off')
            
            # Stage1的gate_map
            gate1 = gate_maps['stage1'][i].squeeze().cpu().numpy()
            gate1_resized = cv2.resize(gate1, (img.shape[1], img.shape[0]))
            im = axes[0, 2].imshow(gate1_resized, cmap='jet', vmin=0, vmax=1)
            axes[0, 2].set_title('Gate Map (Stage1)', fontsize=14)
            axes[0, 2].axis('off')
            plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
            
            axes[1, 0].imshow(baseline_p, cmap='tab10')
            axes[1, 0].set_title('Baseline Prediction', fontsize=14)
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(gated_p, cmap='tab10')
            axes[1, 1].set_title('Gated Fusion Prediction', fontsize=14)
            axes[1, 1].axis('off')
            
            # 差异图（哪些像素预测不同）
            diff = (baseline_p != gated_p).astype(np.uint8)
            axes[1, 2].imshow(diff, cmap='gray')
            axes[1, 2].set_title(f'Prediction Difference\n({diff.sum()} pixels changed)', fontsize=14)
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            base_name = os.path.splitext(filenames[i])[0]
            plt.savefig(os.path.join(save_dir, f'{base_name}_comparison.png'), 
                       dpi=200, bbox_inches='tight')
            plt.close()
    
    print(f"模型对比完成! 保存至: {save_dir}")


if __name__ == "__main__":    
    config = PotsdamConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载门控融合模型
    model = GatedFusionModel(
        num_classes=config.NUM_CLASSES,
        in_channels=3,
        pretrained=False
    ).to(device)
    
    # 加载训练好的权重（如果有）
    checkpoint_path = 'checkpoints/best_model_gated_fusion.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载模型权重: {checkpoint_path}")
    
    # 创建可视化器
    visualizer = GateMapVisualizer(model, device)
    
    # 创建数据加载器
    _, val_loader, _ = create_dataloaders(config)
    
    # 批量可视化
    visualizer.visualize_batch(val_loader, save_dir='results/gate_maps', num_samples=8)