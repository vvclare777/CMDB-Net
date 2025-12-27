import sys
sys.path.append('/mnt/e/Github/CMDB-Net')
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from utils.dataset import PotsdamDataset, create_dataloaders
from models.gated_fusion_model import GatedFusionModel
from metrics import SegmentationMetrics
from configs.potsdam_config import PotsdamConfig

class Visualizer:
    """模型预测和可视化"""
    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        print(f"Loading model: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = GatedFusionModel(
            num_classes=config.NUM_CLASSES,
            in_channels=3,
            pretrained=True
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loading successful! (Epoch: {checkpoint['epoch']})")
        if 'metrics' in checkpoint:
            print(f"Training metrics: mIoU={checkpoint['metrics']['mIoU']:.4f}")
        
        # 类别名称和颜色
        self.class_names = config.CLASS_NAMES
        
        self.class_colors = np.array([
            [255, 255, 255],  # 白色
            [0, 0, 255],      # 蓝色
            [0, 255, 255],    # 青色
            [0, 255, 0],      # 绿色
            [255, 255, 0],    # 黄色
            [255, 0, 0],      # 红色
        ])
    
    def predict_image(self, image_path):
        """预测单张图像"""
        # 读取图像
        image = np.array(Image.open(image_path).convert('RGB'))
        
        # 数据预处理
        transform = PotsdamDataset.get_val_transform(self.config.img_size)
        augmented = transform(image=image)
        img_tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probs, dim=1)
        
        pred_mask = pred.squeeze().cpu().numpy()
        confidence_map = confidence.squeeze().cpu().numpy()
        
        return pred_mask, confidence_map, image
    
    def predict_dataset(self, dataloader, save_dir):
        """预测整个数据集"""
        os.makedirs(save_dir, exist_ok=True)
        # os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'visualizations'), exist_ok=True)
        
        metrics = SegmentationMetrics(num_classes=self.config.NUM_CLASSES)
        
        for batch in tqdm(dataloader, desc="[Test]"):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            filenames = batch['filename']

            # 预测
            with torch.no_grad():
                outputs = self.model(images)
                preds = torch.argmax(outputs[0], dim=1)

            # 更新指标
            metrics.update(preds, masks)

            # 保存预测结果
            for i in range(len(filenames)):
                pred_mask = preds[i].cpu().numpy()
                gt_mask = masks[i].cpu().numpy()

                # # 保存预测mask
                # pred_filename = filenames[i].replace('.jpg', '_pred.png')
                # pred_path = os.path.join(save_dir, 'predictions', pred_filename)
                # Image.fromarray(pred_mask.astype(np.uint8)).save(pred_path)

                # 保存可视化
                self.visualize_prediction(
                    images[i],
                    gt_mask,
                    pred_mask,
                    save_path=os.path.join(save_dir, 'visualizations', filenames[i])
                )
        
        # 打印指标
        metrics.print_results(class_names=self.class_names)
        
        # 保存指标
        all_metrics = metrics.get_all_metrics()
        self.save_metrics(all_metrics, os.path.join(save_dir, 'metrics_' + config.EXP_NAME + '.txt'))
        
        return all_metrics
    
    def visualize_prediction(self, image, gt_mask, pred_mask, save_path):
        """可视化预测结果"""
        # 反归一化图像
        img = image.cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # 转换为RGB
        gt_rgb = self.mask_to_rgb(gt_mask)
        pred_rgb = self.mask_to_rgb(pred_mask)
        
        # 绘图
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(img)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(gt_rgb)
        axes[1].set_title('Ground Truth', fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(pred_rgb)
        axes[2].set_title('Prediction', fontsize=12)
        axes[2].axis('off')
        
        # 叠加显示
        overlay = img.copy()
        overlay = (overlay * 0.6 + pred_rgb / 255.0 * 0.4)
        axes[3].imshow(overlay)
        axes[3].set_title('Overlay', fontsize=12)
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def mask_to_rgb(self, mask):
        """将类别mask转换为RGB"""
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id in range(len(self.class_colors)):
            rgb[mask == class_id] = self.class_colors[class_id]
        return rgb
    
    def visualize_with_legend(self, image_path, save_path):
        """带图例的可视化"""
        pred_mask, confidence, original_img = self.predict_image(image_path)
        pred_rgb = self.mask_to_rgb(pred_mask)
        
        fig = plt.figure(figsize=(16, 6))
        
        # 原图
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(original_img)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 预测结果
        ax2 = plt.subplot(1, 3, 2)
        ax2.imshow(pred_rgb)
        ax2.set_title('Prediction', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # 置信度图
        ax3 = plt.subplot(1, 3, 3)
        im = ax3.imshow(confidence, cmap='jet', vmin=0, vmax=1)
        ax3.set_title('Confidence Map', fontsize=14, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046)
        
        # 添加图例
        legend_elements = [
            Patch(facecolor=self.class_colors[i]/255.0, label=self.class_names[i])
            for i in range(len(self.class_names))
        ]
        fig.legend(
            handles=legend_elements,
            loc='lower center',
            ncol=6,
            fontsize=10,
            frameon=True
        )
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"可视化已保存: {save_path}")
    
    def save_metrics(self, metrics, save_path):
        """保存指标到文件"""
        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("评估指标\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"mIoU:  {metrics['mIoU']:.4f}\n")
            f.write(f"OA:    {metrics['OA']:.4f}\n")
            f.write(f"F1:    {metrics['F1']:.4f}\n")
            f.write(f"Kappa: {metrics['Kappa']:.4f}\n\n")
            
            f.write("Per-class IoU:\n")
            for i, iou in enumerate(metrics['IoU_per_class']):
                f.write(f"  {self.class_names[i]}: {iou:.4f}\n")
            
            f.write("\nPer-class F1:\n")
            for i, f1 in enumerate(metrics['F1_per_class']):
                f.write(f"  {self.class_names[i]}: {f1:.4f}\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"指标已保存: {save_path}")

if __name__ == "__main__":
    config = PotsdamConfig()
    cpt_path = os.path.join('checkpoints', 'best_model_' + config.EXP_NAME + '.pth')
    visualizer = Visualizer(cpt_path, config)

    save_dir = 'results'

    _, _, test_loader = create_dataloaders(config)

    visualizer.predict_dataset(test_loader, save_dir)