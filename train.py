import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.baseline_model import BaselineModel
from utils.loss import Loss
from utils.metrics import SegmentationMetrics
from utils.dataset import create_dataloaders
from configs.potsdam_config import PotsdamConfig
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建TensorBoard writer
        self.writer = SummaryWriter(os.path.join('runs', config.EXP_NAME))
        
        # 创建数据加载器
        print("Creating dataloader...")
        self.train_loader, self.val_loader = create_dataloaders(config)

        # 创建模型
        print("Creating model...")
        self.model = BaselineModel(
            num_classes=config.NUM_CLASSES,
            in_channels=3,
            pretrained=config.PRETRAINED
        ).to(self.device)

        # # 计算类别权重（可选，用于处理类别不平衡）
        # class_weights = self.calculate_class_weights()
        
        # 损失函数
        if config.LOSS == 'ce':
            self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        elif config.LOSS == 'focal':
            self.criterion = Loss(config).focal_loss(alpha=0.25, gamma=2.0, ignore_index=255)
        else:
            raise ValueError(f"Unknown loss: {config.LOSS}")
        
        # 优化器
        if config.OPTIMIZER == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )
        elif config.OPTIMIZER == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )
        elif config.OPTIMIZER == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config.LEARNING_RATE,
                momentum=0.9,
                weight_decay=config.WEIGHT_DECAY
            )

        # 学习率调度器
        if config.SCHEDULER == 'poly':
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda epoch: (1 - epoch / config.NUM_EPOCHS) ** 0.9
            )
        elif config.SCHEDULER == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.NUM_EPOCHS
            )
        elif config.SCHEDULER == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            self.scheduler = None
        
        # 评估指标
        self.metrics = SegmentationMetrics(num_classes=config.NUM_CLASSES)
        
        # 最佳指标
        self.best_miou = 0.0
        
        # 类别名称
        self.class_names = [
            'Impervious surfaces',
            'Building',
            'Low vegetation',
            'Tree',
            'Car',
            'Clutter'
        ]

        # 类别颜色
        self.class_colors = torch.tensor([
            [255, 255, 255],  # 白色
            [0, 0, 255],      # 蓝色
            [0, 255, 255],    # 青色
            [0, 255, 0],      # 绿色
            [255, 255, 0],    # 黄色
            [255, 0, 0],      # 红色
        ], dtype=torch.uint8)

        # 记录配置到TensorBoard
        self.log_config()
        
        print(f"\nTraining configuration:")
        print(f"  Device: {self.device}")
        print(f"  Train set size: {len(self.train_loader.dataset)}")
        print(f"  Validation set size: {len(self.val_loader.dataset)}")
        print(f"  Batch size: {config.BATCH_SIZE}")
        print(f"  Epochs: {config.NUM_EPOCHS}")
        print(f"  Learning rate: {config.LEARNING_RATE}")
        print(f"  Loss function: {config.LOSS}")
        print(f"  Optimizer: {config.OPTIMIZER}\n")
        print(f"  TensorBoard Logs: runs/{config.EXP_NAME}\n")

    # def calculate_class_weights(self):
    #     """计算类别权重（用于处理类别不平衡）"""
    #     # 这里可以根据数据集统计信息计算权重
    #     # 简单起见，这里返回None（均等权重）
    #     return None

    def log_config(self):
        """记录配置到TensorBoard"""
        config_text = f"""
        ### 训练配置
        - **数据集**: {self.config.PROCESSED_DATA_DIR}
        - **图像尺寸**: {self.config.TILE_SIZE}
        - **类别数量**: {self.config.NUM_CLASSES}
        - **Batch Size**: {self.config.BATCH_SIZE}
        - **学习率**: {self.config.LEARNING_RATE}
        - **训练轮数**: {self.config.NUM_EPOCHS}
        - **优化器**: {self.config.OPTIMIZER}
        - **调度器**: {self.config.SCHEDULER}
        - **损失函数**: {self.config.LOSS}
        - **预训练**: {self.config.PRETRAINED}
        """
        self.writer.add_text('Config', config_text, 0)

    def mask_to_rgb(self, mask):
        """将mask转换为RGB用于可视化"""
        # mask: [H, W]
        rgb = torch.zeros(3, mask.shape[0], mask.shape[1], dtype=torch.uint8)
        for class_id in range(len(self.class_colors)):
            rgb[:, mask == class_id] = self.class_colors[class_id].unsqueeze(-1)
        return rgb
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'[Train]')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            # 记录到TensorBoard (每N个batch)
            if batch_idx % 10 == 0:
                global_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss_Step', loss.item(), global_step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], global_step)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

        # 记录epoch平均loss
        self.writer.add_scalar('Train/Loss_Epoch', avg_loss, epoch)
        
        return avg_loss
    
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc=f'[Val]')

        # 用于可视化的样本
        vis_images = []
        vis_gts = []
        vis_preds = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # 预测
                preds = torch.argmax(outputs, dim=1)
                
                # 更新指标
                self.metrics.update(preds, masks)
                
                # 统计
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)

                # 收集前几个batch用于可视化
                if batch_idx < 2:
                    vis_images.append(images.cpu())
                    vis_gts.append(masks.cpu())
                    vis_preds.append(preds.cpu())
                
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        # 计算指标
        all_metrics = self.metrics.get_all_metrics()

        # 记录到TensorBoard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/mIoU', all_metrics['mIoU'], epoch)
        self.writer.add_scalar('Val/OA', all_metrics['OA'], epoch)
        self.writer.add_scalar('Val/F1', all_metrics['F1'], epoch)
        self.writer.add_scalar('Val/Kappa', all_metrics['Kappa'], epoch)

        # 记录每类IoU
        for i, (class_name, iou) in enumerate(zip(self.class_names, all_metrics['IoU_per_class'])):
            self.writer.add_scalar(f'Val/IoU_{class_name}', iou, epoch)

        # 记录每类F1
        for i, (class_name, f1) in enumerate(zip(self.class_names, all_metrics['F1_per_class'])):
            self.writer.add_scalar(f'Val/F1_{class_name}', f1, epoch)

        # 可视化预测结果
        if epoch % self.config.VIS_INTERVAL == 0:
            self.visualize_predictions(
                torch.cat(vis_images, dim=0)[:4],
                torch.cat(vis_gts, dim=0)[:4],
                torch.cat(vis_preds, dim=0)[:4],
                epoch
            )
        
        print(f"\nVerification Result:")
        print(f"  Val Loss: {avg_loss:.4f}")
        print(f"  mIoU: {all_metrics['mIoU']:.4f}")
        print(f"  OA: {all_metrics['OA']:.4f}")
        print(f"  F1: {all_metrics['F1']:.4f}")
        print(f"  Kappa: {all_metrics['Kappa']:.4f}")
        # print(f"  Boundary IoU: {all_metrics['Boundary_IoU']:.4f}")
        
        # 打印每类指标
        print(f"\n  Per-class IoU:")
        for i, iou in enumerate(all_metrics['IoU_per_class']):
            print(f"    {self.class_names[i]}: {iou:.4f}")

        print(f"\n  Per-class F1:")
        for i, f1 in enumerate(all_metrics['F1_per_class']):
            print(f"    {self.class_names[i]}: {f1:.4f}")
        
        return all_metrics
    
    def visualize_predictions(self, images, gts, preds, epoch):
        """可视化预测结果到TensorBoard"""
        # 反归一化图像
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images = images * std + mean
        images = torch.clamp(images, 0, 1)
        
        # 转换mask为RGB
        gt_rgbs = []
        pred_rgbs = []
        for gt, pred in zip(gts, preds):
            gt_rgb = self.mask_to_rgb(gt).float() / 255.0
            pred_rgb = self.mask_to_rgb(pred).float() / 255.0
            gt_rgbs.append(gt_rgb)
            pred_rgbs.append(pred_rgb)
        
        gt_rgbs = torch.stack(gt_rgbs, dim=0)
        pred_rgbs = torch.stack(pred_rgbs, dim=0)
        
        # 拼接图像: [原图 | GT | 预测]
        vis = torch.cat([images, gt_rgbs, pred_rgbs], dim=3)
        
        # 记录到TensorBoard
        self.writer.add_images('Val/Predictions', vis, epoch, dataformats='NCHW')
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        # 保存最佳模型
        if is_best:
            save_path = os.path.join(self.config.CHECKPOINT_DIR, f'best_model_{self.config.EXP_NAME}.pth')
            torch.save(checkpoint, save_path)
            print(f"Saved best model - mIoU: {metrics['mIoU']:.4f}")
    
    def train(self):
        """完整训练流程"""
        print("Start training...")
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            print(f"\nEpoch {epoch}/{self.config.NUM_EPOCHS}")

            # 训练
            train_loss = self.train_epoch(epoch)
            print(f"\nTrain Result:")
            print(f"  Train Loss: {train_loss:.4f}")
            
            # 验证
            val_metrics = self.validate(epoch)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 保存检查点
            is_best = val_metrics['mIoU'] > self.best_miou
            if is_best:
                self.best_miou = val_metrics['mIoU']
            
            self.save_checkpoint(epoch, val_metrics, is_best)
        
        print(f"\nTraining completed! Best mIoU: {self.best_miou:.4f}")

if __name__ == "__main__":
    # 设置设备
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 创建训练器
    trainer = Trainer(PotsdamConfig)
    
    # 训练模型
    trainer.train()