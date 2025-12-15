import numpy as np
import torch
from scipy.ndimage import binary_dilation

class SegmentationMetrics:
    """
    语义分割评估指标
    - mIoU (Mean Intersection over Union)
    - OA (Overall Accuracy)
    - F1 Score
    - Kappa系数
    - Per-class IoU
    """
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """重置混淆矩阵"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, pred, target):
        """更新混淆矩阵"""
        # 转换为numpy
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        # 展平
        pred = pred.flatten()
        target = target.flatten()
        
        # 过滤ignore_index
        mask = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        pred = pred[mask]
        target = target[mask]
        
        # 更新混淆矩阵
        for t, p in zip(target, pred):
            self.confusion_matrix[int(t), int(p)] += 1
    
    def get_miou(self):
        """计算mIoU"""
        intersection = np.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(axis=1) + 
            self.confusion_matrix.sum(axis=0) - 
            intersection
        )
        
        # 避免除零
        iou = intersection / (union + 1e-10)
        
        # 只计算有样本的类别
        valid_classes = union > 0
        miou = np.mean(iou[valid_classes])
        
        return miou, iou
    
    def get_oa(self):
        """计算Overall Accuracy"""
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        oa = correct / (total + 1e-10)
        return oa
    
    def get_f1(self):
        """计算F1 Score"""
        # 每个类别的精确率和召回率
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # 只计算有样本的类别
        valid_classes = (tp + fn) > 0
        mean_f1 = np.mean(f1[valid_classes])
        
        return mean_f1, f1
    
    def get_kappa(self):
        """计算Kappa系数"""
        # 使用混淆矩阵计算
        n = self.confusion_matrix.sum()
        sum_po = np.diag(self.confusion_matrix).sum()
        sum_pe = (
            self.confusion_matrix.sum(axis=0) * 
            self.confusion_matrix.sum(axis=1)
        ).sum()
        
        po = sum_po / (n + 1e-10)
        pe = sum_pe / (n * n + 1e-10)
        
        kappa = (po - pe) / (1 - pe + 1e-10)
        return kappa
    
    def get_boundary_iou(self, pred, target, dilation=2):
        """计算边界IoU 衡量边界区域的分割精度"""
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        # 计算边界
        boundary_ious = []
        for c in range(self.num_classes):
            pred_c = (pred == c).astype(np.uint8)
            target_c = (target == c).astype(np.uint8)
            
            # 膨胀操作获取边界
            pred_boundary = binary_dilation(pred_c, iterations=dilation) & ~pred_c.astype(bool)
            target_boundary = binary_dilation(target_c, iterations=dilation) & ~target_c.astype(bool)
            
            # 计算交并比
            intersection = (pred_boundary & target_boundary).sum()
            union = (pred_boundary | target_boundary).sum()
            
            if union > 0:
                boundary_ious.append(intersection / union)
        
        return np.mean(boundary_ious) if boundary_ious else 0.0
    
    def get_all_metrics(self):
        """获取所有指标"""
        miou, iou_per_class = self.get_miou()
        oa = self.get_oa()
        f1, f1_per_class = self.get_f1()
        kappa = self.get_kappa()
        # biou = self.get_boundary_iou(pred, target)
        
        return {
            'mIoU': miou,
            'OA': oa,
            'F1': f1,
            'Kappa': kappa,
            # 'Boundary_IoU': biou,
            'IoU_per_class': iou_per_class,
            'F1_per_class': f1_per_class
        }
    
    def print_results(self, class_names=None):
        if class_names is None:
            class_names = [f'Class {i}' for i in range(self.num_classes)]
        
        # 获取所有指标
        miou, iou_per_class = self.get_miou()
        oa = self.get_oa()
        f1, f1_per_class = self.get_f1()
        kappa = self.get_kappa()
        
        # 打印整体指标
        print(f"{'=' * 80}")
        print(f"{'评估指标汇总':^60}")
        print(f"{'=' * 80}")
        print(f"{'指标':<20} {'数值':<10}")
        print(f"{'-' * 80}")
        print(f"{'mIoU':<20} {miou:.4f}")
        print(f"{'OA':<20} {oa:.4f}")
        print(f"{'F1 Score':<20} {f1:.4f}")
        print(f"{'Kappa':<20} {kappa:.4f}")
        
        # 打印每个类别的IoU和F1
        print(f"\n{'=' * 80}")
        print(f"{'类别详细指标':^60}")
        print(f"{'=' * 80}")
        print(f"{'类别':<20} {'IoU':<10} {'F1':<10} {'样本数':<10}")
        print(f"{'-' * 80}")
        
        for i, name in enumerate(class_names):
            if i < len(iou_per_class):
                # 计算该类别的总样本数
                total_samples = self.confusion_matrix[i].sum()
                print(f"{name:<20} {iou_per_class[i]:.4f}      {f1_per_class[i]:.4f}      {int(total_samples):<10}")
        
        # 打印混淆矩阵统计
        print(f"\n{'=' * 80}")
        print(f"{'混淆矩阵统计':^60}")
        print(f"{'=' * 80}")
        print(f"总像素数: {int(self.confusion_matrix.sum())}")
        
        # 计算每个类别的预测准确率
        print(f"\n{'类别':<20} {'准确率':<10}")
        print(f"{'-' * 80}")
        for i, name in enumerate(class_names):
            if i < len(iou_per_class):
                correct = self.confusion_matrix[i, i]
                total = self.confusion_matrix[i].sum()
                accuracy = correct / (total + 1e-10)
                print(f"{name:<20} {accuracy:.4f}")
        
        # 返回结果字典
        return {
            'OA': oa,
            'mIoU': miou,
            'F1': f1,
            'Kappa': kappa,
            'IoU_per_class': iou_per_class,
            'F1_per_class': f1_per_class,
            'confusion_matrix': self.confusion_matrix
        }