import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from configs.potsdam_config import PotsdamConfig
from utils.dataset import create_dataloaders
from models.gated_fusion_model import GatedFusionModel
from utils.metrics import SegmentationMetrics
CLASSES = [
    "Impervious",  # 0
    "Building",    # 1
    "LowVegetation",# 2
    "Tree",        # 3
    "Car",         # 4
]

@torch.no_grad()
def evaluate_and_get_cm(model, dataloader, num_classes, device):
    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes) # 初始化
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            logits = model(images)
            preds = torch.argmax(logits[0], dim=1)
            
            # 直接更新混淆矩阵，不保存像素数组
            metrics.update(preds, masks)
            
    # 从 metrics 中获取混淆矩阵
    cm = metrics.confusion_matrix
    return cm, metrics

def compute_metrics_from_cm(cm):
    eps = 1e-6
    num_classes = cm.shape[0]

    IoU = np.zeros(num_classes)
    Precision = np.zeros(num_classes)
    Recall = np.zeros(num_classes)

    for i in range(num_classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP     

        IoU[i] = TP / (TP + FP + FN + eps)
        Precision[i] = TP / (TP + FP + eps)
        Recall[i] = TP / (TP + FN + eps)

    return IoU, Precision, Recall

def plot_confusion_matrix(cm, class_names, normalize=True, save_path=None):
    if normalize:
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-6)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        xticklabels=class_names,
        yticklabels=class_names,
        annot=True,
        fmt=".2f",
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Normalized Confusion Matrix" if normalize else "Confusion Matrix")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    config = PotsdamConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = GatedFusionModel(
        num_classes=PotsdamConfig.NUM_CLASSES,
        in_channels=3,
        pretrained=config.PRETRAINED
    )
    checkpoint = torch.load("checkpoints/best_model_exp1_12_27_no_clutter.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # 准备数据加载器
    _, val_loader, _ = create_dataloaders(config)

    # 计算混淆矩阵 (使用流式计算避免OOM)
    cm, _ = evaluate_and_get_cm(
        model, val_loader, PotsdamConfig.NUM_CLASSES, device
    )

    # 计算指标
    IoU, Precision, Recall = compute_metrics_from_cm(cm)

    print("\nPer-class metrics:")
    for i, class_name in enumerate(CLASSES):
        print(f"{class_name:15s} - IoU: {IoU[i]:.4f}, Precision: {Precision[i]:.4f}, Recall: {Recall[i]:.4f}")

    # 绘制混淆矩阵
    plot_confusion_matrix(cm, CLASSES, normalize=True, save_path="confusion_matrix_2.png")