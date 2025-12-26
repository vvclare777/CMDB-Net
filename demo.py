import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from configs.potsdam_config import PotsdamConfig
from utils.dataset import create_dataloaders
from models.baseline_model import BaselineModel
from utils.metrics import SegmentationMetrics
CLASSES = [
    "Impervious",  # 0
    "Building",    # 1
    "LowVegetation",# 2
    "Tree",        # 3
    "Car",         # 4
    "Clutter"      # 5
]
CLUTTER_ID = 5

@torch.no_grad()
def evaluate_and_get_cm(model, dataloader, num_classes, device):
    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes) # 初始化
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
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

# Clutter被错分成什么类
def analyze_clutter_misclassified(cm, class_names, clutter_id):
    clutter_row = cm[clutter_id]  # GT=Clutter
    total = clutter_row.sum()

    print(f"\nGT = {class_names[clutter_id]} 被错分情况：")
    for i, v in enumerate(clutter_row):
        ratio = v / (total + 1e-6)
        print(f"  → 预测为 {class_names[i]:15s}: {ratio:.3f}")

# 哪些类被错分成Clutter
def analyze_false_clutter(cm, class_names, clutter_id):
    clutter_col = cm[:, clutter_id]  # Pred=Clutter
    total = clutter_col.sum()

    print(f"\n预测为 {class_names[clutter_id]} 的来源：")
    for i, v in enumerate(clutter_col):
        ratio = v / (total + 1e-6)
        print(f"  ← 来自 {class_names[i]:15s}: {ratio:.3f}")

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
    model = BaselineModel(
        num_classes=PotsdamConfig.NUM_CLASSES,
        in_channels=3,
        pretrained=config.PRETRAINED
    )
    checkpoint = torch.load("checkpoints/best_model_exp1_12_24.pth", map_location=device)
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

    # 分析Clutter类的错分情况
    analyze_clutter_misclassified(cm, CLASSES, CLUTTER_ID)
    analyze_false_clutter(cm, CLASSES, CLUTTER_ID)

    # 绘制混淆矩阵
    plot_confusion_matrix(cm, CLASSES, normalize=True, save_path="confusion_matrix.png")

"""
Per-class metrics:
Impervious      - IoU: 0.8198, Precision: 0.9134, Recall: 0.8889
Building        - IoU: 0.9220, Precision: 0.9544, Recall: 0.9645
LowVegetation   - IoU: 0.7791, Precision: 0.8876, Recall: 0.8644
Tree            - IoU: 0.7313, Precision: 0.8204, Recall: 0.8707
Car             - IoU: 0.8181, Precision: 0.8449, Recall: 0.9626
Clutter         - IoU: 0.4211, Precision: 0.5902, Recall: 0.5951

GT = Clutter 被错分情况：
  → 预测为 Impervious     : 0.163
  → 预测为 Building       : 0.107
  → 预测为 LowVegetation  : 0.108
  → 预测为 Tree           : 0.022
  → 预测为 Car            : 0.005
  → 预测为 Clutter        : 0.595

预测为 Clutter 的来源：
  ← 来自 Impervious     : 0.142
  ← 来自 Building       : 0.092
  ← 来自 LowVegetation  : 0.157
  ← 来自 Tree           : 0.018
  ← 来自 Car            : 0.001
  ← 来自 Clutter        : 0.590
"""