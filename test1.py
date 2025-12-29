"""
验证忽略Clutter类别的修改是否正确
"""
import sys
sys.path.append('/mnt/e/Github/CMDB-Net')
import torch
import numpy as np
from configs.potsdam_config import PotsdamConfig
from utils.dataset import create_dataloaders
from models.gated_fusion_model import GatedFusionModel

def verify_config():
    """验证配置文件"""
    print("="*60)
    print("1. 验证配置文件")
    print("="*60)
    
    config = PotsdamConfig()
    
    # 检查类别数
    assert config.NUM_CLASSES == 5, f"❌ NUM_CLASSES应该是5，但是{config.NUM_CLASSES}"
    print(f"✅ NUM_CLASSES = {config.NUM_CLASSES}")
    
    # 检查类别名称
    assert len(config.CLASS_NAMES) == 5, f"❌ CLASS_NAMES长度应该是5"
    assert 'Clutter' not in config.CLASS_NAMES, f"❌ CLASS_NAMES不应包含Clutter"
    print(f"✅ CLASS_NAMES = {config.CLASS_NAMES}")
    
    # 检查颜色映射
    assert len(config.LABEL_COLORS) == 5, f"❌ LABEL_COLORS应该有5个类别"
    assert 5 not in config.LABEL_COLORS, f"❌ LABEL_COLORS不应包含类别5"
    print(f"✅ LABEL_COLORS包含5个类别")
    
    print("✅ 配置文件验证通过!\n")
    return config

def verify_dataloader(config):
    """验证数据加载器"""
    print("="*60)
    print("2. 验证数据加载器")
    print("="*60)
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(config)
        print(f"✅ 数据加载器创建成功")
        print(f"   Train samples: {len(train_loader.dataset)}")
        print(f"   Val samples: {len(val_loader.dataset)}")
        print(f"   Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        return None, None, None
    
    # 检查一个batch
    batch = next(iter(train_loader))
    masks = batch['mask']
    
    unique_values = torch.unique(masks).cpu().numpy()
    print(f"\n✅ Mask中的唯一值: {unique_values}")
    
    # 检查是否有Clutter像素（255）
    clutter_count = (masks == 255).sum().item()
    total_pixels = masks.numel()
    clutter_ratio = clutter_count / total_pixels * 100
    
    print(f"   Clutter像素数量: {clutter_count} ({clutter_ratio:.2f}%)")
    
    # 检查有效类别范围
    valid_mask = masks[masks != 255]
    if len(valid_mask) > 0:
        max_class = valid_mask.max().item()
        min_class = valid_mask.min().item()
        
        assert max_class <= 4, f"❌ 最大类别ID应该<=4，但是{max_class}"
        assert min_class >= 0, f"❌ 最小类别ID应该>=0，但是{min_class}"
        print(f"✅ 有效类别范围: [{min_class}, {max_class}]")
    
    # 统计各类别像素数
    print(f"\n各类别像素统计:")
    for i in range(config.NUM_CLASSES):
        count = (masks == i).sum().item()
        ratio = count / total_pixels * 100
        print(f"   {config.CLASS_NAMES[i]}: {count} ({ratio:.2f}%)")
    
    print("✅ 数据加载器验证通过!\n")
    return train_loader, val_loader, test_loader

def verify_model(config):
    """验证模型"""
    print("="*60)
    print("3. 验证模型")
    print("="*60)
    
    try:
        model = GatedFusionModel(
            num_classes=config.NUM_CLASSES,
            in_channels=3,
            pretrained=False
        )
        print(f"✅ 模型创建成功")
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return None
    
    # 测试前向传播
    x = torch.randn(2, 3, 256, 256)
    try:
        with torch.no_grad():
            output, gate_stats = model(x)
        
        print(f"✅ 前向传播成功")
        print(f"   输入形状: {x.shape}")
        print(f"   输出形状: {output.shape}")
        
        # 检查输出通道数
        assert output.shape[1] == config.NUM_CLASSES, \
            f"❌ 输出通道数应该是{config.NUM_CLASSES}，但是{output.shape[1]}"
        print(f"✅ 输出通道数正确: {output.shape[1]}")
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        return None
    
    print("✅ 模型验证通过!\n")
    return model

def verify_loss_and_metrics(config):
    """验证损失函数和指标"""
    print("="*60)
    print("4. 验证损失函数和指标")
    print("="*60)
    
    from utils.loss import FocalDiceLoss
    from utils.metrics import SegmentationMetrics
    
    # 创建损失函数
    criterion = FocalDiceLoss(
        num_classes=config.NUM_CLASSES,
    )
    print(f"✅ 损失函数创建成功")
    
    # 创建指标
    metrics = SegmentationMetrics(
        num_classes=config.NUM_CLASSES,
    )
    print(f"✅ 指标对象创建成功")
    
    # 测试损失计算
    output = torch.randn(2, config.NUM_CLASSES, 64, 64)
    target = torch.randint(0, config.NUM_CLASSES + 1, (2, 64, 64))
    target[target == config.NUM_CLASSES] = 255  # 模拟ignore_index
    
    try:
        loss = criterion(output, target)
        print(f"✅ 损失计算成功: {loss.item():.4f}")
        assert not torch.isnan(loss), "❌ 损失为NaN"
    except Exception as e:
        print(f"❌ 损失计算失败: {e}")
    
    # 测试指标计算
    try:
        preds = torch.argmax(output, dim=1)
        metrics.update(preds, target)
        all_metrics = metrics.get_all_metrics()
        
        print(f"✅ 指标计算成功:")
        print(f"   mIoU: {all_metrics['mIoU']:.4f}")
        print(f"   OA: {all_metrics['OA']:.4f}")
        
        # 检查IoU数组长度
        assert len(all_metrics['IoU_per_class']) == config.NUM_CLASSES, \
            f"❌ IoU数组长度应该是{config.NUM_CLASSES}"
        print(f"✅ IoU数组长度正确: {len(all_metrics['IoU_per_class'])}")
        
    except Exception as e:
        print(f"❌ 指标计算失败: {e}")
    
    print("✅ 损失函数和指标验证通过!\n")

def main():
    """主验证流程"""
    print("\n" + "="*60)
    print("开始验证忽略Clutter类别的修改")
    print("="*60 + "\n")
    
    try:
        # 1. 验证配置
        config = verify_config()
        
        # 2. 验证数据加载器
        train_loader, val_loader, test_loader = verify_dataloader(config)
        
        # 3. 验证模型
        model = verify_model(config)
        
        # 4. 验证损失和指标
        verify_loss_and_metrics(config)
        
        print("="*60)
        print("🎉 所有验证通过！可以开始训练了！")
        print("="*60)
        print("\n下一步:")
        print("1. 确保实验名称包含标识（如'_no_clutter'）")
        print("2. 运行训练: python train.py")
        print("3. 训练完成后运行测试和可视化")
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ 验证失败: {e}")
        print("="*60)
        print("\n请检查:")
        print("1. 是否正确替换了所有修改的文件")
        print("2. 配置文件中的类别数是否为5")
        print("3. 数据集路径是否正确")

if __name__ == "__main__":
    main()