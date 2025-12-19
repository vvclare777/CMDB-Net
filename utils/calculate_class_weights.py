import torch
from tqdm import tqdm

def calculate_class_weights(train_loader, num_classes, method='sqrt', effective_samples=True, device='cuda', max_weight=3.0):
    """
    计算类别权重以处理类别不平衡问题
    
    支持三种计算方法:
    1. 'inverse': 逆频率 w_c = N / (n_c * C)
    2. 'sqrt': 平方根逆频率 w_c = sqrt(N / n_c) / sum(sqrt(N / n_c))
    3. 'log': 对数逆频率 w_c = log(N / n_c)
    
    有效样本数加权:
    考虑类别之间的重叠，使用有效样本数而非总样本数
    """
    print("\nStart calculating the category weights...")
    
    # 统计每个类别的像素数
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    total_pixels = 0
    
    # 遍历训练集统计
    for batch in tqdm(train_loader, desc='Distribution of statistical categories'):
        masks = batch['mask']  # [B, H, W]
        
        # 统计每个类别的像素数
        for c in range(num_classes):
            class_counts[c] += (masks == c).sum().item()
        
        total_pixels += masks.numel()  # 总像素数
    
    # 打印类别分布
    print("\nCategory distribution statistics:")
    for c in range(num_classes):
        ratio = class_counts[c] / total_pixels * 100
        print(f"  Class {c}: {int(class_counts[c])} pixels ({ratio:.2f}%)")
    
    # 计算有效样本数，处理类别重叠
    if effective_samples:
        # 使用平滑处理避免某些类别样本过少
        # 有效样本 = 实际样本 * (1 - overlap_factor)
        # overlap_factor根据类别频率动态调整
        freq = class_counts / total_pixels
        overlap_factor = 0.1 * (1 - freq)  # 频率越高，重叠越多
        effective_counts = class_counts * (1 - overlap_factor)
        print("\nUsing weighted by valid sample size")
    else:
        effective_counts = class_counts

    # 添加最小计数限制，防止除零和极端权重
    min_count = total_pixels * 0.0001  # 至少0.01%
    effective_counts = torch.clamp(effective_counts, min=min_count)
    
    # 根据方法计算权重
    if method == 'inverse':
        # 逆频率: w_c = N / (n_c * C)
        weights = total_pixels / (effective_counts * num_classes)
        print(f"Using inverse frequency weighting")
        
    elif method == 'sqrt':
        # 平方根逆频率: 更温和的权重调整
        # w_c = sqrt(N / n_c) 然后归一化
        weights = torch.sqrt(total_pixels / effective_counts)
        weights = weights / weights.sum() * num_classes  # 归一化使平均权重为1
        print(f"Using square root inverse frequency weighting")
        
    elif method == 'log':
        # 对数逆频率: 最温和的权重调整
        # w_c = log(N / n_c + 1)
        weights = torch.log(total_pixels / effective_counts + 1)
        weights = weights / weights.sum() * num_classes
        print(f"Using logarithmic inverse frequency weighting")
        
    else:
        raise ValueError(f"Unsupported weighting method: {method}")
    
    # 更严格的权重上下界限制，避免极端值导致梯度爆炸
    min_weight = 0.5
    weights = torch.clamp(weights, min=min_weight, max=max_weight)
    
    # 再次归一化，确保平均权重为1
    weights = weights / weights.mean()
    
    # 打印最终权重
    print("\nFinal class weights:")
    for c in range(num_classes):
        print(f"  Class {c}: {weights[c]:.4f}")

    print(f"\nWeight statistics:")
    print(f"  Minimum weight: {weights.min():.4f}")
    print(f"  Maximum weight: {weights.max():.4f}")
    print(f"  Average weight: {weights.mean():.4f}")
    print(f"  Weight range: {(weights.min() - weights.max()):.4f}")
    
    return weights.to(device)