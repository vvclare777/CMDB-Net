import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from configs.potsdam_config import PotsdamConfig

class PotsdamDataset(Dataset):
    def __init__(self, root_dir, split, transform):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # 数据路径
        self.img_dir = os.path.join(root_dir, split, 'images')
        self.label_dir = os.path.join(root_dir, split, 'labels')
        
        # 获取所有图像文件
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.png'))])
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # 图像和标签路径
        img_name = self.img_files[idx]  # 2_10_0_0.png，label_name与之相同
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name)
        
        # 读取图像和标签并转换为RGB格式，然后转为numpy数组
        image = np.array(Image.open(img_path).convert('RGB'))
        label = np.array(Image.open(label_path).convert('RGB'))
        
        # 将RGB标签转换为类别索引
        mask = rgb_to_mask(label)
        
        # 数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return {
            'image': image,
            'mask': mask.long(),
            'filename': img_name
        }
    
def rgb_to_mask(label_rgb):
    """将RGB标签转换为类别索引"""
    h, w = label_rgb.shape[:2]  # 获取图像高度和宽度
    mask = np.zeros((h, w), dtype=np.uint8)  # 初始化单通道标签图
    
    for class_id, color in PotsdamConfig.LABEL_COLORS.items():
        # 创建掩码：匹配所有通道的颜色
        matches = np.all(label_rgb == color, axis=-1)
        mask[matches] = class_id
    
    return mask

# def mask_to_rgb(mask):
#     """将类别索引转换为RGB可视化"""
#     h, w = mask.shape
#     rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
#     for class_id, color in PotsdamConfig.LABEL_COLORS.items():
#         rgb[mask == class_id] = color
    
#     return rgb

def get_train_transform(img_size):
    """训练集数据增强"""
    return A.Compose([
        A.RandomCrop(width=img_size, height=img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
        ], p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transform(img_size):
    """验证集数据增强"""
    return A.Compose([
        A.CenterCrop(width=img_size, height=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def create_dataloaders(config):
    # 创建数据集
    train_dataset = PotsdamDataset(
        root_dir=config.PROCESSED_DATA_DIR,
        split='train',
        transform=get_train_transform(config.TILE_SIZE)
    )
    
    val_dataset = PotsdamDataset(
        root_dir=config.PROCESSED_DATA_DIR,
        split='val',
        transform=get_val_transform(config.TILE_SIZE)
    )

    test_dataset = PotsdamDataset(
        root_dir=config.PROCESSED_DATA_DIR,
        split='test',
        transform=get_val_transform(config.TILE_SIZE)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader