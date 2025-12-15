import sys
sys.path.append('/mnt/e/Github/demo')
import os
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import random
from configs.potsdam_config import PotsdamConfig

class PotsdamPreprocessor:
    def __init__(self, config):
        self.config = config
        self.label_color = config.LABEL_COLORS
    
    def get_image_label_pairs(self):
        """获取图像-标签对"""
        image_files = sorted([f for f in os.listdir(self.config.RGB_DIR) if f.endswith('.tif')])
        
        pairs = []
        for img_file in image_files:
            img_path = os.path.join(self.config.RGB_DIR, img_file)
            label_file = os.path.basename(img_file).replace('_RGB.tif', '_label.tif')
            label_path = os.path.join(self.config.LABEL_DIR, label_file)
            
            if os.path.exists(label_path):
                pairs.append((img_path, label_path))
            else:
                print(f"Warning: Label path not found {label_path}")
        
        print(f"Found {len(pairs)} image-label pairs")
        return pairs
    
    def split_dataset(self, pairs, train_ratio, val_ratio):
        """划分数据集 7:2:1"""
        random.seed(42)  # 固定随机种子以保证可重复性
        random.shuffle(pairs)
        
        total = len(pairs)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        train_pairs = pairs[:train_end]
        val_pairs = pairs[train_end:val_end]
        test_pairs = pairs[val_end:]
        
        print(f"\nDataset division:")
        print(f"  Training set: {len(train_pairs)} ({len(train_pairs)/total*100:.1f}%)")
        print(f"  Validation set: {len(val_pairs)} ({len(val_pairs)/total*100:.1f}%)")
        print(f"  Test set: {len(test_pairs)} ({len(test_pairs)/total*100:.1f}%)")
        
        return train_pairs, val_pairs, test_pairs
    
    def crop_image(self, image, tile_size, stride):
        """将大图切分为固定大小的小块"""
        h, w = image.shape[:2]
        tiles = []
        positions = []
        
        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                tile = image[y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
                positions.append((y, x))
        
        return tiles, positions
    
    def process_pair(self, img_path, label_path, output_img_dir, output_label_dir):
        """处理单个图像-标签对"""
        # 读取图像和标签
        image = np.array(Image.open(img_path))
        label = np.array(Image.open(label_path))
        
        # 切片
        img_tiles, positions = self.crop_image(image, self.config.TILE_SIZE, self.config.STRIDE)
        label_tiles, _ = self.crop_image(label, self.config.TILE_SIZE, self.config.STRIDE)
        
        saved_count = 0  # 保存的切片计数
        base_name = os.path.basename(img_path).replace('top_potsdam_', '').replace('_RGB.tif', '')  # 2_10
        
        for idx, (img_tile, label_tile, (y, x)) in enumerate(zip(img_tiles, label_tiles, positions)):            
            # 保存切片
            tile_name = f"{base_name}_{y}_{x}.png"
            
            img_save_path = os.path.join(output_img_dir, tile_name)
            label_save_path = os.path.join(output_label_dir, tile_name)
            
            Image.fromarray(img_tile).save(img_save_path)
            Image.fromarray(label_tile).save(label_save_path)
            
            saved_count += 1
        
        return saved_count
    
    def save_statistics(self, train_tiles, val_tiles, test_tiles, train_imgs, val_imgs, test_imgs):
        """保存统计信息"""
        stats_file = os.path.join(self.config.PROCESSED_DATA_DIR, 'dataset_statistics.txt')
        
        with open(stats_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Statistical information of the Potsdam dataset\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Tile size: {self.config.TILE_SIZE}x{self.config.TILE_SIZE}\n")
            f.write(f"Overlap pixels: {self.config.OVERLAP}\n")
            f.write(f"Stride: {self.config.STRIDE}\n\n")
            
            f.write("Number of original images:\n")
            f.write(f"  Training set: {train_imgs}\n")
            f.write(f"  Validation set: {val_imgs}\n")
            f.write(f"  Test set: {test_imgs}\n")
            f.write(f"  Total: {train_imgs + val_imgs + test_imgs}\n\n")
            
            f.write("Number of tiles:\n")
            f.write(f"  Training set: {train_tiles}\n")
            f.write(f"  Validation set: {val_tiles}\n")
            f.write(f"  Test set: {test_tiles}\n")
            f.write(f"  Total: {train_tiles + val_tiles + test_tiles}\n\n")
            
            f.write("Dataset paths:\n")
            f.write(f"  Training images: {self.config.TRAIN_IMG_DIR}\n")
            f.write(f"  Training labels: {self.config.TRAIN_LABEL_DIR}\n")
            f.write(f"  Validation images: {self.config.VAL_IMG_DIR}\n")
            f.write(f"  Validation labels: {self.config.VAL_LABEL_DIR}\n")
            f.write(f"  Test images: {self.config.TEST_IMG_DIR}\n")
            f.write(f"  Test labels: {self.config.TEST_LABEL_DIR}\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"\nStatistics saved to: {stats_file}")
    
    def process_dataset(self):
        """处理整个数据集"""
        print("=" * 80)
        print("Preprocessing of the Potsdam dataset...")
        print("=" * 80)
        print(f"Original data path: {self.config.RAW_DATA_DIR}")
        print(f"Output path: {self.config.PROCESSED_DATA_DIR}")
        print(f"Tile size: {self.config.TILE_SIZE}x{self.config.TILE_SIZE}")
        print(f"Overlap: {self.config.OVERLAP} pixels")
        print(f"Stride: {self.config.STRIDE} pixels")
        print("=" * 80)
        
        # 获取图像-标签对
        pairs = self.get_image_label_pairs()
        
        # 划分数据集
        train_pairs, val_pairs, test_pairs = self.split_dataset(pairs, self.config.TRAIN_RATIO, self.config.VAL_RATIO)

        # 处理训练集
        print("\nProcessing training set...")
        train_total = 0
        for (img_path, label_path) in tqdm(train_pairs, desc="Training set"):
            count = self.process_pair(
                img_path, label_path,
                self.config.TRAIN_IMG_DIR, self.config.TRAIN_LABEL_DIR,
            )
            train_total += count
        
        # 处理验证集
        print("\nProcessing validation set...")
        val_total = 0
        for (img_path, label_path) in tqdm(val_pairs, desc="Validation set"):
            count = self.process_pair(
                img_path, label_path,
                self.config.VAL_IMG_DIR, self.config.VAL_LABEL_DIR,
            )
            val_total += count
        
        # 处理测试集
        print("\nProcessing test set...")
        test_total = 0
        for (img_path, label_path) in tqdm(test_pairs, desc="Test set"):
            count = self.process_pair(
                img_path, label_path,
                self.config.TEST_IMG_DIR, self.config.TEST_LABEL_DIR,
            )
            test_total += count
        
        # 统计信息
        print("\n" + "=" * 80)
        print("Preprocessing completed!")
        print("=" * 80)
        print(f"Training set tiles: {train_total}")
        print(f"Validation set tiles: {val_total}")
        print(f"Test set tiles: {test_total}")
        print(f"Total tiles: {train_total + val_total + test_total}")
        print("=" * 80)
        
        # 保存统计信息
        self.save_statistics(train_total, val_total, test_total, len(train_pairs), len(val_pairs), len(test_pairs))

if __name__ == "__main__":
    PotsdamConfig.create_dirs()
    preprocessor = PotsdamPreprocessor(PotsdamConfig)
    preprocessor.process_dataset()
