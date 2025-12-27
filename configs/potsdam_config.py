import os

class PotsdamConfig:
    EXP_NAME = 'exp3_12_26'

    """Potsdam数据集配置"""
    # Potsdam标签颜色映射 (RGB)
    LABEL_COLORS = {
        0: [255, 255, 255],  # Impervious surfaces - 白色
        1: [0, 0, 255],      # Building - 蓝色
        2: [0, 255, 255],    # Low vegetation - 青色
        3: [0, 255, 0],      # Tree - 绿色
        4: [255, 255, 0],    # Car - 黄色
        5: [255, 0, 0],      # Clutter - 红色
    }
    CLASS_NAMES = [
        'Impervious surfaces',
        'Building',
        'Low vegetation',
        'Tree',
        'Car',
        'Clutter'
    ]
    NUM_CLASSES = 6

    # 数据集路径配置
    RAW_DATA_DIR = 'data/Potsdam'
    PROCESSED_DATA_DIR = 'data/Potsdam_processed'

    # 原始数据子目录
    RGB_DIR = os.path.join(RAW_DATA_DIR, 'RGB')
    LABEL_DIR = os.path.join(RAW_DATA_DIR, 'Label')

    # 处理后数据目录
    TRAIN_IMG_DIR = os.path.join(PROCESSED_DATA_DIR, 'train/images')
    TRAIN_LABEL_DIR = os.path.join(PROCESSED_DATA_DIR, 'train/labels')
    VAL_IMG_DIR = os.path.join(PROCESSED_DATA_DIR, 'val/images')
    VAL_LABEL_DIR = os.path.join(PROCESSED_DATA_DIR, 'val/labels')
    TEST_IMG_DIR = os.path.join(PROCESSED_DATA_DIR, 'test/images')
    TEST_LABEL_DIR = os.path.join(PROCESSED_DATA_DIR, 'test/labels')

    # 检查点保存
    CHECKPOINT_DIR = 'checkpoints'

    # 数据集分割配置
    TRAIN_RATIO = 0.7  # 训练集比例
    VAL_RATIO = 0.2   # 验证集比例

    # 图像切片配置
    TILE_SIZE = 512  # 切片大小
    OVERLAP = 256    # 重叠区域大小
    STRIDE = TILE_SIZE - OVERLAP  # 步长
    IMG_SIZE = 256  # 输入图像大小

    # 模型训练配置
    VIS_INTERVAL = 5  # 每5个epoch可视化一次预测结果
    BATCH_SIZE = 4
    NUM_EPOCHS = 50
    NUM_WORKERS = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    PRETRAINED = True
    LOSS = 'focal_dice'
    OPTIMIZER = 'adamw'
    SCHEDULER = 'cosine'

    @staticmethod
    def create_dirs():
        dirs = [
            PotsdamConfig.PROCESSED_DATA_DIR,
            PotsdamConfig.TRAIN_IMG_DIR,
            PotsdamConfig.TRAIN_LABEL_DIR,
            PotsdamConfig.VAL_IMG_DIR,
            PotsdamConfig.VAL_LABEL_DIR,
            PotsdamConfig.TEST_IMG_DIR,
            PotsdamConfig.TEST_LABEL_DIR,
            PotsdamConfig.CHECKPOINT_DIR,
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)