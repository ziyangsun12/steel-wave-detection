#!/usr/bin/env python3
"""
YOLOv8迁移学习脚本
用于训练识别卷钢的模型
"""

import os
import shutil
import argparse
from ultralytics import YOLO

def prepare_dataset(data_dir):
    """准备数据集
    
    Args:
        data_dir: 数据集目录
    """
    # 创建数据集目录结构
    dataset_dir = os.path.join(data_dir, 'steel_coil_dataset')
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    
    for dir_path in [dataset_dir, images_dir, labels_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # 创建训练和验证子目录
    for split in ['train', 'val']:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
    
    return dataset_dir

def create_data_yaml(dataset_dir, class_names):
    """创建data.yaml文件
    
    Args:
        dataset_dir: 数据集目录
        class_names: 类别名称列表
    """
    yaml_content = f"""
path: {dataset_dir}
train: images/train
val: images/val

nc: {len(class_names)}
names: {class_names}
"""
    
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    return yaml_path

def train_model(data_yaml, epochs=50, batch=8, imgsz=640):
    """训练模型
    
    Args:
        data_yaml: data.yaml文件路径
        epochs: 训练轮数
        batch: 批次大小
        imgsz: 图像大小
    """
    # 加载预训练模型
    model = YOLO('yolov8n.pt')
    
    # 训练模型
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device='cpu',  # 使用CPU
        project='runs/train',
        name='steel_coil'
    )
    
    # 返回最佳模型路径
    best_model = os.path.join('runs', 'train', 'steel_coil', 'weights', 'best.pt')
    return best_model

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLOv8迁移学习训练卷钢识别模型')
    parser.add_argument('--data-dir', type=str, default='data', help='数据集目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch', type=int, default=8, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='图像大小')
    args = parser.parse_args()
    
    print('准备数据集...')
    dataset_dir = prepare_dataset(args.data_dir)
    
    # 定义类别
    class_names = ['steel_coil']
    
    print('创建data.yaml文件...')
    data_yaml = create_data_yaml(dataset_dir, class_names)
    
    print('开始训练模型...')
    best_model = train_model(data_yaml, args.epochs, args.batch, args.imgsz)
    
    print(f'训练完成！最佳模型路径: {best_model}')
    print('请将训练好的模型复制到models目录下，并重命名为yolov8_steel_coil.pt')

if __name__ == '__main__':
    main()
