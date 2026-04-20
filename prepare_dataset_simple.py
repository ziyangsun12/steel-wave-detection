#!/usr/bin/env python3
"""
简单的数据集准备脚本
从data文件夹的子文件夹中读取视频，创建数据集结构
"""

import os
import argparse

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='简单的数据集准备脚本')
    parser.add_argument('--data-dir', type=str, default='data', help='数据文件夹路径')
    parser.add_argument('--output-dir', type=str, default='data/steel_coil_dataset', help='输出目录')
    args = parser.parse_args()
    
    # 创建输出目录结构
    images_train_dir = os.path.join(args.output_dir, 'images', 'train')
    images_val_dir = os.path.join(args.output_dir, 'images', 'val')
    labels_train_dir = os.path.join(args.output_dir, 'labels', 'train')
    labels_val_dir = os.path.join(args.output_dir, 'labels', 'val')
    
    for dir_path in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 遍历data文件夹及其子文件夹
    video_files = []
    for root, dirs, files in os.walk(args.data_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                video_files.append(video_path)
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频
    for i, video_path in enumerate(video_files):
        # 从文件名提取标签
        video_name = os.path.basename(video_path)
        label = os.path.splitext(video_name)[0]
        
        # 决定是训练集还是验证集（80%训练，20%验证）
        if i % 5 == 0:  # 20% 验证集
            output_images_dir = images_val_dir
            output_labels_dir = labels_val_dir
        else:  # 80% 训练集
            output_images_dir = images_train_dir
            output_labels_dir = labels_train_dir
        
        # 创建标签文件（示例）
        label_filename = f"{label}_0000.txt"
        label_path = os.path.join(output_labels_dir, label_filename)
        
        with open(label_path, 'w') as f:
            # YOLO格式: class_id x_center y_center width height
            f.write(f"0 0.5 0.5 0.5 0.5\n")
        
        print(f"处理视频: {video_name} -> 标签: {label}")
    
    # 创建data.yaml文件
    data_yaml_path = os.path.join(args.output_dir, 'data.yaml')
    with open(data_yaml_path, 'w') as f:
        f.write(f"""
path: {os.path.abspath(args.output_dir)}
train: images/train
val: images/val

nc: 1
names: ['steel_coil']
""")
    
    print(f"\n数据集准备完成！")
    print(f"数据集目录: {os.path.abspath(args.output_dir)}")
    print(f"data.yaml文件: {os.path.abspath(data_yaml_path)}")
    print("\n注意：由于系统限制，本脚本仅创建了数据集结构和标签文件，")
    print("但没有从视频中提取帧。请手动从视频中提取帧并放入对应的images目录。")

if __name__ == '__main__':
    main()
