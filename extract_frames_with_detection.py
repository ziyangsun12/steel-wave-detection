#!/usr/bin/env python3
"""
Extract frames from videos with steel coil detection
"""

import os
import cv2
import numpy as np

def detect_steel_coil(image):
    """
    Detect steel coil in image
    Returns (has_coil, bounding_box)
    bounding_box: [x, y, width, height] in pixel coordinates
    """
    # Convert to HSV color space for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define red color range (adjusted for high-temperature steel)
    lower_red1 = np.array([0, 20, 20])
    upper_red1 = np.array([30, 255, 255])
    lower_red2 = np.array([130, 20, 20])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2
    
    # Calculate red area percentage
    red_area = cv2.countNonZero(mask)
    total_area = image.shape[0] * image.shape[1]
    red_percentage = red_area / total_area
    
    # Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if any contour is large enough and rectangular
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 3000:  # Lowered minimum area threshold
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio (should be rectangular)
            aspect_ratio = w / h
            if 0.8 < aspect_ratio < 20:  # Widened aspect ratio range
                # Check if the contour is in the center part of the image (where steel coil typically appears)
                img_h, img_w = image.shape[:2]
                center_x = x + w/2
                center_y = y + h/2
                
                if 0.2 < center_x/img_w < 0.8 and 0.2 < center_y/img_h < 0.8:
                    print(f"Steel coil detected: area={area}, aspect_ratio={aspect_ratio}, red_percentage={red_percentage:.2f}")
                    return True, [x, y, w, h]
    
    # Also check for high red percentage (common in high-temperature steel)
    if red_percentage > 0.15:
        print(f"Steel coil detected by red percentage: {red_percentage:.2f}")
        # Create a bounding box around the red area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return True, [x, y, w, h]
    
    print(f"No steel coil detected: red_percentage={red_percentage:.2f}")
    return False, None

def extract_frames_from_video(video_path, output_dir, frames_per_video=100):
    """
    Extract frames from a video file, only keeping frames with steel coil
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return 0
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame step to extract approximately frames_per_video frames
    step = max(1, total_frames // (frames_per_video * 1.5))  # Check more frames
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame every step frames
        if frame_count % step == 0:
            # Detect steel coil
            has_coil, bbox = detect_steel_coil(frame)
            
            if has_coil:
                # Create filename with sanitized video name
                video_name = os.path.basename(video_path)
                video_name = os.path.splitext(video_name)[0]
                # Sanitize video name to only include ASCII characters
                video_name = ''.join(c for c in video_name if ord(c) < 128)
                video_name = video_name.strip().replace(' ', '_')
                # Limit filename length to avoid issues
                video_name = video_name[:50]
                frame_filename = f"{video_name}_{extracted_count:04d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                
                # Save frame
                try:
                    cv2.imwrite(frame_path, frame)
                    extracted_count += 1
                except Exception as e:
                    print(f"Error saving frame: {e}")
                
                # Stop if we've extracted enough frames
                if extracted_count >= frames_per_video:
                    break
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {extracted_count} frames with steel coil from {video_path}")
    return extracted_count

def extract_frames_from_directory(data_dir, output_dir, frames_per_video=100):
    """
    Extract frames from all videos in a directory
    """
    # Create output directory structure
    train_dir = os.path.join(output_dir, 'images', 'train')
    val_dir = os.path.join(output_dir, 'images', 'val')
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    
    # Get list of video files
    video_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_files.append(os.path.join(root, file))
    
    print(f"Found {len(video_files)} video files")
    
    total_extracted = 0
    
    # Process each video
    for i, video_file in enumerate(video_files):
        # Determine if it's training or validation set (80% training, 20% validation)
        if i % 5 == 0:  # 20% validation set
            output_subdir = val_dir
        else:  # 80% training set
            output_subdir = train_dir
        
        extracted = extract_frames_from_video(video_file, output_subdir, frames_per_video)
        total_extracted += extracted
    
    print(f"Frame extraction completed! Total frames extracted: {total_extracted}")
    return total_extracted

def create_label_files(output_dir):
    """
    Create label files for all images
    """
    # Create label directories
    train_labels_dir = os.path.join(output_dir, 'labels', 'train')
    val_labels_dir = os.path.join(output_dir, 'labels', 'val')
    
    if not os.path.exists(train_labels_dir):
        os.makedirs(train_labels_dir)
        print(f"Created directory: {train_labels_dir}")
    if not os.path.exists(val_labels_dir):
        os.makedirs(val_labels_dir)
        print(f"Created directory: {val_labels_dir}")
    
    # Process train images
    train_images_dir = os.path.join(output_dir, 'images', 'train')
    if os.path.exists(train_images_dir):
        image_files = [f for f in os.listdir(train_images_dir) if f.endswith('.jpg')]
        print(f"Found {len(image_files)} image files in {train_images_dir}")
        
        for image_file in image_files:
            # Read image to detect steel coil
            image_path = os.path.join(train_images_dir, image_file)
            image = cv2.imread(image_path)
            
            if image is not None:
                # Detect steel coil
                has_coil, bbox = detect_steel_coil(image)
                
                if has_coil:
                    # Create label file
                    label_file = image_file.replace('.jpg', '.txt')
                    label_path = os.path.join(train_labels_dir, label_file)
                    
                    # Convert to YOLO format (normalized coordinates)
                    img_h, img_w = image.shape[:2]
                    x, y, w, h = bbox
                    
                    # Calculate normalized coordinates
                    x_center = (x + w/2) / img_w
                    y_center = (y + h/2) / img_h
                    width = w / img_w
                    height = h / img_h
                    
                    # Write YOLO format label
                    with open(label_path, 'w') as f:
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    print(f"Created label file: {label_path}")
                else:
                    # Remove image if no coil detected
                    os.remove(image_path)
                    print(f"Removed image without coil: {image_path}")
            else:
                print(f"Could not read image: {image_path}")
    
    # Process val images
    val_images_dir = os.path.join(output_dir, 'images', 'val')
    if os.path.exists(val_images_dir):
        image_files = [f for f in os.listdir(val_images_dir) if f.endswith('.jpg')]
        print(f"Found {len(image_files)} image files in {val_images_dir}")
        
        for image_file in image_files:
            # Read image to detect steel coil
            image_path = os.path.join(val_images_dir, image_file)
            image = cv2.imread(image_path)
            
            if image is not None:
                # Detect steel coil
                has_coil, bbox = detect_steel_coil(image)
                
                if has_coil:
                    # Create label file
                    label_file = image_file.replace('.jpg', '.txt')
                    label_path = os.path.join(val_labels_dir, label_file)
                    
                    # Convert to YOLO format (normalized coordinates)
                    img_h, img_w = image.shape[:2]
                    x, y, w, h = bbox
                    
                    # Calculate normalized coordinates
                    x_center = (x + w/2) / img_w
                    y_center = (y + h/2) / img_h
                    width = w / img_w
                    height = h / img_h
                    
                    # Write YOLO format label
                    with open(label_path, 'w') as f:
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    print(f"Created label file: {label_path}")
                else:
                    # Remove image if no coil detected
                    os.remove(image_path)
                    print(f"Removed image without coil: {image_path}")
            else:
                print(f"Could not read image: {image_path}")
    
    print("Label files created!")

def create_data_yaml(output_dir):
    """
    Create data.yaml file
    """
    data_yaml_path = os.path.join(output_dir, 'data.yaml')
    
    content = f"""path: {output_dir}
train: images/train
val: images/val

nc: 1
names: ['steel_coil']
"""
    
    with open(data_yaml_path, 'w') as f:
        f.write(content)
    
    print(f"data.yaml file created at: {data_yaml_path}")

def rename_files_to_english(output_dir):
    """
    Rename image and label files to English
    """
    # Process train directory
    train_images_dir = os.path.join(output_dir, 'images', 'train')
    train_labels_dir = os.path.join(output_dir, 'labels', 'train')
    
    if os.path.exists(train_images_dir):
        image_files = [f for f in os.listdir(train_images_dir) if f.endswith('.jpg')]
        for i, image_file in enumerate(image_files):
            # Rename image file
            old_image_path = os.path.join(train_images_dir, image_file)
            new_image_name = f"steel_coil_train_{i:04d}.jpg"
            new_image_path = os.path.join(train_images_dir, new_image_name)
            if os.path.exists(new_image_path):
                os.remove(new_image_path)
            os.rename(old_image_path, new_image_path)
            
            # Rename corresponding label file
            old_label_name = image_file.replace('.jpg', '.txt')
            old_label_path = os.path.join(train_labels_dir, old_label_name)
            if os.path.exists(old_label_path):
                new_label_name = new_image_name.replace('.jpg', '.txt')
                new_label_path = os.path.join(train_labels_dir, new_label_name)
                if os.path.exists(new_label_path):
                    os.remove(new_label_path)
                os.rename(old_label_path, new_label_path)
    
    # Process val directory
    val_images_dir = os.path.join(output_dir, 'images', 'val')
    val_labels_dir = os.path.join(output_dir, 'labels', 'val')
    
    if os.path.exists(val_images_dir):
        image_files = [f for f in os.listdir(val_images_dir) if f.endswith('.jpg')]
        for i, image_file in enumerate(image_files):
            # Rename image file
            old_image_path = os.path.join(val_images_dir, image_file)
            new_image_name = f"steel_coil_val_{i:04d}.jpg"
            new_image_path = os.path.join(val_images_dir, new_image_name)
            if os.path.exists(new_image_path):
                os.remove(new_image_path)
            os.rename(old_image_path, new_image_path)
            
            # Rename corresponding label file
            old_label_name = image_file.replace('.jpg', '.txt')
            old_label_path = os.path.join(val_labels_dir, old_label_name)
            if os.path.exists(old_label_path):
                new_label_name = new_image_name.replace('.jpg', '.txt')
                new_label_path = os.path.join(val_labels_dir, new_label_name)
                if os.path.exists(new_label_path):
                    os.remove(new_label_path)
                os.rename(old_label_path, new_label_path)
    
    print("Files renamed to English!")

if __name__ == "__main__":
    try:
        data_dir = "data"
        output_dir = "data/steel_coil_dataset"
        
        # Clear existing dataset
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        # Extract frames with steel coil detection
        extract_frames_from_directory(data_dir, output_dir, frames_per_video=50)
        
        # Create label files with actual bounding boxes
        create_label_files(output_dir)
        
        # Rename files to English
        rename_files_to_english(output_dir)
        
        # Create data.yaml file
        create_data_yaml(output_dir)
        
        print("\nDataset preparation completed!")
        print(f"Dataset directory: {os.path.abspath(output_dir)}")
        print(f"data.yaml file: {os.path.abspath(os.path.join(output_dir, 'data.yaml'))}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
