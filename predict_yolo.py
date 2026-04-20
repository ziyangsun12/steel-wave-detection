#!/usr/bin/env python3
"""
YOLOv8 prediction script for steel coil detection
"""

import os
import cv2
import numpy as np
import time

def load_yolo_model():
    """
    Load YOLOv8 model using Ultralytics
    """
    from ultralytics import YOLO
    # Load our trained model
    model_path = "models/yolov8_steel_coil.pt"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None
    
    # Load model
    model = YOLO(model_path)
    return model

def preprocess_image(image):
    """
    Preprocess image for YOLOv8
    """
    blob = cv2.dnn.blobFromImage(
        image, 1/255.0, (640, 640), swapRB=True, crop=False
    )
    return blob

def postprocess_output(outputs, image_shape, conf_threshold=0.5):
    """
    Postprocess YOLOv8 output
    """
    boxes = []
    confidences = []
    class_ids = []
    
    # YOLOv8 output shape: (1, 84, 8400)
    output = outputs[0]
    rows = output.shape[1]
    
    img_h, img_w = image_shape[:2]
    x_factor = img_w / 640
    y_factor = img_h / 640
    
    for i in range(rows):
        row = output[0, i]
        confidence = row[4]
        
        if confidence >= conf_threshold:
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)
            
            # Only consider 'person' class (class 0) as steel coil for now
            # We'll fine-tune this later
            if class_id == 0:
                continue
            
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            
            left = int((cx - w/2) * x_factor)
            top = int((cy - h/2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            
            boxes.append([left, top, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.45)
    
    results = []
    for i in indices:
        box = boxes[i]
        left, top, width, height = box
        results.append({
            'box': [left, top, width, height],
            'confidence': confidences[i],
            'class_id': class_ids[i]
        })
    
    return results

def detect_steel_coil(image_path):
    """
    Detect steel coil in image
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return False, []
    
    # Load model
    model = load_yolo_model()
    if model is None:
        # Fallback: color-based detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([150, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2
        
        # Check if there's enough red area
        red_area = cv2.countNonZero(mask)
        total_area = image.shape[0] * image.shape[1]
        if red_area / total_area > 0.05:  # 5% of image area
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            results = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 10000:
                    x, y, w, h = cv2.boundingRect(contour)
                    results.append({
                        'box': [x, y, w, h],
                        'confidence': 0.8,
                        'class_id': 0
                    })
            return True, results
        return False, []
    
    # Forward pass
    start_time = time.time()
    results = model(image)
    end_time = time.time()
    
    # Process results
    processed_results = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            class_id = box.cls[0]
            
            # Convert to [x, y, width, height]
            x = int(x1)
            y = int(y1)
            width = int(x2 - x1)
            height = int(y2 - y1)
            
            processed_results.append({
                'box': [x, y, width, height],
                'confidence': float(confidence),
                'class_id': int(class_id)
            })
    
    # Check if steel coil is detected
    has_coil = len(processed_results) > 0
    
    return has_coil, processed_results

def process_image(image_path, output_path):
    """
    Process image and save result
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return
    
    # Detect steel coil
    has_coil, results = detect_steel_coil(image_path)
    
    # Draw bounding boxes
    for result in results:
        box = result['box']
        left, top, width, height = box
        confidence = result['confidence']
        
        # Draw rectangle
        cv2.rectangle(image, (left, top), (left + width, top + height), (0, 255, 0), 2)
        
        # Draw label
        label = f"Steel Coil: {confidence:.2f}"
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # If no YOLO results but color-based detection found coil
    if not results and has_coil:
        # Find contours of red areas
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([150, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, "Steel Coil", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save result
    cv2.imwrite(output_path, image)
    print(f"Processed: {image_path} -> {output_path}")

def process_directory(input_dir, output_dir):
    """
    Process all images in directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    
    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)
        process_image(input_path, output_path)

if __name__ == "__main__":
    input_dir = "data/steel_coil_dataset/images/train"
    output_dir = "output/predictions"
    process_directory(input_dir, output_dir)
