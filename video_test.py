import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

# Constants
MODEL_PATH = 'complete_optimized_model.pth'
YOLO_MODEL_PATH = 'yolo11n.pt'  # Path to YOLO fruit detection model
CLASS_NAMES = [
    'freshapples', 'freshbanana', 'freshmango', 'freshoranges', 'freshtomato',
    'rottenapples', 'rottenbanana', 'rottenmango', 'rottenoranges', 'rottentomato'
]
INPUT_SIZE = (224, 224)
DETECTION_CONFIDENCE = 0.5  # Minimum confidence for fruit detection

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models():
    # Load YOLO fruit detection model
    yolo_model = YOLO(YOLO_MODEL_PATH).to(device)
    
    # Load EfficientNet freshness classifier
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES)))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()
    
    return yolo_model, model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

def realtime_classification():
    yolo_model, classifier = load_models()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    prev_time = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Stage 1: Fruit detection with YOLO
        yolo_results = yolo_model(frame, verbose=False)[0]
        detections = []
        
        for box in yolo_results.boxes:
            if box.conf.item() > DETECTION_CONFIDENCE:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((x1, y1, x2, y2))
                
                # Draw detection bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Stage 2: Freshness classification only on detected fruits
        if detections:
            for x1, y1, x2, y2 in detections:
                # Crop and preprocess detected fruit
                fruit_img = frame[y1:y2, x1:x2]
                if fruit_img.size == 0:  # Skip empty crops
                    continue
                    
                pil_img = Image.fromarray(cv2.cvtColor(fruit_img, cv2.COLOR_BGR2RGB))
                input_tensor = preprocess_image(pil_img)
                
                # Classify freshness
                with torch.no_grad():
                    outputs = classifier(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    confidence = confidence.item()
                    class_idx = predicted.item()
                
                class_name = CLASS_NAMES[class_idx]
                status = "FRESH" if 'fresh' in class_name else "ROTTEN"
                fruit_name = class_name.replace('fresh', '').replace('rotten', '').capitalize()
                
                # Display freshness info
                text_color = (0, 255, 0) if status == "FRESH" else (0, 0, 255)
                cv2.putText(frame, f"{fruit_name}: {status}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                cv2.putText(frame, f"{confidence:.0%}", (x1, y1-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Fruit Freshness Classifier', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_classification()