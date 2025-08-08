from flask import Flask, render_template, Response, request, jsonify
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import os
import time
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Constants
YOLO_MODEL_PATH = 'yolo11n.pt'
CLASSIFIER_PATH = 'complete_optimized_model.pth'
CLASS_NAMES = [
    'freshapples', 'freshbanana', 'freshmango', 'freshoranges', 'freshtomato',
    'rottenapples', 'rottenbanana', 'rottenmango', 'rottenoranges', 'rottentomato'
]
INPUT_SIZE = (224, 224)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for camera capture
camera = None
last_capture_time = 0
captured_frames = []
current_status = "Ready"
prediction_result = "No fruit detected"

# Load models
def load_models():
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    checkpoint = torch.load(CLASSIFIER_PATH, map_location='cpu')
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return yolo_model, model

yolo_model, classifier = load_models()

# Video capture generator
def gen_frames():
    global camera, last_capture_time, captured_frames, current_status, prediction_result
    
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Process frame every 2 seconds
        current_time = time.time()
        if current_time - last_capture_time > 2:
            last_capture_time = current_time
            processed_frame, status_update = process_frame(frame.copy())
            current_status = status_update
            frame = processed_frame
        
        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def process_frame(frame):
    global captured_frames, prediction_result
    
    # Fruit detection
    results = yolo_model(frame, verbose=False)[0]
    valid_detections = []
    
    for box in results.boxes:
        if box.conf.item() > 0.7:  # Confidence threshold
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            valid_detections.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    status = "Scanning for fruits..."
    
    if valid_detections:
        status = "Fruit detected - analyzing..."
        for x1, y1, x2, y2 in valid_detections:
            fruit_img = frame[y1:y2, x1:x2]
            
            if fruit_img.size > 0:
                # Preprocess and classify
                pil_img = Image.fromarray(cv2.cvtColor(fruit_img, cv2.COLOR_BGR2RGB))
                input_tensor = transforms.Compose([
                    transforms.Resize(INPUT_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])(pil_img).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = classifier(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    class_idx = predicted.item()
                
                class_name = CLASS_NAMES[class_idx]
                status = "Fresh" if 'fresh' in class_name else "Rotten"
                fruit_name = class_name.replace('fresh', '').replace('rotten', '').capitalize()
                prediction_result = f"{fruit_name}: {status} (Confidence: {confidence.item():.2f})"
                
                # Add text to frame
                text_color = (0, 255, 0) if status == "Fresh" else (0, 0, 255)
                cv2.putText(frame, f"{fruit_name}: {status}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    return frame, status

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_status')
def camera_status():
    global current_status, prediction_result
    return jsonify({
        'status': current_status,
        'prediction': prediction_result,
        'timer_active': True,
        'frame1_captured': False,
        'frame2_captured': False
    })

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        image = cv2.imread(filepath)
        processed_image, _ = process_frame(image)
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + filename)
        cv2.imwrite(processed_path, processed_image)
        
        # Get prediction (simplified for example)
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_tensor = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(pil_img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = classifier(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            class_idx = predicted.item()
        
        class_name = CLASS_NAMES[class_idx]
        status = "Fresh" if 'fresh' in class_name else "Rotten"
        fruit_name = class_name.replace('fresh', '').replace('rotten', '').capitalize()
        
        return jsonify({
            'prediction': f"{fruit_name}: {status}",
            'confidence': f"{confidence.item():.2f}",
            'image_url': f"/static/uploads/processed_{filename}"
        })
    
    return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)