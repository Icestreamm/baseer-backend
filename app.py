"""
Render Backend - YOLO Image Processing API
Handles image processing requests from the Expo app
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import io
from PIL import Image
import os
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for Expo app

# YOLO model placeholder - replace with your actual YOLO model loading
# Example: from ultralytics import YOLO
# yolo_model = YOLO('yolov8n.pt')  # or your custom model

def load_yolo_model():
    """
    Load YOLO model. Replace this with your actual model loading code.
    For now, returns None as placeholder.
    """
    try:
        # Example with ultralytics:
        # from ultralytics import YOLO
        # model_path = os.getenv('YOLO_MODEL_PATH', 'yolov8n.pt')
        # return YOLO(model_path)
        
        # Example with torch:
        # import torch
        # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # return model
        
        logger.warning("YOLO model not loaded - using placeholder. Implement load_yolo_model() with your model.")
        return None
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        return None

# Global model variable
yolo_model = load_yolo_model()

def download_image_from_url(image_url: str) -> bytes:
    """
    Download image from Supabase Storage URL
    
    Args:
        image_url: Full URL to the image in Supabase Storage
        
    Returns:
        Image bytes
    """
    try:
        logger.info(f"Downloading image from: {image_url}")
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        if not response.headers.get('content-type', '').startswith('image/'):
            raise ValueError(f"URL does not point to an image. Content-Type: {response.headers.get('content-type')}")
        
        return response.content
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading image: {e}")
        raise Exception(f"Failed to download image: {str(e)}")

def process_image_with_yolo(image_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Process image with YOLO model
    
    Args:
        image_bytes: Image file bytes
        
    Returns:
        List of detections with class, confidence, and bbox
    """
    if yolo_model is None:
        # Placeholder detections for testing
        logger.warning("YOLO model not available - returning placeholder detections")
        return [
            {
                "class": "scratch",
                "confidence": 0.85,
                "bbox": [100, 150, 200, 250]
            },
            {
                "class": "dent",
                "confidence": 0.92,
                "bbox": [300, 400, 150, 180]
            }
        ]
    
    try:
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Run YOLO inference
        # Example with ultralytics:
        # results = yolo_model(image)
        # detections = []
        # for result in results:
        #     boxes = result.boxes
        #     for box in boxes:
        #         detections.append({
        #             "class": yolo_model.names[int(box.cls)],
        #             "confidence": float(box.conf),
        #             "bbox": [
        #                 int(box.xyxy[0][0]),  # x
        #                 int(box.xyxy[0][1]),  # y
        #                 int(box.xyxy[0][2] - box.xyxy[0][0]),  # width
        #                 int(box.xyxy[0][3] - box.xyxy[0][1])   # height
        #             ]
        #         })
        
        # Example with torch/yolov5:
        # results = yolo_model(image)
        # detections = []
        # for *box, conf, cls in results.xyxy[0]:
        #     detections.append({
        #         "class": yolo_model.names[int(cls)],
        #         "confidence": float(conf),
        #         "bbox": [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])]
        #     })
        
        # For now, return placeholder
        return [
            {
                "class": "scratch",
                "confidence": 0.85,
                "bbox": [100, 150, 200, 250]
            }
        ]
    except Exception as e:
        logger.error(f"Error processing image with YOLO: {e}")
        raise Exception(f"YOLO processing failed: {str(e)}")

def generate_summary(detections: List[Dict[str, Any]]) -> str:
    """
    Generate human-readable summary of detections
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Plain text summary
    """
    if not detections:
        return "No damage detected in the image."
    
    # Group by class
    class_counts = {}
    for det in detections:
        class_name = det.get('class', 'unknown')
        if class_name not in class_counts:
            class_counts[class_name] = []
        class_counts[class_name].append(det.get('confidence', 0))
    
    # Build summary
    parts = []
    for class_name, confidences in class_counts.items():
        count = len(confidences)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        conf_percent = int(avg_conf * 100)
        
        if count == 1:
            parts.append(f"1 {class_name} ({conf_percent}% confidence)")
        else:
            parts.append(f"{count} {class_name}s (avg {conf_percent}% confidence)")
    
    summary = f"Found {len(detections)} damage area(s): " + ", ".join(parts)
    return summary

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "yolo_model_loaded": yolo_model is not None
    }), 200

@app.route('/yolo/process-url', methods=['POST'])
def process_image_url():
    """
    Process image from URL through YOLO
    
    Request body:
    {
        "image_url": "https://...",
        "user_id": "uuid" (optional)
    }
    
    Response:
    {
        "success": true,
        "detections": [...],
        "plain_text_summary": "..."
    }
    """
    try:
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        image_url = data.get('image_url')
        user_id = data.get('user_id')  # Optional, for logging
        
        if not image_url:
            return jsonify({
                "success": False,
                "error": "image_url is required"
            }), 400
        
        logger.info(f"Processing image for user {user_id}: {image_url}")
        
        # Download image
        try:
            image_bytes = download_image_from_url(image_url)
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Failed to download image: {str(e)}"
            }), 400
        
        # Process with YOLO
        try:
            detections = process_image_with_yolo(image_bytes)
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"YOLO processing failed: {str(e)}"
            }), 500
        
        # Generate summary
        summary = generate_summary(detections)
        
        logger.info(f"Successfully processed image: {len(detections)} detections found")
        
        # Return results
        return jsonify({
            "success": True,
            "detections": detections,
            "plain_text_summary": summary
        }), 200
        
    except Exception as e:
        logger.error(f"Unexpected error in process_image_url: {e}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        "service": "YOLO Image Processing API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "process": "/yolo/process-url"
        }
    }), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
