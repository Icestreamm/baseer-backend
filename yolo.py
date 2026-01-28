"""
YOLO Model Inference Router
Handles image processing using handle_best.pt model
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import os
from typing import Optional

router = APIRouter(prefix="/yolo", tags=["yolo"])

# Global variable to store the loaded model
model: Optional[YOLO] = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "handle_best.pt")

def load_model() -> YOLO:
    """Load the YOLO model once at startup"""
    global model
    if model is None:
        # Try multiple possible paths
        possible_paths = [
            MODEL_PATH,
            "handle_best.pt",
            os.path.join("models", "handle_best.pt"),
            os.path.join(os.path.dirname(__file__), "..", "handle_best.pt"),
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(
                f"Model file handle_best.pt not found. Tried: {possible_paths}. "
                "Please upload it to the server."
            )
        
        model = YOLO(model_path)
        print(f"Model {model_path} loaded successfully")
    return model

@router.get("/health")
async def yolo_health():
    """Health check endpoint for YOLO service"""
    try:
        yolo_model = load_model()
        return {
            "status": "healthy",
            "model_loaded": yolo_model is not None,
            "model_path": MODEL_PATH
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": str(e)
        }

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict endpoint that processes image in memory without saving
    
    Args:
        file: Image file uploaded via multipart/form-data
    
    Returns:
        JSON response with detection results
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data into memory (without saving to disk)
        image_data = await file.read()
        
        # Convert to PIL Image for processing
        image = Image.open(io.BytesIO(image_data))
        
        # Convert RGBA to RGB if necessary (YOLO expects RGB)
        if image.mode == 'RGBA':
            # Create a white background
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])  # Use alpha channel as mask
            image = rgb_image
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Load model if not already loaded
        yolo_model = load_model()
        
        # Run inference
        results = yolo_model(image)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes[i]
                detection = {
                    "class": int(box.cls[0]),
                    "class_name": yolo_model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3]),
                    }
                }
                detections.append(detection)
        
        # Prepare response
        response = {
            "success": True,
            "detections": detections,
            "detection_count": len(detections),
            "image_size": {
                "width": image.width,
                "height": image.height
            }
        }
        
        return JSONResponse(content=response)
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model file not found: {str(e)}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
