"""
Flask API Server for YOLO Model Inference
Deploy this on Render to process images with your local YOLO model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React Native app

# Load YOLO model
MODEL_PATH = os.getenv('MODEL_PATH', 'handle_best.pt')
try:
    model = YOLO(MODEL_PATH)
    print(f'Model loaded successfully: {MODEL_PATH}')
except Exception as e:
    print(f'Error loading model: {e}')
    model = None

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - helps verify server is running"""
    return jsonify({
        'status': 'YOLO Model Server is running',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)'
        },
        'model_loaded': model is not None,
        'model_path': MODEL_PATH
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH
    })

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    """
    Predict endpoint - matches Roboflow API format
    Accepts base64 encoded image and returns predictions
    GET method for testing, POST for actual predictions
    """
    # Handle GET requests for testing
    if request.method == 'GET':
        return jsonify({
            'status': 'predict endpoint is available',
            'method': 'Use POST to send image data',
            'content_type': 'application/x-www-form-urlencoded or application/json'
        })
    
    try:
        # Get base64 image from request
        if request.content_type == 'application/x-www-form-urlencoded':
            # Raw base64 string in body (like Roboflow)
            image_base64 = request.data.decode('utf-8')
        elif request.content_type == 'application/json':
            # JSON with image field
            data = request.json
            image_base64 = data.get('image', '')
        else:
            return jsonify({
                'error': 'Invalid content type. Use application/x-www-form-urlencoded or application/json'
            }), 400

        if not image_base64:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode base64 image
        try:
            img_data = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(img_data))
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {str(e)}'}), 400

        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'YOLO model failed to load. Check server logs.'
            }), 500

        # Run inference
        results = model(img, verbose=False)
        
        # Convert results to JSON format (similar to Roboflow)
        result = results[0]
        
        # Extract predictions
        predictions = []
        if result.boxes is not None:
            for box in result.boxes:
                predictions.append({
                    'class': result.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'x': float(box.xyxy[0][0]),
                    'y': float(box.xyxy[0][1]),
                    'width': float(box.xyxy[0][2] - box.xyxy[0][0]),
                    'height': float(box.xyxy[0][3] - box.xyxy[0][1]),
                })

        return jsonify({
            'predictions': predictions,
            'image': {
                'width': img.width,
                'height': img.height
            }
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Failed to process image'
        }), 500

@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def catch_all(path):
    """Catch-all route for debugging - shows what routes are available"""
    return jsonify({
        'error': f'Route not found: /{path}',
        'available_routes': {
            'GET /': 'Server info',
            'GET /health': 'Health check',
            'POST /predict': 'Image prediction',
            'GET /predict': 'Predict endpoint info'
        },
        'request_method': request.method,
        'request_path': request.path,
        'request_url': request.url
    }), 404

if __name__ == '__main__':
    port = int(os.getenv('PORT', 9001))
    
    # Print all registered routes for debugging
    print('=' * 60)
    print('Registered Flask Routes:')
    for rule in app.url_map.iter_rules():
        print(f'  {rule.rule} -> {rule.endpoint} [{", ".join(rule.methods)}]')
    print('=' * 60)
    
    # Increase timeout for large images and model inference
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
