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

# Register predict route - MUST be before catch-all
@app.route('/predict', methods=['POST', 'GET', 'OPTIONS'])
def predict():
    """
    Predict endpoint - matches Roboflow API format
    Accepts base64 encoded image and returns predictions
    GET method for testing, POST for actual predictions
    """
    print(f'Predict endpoint called with method: {request.method}')
    
    # Handle OPTIONS for CORS preflight
    if request.method == 'OPTIONS':
        return '', 200
    
    # Handle GET requests for testing
    if request.method == 'GET':
        return jsonify({
            'status': 'predict endpoint is available',
            'method': 'Use POST to send image data',
            'content_type': 'application/x-www-form-urlencoded or application/json',
            'endpoint': '/predict',
            'registered': True
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

# Common bot/scanner paths that should be handled silently
BOT_PATTERNS = [
    '/admin', '/wp-admin', '/wp-login', '/.env', '/.git',
    '/phpmyadmin', '/mysql', '/sql', '/backup', '/config',
    '/api', '/v1', '/v2', '/graphql', '/swagger', '/docs',
    '/login', '/signin', '/register', '/signup',
    '/.well-known', '/robots.txt', '/sitemap.xml',
    '/test', '/debug', '/console', '/shell',
    # Add more common probe patterns here
]

def is_bot_request(path, user_agent=''):
    """Detect if request is likely from a bot/scanner"""
    path_lower = path.lower()
    user_agent_lower = (user_agent or '').lower()
    
    # Check against known bot patterns
    for pattern in BOT_PATTERNS:
        if path_lower.startswith(pattern.lower()):
            return True
    
    # Check for common bot user agents
    bot_agents = ['bot', 'crawler', 'spider', 'scanner', 'curl', 'wget', 'python-requests']
    if any(agent in user_agent_lower for agent in bot_agents):
        return True
    
    # Check for suspicious path patterns
    suspicious = ['.php', '.asp', '.jsp', '.exe', '.sh', '.py']
    if any(ext in path_lower for ext in suspicious):
        return True
    
    return False

# Catch-all route MUST be last - handles 404s
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'HEAD'])
def catch_all(path):
    """Catch-all route - handles 404s with smart bot detection"""
    user_agent = request.headers.get('User-Agent', '')
    is_bot = is_bot_request(f'/{path}', user_agent)
    
    # Only log non-bot requests to reduce log noise
    if not is_bot:
        print(f'[Real User] 404 for: /{path} (method: {request.method}, UA: {user_agent[:50]})')
    # Bot requests are silently handled (no logging)
    
    # Return minimal response for bots, helpful response for real users
    if is_bot:
        # Minimal response for bots - no details
        return jsonify({'detail': 'Not Found'}), 404
    else:
        # Helpful response for real users
        return jsonify({
            'error': 'Route not found',
            'path': f'/{path}',
            'available_endpoints': {
                'GET /': 'Server info',
                'GET /health': 'Health check',
                'POST /predict': 'Image prediction (YOLO)',
                'GET /predict': 'Predict endpoint info'
            }
        }), 404

if __name__ == '__main__':
    port = int(os.getenv('PORT', 9001))
    
    # Print startup info and verify routes
    print('=' * 60)
    print('Flask Application Starting...')
    print('=' * 60)
    
    # Verify predict route exists
    predict_routes = [rule for rule in app.url_map.iter_rules() if 'predict' in rule.rule]
    if predict_routes:
        print(f'✅ /predict route REGISTERED: {predict_routes[0].rule} [{", ".join([m for m in predict_routes[0].methods if m not in ["HEAD", "OPTIONS"]])}]')
    else:
        print('❌ ERROR: /predict route NOT FOUND!')
        print('This is a critical error - the route should be registered.')
    
    print('=' * 60)
    print('Registered Routes:')
    for rule in app.url_map.iter_rules():
        if rule.rule != '/<path:path>':  # Don't show catch-all
            methods = ", ".join([m for m in rule.methods if m not in ["HEAD", "OPTIONS"]])
            print(f'  {rule.rule} -> {rule.endpoint} [{methods}]')
    print('=' * 60)
    print('Bot traffic will be handled silently (minimal logging)')
    print('=' * 60)
    
    # Increase timeout for large images and model inference
    print(f'Starting Flask server on port {port}...')
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
