"""
Flask API Server for YOLO Model Inference
Runs YOLO model (handle_best.pt) behind a clean /predict API.
Model loads lazily on first prediction to avoid Render startup timeout.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)  # Allow calls from your Expo / React Native iOS app

# Global model variable — will be loaded only when needed
model = None
MODEL_PATH = os.getenv("MODEL_PATH", "handle_best.pt")

def load_model():
    """Load the YOLO model only when first needed (lazy loading)."""
    global model
    if model is not None:
        return  # already loaded

    try:
        print(f"[YOLO] Loading model from {MODEL_PATH} ... (may take 30-120s first time)")
        model = YOLO(MODEL_PATH)
        print(f"[YOLO] Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"[YOLO] ERROR loading model from {MODEL_PATH}: {e}")
        model = None


@app.route("/", methods=["GET"])
def root():
    """Basic info endpoint."""
    return jsonify(
        {
            "status": "ok",
            "message": "YOLO inference server (model loads on first /predict)",
            "endpoints": {
                "GET /health": "health check",
                "POST /predict": "run YOLO on base64 image",
            },
            "model_loaded": model is not None,
            "model_path": MODEL_PATH,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    """Health check — fast response, no model loading here."""
    return jsonify(
        {
            "status": "healthy" if model is not None else "model_not_loaded_yet",
            "model_loaded": model is not None,
            "model_path": MODEL_PATH,
        }
    )


@app.route("/predict", methods=["POST", "GET"])
def predict():
    """
    Run YOLO on an image.
    - GET: simple info response (for browser testing)
    - POST: perform inference
    Supported POST body formats:
      1) Raw base64 string (Content-Type: application/x-www-form-urlencoded or text/plain)
      2) JSON: { "image": "<base64-string>" } (Content-Type: application/json)
    """
    if request.method == "GET":
        return jsonify(
            {
                "status": "ok",
                "usage": "Send POST with base64 image to get YOLO predictions.",
                "body_options": {
                    "json": {"image": "<base64-string>"},
                    "raw": "raw base64 body (e.g. from mobile app)",
                },
            }
        )

    # Load model lazily on first real prediction request
    load_model()

    if model is None:
        return (
            jsonify(
                {
                    "error": "model_not_loaded",
                    "message": f"Failed to load YOLO model from {MODEL_PATH}. Check server logs.",
                }
            ),
            500,
        )

    # Extract base64 image from request
    content_type = (request.headers.get("Content-Type") or "").lower()
    image_base64 = None

    if "application/json" in content_type:
        data = request.get_json(silent=True) or {}
        image_base64 = data.get("image")
    else:
        # Assume raw base64 body (common for mobile apps)
        image_base64 = request.get_data(as_text=True).strip()

    if not image_base64:
        return (
            jsonify(
                {
                    "error": "no_image",
                    "message": "No image data provided. Send base64 in JSON or raw body.",
                }
            ),
            400,
        )

    try:
        # Decode base64 → PIL Image
        img_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return (
            jsonify(
                {
                    "error": "invalid_image",
                    "message": f"Could not decode base64 image: {str(e)}",
                }
            ),
            400,
        )

    try:
        # Run YOLO inference
        results = model(img, verbose=False)
        result = results[0]
        predictions = []

        if result.boxes is not None:
            for box in result.boxes:
                predictions.append(
                    {
                        "class": result.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "x": float(box.xyxy[0][0]),
                        "y": float(box.xyxy[0][1]),
                        "width": float(box.xyxy[0][2] - box.xyxy[0][0]),
                        "height": float(box.xyxy[0][3] - box.xyxy[0][1]),
                    }
                )

        return jsonify(
            {
                "predictions": predictions,
                "image": {"width": img.width, "height": img.height},
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "error": "inference_error",
                    "message": f"Failed to run YOLO inference: {str(e)}",
                }
            ),
            500,
        )


@app.errorhandler(404)
def not_found(_):
    """Clean 404 response."""
    return jsonify({"detail": "Not Found"}), 404


# IMPORTANT: Do NOT add app.run() here — Render uses gunicorn
# No if __name__ == "__main__" block needed
