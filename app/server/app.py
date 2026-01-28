"""
Flask API Server for YOLO Model Inference
Runs your local YOLO model (handle_best.pt) behind a clean /predict API.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)  # Allow calls from your Expo / React Native app

# Load YOLO model once at startup
MODEL_PATH = os.getenv("MODEL_PATH", "handle_best.pt")
try:
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
            "message": "YOLO inference server",
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
    """Health check used by your app before calling /predict."""
    return jsonify(
        {
            "status": "healthy" if model is not None else "model_not_loaded",
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

    POST body formats supported:
      1) Raw base64 in body with Content-Type: application/x-www-form-urlencoded
      2) JSON: { "image": "<base64-string>" } with Content-Type: application/json
    """
    if request.method == "GET":
        return jsonify(
            {
                "status": "ok",
                "usage": "POST base64 image to this endpoint to get YOLO predictions.",
                "expect": {
                    "json": {"image": "<base64>"},
                    "or_raw": "raw base64 body with Content-Type application/x-www-form-urlencoded",
                },
            }
        )

    if model is None:
        return (
            jsonify(
                {
                    "error": "model_not_loaded",
                    "message": f"YOLO model failed to load from {MODEL_PATH}",
                }
            ),
            500,
        )

    content_type = (request.headers.get("Content-Type") or "").lower()
    image_base64 = None

    # JSON: { "image": "<base64>" }
    if "application/json" in content_type:
        data = request.get_json(silent=True) or {}
        image_base64 = data.get("image")
    else:
        # Treat body as raw base64 (what your mobile app sends)
        image_base64 = request.get_data(as_text=True).strip()

    if not image_base64:
        return (
            jsonify(
                {
                    "error": "no_image",
                    "message": "No image data provided. Send base64 in JSON {\"image\": \"...\"} or raw body.",
                }
            ),
            400,
        )

    try:
        # Decode base64 into a PIL image
        img_bytes = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return (
            jsonify(
                {
                    "error": "invalid_image",
                    "message": f"Could not decode base64 image: {e}",
                }
            ),
            400,
        )

    try:
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
                    "message": f"Failed to run YOLO inference: {e}",
                }
            ),
            500,
        )


@app.errorhandler(404)
def not_found(_):
    """
    Simple, clean 404 handler.
    Keeps logs minimal; no noisy stack traces for scanners.
    """
    return jsonify({"detail": "Not Found"}), 404


if __name__ == "__main__":
    port = int(os.getenv("PORT", 9001))
    print(f"[YOLO] Starting Flask server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
