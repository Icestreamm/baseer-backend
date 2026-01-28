# YOLO Model Server

Flask API server for YOLO model inference, deployable on Render.

## Setup Instructions

### 1. Add Your Model File

1. Place your YOLO model file (`.pt` file) in the `server/` directory
2. Update `app.py` line 15: Replace `'your_model.pt'` with your actual model filename
3. Update `Dockerfile` line 20: Uncomment and replace `your_model.pt` with your model filename

### 2. Local Testing

```bash
cd server
pip install -r requirements.txt
python app.py
```

Server will run on `http://localhost:9001`

### 3. Deploy to Render

1. **Push to GitHub:**
   - Commit all files including your `.pt` model file
   - Push to your GitHub repository

2. **Create Render Service:**
   - Go to https://render.com
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Select the repository and branch
   - Configure:
     - **Name:** `yolo-model-server` (or your choice)
     - **Environment:** `Docker`
     - **Dockerfile Path:** `server/Dockerfile`
     - **Root Directory:** `server`
     - **Port:** `9001` (or leave default)
   - Click "Create Web Service"

3. **Get Your Render URL:**
   - After deployment, Render will provide a URL like: `https://your-app.onrender.com`
   - Copy this URL - you'll need it in the app

### 4. Update App Configuration

In `app/index.tsx`, update the `YOLO_LOCAL_CONFIG`:
- `apiUrl`: Your Render URL (e.g., `https://your-app.onrender.com`)

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Process image (accepts base64 image)

## Response Format

Matches Roboflow format:
```json
{
  "predictions": [
    {
      "class": "door-handle",
      "confidence": 0.95,
      "x": 100,
      "y": 200,
      "width": 50,
      "height": 50
    }
  ],
  "image": {
    "width": 640,
    "height": 480
  }
}
```
