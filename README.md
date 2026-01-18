# Baseer Backend

FastAPI backend for Baseer car damage assessment application.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run locally:
```bash
uvicorn app.main:app --reload --port 8000
```

3. Test:
- Health: http://localhost:8000/health
- Root: http://localhost:8000/

## Deployment

Deploy to Render.com using `render.yaml` configuration.
