from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(
    title="Baseer Car Damage Assessment API",
    description="AI-powered car damage assessment using YOLO models",
    version="1.0.0"
)

# CORS middleware - allows mobile app to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

@app.get("/")
async def root():
    return {
        "message": "Baseer Car Damage Assessment API",
        "status": "running",
        "note": "Processing endpoint is a stub - YOLO models not yet implemented"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}
