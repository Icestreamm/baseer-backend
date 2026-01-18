"""
Model Loader - Downloads and loads YOLO models from Supabase Storage
"""

import os
import httpx
import aiofiles
from ultralytics import YOLO
from app.config import config
from typing import Dict, Optional

# Cache loaded models
_models_cache: Dict[str, YOLO] = {}

async def download_model_from_supabase(model_name: str, model_path: str) -> bool:
    """
    Download a YOLO model from Supabase Storage.
    
    Args:
        model_name: Name of the model (e.g., 'handle', 'component')
        model_path: Local path to save the model
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Construct Supabase Storage URL
        # Format: https://PROJECT_ID.supabase.co/storage/v1/object/public/BUCKET_NAME/PATH
        supabase_url = config.SUPABASE_URL
        if not supabase_url:
            print(f"Warning: SUPABASE_URL not configured")
            return False
        
        # Construct model URL
        model_url = f"{supabase_url}/storage/v1/object/public/{config.MODELS_BUCKET}/{model_name}/best.pt"
        
        print(f"Downloading {model_name} from {model_url}")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Download model
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout for large files
            response = await client.get(model_url)
            response.raise_for_status()
            
            # Save to disk
            async with aiofiles.open(model_path, 'wb') as f:
                await f.write(response.content)
        
        print(f"Successfully downloaded {model_name} to {model_path}")
        return True
        
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        return False

async def load_models() -> Dict[str, YOLO]:
    """
    Load all YOLO models. Downloads from Supabase if not cached.
    
    Returns:
        Dict of model_name -> YOLO model
    """
    global _models_cache
    
    models = {}
    
    for model_name, model_path in config.MODEL_PATHS.items():
        # Check if already loaded
        if model_name in _models_cache:
            models[model_name] = _models_cache[model_name]
            continue
        
        # Check if file exists locally
        if not os.path.exists(model_path):
            print(f"Model {model_name} not found locally, downloading from Supabase...")
            success = await download_model_from_supabase(model_name, model_path)
            if not success:
                print(f"Warning: Failed to download {model_name}, skipping...")
                continue
        
        # Load model
        try:
            print(f"Loading model: {model_name}")
            model = YOLO(model_path)
            models[model_name] = model
            _models_cache[model_name] = model
            print(f"Successfully loaded {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue
    
    return models
