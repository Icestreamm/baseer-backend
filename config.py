import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Supabase
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
    
    # Model storage
    MODELS_BUCKET = "yolo-models"
    MODELS_BASE_PATH = os.getenv("MODELS_BASE_PATH", "./models")
    
    # Storage paths
    PHOTOS_BASE_PATH = os.getenv("PHOTOS_BASE_PATH", "./temp_photos")
    PDFS_BASE_PATH = os.getenv("PDFS_BASE_PATH", "./temp_pdfs")
    
    # Processing
    MAX_PHOTOS_PER_ASSESSMENT = 10
    CONFIDENCE_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.5
    
    # Model paths (will be downloaded from Supabase Storage)
    MODEL_PATHS = {
        'handle': f"{MODELS_BASE_PATH}/handle/best.pt",
        'component': f"{MODELS_BASE_PATH}/component/best.pt",
        'side_hunter': f"{MODELS_BASE_PATH}/side_hunter/best.pt",
        'side_kulas': f"{MODELS_BASE_PATH}/side_kulas/best.pt",
        'damage_sindhu': f"{MODELS_BASE_PATH}/damage_sindhu/best.pt",
        'damage_cddce': f"{MODELS_BASE_PATH}/damage_cddce/best.pt",
        'damage_capstone': f"{MODELS_BASE_PATH}/damage_capstone/best.pt",
    }

config = Config()
