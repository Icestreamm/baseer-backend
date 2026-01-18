from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List
from app.models.damage_processor import DamageProcessor

router = APIRouter()

class ProcessAssessmentRequest(BaseModel):
    assessment_id: str
    photo_urls: List[str]
    car_make: str
    car_model: str
    car_year: int
    tire_diameter: float
    handle_width: float
    license_width: float
    luxury_index: float
    currency: str
    currency_exchange_rate: float
    country_lux_factor: float
    tax_rate: float

class ProcessAssessmentResponse(BaseModel):
    assessment_id: str
    status: str
    message: str

class AssessmentStatusResponse(BaseModel):
    assessment_id: str
    status: str
    progress: int = 0
    error: str = None

@router.post("/assessments/process", response_model=ProcessAssessmentResponse)
async def process_assessment(
    request: ProcessAssessmentRequest,
    background_tasks: BackgroundTasks
):
    """
    Start processing an assessment with YOLO models.
    Processing happens in background.
    """
    assessment_id = request.assessment_id
    
    # Start processing in background
    processor = DamageProcessor()
    background_tasks.add_task(
        processor.process_assessment,
        request.dict()
    )
    
    return ProcessAssessmentResponse(
        assessment_id=assessment_id,
        status="processing",
        message="Assessment processing started"
    )

@router.get("/assessments/{assessment_id}/status", response_model=AssessmentStatusResponse)
async def get_assessment_status(assessment_id: str):
    """Get processing status of an assessment from Supabase"""
    from app.config import config
    from supabase import create_client, Client
    
    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_KEY:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
    
    try:
        response = supabase.table('assessments').select('status, metadata').eq('id', assessment_id).single().execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Assessment not found")
        
        assessment = response.data
        status = assessment.get('status', 'pending')
        metadata = assessment.get('metadata', {})
        error = metadata.get('error') if status == 'failed' else None
        
        # Get progress from metadata if available
        progress = metadata.get('progress', 0)
        
        # Fallback: estimate progress based on status
        if progress == 0:
            if status == 'processing':
                progress = 50  # Processing
            elif status == 'completed':
                progress = 100
            elif status == 'failed':
                progress = 0
        
        return AssessmentStatusResponse(
            assessment_id=assessment_id,
            status=status,
            progress=progress,
            error=error
        )
    except Exception as e:
        print(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
