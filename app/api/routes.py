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
    estimated_cost: float = None
    currency: str = None

@router.post("/assessments/process", response_model=ProcessAssessmentResponse)
async def process_assessment(
    request: ProcessAssessmentRequest,
    background_tasks: BackgroundTasks
):
    """
    Start processing an assessment with YOLO models.
    Processing happens in background - returns immediately.
    
    IMPORTANT: This endpoint MUST return immediately (< 1 second) to prevent client timeout.
    Actual processing (which can take 2-5 minutes) happens in the background task.
    """
    assessment_id = request.assessment_id
    
    # Validate request
    if not assessment_id:
        raise HTTPException(status_code=400, detail="assessment_id is required")
    
    if not request.photo_urls or len(request.photo_urls) == 0:
        raise HTTPException(status_code=400, detail="At least one photo URL is required")
    
    # Set initial status to 'processing' in database BEFORE starting background task
    from app.config import config
    from supabase import create_client
    if config.SUPABASE_URL and config.SUPABASE_SERVICE_KEY:
        try:
            supabase = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY)
            supabase.table('assessments').update({
                'status': 'processing',
                'metadata': {'message': 'Processing started', 'progress': 0}
            }).eq('id', assessment_id).execute()
        except Exception as e:
            print(f"Warning: Could not set initial status: {e}")
    
    # IMPORTANT: Create processor and add background task
    # The background task runs ASYNCHRONOUSLY - this function returns immediately
    processor = DamageProcessor()
    
    # Create async wrapper for background processing
    async def run_processing():
        try:
            print(f"🚀 Background task started for assessment {assessment_id}")
            await processor.process_assessment(request.dict())
            print(f"✅ Background task completed for assessment {assessment_id}")
        except Exception as e:
            print(f"❌ Background processing error for {assessment_id}: {e}")
            import traceback
            traceback.print_exc()
            # Error is already saved to database in processor._update_status
    
    # Add background task - FastAPI queues this and returns immediately
    # DO NOT await this - that would block the response
    background_tasks.add_task(run_processing)
    
    print(f"✅ Assessment {assessment_id} queued for processing. Returning immediately to client.")
    
    # Return IMMEDIATELY - client can poll /status endpoint for updates
    return ProcessAssessmentResponse(
        assessment_id=assessment_id,
        status="processing",
        message="Assessment processing started. Use /status endpoint to check progress."
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
        # Get assessment with estimated_cost and currency for completed assessments
        response = supabase.table('assessments').select(
            'status, metadata, estimated_cost, currency'
        ).eq('id', assessment_id).single().execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Assessment not found")
        
        assessment = response.data
        status = assessment.get('status', 'pending')
        metadata = assessment.get('metadata', {}) or {}
        
        # Get error message from metadata.message or metadata.error
        error = None
        if status == 'failed':
            error = metadata.get('error') or metadata.get('message') or 'Processing failed'
        
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
        
        # Get estimated cost and currency for completed assessments
        estimated_cost = None
        currency = None
        if status == 'completed':
            estimated_cost = assessment.get('estimated_cost')
            currency = assessment.get('currency')
        
        return AssessmentStatusResponse(
            assessment_id=assessment_id,
            status=status,
            progress=progress,
            error=error,
            estimated_cost=estimated_cost,
            currency=currency
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting status: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
