"""
Damage Processor - Main processing class
Ported from Python V3.8 code
"""

import os
import asyncio
import httpx
import aiofiles
import cv2
import numpy as np
from PIL import Image as PILImage
from ultralytics.utils.plotting import Annotator
from typing import Dict, List, Tuple
from app.config import config
from app.utils.scale_calculator import calculate_scale
from app.utils.consensus import get_multi_model_consensus
from app.utils.cost_calculator import calculate_costs
from app.utils.model_loader import load_models
from app.utils.pdf_generator import generate_invoice_pdf, generate_analysis_pdf
from supabase import create_client, Client
from datetime import datetime

class DamageProcessor:
    def __init__(self):
        self.supabase: Client = None
        if config.SUPABASE_URL and config.SUPABASE_SERVICE_KEY:
            self.supabase = create_client(
                config.SUPABASE_URL,
                config.SUPABASE_SERVICE_KEY
            )
        self.models = {}
        self.models_loaded = False
    
    async def load_models_if_needed(self):
        """Load YOLO models (only once)"""
        if not self.models_loaded:
            print("Loading YOLO models...")
            self.models = await load_models()
            self.models_loaded = True
            print(f"Loaded {len(self.models)} models")
    
    async def process_assessment(self, assessment_data: Dict):
        """
        Main processing function - ports the Python V3.8 processing logic EXACTLY
        Following the exact same procedure as the Python script
        """
        assessment_id = assessment_data['assessment_id']
        
        try:
            print(f"\n{'='*90}")
            print(f"STARTING ASSESSMENT PROCESSING: {assessment_id}")
            print(f"{'='*90}\n")
            
            # Update status to processing
            await self._update_status(assessment_id, 'processing', 'Initializing...', 5)
            
            # STEP 1: Load all models FIRST (matches Python script line 817-825)
            print("Loading YOLO models...")
            await self._update_status(assessment_id, 'processing', 'Loading AI models...', 10)
            await self.load_models_if_needed()
            
            if not self.models or len(self.models) < 7:
                missing = set(['handle', 'component', 'side_hunter', 'side_kulas', 
                              'damage_sindhu', 'damage_cddce', 'damage_capstone']) - set(self.models.keys())
                raise Exception(f"Failed to load YOLO models. Missing: {missing}")
            
            print(f"✅ Loaded {len(self.models)} models: {list(self.models.keys())}")
            
            # Download photos from Supabase Storage
            photo_paths = await self._download_photos(
                assessment_data['photo_urls'],
                assessment_id
            )
            
            if not photo_paths:
                raise Exception("Failed to download photos")
            
            # Process each photo
            photo_results = []
            paint_costs_jod = []
            total_consensus_area = 0.0
            t1_total = t2_total = t3_total = 0.0
            
            # Global flags for component damage
            global_has_windshield_damage = False
            global_has_light_damage = False
            global_has_tire_damage = False
            
            # Create temp directory for this assessment
            temp_dir = os.path.join(config.PHOTOS_BASE_PATH, f"temp_{assessment_id}")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Update progress: photos downloaded
            await self._update_status(assessment_id, 'processing', f'Processing {len(photo_paths)} photos...')
            
            for photo_num, photo_path in enumerate(photo_paths, 1):
                print(f"\n{'='*50} PROCESSING PHOTO {photo_num}/{len(photo_paths)}: {photo_path} {'='*50}")
                
                # Get image dimensions
                img = PILImage.open(photo_path)
                w_px, h_px = img.size
                print(f"Image size: {w_px} × {h_px} px\n")
                
                # STEP 1: Run handle and component models FIRST for scale calculation
                print(f"Step 1: Calculating scale...")
                handle_res = self.models['handle'](photo_path, conf=0.4, verbose=False)[0]
                component_res = self.models['component'](photo_path, conf=0.4, verbose=False)[0]
                
                # STEP 2: Calculate scale (MUST BE DONE BEFORE DAMAGE MODELS)
                scale_data = calculate_scale(
                    handle_res,
                    component_res,
                    assessment_data['tire_diameter'],
                    assessment_data['handle_width'],
                    assessment_data['license_width'],
                    w_px,
                    h_px
                )
                
                scale_cm_per_px = scale_data['scale_cm_per_px']
                tire_boxes = scale_data['tire_boxes']
                print(f"Scale: {scale_data['source']} → {scale_cm_per_px:.6f} cm/px")
                print(f"Estimated image width: {w_px * scale_cm_per_px:.1f} cm\n")
                
                # STEP 3: Run orientation models (for reporting/logging)
                print(f"Step 2: Detecting orientation...")
                hunter_res = self.models['side_hunter'](photo_path, conf=0.5, verbose=False)[0]
                kulas_res = self.models['side_kulas'](photo_path, conf=0.4, verbose=False)[0]
                
                # STEP 4: Run damage models (using scale already calculated)
                print(f"Step 3: Running damage detection models...")
                sindhu_res = self.models['damage_sindhu'](photo_path, conf=0.3, verbose=False)[0]
                cddce_res = self.models['damage_cddce'](photo_path, conf=0.3, verbose=False)[0]
                capstone_res = self.models['damage_capstone'](photo_path, conf=0.3, verbose=False)[0]
                
                # STEP 5: Calculate damage areas from individual models (for reporting)
                t1 = self._calculate_damage_area(sindhu_res, scale_cm_per_px)
                t2 = self._calculate_damage_area(cddce_res, scale_cm_per_px)
                t3 = self._calculate_damage_area(capstone_res, scale_cm_per_px)
                t1_total += t1
                t2_total += t2
                t3_total += t3
                print(f"Damage areas: Sindhu={t1:.1f} cm², CDDCE={t2:.1f} cm², Capstone={t3:.1f} cm²")
                
                # STEP 6: Get consensus damage (2+ models must agree)
                print(f"Step 4: Calculating consensus damage...")
                consensus = get_multi_model_consensus(
                    [sindhu_res, cddce_res, capstone_res],
                    iou_threshold=config.IOU_THRESHOLD
                )
                print(f"Found {len(consensus)} consensus damage items")
                
                # STEP 7: Calculate consensus damage area
                t_cons = self._calculate_consensus_area(consensus, scale_cm_per_px)
                total_consensus_area += t_cons
                
                # STEP 8: Generate annotated consensus image
                consensus_path = None
                if consensus:
                    print(f"Step 5: Generating annotated image...")
                    consensus_path = await self._generate_consensus_image(
                        photo_path,
                        consensus,
                        temp_dir,
                        photo_num
                    )
                
                # STEP 9: Calculate paint area and component flags for this photo
                print(f"Step 6: Calculating costs for photo {photo_num}...")
                paint_area, is_photo_windshield, is_photo_light, is_photo_tire = \
                    self._calculate_paint_area(consensus, tire_boxes, scale_cm_per_px)
                
                # STEP 10: Paint cost for this photo (formula: 0.019157 * area + 2.093)
                paint_cost_photo_jod = (0.019157 * paint_area + 2.093) if paint_area > 0 else 0
                paint_costs_jod.append((photo_num, paint_area, paint_cost_photo_jod))
                print(f"Paint area: {paint_area:.1f} cm², Cost: {paint_cost_photo_jod:.2f} JOD")
                
                # STEP 11: Update global flags (component damage charged once if found anywhere)
                global_has_windshield_damage = global_has_windshield_damage or is_photo_windshield
                global_has_light_damage = global_has_light_damage or is_photo_light
                global_has_tire_damage = global_has_tire_damage or is_photo_tire
                
                # Update progress: photo processed
                progress = int((photo_num / len(photo_paths)) * 80)  # 0-80% for photo processing
                await self._update_status(assessment_id, 'processing', f'Processed photo {photo_num}/{len(photo_paths)}...', progress)
                
                # Store photo results
                photo_results.append({
                    'photo_num': photo_num,
                    'photo_path': photo_path,
                    'scale_data': scale_data,
                    'consensus': consensus,
                    'consensus_path': consensus_path,
                    'paint_area': paint_area,
                    'tire_boxes': tire_boxes,
                })
            
            # STEP 12: Calculate final costs (following Python script procedure)
            print(f"\n{'='*50} CALCULATING FINAL COSTS {'='*50}")
            await self._update_status(assessment_id, 'processing', 'Calculating final costs...', 85)
            
            cost_data = calculate_costs(
                photo_results,
                paint_costs_jod,
                global_has_windshield_damage,
                global_has_light_damage,
                global_has_tire_damage,
                assessment_data['currency_exchange_rate'],
                assessment_data['tax_rate'],
                assessment_data['luxury_index'],
                assessment_data['country_lux_factor'],
                assessment_data['currency']
            )
            
            print(f"Cost breakdown:")
            print(f"  Paint total: {cost_data['paint_total_jod']:.2f} JOD")
            print(f"  Light cost: {cost_data['light_cost_jod']:.2f} JOD")
            print(f"  Windshield cost: {cost_data['windshield_cost_jod']:.2f} JOD")
            print(f"  Tire cost: {cost_data['tire_cost_jod']:.2f} JOD")
            print(f"  Subtotal (base): {cost_data['subtotal_local_base']:.2f} {cost_data['currency']}")
            print(f"  Tax ({cost_data['tax_rate']*100:.2f}%): {cost_data['tax_amount_on_base_local']:.2f} {cost_data['currency']}")
            print(f"  Subtotal (post-tax): {cost_data['subtotal_post_base_tax']:.2f} {cost_data['currency']}")
            print(f"  Luxury factor (car): {cost_data['luxury_index']:.2f}")
            print(f"  Country lux factor: {cost_data['country_lux_factor']:.3f}")
            print(f"  FINAL COST: {cost_data['final_local_cost']:.2f} {cost_data['currency']}")
            
            # STEP 13: Generate PDFs
            print(f"\n{'='*50} GENERATING PDF REPORTS {'='*50}")
            await self._update_status(assessment_id, 'processing', 'Generating PDF reports...', 88)
            
            pdf_urls = await self._generate_and_upload_pdfs(
                assessment_id,
                photo_results,
                cost_data,
                assessment_data
            )
            
            # STEP 14: Save results to Supabase
            print(f"\n{'='*50} SAVING RESULTS {'='*50}")
            await self._update_status(assessment_id, 'processing', 'Saving results...', 92)
            
            await self._save_results(
                assessment_id,
                cost_data,
                total_consensus_area,
                t1_total,
                t2_total,
                t3_total,
                global_has_windshield_damage,
                global_has_light_damage,
                pdf_urls
            )
            
            # STEP 14: Update status to completed (status already set in _save_results)
            print(f"\n{'='*50} ASSESSMENT COMPLETE {'='*50}")
            print(f"✅ Assessment {assessment_id} processed successfully!")
            print(f"   Total Consensus Damage: {total_consensus_area:.1f} cm²")
            print(f"   Final Cost: {cost_data['final_local_cost']:.2f} {cost_data['currency']}")
            print(f"{'='*90}\n")
            
            # Ensure status is set to completed (in case _save_results didn't update it)
            await self._update_status(assessment_id, 'completed', 'Assessment processing completed!', 100)
            
        except Exception as e:
            print(f"\n❌ ERROR processing assessment {assessment_id}: {e}")
            import traceback
            traceback.print_exc()
            error_message = str(e)
            await self._update_status(assessment_id, 'failed', error_message, 0)
            # Don't re-raise - let it fail gracefully and mark as failed in database
            print(f"Assessment {assessment_id} marked as failed in database")
    
    def _calculate_damage_area(self, results, scale_cm_per_px: float) -> float:
        """Calculate total damage area in cm² from a single model"""
        total = 0.0
        if results.boxes is not None:
            for box in results.boxes:
                w_px = box.xywh[0][2].item()
                h_px = box.xywh[0][3].item()
                w_cm = w_px * scale_cm_per_px
                h_cm = h_px * scale_cm_per_px
                area = w_cm * h_cm
                total += area
        return total
    
    def _calculate_consensus_area(self, consensus: List[Dict], scale_cm_per_px: float) -> float:
        """Calculate total consensus damage area in cm²"""
        total = 0.0
        for item in consensus:
            xyxy = item['xyxy']
            w_px = xyxy[2] - xyxy[0]
            h_px = xyxy[3] - xyxy[1]
            w_cm = w_px * scale_cm_per_px
            h_cm = h_px * scale_cm_per_px
            area = w_cm * h_cm
            total += area
        return total
    
    def _calculate_paint_area(
        self,
        consensus: List[Dict],
        tire_boxes: List,
        scale_cm_per_px: float
    ) -> Tuple[float, bool, bool, bool]:
        """
        Calculate paint damage area and check for component damage.
        Matches Python script logic EXACTLY (lines 979-1019).
        
        Procedure:
        1. Windshield damage: Excluded from paint area, sets flag
        2. Light damage: Excluded from paint area, sets flag  
        3. Other damages: Check if overlaps with tire
           - If >50% overlap with tire: Tire damage, excluded from paint area
           - If no overlap: Add to paint area
        
        Returns: (paint_area_cm2, has_windshield, has_light, has_tire)
        """
        paint_area = 0.0
        has_windshield = False
        has_light = False
        has_tire = False
        
        # Process each consensus item (matches Python script lines 985-1011)
        for item in consensus:
            # Get detected class name (lowercase for matching)
            detected_class_lower = item.get('detected_class', '').lower() if item.get('detected_class') else ''
            
            # STEP 1: Check for windshield damage (excluded from paint area)
            if item.get('is_windshield', False) or ('windshield' in detected_class_lower):
                has_windshield = True
                continue  # Don't add to paint area (matches Python line 989)
            
            # STEP 2: Check for light damage (excluded from paint area)
            if item.get('is_light', False) or ('light' in detected_class_lower):
                has_light = True
                continue  # Don't add to paint area (matches Python line 991)
            
            # STEP 3: Other damages - check tire overlap (matches Python lines 992-1011)
            xyxy = item['xyxy']
            damage_area_px = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            area_cm2 = damage_area_px * (scale_cm_per_px ** 2)
            
            # Check if damage overlaps with tire (>50% overlap = tire damage)
            overlapped_tire = False
            for tire_box in tire_boxes:
                xi1 = max(xyxy[0], tire_box[0])
                yi1 = max(xyxy[1], tire_box[1])
                xi2 = min(xyxy[2], tire_box[2])
                yi2 = min(xyxy[3], tire_box[3])
                inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                
                # If >50% overlap with tire, it's tire damage (matches Python line 1005)
                if inter / (damage_area_px + 1e-6) > 0.5:
                    overlapped_tire = True
                    has_tire = True
                    break
            
            # Only add to paint area if NOT overlapped with tire (matches Python line 1010)
            if not overlapped_tire:
                paint_area += area_cm2
        
        return (paint_area, has_windshield, has_light, has_tire)
    
    async def _generate_consensus_image(
        self,
        photo_path: str,
        consensus: List[Dict],
        temp_dir: str,
        photo_num: int
    ) -> str:
        """Generate annotated consensus image"""
        img = cv2.imread(photo_path)
        if img is None:
            return None
        
        annotator = Annotator(img, line_width=4)
        for item in consensus:
            xyxy = item['xyxy']
            box = xyxy.astype(int).tolist() if isinstance(xyxy, np.ndarray) else [int(x) for x in xyxy]
            
            if item.get('is_windshield', False):
                label = f"Windshield {item['conf']:.2f}"
                annotator.box_label(box, label, color=(0, 255, 0))  # Green
            elif item.get('is_light', False):
                label = f"Light {item['conf']:.2f}"
                annotator.box_label(box, label, color=(255, 255, 0))  # Yellow
            else:
                label = f"Damage {item['conf']:.2f}"
                annotator.box_label(box, label, color=(0, 0, 255))  # Red
        
        annotated = annotator.result()
        consensus_path = os.path.join(temp_dir, f"photo{photo_num}_consensus.jpg")
        cv2.imwrite(consensus_path, annotated)
        
        return consensus_path
    
    async def _download_photos(self, photo_urls: List[str], assessment_id: str) -> List[str]:
        """Download photos from Supabase Storage URLs"""
        os.makedirs(f"{config.PHOTOS_BASE_PATH}/{assessment_id}", exist_ok=True)
        photo_paths = []
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for i, url in enumerate(photo_urls):
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    photo_path = f"{config.PHOTOS_BASE_PATH}/{assessment_id}/photo_{i+1}.jpg"
                    
                    async with aiofiles.open(photo_path, 'wb') as f:
                        await f.write(response.content)
                    
                    photo_paths.append(photo_path)
                except Exception as e:
                    print(f"Error downloading photo {i+1}: {e}")
                    continue
        
        return photo_paths
    
    async def _update_status(self, assessment_id: str, status: str, message: str = None, progress: int = None):
        """Update assessment status in Supabase"""
        if not self.supabase:
            print(f"Warning: Supabase not configured, cannot update status")
            return
        
        update_data = {'status': status}
        metadata = {}
        
        if message:
            metadata['message'] = message
        
        if progress is not None:
            metadata['progress'] = progress
        
        if metadata:
            # Merge with existing metadata
            try:
                existing = self.supabase.table('assessments').select('metadata').eq('id', assessment_id).single().execute()
                if existing.data and existing.data.get('metadata'):
                    # Merge metadata, but keep existing keys if they exist
                    existing_metadata = existing.data['metadata'] or {}
                    metadata = {**existing_metadata, **metadata}
            except Exception as e:
                # If we can't fetch existing metadata, just use new metadata
                print(f"Note: Could not fetch existing metadata: {e}")
            
            update_data['metadata'] = metadata
        
        try:
            response = self.supabase.table('assessments').update(update_data).eq('id', assessment_id).execute()
            if message:
                print(f"📊 Status update: {status} - {message} (progress: {progress}%)")
            elif status in ['completed', 'failed']:
                print(f"📊 Status update: {status}")
        except Exception as e:
            print(f"❌ Error updating status in Supabase: {e}")
            import traceback
            traceback.print_exc()
    
    async def _generate_and_upload_pdfs(
        self,
        assessment_id: str,
        photo_results: List[Dict],
        cost_data: Dict,
        assessment_data: Dict
    ) -> Dict[str, str]:
        """Generate and upload PDF reports to Supabase Storage"""
        pdf_urls = {'invoice_url': None, 'analysis_url': None}
        
        if not self.supabase:
            print(f"Warning: Supabase not configured, cannot upload PDFs")
            return pdf_urls
        
        try:
            # Fetch assessment data from database to get user_id and other info
            user_id = None
            customer_name = 'Client'
            country = 'Unknown'
            
            try:
                assessment = self.supabase.table('assessments').select(
                    'user_id, metadata, car_make, car_model, car_year'
                ).eq('id', assessment_id).single().execute()
                
                if assessment.data:
                    user_id = assessment.data.get('user_id')
                    metadata = assessment.data.get('metadata', {}) or {}
                    
                    # Try to get customer name from metadata or construct from car info
                    customer_name = metadata.get('customer_name', 'Client')
                    if customer_name == 'Client':
                        # Construct from car info
                        car_make = assessment.data.get('car_make', '')
                        car_model = assessment.data.get('car_model', '')
                        car_year = assessment.data.get('car_year', '')
                        if car_make or car_model:
                            customer_name = f"{car_make} {car_model} {car_year}".strip()
                    
                    country = metadata.get('country', assessment_data.get('country', 'Unknown'))
            except Exception as e:
                print(f"Warning: Could not fetch assessment data: {e}")
                # Try to get user_id from assessment_data as fallback
                user_id = assessment_data.get('user_id')
            
            if not user_id:
                print(f"Warning: No user_id found, skipping PDF upload")
                return pdf_urls
            
            # Prepare report data for PDF
            report_data = {
                'date': datetime.now().strftime("%Y-%m-%d"),
                'customer_name': customer_name,
                'country': country,
                'currency': cost_data.get('currency', 'JOD'),
            }
            
            # Prepare analysis text
            analysis_lines = []
            analysis_lines.append(f"Assessment ID: {assessment_id}")
            analysis_lines.append(f"Date: {report_data['date']}")
            analysis_lines.append(f"Customer: {report_data['customer_name']}")
            analysis_lines.append(f"Country: {report_data['country']}")
            analysis_lines.append(f"Currency: {report_data['currency']}")
            analysis_lines.append("")
            analysis_lines.append("Cost Summary:")
            analysis_lines.append(f"  Paint Total: {cost_data.get('paint_total_jod', 0):.2f} JOD")
            analysis_lines.append(f"  Light Cost: {cost_data.get('light_cost_jod', 0):.2f} JOD")
            analysis_lines.append(f"  Windshield Cost: {cost_data.get('windshield_cost_jod', 0):.2f} JOD")
            analysis_lines.append(f"  Tire Cost: {cost_data.get('tire_cost_jod', 0):.2f} JOD")
            analysis_lines.append(f"  Final Cost: {cost_data.get('final_local_cost', 0):.2f} {report_data['currency']}")
            analysis_text = '\n'.join(analysis_lines)
            
            # Generate PDFs locally
            pdf_dir = os.path.join(config.PDFS_BASE_PATH, assessment_id)
            os.makedirs(pdf_dir, exist_ok=True)
            
            invoice_pdf_path = os.path.join(pdf_dir, f"{assessment_id}-invoice.pdf")
            analysis_pdf_path = os.path.join(pdf_dir, f"{assessment_id}-analysis.pdf")
            
            # Generate invoice PDF
            print(f"Generating invoice PDF...")
            generate_invoice_pdf(
                assessment_id,
                invoice_pdf_path,
                photo_results,
                cost_data,
                report_data
            )
            
            # Generate analysis PDF
            print(f"Generating analysis PDF...")
            generate_analysis_pdf(
                assessment_id,
                analysis_pdf_path,
                analysis_text,
                photo_results
            )
            
            # Upload to Supabase Storage
            pdfs_bucket = "assessment-pdfs"
            storage_path_prefix = f"{user_id}/{assessment_id}"
            
            # Upload invoice PDF
            if os.path.exists(invoice_pdf_path):
                with open(invoice_pdf_path, 'rb') as f:
                    invoice_data = f.read()
                    invoice_storage_path = f"{storage_path_prefix}/invoice.pdf"
                    try:
                        # Supabase Python client syntax
                        result = self.supabase.storage.from_(pdfs_bucket).upload(
                            invoice_storage_path,
                            invoice_data,
                            file_options={"content-type": "application/pdf", "upsert": "true"}
                        )
                        # Get public URL
                        public_url_result = self.supabase.storage.from_(pdfs_bucket).get_public_url(invoice_storage_path)
                        pdf_urls['invoice_url'] = public_url_result
                        print(f"✅ Invoice PDF uploaded: {public_url_result}")
                    except Exception as e:
                        print(f"⚠️  Error uploading invoice PDF: {e}")
                        # Try alternative method if first fails
                        try:
                            # Alternative: direct URL construction
                            pdf_urls['invoice_url'] = f"{config.SUPABASE_URL}/storage/v1/object/public/{pdfs_bucket}/{invoice_storage_path}"
                            print(f"✅ Invoice PDF URL constructed: {pdf_urls['invoice_url']}")
                        except:
                            pass
            
            # Upload analysis PDF
            if os.path.exists(analysis_pdf_path):
                with open(analysis_pdf_path, 'rb') as f:
                    analysis_data = f.read()
                    analysis_storage_path = f"{storage_path_prefix}/analysis.pdf"
                    try:
                        # Supabase Python client syntax
                        result = self.supabase.storage.from_(pdfs_bucket).upload(
                            analysis_storage_path,
                            analysis_data,
                            file_options={"content-type": "application/pdf", "upsert": "true"}
                        )
                        # Get public URL
                        public_url_result = self.supabase.storage.from_(pdfs_bucket).get_public_url(analysis_storage_path)
                        pdf_urls['analysis_url'] = public_url_result
                        print(f"✅ Analysis PDF uploaded: {public_url_result}")
                    except Exception as e:
                        print(f"⚠️  Error uploading analysis PDF: {e}")
                        # Try alternative method if first fails
                        try:
                            # Alternative: direct URL construction
                            pdf_urls['analysis_url'] = f"{config.SUPABASE_URL}/storage/v1/object/public/{pdfs_bucket}/{analysis_storage_path}"
                            print(f"✅ Analysis PDF URL constructed: {pdf_urls['analysis_url']}")
                        except:
                            pass
            
        except Exception as e:
            print(f"❌ Error generating/uploading PDFs: {e}")
            import traceback
            traceback.print_exc()
            # Don't fail the whole process if PDF generation fails
        
        return pdf_urls
    
    async def _save_results(
        self,
        assessment_id: str,
        cost_data: Dict,
        total_consensus_area: float,
        t1_total: float,
        t2_total: float,
        t3_total: float,
        global_has_windshield_damage: bool,
        global_has_light_damage: bool,
        pdf_urls: Dict[str, str] = None
    ):
        """Save processing results to Supabase (matches Python script report_data structure)"""
        if not self.supabase:
            print(f"Warning: Supabase not configured, cannot save results")
            return
        
        # Prepare update data matching Python script structure (lines 1100-1124)
        update_data = {
            'status': 'completed',  # Mark as completed
            'estimated_cost': cost_data['final_local_cost'],
            'subtotal_base': cost_data['subtotal_local_base'],
            'tax_rate': cost_data['tax_rate'],
            'tax_amount': cost_data['tax_amount_on_base_local'],
            'subtotal_post_tax': cost_data['subtotal_post_base_tax'],
            'luxury_factor': cost_data['luxury_index'],
            'country_lux_factor': cost_data['country_lux_factor'],
            'currency': cost_data['currency'],
        }
        
        # Add PDF URLs if available
        if pdf_urls:
            if pdf_urls.get('invoice_url'):
                update_data['pdf_url'] = pdf_urls['invoice_url']
            if pdf_urls.get('analysis_url'):
                update_data['metadata'] = update_data.get('metadata', {})
                update_data['metadata']['analysis_pdf_url'] = pdf_urls['analysis_url']
        
        update_data['metadata'] = update_data.get('metadata', {})
        update_data['metadata'].update({
            # Match Python script report_data keys
            'consensus_damage_cm2': round(total_consensus_area, 1),
                'Paint Costs Local': cost_data['paint_costs_local'],  # Match Python key
                'Lights Repair Cost (Local)': round(cost_data['light_cost_local'], 2),
                'Windshield Repair Cost (Local)': round(cost_data['windshield_cost_local'], 2),
                'Tire Repair Cost (Local)': round(cost_data['tire_cost_local'], 2),
                'Subtotal Cost (Local)': round(cost_data['subtotal_local_base'], 2),
                'Tax Amount (Local)': round(cost_data['tax_amount_on_base_local'], 2),
                'Subtotal Post Base Tax (Local)': round(cost_data['subtotal_post_base_tax'], 2),
                'Luxury Factor': round(cost_data['luxury_index'], 2),
                'Final Cost (Local)': round(cost_data['final_local_cost'], 2),
                'Sindhu Damage (cm²)': round(t1_total, 1),
                'CDDCE Damage (cm²)': round(t2_total, 1),
                'Capstone Damage (cm²)': round(t3_total, 1),
                'Tax Rate': cost_data['tax_rate'],
                'Light Damage Found': global_has_light_damage,
                'Windshield Damage Found': global_has_windshield_damage,
                # Also keep shorter keys for API compatibility
                'paint_costs': cost_data['paint_costs_local'],
                'light_repair_cost': cost_data['light_cost_local'],
                'windshield_repair_cost': cost_data['windshield_cost_local'],
                'tire_repair_cost': cost_data['tire_cost_local'],
                'sindhu_damage_cm2': round(t1_total, 1),
                'cddce_damage_cm2': round(t2_total, 1),
                'capstone_damage_cm2': round(t3_total, 1),
            }
        }
        
        try:
            response = self.supabase.table('assessments').update(update_data).eq('id', assessment_id).execute()
            if response.data:
                print(f"✅ Results saved to Supabase for assessment {assessment_id}")
                print(f"   Final Cost: {cost_data['final_local_cost']:.2f} {cost_data['currency']}")
                print(f"   Consensus Damage: {total_consensus_area:.1f} cm²")
            else:
                print(f"⚠️  No data returned from Supabase update")
        except Exception as e:
            print(f"❌ Error saving results to Supabase: {e}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise to mark assessment as failed
