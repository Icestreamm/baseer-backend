"""
Scale Calculator - Ported from Python V3.8 code
Calculates scale (cm per pixel) based on detected reference objects
"""

import numpy as np
from typing import Dict, Optional, Tuple

def calculate_scale(
    handle_res,
    component_res,
    tire_diameter: float,
    handle_width: float,
    license_width: float,
    image_width_px: int,
    image_height_px: int
) -> Dict:
    """
    Calculate scale based on detected reference objects.
    Priority: Tire > Handle > License Plate > Headlight > Fallback
    
    Returns:
        Dict with scale_cm_per_px, source, and detection details
    """
    best_handle_px = 0
    best_tire_px = 0
    best_headlight_px = 0
    best_license_px = 0
    tire_boxes = []
    windshield_boxes = []
    conf_th = 0.5
    
    # Process handle detection
    if handle_res.boxes is not None:
        for box in handle_res.boxes:
            name = handle_res.names[int(box.cls)].lower()
            if "handle" in name and box.conf > conf_th:
                width_px = box.xywh[0][2].item()
                best_handle_px = max(best_handle_px, width_px)
    
    # Process component detection (tire, headlight, license, windshield)
    if component_res.boxes is not None:
        for box in component_res.boxes:
            name = component_res.names[int(box.cls)].lower()
            
            # Tire/Wheel detection
            if ("wheel" in name or "tire" in name) and box.conf > conf_th:
                width_px = min(box.xywh[0][2].item(), box.xywh[0][3].item())
                best_tire_px = max(best_tire_px, width_px)
                tire_boxes.append(box.xyxy[0].cpu().numpy())
            
            # Headlight detection
            if "headlight" in name and box.conf > 0.3:
                width_px = box.xywh[0][2].item()
                best_headlight_px = max(best_headlight_px, width_px)
            
            # License plate detection
            if ("license" in name or "plate" in name) and box.conf > 0.65:
                width_px = box.xywh[0][2].item()
                best_license_px = max(best_license_px, width_px)
            
            # Windshield detection (for later use)
            if "windshield" in name and box.conf > conf_th:
                windshield_boxes.append(box.xyxy[0].cpu().numpy())
    
    # Calculate scales
    tire_scale = tire_diameter / best_tire_px if best_tire_px > 0 else None
    handle_scale = handle_width / best_handle_px if best_handle_px > 0 else None
    license_scale = license_width / best_license_px if best_license_px > 0 else None
    headlight_scale = 33.0 / best_headlight_px if best_headlight_px > 0 else None  # Fixed 33cm
    
    # Determine final scale based on priority
    if tire_scale:
        scale_cm_per_px = tire_scale
        source = "TIRE/WHEEL-BASED (Priority 1)"
    elif handle_scale:
        scale_cm_per_px = handle_scale
        source = "HANDLE-BASED (Priority 2)"
    elif license_scale:
        scale_cm_per_px = license_scale
        source = "LICENSE PLATE-BASED (Priority 3)"
    elif headlight_scale:
        scale_cm_per_px = headlight_scale
        source = "HEADLIGHT-BASED (33 cm fixed – Priority 4)"
    else:
        # Fallback: assume image width = 1 meter
        scale_cm_per_px = 100.0 / image_width_px
        source = "FALLBACK (Image width = 1 meter)"
    
    return {
        'scale_cm_per_px': scale_cm_per_px,
        'source': source,
        'tire_detected': best_tire_px > 0,
        'handle_detected': best_handle_px > 0,
        'license_detected': best_license_px > 0,
        'headlight_detected': best_headlight_px > 0,
        'tire_boxes': tire_boxes,
        'windshield_boxes': windshield_boxes,
        'estimated_image_width_cm': image_width_px * scale_cm_per_px,
        'best_tire_px': best_tire_px,
        'best_handle_px': best_handle_px,
        'best_license_px': best_license_px,
        'best_headlight_px': best_headlight_px,
    }
