"""
Consensus Algorithm - Ported from Python V3.8 code
Combines results from multiple YOLO models using IoU-based matching
"""

import numpy as np
from typing import List, Dict

def get_multi_model_consensus(results_list: List, iou_threshold: float = 0.5) -> List[Dict]:
    """
    Get consensus damage detection from multiple models.
    Only damage detected by 2+ models is counted.
    
    Args:
        results_list: List of YOLO results objects
        iou_threshold: IoU threshold for matching boxes (default 0.5)
    
    Returns:
        List of consensus damage items with bounding boxes and metadata
    """
    all_boxes = []
    
    # Collect all boxes from all models
    for res in results_list:
        if res.boxes is not None:
            for box in res.boxes:
                all_boxes.append({
                    'xyxy': box.xyxy[0].cpu().numpy(),
                    'conf': box.conf.item(),
                    'cls': int(box.cls.item()),
                    'source': res,
                    'class_name': res.names[int(box.cls.item())]
                })
    
    consensus = []
    used = [False] * len(all_boxes)
    
    # Match boxes using IoU
    for i, b1 in enumerate(all_boxes):
        if used[i]:
            continue
        
        matches = [b1]
        
        for j, b2 in enumerate(all_boxes):
            if i == j or used[j]:
                continue
            
            # Calculate IoU
            xi1 = max(b1['xyxy'][0], b2['xyxy'][0])
            yi1 = max(b1['xyxy'][1], b2['xyxy'][1])
            xi2 = min(b1['xyxy'][2], b2['xyxy'][2])
            yi2 = min(b1['xyxy'][3], b2['xyxy'][3])
            
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = (b1['xyxy'][2] - b1['xyxy'][0]) * (b1['xyxy'][3] - b1['xyxy'][1])
            box2_area = (b2['xyxy'][2] - b2['xyxy'][0]) * (b2['xyxy'][3] - b2['xyxy'][1])
            union = box1_area + box2_area - inter
            
            if union > 0 and inter / union > iou_threshold:
                matches.append(b2)
                used[j] = True
        
        # Only add to consensus if 2+ models agree
        if len(matches) >= 2:
            avg_box = np.mean([m['xyxy'] for m in matches], axis=0)
            current_consensus_items = []
            
            # Check for Windshield consensus
            windshield_count = sum(1 for m in matches if 'windshield' in m['class_name'].lower())
            if windshield_count >= 2:
                current_consensus_items.append({
                    'xyxy': avg_box,
                    'conf': np.mean([m['conf'] for m in matches]),
                    'cls': matches[0]['cls'],
                    'model_names': matches[0]['source'].names,
                    'detected_class': 'Windshield',
                    'is_windshield': True,
                    'is_light': False
                })
            
            # Check for Light consensus
            light_count = sum(1 for m in matches if 'light' in m['class_name'].lower())
            if light_count >= 2:
                current_consensus_items.append({
                    'xyxy': avg_box,
                    'conf': np.mean([m['conf'] for m in matches]),
                    'cls': matches[0]['cls'],
                    'model_names': matches[0]['source'].names,
                    'detected_class': 'Light',
                    'is_windshield': False,
                    'is_light': True
                })
            
            # Check for other damage consensus
            has_specific_damage = (windshield_count >= 2) or (light_count >= 2)
            if not has_specific_damage:
                other_damage_count = sum(1 for m in matches
                                       if 'windshield' not in m['class_name'].lower() and
                                          'light' not in m['class_name'].lower())
                if other_damage_count >= 2:
                    current_consensus_items.append({
                        'xyxy': avg_box,
                        'conf': np.mean([m['conf'] for m in matches]),
                        'cls': matches[0]['cls'],
                        'model_names': matches[0]['source'].names,
                        'detected_class': 'Damage',
                        'is_windshield': False,
                        'is_light': False
                    })
            
            consensus.extend(current_consensus_items)
        
        used[i] = True
    
    return consensus
