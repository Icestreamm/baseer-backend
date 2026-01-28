"""
Cost Calculator - Ported from Python V3.8 code
Calculates repair costs based on damage areas and component replacements
"""

from typing import List, Dict, Tuple

def calculate_costs(
    photo_results: List[Dict],
    paint_costs_jod: List[Tuple[int, float, float]],  # (photo_num, area_cm2, cost_jod)
    global_has_windshield_damage: bool,
    global_has_light_damage: bool,
    global_has_tire_damage: bool,
    jod_to_local: float,
    country_tax_rate: float,
    luxury_index: float,
    country_lux_factor: float,
    currency: str
) -> Dict:
    """
    Calculate total repair costs.
    
    Cost formula:
    - Paint: 0.019157 * area_cm2 + 2.093 (per photo)
    - Light: 30 JOD (once if detected)
    - Windshield: 50 JOD (once if detected)
    - Tire: 20 JOD (once if detected)
    
    Then:
    1. Sum all costs in JOD
    2. Convert to local currency
    3. Apply tax
    4. Apply luxury factors
    
    Returns:
        Dict with all cost breakdown
    """
    # Paint costs total
    paint_total_jod = sum(cost for _, _, cost in paint_costs_jod)
    
    # Component costs (charged once globally)
    light_cost_jod = 30.0 if global_has_light_damage else 0.0
    windshield_cost_jod = 50.0 if global_has_windshield_damage else 0.0
    tire_cost_jod = 20.0 if global_has_tire_damage else 0.0
    
    # Base subtotal in JOD
    subtotal_jod_base = paint_total_jod + light_cost_jod + windshield_cost_jod + tire_cost_jod
    
    # Convert to local currency
    subtotal_local_base = subtotal_jod_base * jod_to_local
    
    # Calculate tax on base subtotal
    tax_amount_on_base_local = subtotal_local_base * country_tax_rate
    
    # Subtotal after tax
    subtotal_post_base_tax = subtotal_local_base + tax_amount_on_base_local
    
    # Apply luxury factors
    final_local_cost = subtotal_post_base_tax * luxury_index * country_lux_factor
    
    # Individual costs in local currency
    paint_costs_local = [(num, area, cost * jod_to_local) for num, area, cost in paint_costs_jod]
    light_cost_local = light_cost_jod * jod_to_local
    windshield_cost_local = windshield_cost_jod * jod_to_local
    tire_cost_local = tire_cost_jod * jod_to_local
    
    return {
        'paint_costs_jod': paint_costs_jod,
        'paint_costs_local': paint_costs_local,
        'paint_total_jod': paint_total_jod,
        'light_cost_jod': light_cost_jod,
        'light_cost_local': light_cost_local,
        'windshield_cost_jod': windshield_cost_jod,
        'windshield_cost_local': windshield_cost_local,
        'tire_cost_jod': tire_cost_jod,
        'tire_cost_local': tire_cost_local,
        'subtotal_jod_base': subtotal_jod_base,
        'subtotal_local_base': subtotal_local_base,
        'tax_rate': country_tax_rate,
        'tax_amount_on_base_local': tax_amount_on_base_local,
        'subtotal_post_base_tax': subtotal_post_base_tax,
        'luxury_index': luxury_index,
        'country_lux_factor': country_lux_factor,
        'final_local_cost': final_local_cost,
        'currency': currency,
    }
