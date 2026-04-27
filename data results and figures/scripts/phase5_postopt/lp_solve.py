import sys
import os
from pathlib import Path
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))
from phase0_setup.datatypes import Task

def consolidate_packets(sol_routes, packets):
    """
    Consolidates consecutive packets for the same customer on the same vehicle.
    This fulfills the discrete allocation requirement and represents a simplified
    Phase B Post-Optimization (merging splits).
    """
    packet_dict = {p.task_id: p for p in packets}
    final_routes = []
    
    for r in sol_routes:
        if not r['packets']:
            continue
            
        merged_customers = []
        current_cust = None
        current_w = 0.0
        current_v = 0.0
        
        for pid in r['packets']:
            p = packet_dict[pid]
            if current_cust is None:
                current_cust = p.customer_id
                current_w = p.w
                current_v = p.v
            elif p.customer_id == current_cust:
                current_w += p.w
                current_v += p.v
            else:
                merged_customers.append({
                    'customer_id': current_cust,
                    'delivered_w': current_w,
                    'delivered_v': current_v
                })
                current_cust = p.customer_id
                current_w = p.w
                current_v = p.v
                
        if current_cust is not None:
            merged_customers.append({
                'customer_id': current_cust,
                'delivered_w': current_w,
                'delivered_v': current_v
            })
            
        final_routes.append({
            'v_type': r['v_type'],
            'visits': merged_customers,
            'cost': r['cost'],
            'details': r.get('details', {})
        })
        
    return final_routes

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    results_dir = base_dir / 'results'
    precomp_dir = results_dir / 'precomputed'
    
    with open(precomp_dir / 'packets.json', 'r') as f:
        packet_dicts = json.load(f)
    packets = [Task(**d) for d in packet_dicts]
    
    sol_path = results_dir / 'phaseA_solution.json'
    if os.path.exists(sol_path):
        with open(sol_path, 'r') as f:
            sol_routes = json.load(f)
            
        final_routes = consolidate_packets(sol_routes, packets)
        
        with open(results_dir / 'phaseB_solution.json', 'w') as f:
            json.dump(final_routes, f, indent=2)
            
        print(f"Phase 5 (Post-Optimisation/Consolidation) completed. Total consolidated routes: {len(final_routes)}")
    else:
        print("No Phase A solution found.")
