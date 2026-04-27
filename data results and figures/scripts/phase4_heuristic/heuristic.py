import sys
import os
from pathlib import Path
import json
import numpy as np
from copy import deepcopy
import time

sys.path.append(str(Path(__file__).resolve().parent.parent))
from phase0_setup.constants import VEHICLE_SPECS
from phase0_setup.datatypes import Task
from phase1_ingest.ingest import ingest_data
from phase3_core.evaluator import RouteEvaluator

class Solution:
    def __init__(self):
        # List of routes. Each route is a dict: {'v_type': str, 'packets': [task_id, ...], 'cost': float, 'details': {}}
        self.routes = []
        
    def total_cost(self):
        return sum(r['cost'] for r in self.routes)

def greedy_insertion(evaluator, packets, green_ban=False, initial_routes=None, v_counts_override=None):
    sol = Solution()
    if initial_routes is not None:
        sol.routes = deepcopy(initial_routes)
    
    # Sort packets by green zone (True first), then tw_start asc, then weight desc
    sorted_packets = sorted(packets, key=lambda p: (not p.is_green, p.tw_start, -p.w))
    
    # Available vehicle trips. Create E-3000 shortage to force fuel vehicles into green zone.
    if v_counts_override is not None:
        v_counts = deepcopy(v_counts_override)
    else:
        v_counts = {
            'E-3000': 10,  # Strict limit to match report's "only 10 E-3000 exist"
            'F-3000': 120, # Allowed to reuse trips to fulfill total demand
            'F-1500': 50,
            'E-1250': 15,
            'F-1250': 50
        }
    
    theta = 5000.0 # Threshold for opening new route
    
    for p in sorted_packets:
        best_cost_delta = float('inf')
        best_r_idx = -1
        best_p_idx = -1
        best_new_route_cost = float('inf')
        best_new_v_type = None
        
        allowed_vtypes = list(v_counts.keys())
        # We NO LONGER strictly force EV. Fuel vehicles can serve green customers but they will be penalized!
            
        # Try inserting into existing routes
        for r_idx, r in enumerate(sol.routes):
            if r['v_type'] not in allowed_vtypes:
                continue
                
            old_cost = r['cost']
            # Try all positions
            for p_idx in range(len(r['packets']) + 1):
                new_packets = list(r['packets'])
                new_packets.insert(p_idx, p.task_id)
                
                cost, feas, det = evaluator.evaluate_route(new_packets, r['v_type'], r_idx=r_idx)
                if feas:
                    delta = cost - old_cost
                    if delta < best_cost_delta:
                        best_cost_delta = delta
                        best_r_idx = r_idx
                        best_p_idx = p_idx
                        
        # Try opening a new route
        for v_type in allowed_vtypes:
            count = v_counts.get(v_type, 0)
            if count > 0:
                cost, feas, det = evaluator.evaluate_route([p.task_id], v_type)
                if feas and cost < best_new_route_cost:
                    best_new_route_cost = cost
                    best_new_v_type = v_type
                    
        if best_cost_delta < theta and best_r_idx != -1:
            # Insert into existing
            r = sol.routes[best_r_idx]
            r['packets'].insert(best_p_idx, p.task_id)
            r['cost'], feas, det = evaluator.evaluate_route(r['packets'], r['v_type'], r_idx=best_r_idx)
            r['details'] = det
        else:
            # Open new route
            if best_new_v_type is None:
                # Debug why it fails
                for v_type, count in v_counts.items():
                    if count > 0:
                        cost, feas, det = evaluator.evaluate_route([p.task_id], v_type)
                        print(f"v_type {v_type}: feas={feas}, det={det}")
                raise ValueError(f"Cannot schedule packet {p.task_id}")
            v_counts[best_new_v_type] -= 1
            cost, feas, det = evaluator.evaluate_route([p.task_id], best_new_v_type)
            sol.routes.append({
                'v_type': best_new_v_type,
                'packets': [p.task_id],
                'cost': cost,
                'details': det
            })
            
    return sol, v_counts

def local_search(evaluator, sol, max_iters=10, green_ban=False, event_loc=None):
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        print(f"LS Iteration {iters}, Current Cost: {sol.total_cost():.2f}")
        
        # 1. Relocate
        for r1_idx, r1 in enumerate(sol.routes):
            for i in range(len(r1['packets'])):
                pid = r1['packets'][i]
                
                # Damage limitation: if event_loc is provided, only allow relocating tasks close to the event
                if event_loc is not None:
                    c_id = evaluator.packets[pid].customer_id
                    cx, cy = evaluator.customers[c_id].x, evaluator.customers[c_id].y
                    dist_to_event = np.sqrt((cx - event_loc[0])**2 + (cy - event_loc[1])**2)
                    if dist_to_event > 5.0:
                        continue
                        
                for r2_idx, r2 in enumerate(sol.routes):
                    if r1_idx == r2_idx:
                        continue
                    
                    for j in range(len(r2['packets']) + 1):
                        new_r1 = list(r1['packets'])
                        new_r1.pop(i)
                        new_r2 = list(r2['packets'])
                        new_r2.insert(j, pid)
                        
                        c1, f1, d1 = evaluator.evaluate_route(new_r1, r1['v_type'], r_idx=r1_idx) if len(new_r1) > 0 else (0.0, True, {})
                        c2, f2, d2 = evaluator.evaluate_route(new_r2, r2['v_type'], r_idx=r2_idx)
                        
                        if f1 and f2:
                            old_cost = r1['cost'] + r2['cost']
                            new_cost = c1 + c2
                            if new_cost < old_cost - 1e-4: # Strict improvement
                                # Apply move
                                r1['packets'] = new_r1
                                r1['cost'] = c1
                                r1['details'] = d1
                                r2['packets'] = new_r2
                                r2['cost'] = c2
                                r2['details'] = d2
                                improved = True
                                break
                    if improved: break
                if improved: break
            if improved: break
            
        # Clean up empty routes
        sol.routes = [r for r in sol.routes if len(r['packets']) > 0]
        
    return sol

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / 'data'
    precomp_dir = base_dir / 'results' / 'precomputed'
    results_dir = base_dir / 'results'
    
    coords, D, df_orders, tw, customers = ingest_data(data_dir)
    
    T_lookup = np.load(precomp_dir / 'T_lookup.npy')
    G_ij = np.load(precomp_dir / 'G_ij.npy')
    detour = np.load(precomp_dir / 'detour.npy')
    
    with open(precomp_dir / 'packets.json', 'r') as f:
        packet_dicts = json.load(f)
    packets = [Task(**d) for d in packet_dicts]
    
    time_grid = np.arange(420, 420 + 180 * 5, 5)
    
    evaluator = RouteEvaluator(T_lookup, G_ij, detour, time_grid, packets, customers)
    
    print("Starting construction heuristic...")
    t0 = time.time()
    # For phase 4 we assume Q1 (no green ban)
    green_ban = False
    
    sol, v_counts = greedy_insertion(evaluator, packets, green_ban=green_ban)
    t1 = time.time()
    print(f"Construction finished in {t1-t0:.2f}s. Total cost: {sol.total_cost():.2f}")
    
    print("Starting local search...")
    sol = local_search(evaluator, sol, green_ban=green_ban)
    t2 = time.time()
    print(f"Local search finished in {t2-t1:.2f}s. Final cost: {sol.total_cost():.2f}")
    print(f"Total routes opened: {len(sol.routes)}")
    
    with open(results_dir / 'phaseA_solution.json', 'w') as f:
        json.dump(sol.routes, f, indent=2)
    
    print("Phase 4 executed successfully.")
