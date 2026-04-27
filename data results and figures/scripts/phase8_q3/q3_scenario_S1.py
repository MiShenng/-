import sys
import os
from pathlib import Path
import json
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from phase0_setup.datatypes import Task
from phase1_ingest.ingest import ingest_data
from phase3_core.evaluator import RouteEvaluator
from phase4_heuristic.heuristic import greedy_insertion, local_search
from phase5_postopt.lp_solve import consolidate_packets
from phase8_q3.dynamic_dispatcher import RollingHorizonDispatcher, StabilityEvaluatorWrapper
from phase6_q1.q1_report import generate_summary

def run_scenario_s1():
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / 'data'
    precomp_dir = base_dir / 'results' / 'precomputed'
    results_dir = base_dir / 'results'
    q3_dir = results_dir / 'q3'
    q3_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Environment
    coords, D, df_orders, tw, customers = ingest_data(data_dir)
    T_lookup = np.load(precomp_dir / 'T_lookup.npy')
    G_ij = np.load(precomp_dir / 'G_ij.npy')
    detour = np.load(precomp_dir / 'detour.npy')
    
    with open(precomp_dir / 'packets.json', 'r') as f:
        packet_dicts = json.load(f)
    packets = [Task(**d) for d in packet_dicts]
    time_grid = np.arange(420, 420 + 180 * 5, 5)
    
    # Base evaluator
    base_evaluator = RouteEvaluator(T_lookup, G_ij, detour, time_grid, packets, customers, green_ban=False)
    
    # 2. Load Q1 Baseline routes
    with open(results_dir / 'phaseA_solution.json', 'r') as f:
        baseline_routes = json.load(f)
        
    print(f"Loaded baseline with {len(baseline_routes)} routes.")
    
    # 3. Create Event: New Order at 11:00 (660 mins)
    event_time = 660.0
    # Let's say customer 98 placed a new order of 1000kg.
    # Customer 98 is in the green zone.
    # Time window: 11:00 to 18:00
    new_task_id = max(p.task_id for p in packets) + 1
    new_packet = Task(
        task_id=new_task_id,
        customer_id=98,
        w=1000.0,
        v=3.0,
        tw_start=event_time,
        tw_end=1080.0,
        service_time=20.0,
        is_green=True
    )
    packets.append(new_packet)
    # Update evaluator's packet dict
    base_evaluator.packets[new_task_id] = new_packet
    
    print(f"Injected Event: New Order (Task {new_task_id}) at 11:00.")
    
    # 4. Freeze and track assignments
    dispatcher = RollingHorizonDispatcher(base_evaluator)
    initial_routes, baseline_assignments = dispatcher.freeze_and_split(baseline_routes, event_time)
    
    print(f"Tracking assignments for stability penalty. Total packets tracked: {len(baseline_assignments)}")
    
    # 5. Stability Wrapper
    stability_evaluator = StabilityEvaluatorWrapper(base_evaluator, baseline_assignments, lambda_penalty=100.0)
    
    # 6. Insert the new order into the best existing route or new route
    print("Inserting new order into baseline plan...")
    from phase4_heuristic.heuristic import Solution
    sol = Solution()
    sol.routes = initial_routes
    
    best_cost_delta = float('inf')
    best_r_idx = -1
    best_p_idx = -1
    best_det = None
    
    for r_idx, r in enumerate(sol.routes):
        old_cost, _, _ = stability_evaluator.evaluate_route(r['packets'], r['v_type'], r_idx=r_idx)
        for p_idx in range(len(r['packets']) + 1):
            new_packets = list(r['packets'])
            new_packets.insert(p_idx, new_task_id)
            cost, feas, det = stability_evaluator.evaluate_route(new_packets, r['v_type'], r_idx=r_idx)
            if feas:
                delta = cost - old_cost
                if delta < best_cost_delta:
                    best_cost_delta = delta
                    best_r_idx = r_idx
                    best_p_idx = p_idx
                    best_det = det
                    
    if best_r_idx != -1:
        sol.routes[best_r_idx]['packets'].insert(best_p_idx, new_task_id)
        sol.routes[best_r_idx]['cost'] += best_cost_delta
        sol.routes[best_r_idx]['details'] = best_det
        print(f"Inserted into Route {best_r_idx} at position {best_p_idx}. Marginal cost: {best_cost_delta:.2f}")
    else:
        print("Could not insert into existing routes. Opening new route.")
        cost, feas, det = stability_evaluator.evaluate_route([new_task_id], 'E-3000', r_idx=-1)
        sol.routes.append({
            'v_type': 'E-3000',
            'packets': [new_task_id],
            'cost': cost,
            'details': det
        })
        
    # 7. Local Search with Damage Limitation
    c_idx = new_packet.customer_id
    event_loc = (customers[c_idx].x, customers[c_idx].y)
    print("Running damage-limited local search...")
    sol = local_search(stability_evaluator, sol, max_iters=10, green_ban=False, event_loc=event_loc)
    
    # Consolidate and finalize
    # We must use base_evaluator to get the real cost without stability lambda included!
    final_cost = 0.0
    for r in sol.routes:
        c, f, d = base_evaluator.evaluate_route(r['packets'], r['v_type'])
        r['cost'] = c
        r['details'] = d
        final_cost += c
        
    print(f"Dynamic rescheduling complete. Final cost: {final_cost:.2f}")
    
    with open(q3_dir / 'S1_plan.json', 'w') as f:
        json.dump(sol.routes, f, indent=2)
        
    # Check deviations
    deviations = 0
    for r_idx, r in enumerate(sol.routes):
        for pid in r['packets']:
            if pid in baseline_assignments and baseline_assignments[pid] != r_idx:
                deviations += 1
                
    print(f"Total task reassignments (deviations from baseline): {deviations}")

if __name__ == "__main__":
    run_scenario_s1()
