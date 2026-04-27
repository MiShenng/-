import sys
import os
from pathlib import Path
import copy
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from phase0_setup.constants import VEHICLE_SPECS, WAIT_PENALTY, LATE_PENALTY, PRICE_ELEC, PRICE_FUEL, CO2_COST, ETA_FUEL

class Event:
    def __init__(self, e_type, clock, payload):
        self.type = e_type  # 'new_order', 'cancel', etc.
        self.clock = clock  # Time in minutes from midnight
        self.payload = payload # e.g. Task object

class RollingHorizonDispatcher:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        
    def trace_route(self, route_packets, v_type):
        """
        Simulates the route and returns the exact timing of each node.
        Returns a list of dicts with keys: 'packet_id', 't_dep', 't_arr', 't_start_svc', 't_end_svc'
        """
        nodes = [0] + [self.evaluator.packets[pid].customer_id for pid in route_packets] + [0]
        svc_times = [0.0] + [self.evaluator.packets[pid].service_time for pid in route_packets] + [0.0]
        tw_starts = [0.0] + [self.evaluator.packets[pid].tw_start for pid in route_packets] + [0.0]
        tw_ends = [24*60.0] + [self.evaluator.packets[pid].tw_end for pid in route_packets] + [24*60.0]
        
        # We need the optimal departure from DC, which we stored in the details
        # However, to be exact, we can just re-run the grid search
        best_cost, feas, det = self.evaluator.evaluate_route(route_packets, v_type)
        if not feas:
            return []
            
        t_curr = det['optimal_dep']
        is_ev = (VEHICLE_SPECS[v_type]['type'] == 'electric')
        
        trace = []
        for k in range(1, len(nodes)):
            i = nodes[k-1]
            j = nodes[k]
            
            use_detour = False
            if self.evaluator.green_ban and not is_ev and self.evaluator.G[i, j]:
                tt_est = self.evaluator.get_travel_time(i, j, t_curr, False)
                if 480 <= t_curr + tt_est <= 960 or 480 <= t_curr <= 960:
                    if self.evaluator.detour[i, j] >= 1e8:
                        t_curr = max(t_curr, 960.0)
                    else:
                        use_detour = True
                        
            t_dep = t_curr
            tt = self.evaluator.get_travel_time(i, j, t_curr, use_detour)
            t_arr = t_curr + tt
            
            if t_arr < tw_starts[k]:
                t_start_svc = tw_starts[k]
            else:
                t_start_svc = t_arr
                
            t_curr = t_start_svc + svc_times[k]
            
            if k < len(nodes) - 1:
                trace.append({
                    'packet_id': route_packets[k-1],
                    't_dep': t_dep, # Time left previous node
                    't_arr': t_arr,
                    't_start_svc': t_start_svc,
                    't_end_svc': t_curr
                })
        return trace

    def freeze_and_split(self, routes, event_time):
        baseline_assignments = {}
        for r_idx, r in enumerate(routes):
            trace = self.trace_route(r['packets'], r['v_type'])
            for step in trace:
                baseline_assignments[step['packet_id']] = r_idx
                
        # Return deepcopy of routes to act as the starting point
        return copy.deepcopy(routes), baseline_assignments

class StabilityEvaluatorWrapper:
    def __init__(self, evaluator, baseline_assignments, lambda_penalty=100.0):
        self.evaluator = evaluator
        self.baseline_assignments = baseline_assignments
        self.lambda_penalty = lambda_penalty
        
    def evaluate_route(self, packet_ids, v_type, r_idx=-1):
        cost, feas, det = self.evaluator.evaluate_route(packet_ids, v_type, r_idx)
        if not feas:
            return cost, feas, det
            
        stability_cost = 0.0
        for pid in packet_ids:
            if pid in self.baseline_assignments:
                if self.baseline_assignments[pid] != r_idx:
                    stability_cost += self.lambda_penalty
                    
        return cost + stability_cost, feas, det
        
    def __getattr__(self, item):
        return getattr(self.evaluator, item)
