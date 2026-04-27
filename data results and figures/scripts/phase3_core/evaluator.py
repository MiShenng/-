import numpy as np
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from phase0_setup.constants import VEHICLE_SPECS, REGIMES, WAIT_PENALTY, LATE_PENALTY, PRICE_ELEC, PRICE_FUEL, CO2_COST, ETA_ELEC, ETA_FUEL, SPEED_PARAMS, R_GREEN

class RouteEvaluator:
    def __init__(self, T_lookup, G_ij, detour, time_grid, packets, customers, green_ban=False):
        self.green_ban = green_ban
        self.T = T_lookup
        self.G = G_ij
        self.detour = detour
        self.time_grid = time_grid
        self.packets = {p.task_id: p for p in packets}
        self.customers = customers
        
        # Dynamic overrides for sensitivity analysis
        self.PRICE_FUEL = PRICE_FUEL
        self.CO2_COST = CO2_COST
        self.PRICE_ELEC = PRICE_ELEC
        
        # Calculate pace table
        self.pace_table = {}
        for name, (mu_v, sigma_v) in SPEED_PARAMS.items():
            mu_p = 60.0 / mu_v
            sigma_p = (60.0 * sigma_v) / (mu_v ** 2)
            self.pace_table[name] = {'mu_p': mu_p, 'sigma_p': sigma_p}
            
    def get_travel_time(self, i, j, t_dep, use_detour=False):
        if i == j:
            return 0.0
            
        t_dep_clip = np.clip(t_dep, self.time_grid[0], self.time_grid[-1])
        idx = (t_dep_clip - self.time_grid[0]) / 5.0
        idx0 = int(np.floor(idx))
        idx1 = min(idx0 + 1, len(self.time_grid) - 1)
        w = idx - idx0
        
        tt = (1 - w) * self.T[i, j, idx0] + w * self.T[i, j, idx1]
        
        if use_detour:
            P_i = np.array([self.customers[i].x, self.customers[i].y])
            P_j = np.array([self.customers[j].x, self.customers[j].y])
            dist_straight = np.linalg.norm(P_i - P_j)
            dist_ratio = self.detour[i, j] / max(1e-6, dist_straight)
            tt *= dist_ratio
            
        return tt
        
    def base_invariants(self, service_time, tw_start, tw_end):
        return (service_time, tw_start, tw_end, 0.0, 0.0)
        
    def concat_invariants(self, inv1, inv2, travel_time):
        D1, E1, L1, W1, P1 = inv1
        D2, E2, L2, W2, P2 = inv2
        
        D_new = D1 + travel_time + D2
        E_new = max(E1, E2 - D1 - travel_time)
        L_new = min(L1, L2 - D1 - travel_time)
        
        W_new = W1 + W2 + WAIT_PENALTY * max(0, E2 - E1 - D1 - travel_time)
        P_new = P1 + P2 + LATE_PENALTY * max(0, L1 + D1 + travel_time - L2)
        
        return (D_new, E_new, L_new, W_new, P_new)

    def evaluate_route(self, packet_ids, v_type, r_idx=-1):
        nodes = [0] + [self.packets[pid].customer_id for pid in packet_ids] + [0]
        svc_times = [0.0] + [self.packets[pid].service_time for pid in packet_ids] + [0.0]
        tw_starts = [0.0] + [self.packets[pid].tw_start for pid in packet_ids] + [0.0]
        tw_ends = [24*60.0] + [self.packets[pid].tw_end for pid in packet_ids] + [24*60.0]
        
        total_w = sum(self.packets[pid].w for pid in packet_ids)
        total_v = sum(self.packets[pid].v for pid in packet_ids)
        spec = VEHICLE_SPECS[v_type]
        
        if total_w > spec['Q_w'] + 1e-6 or total_v > spec['Q_v'] + 1e-6:
            return float('inf'), False, {"reason": "Capacity"}
            
        is_ev = (spec['type'] == 'electric')
        fixed_cost = 400.0
        alpha_star = spec['alpha']
        
        best_cost = float('inf')
        best_details = {}
        
        # Grid search over possible departure times from DC
        # Since wait penalty is 1/3 and late is 5/6, we should depart as late as possible to avoid wait, 
        # but early enough to avoid late penalty. We can just test departure times from 420 to 1200, step 30.
        # Actually, let's just check departure times that would align with the start of each TW.
        candidate_deps = [420.0]
        for k in range(1, len(nodes)):
            # roughly estimate back-propagation of tw_starts
            candidate_deps.append(max(420.0, tw_starts[k] - 120.0))
            candidate_deps.append(max(420.0, tw_starts[k] - 60.0))
            candidate_deps.append(max(420.0, tw_starts[k]))
            
        for dep_DC in sorted(list(set(candidate_deps))):
            route_dist = 0.0
            energy_cost = 0.0
            soft_penalty = 0.0
            current_w = total_w
            
            t_curr = dep_DC
            feasible = True
            
            for k in range(1, len(nodes)):
                i = nodes[k-1]
                j = nodes[k]
                
                use_detour = False
                if self.green_ban and not is_ev and self.G[i, j]:
                    tt_est = self.get_travel_time(i, j, t_curr, False)
                    t_arr_est = t_curr + tt_est
                    if 480 <= t_arr_est <= 960 or 480 <= t_curr <= 960:
                        if self.detour[i, j] >= 1e8:
                            t_curr = max(t_curr, 960.0)
                        else:
                            use_detour = True
                
                P_i = np.array([self.customers[i].x, self.customers[i].y])
                P_j = np.array([self.customers[j].x, self.customers[j].y])
                dist_straight = np.linalg.norm(P_i - P_j)
                d_ij = self.detour[i, j] if use_detour else dist_straight
                route_dist += d_ij
                
                tt = self.get_travel_time(i, j, t_curr, use_detour)
                t_arr = t_curr + tt
                
                if tt > 1e-6 and d_ij > 1e-6:
                    v_kmh = d_ij / (tt / 60.0)
                    rho = current_w / spec['Q_w']
                    
                    if is_ev:
                        epk = 0.0014 * v_kmh**2 - 0.12 * v_kmh + 36.19
                        rate_per_km = (epk / 100.0) * (1 + spec['alpha'] * rho)
                        energy_cost += rate_per_km * d_ij * self.PRICE_ELEC
                    else:
                        fpk = 0.0025 * v_kmh**2 - 0.2554 * v_kmh + 31.75
                        rate_per_km = (fpk / 100.0) * (1 + spec['alpha'] * rho)
                        fuel_c = rate_per_km * d_ij * self.PRICE_FUEL
                        carbon_c = rate_per_km * d_ij * ETA_FUEL * self.CO2_COST
                        energy_cost += fuel_c + carbon_c
                    
                if k < len(nodes) - 1:
                    pid = packet_ids[k-1]
                    current_w -= self.packets[pid].w
                
                # Penalty at node k
                if t_arr < tw_starts[k]:
                    soft_penalty += WAIT_PENALTY * (tw_starts[k] - t_arr)
                    t_curr = tw_starts[k] + svc_times[k]
                elif t_arr > tw_ends[k]:
                    soft_penalty += LATE_PENALTY * (t_arr - tw_ends[k])
                    t_curr = t_arr + svc_times[k]
                else:
                    t_curr = t_arr + svc_times[k]
                    
            total_cost = fixed_cost + energy_cost + soft_penalty
            if total_cost < best_cost:
                best_cost = total_cost
                best_details = {
                    "fixed_cost": fixed_cost,
                    "energy_cost": energy_cost,
                    "penalty": soft_penalty,
                    "dist": route_dist,
                    "optimal_dep": dep_DC
                }
                
        return best_cost, True, best_details

if __name__ == "__main__":
    # Simple test logic
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / 'data'
    precomp_dir = base_dir / 'results' / 'precomputed'
    
    from phase1_ingest.ingest import ingest_data
    coords, D, df_orders, tw, customers = ingest_data(data_dir)
    
    import json
    T_lookup = np.load(precomp_dir / 'T_lookup.npy')
    G_ij = np.load(precomp_dir / 'G_ij.npy')
    detour = np.load(precomp_dir / 'detour.npy')
    
    from phase0_setup.datatypes import Task
    with open(precomp_dir / 'packets.json', 'r') as f:
        packet_dicts = json.load(f)
    packets = [Task(**d) for d in packet_dicts]
    
    time_grid = np.arange(420, 420 + 180 * 5, 5)
    
    evaluator = RouteEvaluator(T_lookup, G_ij, detour, time_grid, packets, customers)
    
    # Test with a single packet route
    if len(packets) > 0:
        cost, feas, det = evaluator.evaluate_route([packets[0].task_id], 'F-3000')
        print(f"Test Route F-3000 Cost: {cost:.2f}, Feasible: {feas}, Details: {det}")
        
        cost, feas, det = evaluator.evaluate_route([packets[0].task_id], 'E-3000')
        print(f"Test Route E-3000 Cost: {cost:.2f}, Feasible: {feas}, Details: {det}")
    print("Phase 3 executed successfully.")
