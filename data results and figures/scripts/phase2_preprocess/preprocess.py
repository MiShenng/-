import sys
import os
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from phase0_setup.constants import SPEED_PARAMS, REGIMES, R_GREEN, VEHICLE_SPECS, S0, BETA_W, BETA_V
from phase0_setup.datatypes import Task
from phase1_ingest.ingest import ingest_data

def calibrate_pace():
    pace_table = {}
    for name, (mu_v, sigma_v) in SPEED_PARAMS.items():
        mu_p = 60.0 / mu_v
        sigma_p = (60.0 * sigma_v) / (mu_v ** 2)
        pace_table[name] = {'mu_p': mu_p, 'sigma_p': sigma_p}
    return pace_table

def get_regime_at_time(t):
    t_mod = t % (24 * 60)
    for start, end, name in REGIMES:
        if start <= t_mod < end:
            return name
    return 'smooth' # Default fallback

def build_travel_time_lookup(D, pace_table):
    n = D.shape[0]
    T_lookup = np.zeros((n, n, 180))
    time_grid = np.arange(420, 420 + 180 * 5, 5) # 07:00 to 21:55
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d_ij = D[i, j]
            for t_idx, t0 in enumerate(time_grid):
                d_rem = d_ij
                t_curr = t0
                while d_rem > 1e-6:
                    t_mod = t_curr % (24 * 60)
                    # Find current regime bounds
                    r_name = 'smooth'
                    r_end = 24 * 60
                    for start, end, name in REGIMES:
                        if start <= t_mod < end:
                            r_name = name
                            r_end = t_curr - t_mod + end # Map back to absolute time
                            break
                    
                    mu_p = pace_table[r_name]['mu_p']
                    time_avail = r_end - t_curr
                    dist_avail = time_avail / mu_p
                    
                    if d_rem <= dist_avail:
                        t_curr += d_rem * mu_p
                        d_rem = 0
                    else:
                        t_curr = r_end
                        d_rem -= dist_avail
                        
                T_lookup[i, j, t_idx] = t_curr - t0
                
    return T_lookup, time_grid

def compute_G_ij(coords):
    n = len(coords)
    G = np.zeros((n, n), dtype=bool)
    for i in range(n):
        P_i = np.array(coords[i])
        for j in range(n):
            if i == j:
                continue
            P_j = np.array(coords[j])
            
            dP = P_j - P_i
            A = np.dot(dP, dP)
            B = 2 * np.dot(P_i, dP)
            C = np.dot(P_i, P_i)
            
            if A == 0:
                continue
                
            t_star = -B / (2 * A)
            
            if 0 <= t_star <= 1 and (C - B**2 / (4 * A)) < R_GREEN**2:
                G[i, j] = True
            elif t_star < 0 and C < R_GREEN**2:
                G[i, j] = True
            elif t_star > 1 and (A + B + C) < R_GREEN**2:
                G[i, j] = True
                
    return G

def compute_detour(D, coords, G):
    n = len(coords)
    detour = np.copy(D)
    for i in range(n):
        P_i = np.array(coords[i])
        D_i = np.linalg.norm(P_i)
        for j in range(n):
            if not G[i, j]:
                continue
                
            P_j = np.array(coords[j])
            D_j = np.linalg.norm(P_j)
            
            if D_i <= R_GREEN or D_j <= R_GREEN:
                detour[i, j] = 1e9 # Infeasible
                continue
                
            l_i = np.sqrt(D_i**2 - R_GREEN**2)
            l_j = np.sqrt(D_j**2 - R_GREEN**2)
            
            dot_val = np.dot(P_i, P_j) / (D_i * D_j)
            dot_val = np.clip(dot_val, -1.0, 1.0)
            delta_theta = np.arccos(dot_val)
            
            gamma_i = np.arccos(R_GREEN / D_i)
            gamma_j = np.arccos(R_GREEN / D_j)
            
            theta_arc = delta_theta - gamma_i - gamma_j
            if theta_arc > 0:
                detour[i, j] = l_i + l_j + R_GREEN * theta_arc
            else:
                detour[i, j] = D[i, j]
                
    return detour

def discretise_packets(customers):
    packets = []
    Q_w_max = VEHICLE_SPECS['F-3000']['Q_w']
    Q_v_max = VEHICLE_SPECS['F-3000']['Q_v']
    
    task_id = 0
    for i in range(1, 99):
        c = customers[i]
        if c.demand_w == 0 and c.demand_v == 0:
            continue
            
        n_w = int(np.ceil(c.demand_w / Q_w_max))
        n_v = int(np.ceil(c.demand_v / Q_v_max))
        n_i = max(1, n_w, n_v)
        
        w_split = c.demand_w / n_i
        v_split = c.demand_v / n_i
        svc_time = S0 + BETA_W * w_split + BETA_V * v_split
        
        c.is_green = bool(np.linalg.norm(np.array([c.x, c.y])) <= R_GREEN)
        
        for _ in range(n_i):
            packets.append(Task(
                task_id=task_id,
                customer_id=c.id,
                w=w_split,
                v=v_split,
                tw_start=c.tw_start,
                tw_end=c.tw_end,
                is_green=c.is_green,
                service_time=svc_time
            ))
            task_id += 1
            
    return packets

def plot_demand(customers, packets, figures_dir):
    cust_w = [c.demand_w for c in customers.values() if c.id != 0 and c.demand_w > 0]
    pkt_w = [p.w for p in packets]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(cust_w, bins=20, color='skyblue', edgecolor='black')
    ax1.axvline(3000, color='red', linestyle='dashed', linewidth=2, label='3000 kg Limit')
    ax1.set_title('Customer Demand (Weight)')
    ax1.legend()
    
    ax2.hist(pkt_w, bins=20, color='lightgreen', edgecolor='black')
    ax2.axvline(3000, color='red', linestyle='dashed', linewidth=2)
    ax2.set_title('Packet Demand (Weight) after Splitting')
    
    plt.suptitle('Figure F4: Demand Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'f04_demand_distribution.png'), dpi=300)
    plt.close()

def plot_pace(pace_table, figures_dir):
    times = np.arange(420, 1320)
    mu_vals = []
    sigma_vals = []
    for t in times:
        r = get_regime_at_time(t)
        mu_vals.append(pace_table[r]['mu_p'])
        sigma_vals.append(pace_table[r]['sigma_p'])
        
    mu_vals = np.array(mu_vals)
    sigma_vals = np.array(sigma_vals)
    
    plt.figure(figsize=(10, 5))
    plt.plot(times / 60, mu_vals, label='Mean Pace (min/km)', color='blue')
    plt.fill_between(times / 60, mu_vals - sigma_vals, mu_vals + sigma_vals, color='blue', alpha=0.2, label='±1 Sigma')
    
    plt.axvspan(8, 16, color='red', alpha=0.1, label='Ban Window')
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Pace (min/km)')
    plt.title('Figure F2: Time-Dependent Pace Profile')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, 'f02_pace_profile.png'), dpi=300)
    plt.close()

def plot_g_ij(G, figures_dir):
    plt.figure(figsize=(8, 8))
    plt.imshow(G, cmap='Blues', aspect='auto')
    plt.title('Figure F3: Arc Penetration Matrix $G_{ij}$')
    plt.colorbar(label='Penetrates Green Zone')
    plt.savefig(os.path.join(figures_dir, 'f03_arc_penetration.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / 'data'
    results_dir = base_dir / 'results'
    precomp_dir = results_dir / 'precomputed'
    figures_dir = base_dir / 'figures'
    
    coords, D, df_orders, tw, customers = ingest_data(data_dir)
    
    pace_table = calibrate_pace()
    T_lookup, time_grid = build_travel_time_lookup(D, pace_table)
    
    G = compute_G_ij(coords)
    detour = compute_detour(D, coords, G)
    
    packets = discretise_packets(customers)
    
    precomp_dir.mkdir(parents=True, exist_ok=True)
    np.save(precomp_dir / 'T_lookup.npy', T_lookup)
    np.save(precomp_dir / 'G_ij.npy', G)
    np.save(precomp_dir / 'detour.npy', detour)
    
    with open(precomp_dir / 'packets.json', 'w') as f:
        json.dump([vars(p) for p in packets], f, indent=2)
        
    plot_demand(customers, packets, figures_dir)
    plot_pace(pace_table, figures_dir)
    plot_g_ij(G, figures_dir)
    
    print(f"Phase 2 completed. Discretized into {len(packets)} packets.")
