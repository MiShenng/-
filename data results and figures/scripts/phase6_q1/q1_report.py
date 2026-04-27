import sys
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.append(str(Path(__file__).resolve().parent.parent))
from phase0_setup.constants import VEHICLE_SPECS, R_GREEN
from phase0_setup.datatypes import Task
from phase1_ingest.ingest import ingest_data

def generate_summary(routes):
    by_vtype = {}
    for r in routes:
        vt = r['v_type']
        if vt not in by_vtype:
            by_vtype[vt] = 0
        by_vtype[vt] += 1
        
    physical_startup = 0
    for vt, trips in by_vtype.items():
        physical_used = min(trips, VEHICLE_SPECS[vt]['count'])
        physical_startup += physical_used * 400.0
        
    summary = {
        'n_routes': len(routes),
        'total_km': sum(r['details']['dist'] for r in routes),
        'startup_cost': physical_startup,
        'energy_cost': sum(r['details']['energy_cost'] for r in routes),
        'penalty_cost': sum(r['details']['penalty'] for r in routes),
        'by_vtype': by_vtype
    }
    
    summary['total_cost'] = summary['startup_cost'] + summary['energy_cost'] + summary['penalty_cost']
        
    return summary

def generate_csvs(routes, q1_dir):
    stops_data = []
    routes_data = []
    
    for r_idx, r in enumerate(routes):
        vt = r['v_type']
        details = r['details']
        
        # Simplified reconstruction for stops
        stops_data.append({
            'route_id': r_idx,
            'vehicle_type': vt,
            'stop_order': 0,
            'customer_id': 0, # DC departure
            'delivered_w': 0,
            'delivered_v': 0
        })
        
        tot_w = 0
        tot_v = 0
        for s_idx, visit in enumerate(r['visits']):
            stops_data.append({
                'route_id': r_idx,
                'vehicle_type': vt,
                'stop_order': s_idx + 1,
                'customer_id': visit['customer_id'],
                'delivered_w': visit['delivered_w'],
                'delivered_v': visit['delivered_v']
            })
            tot_w += visit['delivered_w']
            tot_v += visit['delivered_v']
            
        stops_data.append({
            'route_id': r_idx,
            'vehicle_type': vt,
            'stop_order': len(r['visits']) + 1,
            'customer_id': 0, # DC return
            'delivered_w': 0,
            'delivered_v': 0
        })
        
        routes_data.append({
            'route_id': r_idx,
            'vehicle_type': vt,
            'n_stops': len(r['visits']),
            'total_load_w': tot_w,
            'total_load_v': tot_v,
            'distance_km': details['dist'],
            'startup_cost': details['fixed_cost'],
            'energy_cost': details['energy_cost'],
            'penalty_cost': details['penalty'],
            'total_cost': r['cost'],
            'depart_dc': details['optimal_dep']
        })
        
    pd.DataFrame(stops_data).to_csv(q1_dir / 'stops.csv', index=False)
    pd.DataFrame(routes_data).to_csv(q1_dir / 'routes_summary.csv', index=False)

def plot_routes(routes, customers, figures_dir):
    plt.figure(figsize=(12, 12))
    
    # Plot customers
    x_green, y_green = [], []
    x_reg, y_reg = [], []
    for i in range(1, 99):
        c = customers[i]
        if np.sqrt(c.x**2 + c.y**2) <= R_GREEN:
            x_green.append(c.x)
            y_green.append(c.y)
        else:
            x_reg.append(c.x)
            y_reg.append(c.y)
            
    plt.scatter(x_reg, y_reg, c='lightgray', s=20, alpha=0.5, zorder=1)
    plt.scatter(x_green, y_green, c='lightgreen', s=20, alpha=0.5, zorder=1)
    plt.scatter(0, 0, c='red', marker='*', s=200, zorder=3, label='DC')
    
    circle = plt.Circle((0, 0), R_GREEN, color='green', fill=False, linestyle='--', alpha=0.5)
    plt.gca().add_patch(circle)
    
    colors = {'F-3000': 'blue', 'F-1500': 'purple', 'F-1250': 'orange', 'E-3000': 'green', 'E-1250': 'cyan'}
    
    for r in routes:
        vt = r['v_type']
        pts = [(0, 0)] + [(customers[v['customer_id']].x, customers[v['customer_id']].y) for v in r['visits']] + [(0, 0)]
        pts = np.array(pts)
        plt.plot(pts[:,0], pts[:,1], color=colors[vt], alpha=0.4, linewidth=1)
        
    # Custom legend for paths
    handles = [plt.Line2D([0], [0], color=c, lw=2, label=vt) for vt, c in colors.items()]
    plt.legend(handles=handles)
    
    plt.title('Figure F7: Q1 Static Scheduling Routes')
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(figures_dir, 'f07_q1_routes.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_cost_breakdown(summary, figures_dir):
    costs = {
        'Startup': summary['startup_cost'],
        'Energy': summary['energy_cost'],
        'Penalty': summary['penalty_cost']
    }
    
    plt.figure(figsize=(8, 6))
    plt.bar(costs.keys(), costs.values(), color=['gray', 'orange', 'red'])
    for i, v in enumerate(costs.values()):
        plt.text(i, v + 500, f'¥{v:.0f}', ha='center')
        
    plt.title(f'Figure F8: Q1 Cost Breakdown (Total: ¥{summary["total_cost"]:.0f})')
    plt.ylabel('Cost (¥)')
    plt.savefig(os.path.join(figures_dir, 'f08_q1_cost_breakdown.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fleet_usage(summary, figures_dir):
    vt_used = summary['by_vtype']
    vt_avail = {v: VEHICLE_SPECS[v]['count'] for v in VEHICLE_SPECS}
    
    vtypes = list(VEHICLE_SPECS.keys())
    used = [vt_used.get(v, 0) for v in vtypes]
    avail = [vt_avail[v] for v in vtypes]
    
    x = np.arange(len(vtypes))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, used, width, label='Trips Opened', color='blue')
    plt.bar(x + width/2, avail, width, label='Physical Vehicles Available', color='gray', alpha=0.5)
    
    plt.xticks(x, vtypes)
    plt.ylabel('Count')
    plt.title('Figure F10: Q1 Fleet Usage (Trips vs Physical Constraints)')
    plt.legend()
    
    plt.savefig(os.path.join(figures_dir, 'f10_q1_fleet_usage.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / 'data'
    results_dir = base_dir / 'results'
    q1_dir = results_dir / 'q1'
    figures_dir = base_dir / 'figures'
    
    q1_dir.mkdir(parents=True, exist_ok=True)
    
    coords, D, df_orders, tw, customers = ingest_data(data_dir)
    
    with open(results_dir / 'phaseB_solution.json', 'r') as f:
        routes = json.load(f)
        
    summary = generate_summary(routes)
    with open(q1_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    generate_csvs(routes, q1_dir)
    plot_routes(routes, customers, figures_dir)
    plot_cost_breakdown(summary, figures_dir)
    plot_fleet_usage(summary, figures_dir)
    
    print("Phase 6 (Q1 Reporting) completed successfully.")
