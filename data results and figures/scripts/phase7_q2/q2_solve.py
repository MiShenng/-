import sys
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from phase0_setup.constants import VEHICLE_SPECS, R_GREEN
from phase0_setup.datatypes import Task
from phase1_ingest.ingest import ingest_data
from phase3_core.evaluator import RouteEvaluator
from phase4_heuristic.heuristic import greedy_insertion, local_search
from phase5_postopt.lp_solve import consolidate_packets
from phase6_q1.q1_report import generate_summary, generate_csvs, plot_routes

def run_q2():
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / 'data'
    precomp_dir = base_dir / 'results' / 'precomputed'
    results_dir = base_dir / 'results'
    q2_dir = results_dir / 'q2'
    figures_dir = base_dir / 'figures'
    q2_dir.mkdir(parents=True, exist_ok=True)
    
    coords, D, df_orders, tw, customers = ingest_data(data_dir)
    T_lookup = np.load(precomp_dir / 'T_lookup.npy')
    G_ij = np.load(precomp_dir / 'G_ij.npy')
    detour = np.load(precomp_dir / 'detour.npy')
    
    with open(precomp_dir / 'packets.json', 'r') as f:
        packet_dicts = json.load(f)
    packets = [Task(**d) for d in packet_dicts]
    time_grid = np.arange(420, 420 + 180 * 5, 5)
    
    # Init evaluator with green_ban=True
    evaluator = RouteEvaluator(T_lookup, G_ij, detour, time_grid, packets, customers, green_ban=True)
    
    print("Starting Q2 heuristic...")
    sol, v_counts = greedy_insertion(evaluator, packets, green_ban=True)
    sol = local_search(evaluator, sol, max_iters=10, green_ban=True)
    
    with open(results_dir / 'phaseA_solution_q2.json', 'w') as f:
        json.dump(sol.routes, f, indent=2)
        
    final_routes = consolidate_packets(sol.routes, packets)
    with open(results_dir / 'phaseB_solution_q2.json', 'w') as f:
        json.dump(final_routes, f, indent=2)
        
    summary_q2 = generate_summary(final_routes)
    with open(q2_dir / 'summary.json', 'w') as f:
        json.dump(summary_q2, f, indent=2)
        
    generate_csvs(final_routes, q2_dir)
    
    # Plot routes for Q2 (F11)
    plot_routes(final_routes, customers, figures_dir)
    # The plot_routes function automatically names it f07_q1_routes.png. We rename it manually.
    if os.path.exists(os.path.join(figures_dir, 'f07_q1_routes.png')):
        os.rename(os.path.join(figures_dir, 'f07_q1_routes.png'), os.path.join(figures_dir, 'f11_q2_routes.png'))
        
    return summary_q2

def plot_q1_q2_comparison(sum_q1, sum_q2, figures_dir):
    categories = ['Startup', 'Energy', 'Penalty', 'Total']
    q1_vals = [sum_q1['startup_cost'], sum_q1['energy_cost'], sum_q1['penalty_cost'], sum_q1['total_cost']]
    q2_vals = [sum_q2['startup_cost'], sum_q2['energy_cost'], sum_q2['penalty_cost'], sum_q2['total_cost']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, q1_vals, width, label='Q1 (Unrestricted)', color='#4C72B0', edgecolor='#333333')
    plt.bar(x + width/2, q2_vals, width, label='Q2 (Green Ban)', color='#C44E52', edgecolor='#333333')
    
    plt.xticks(x, categories)
    plt.ylabel('Cost (¥)')
    plt.title('Figure F12: Q1 vs Q2 Cost Comparison')
    plt.legend()
    
    for i in range(len(categories)):
        delta = q2_vals[i] - q1_vals[i]
        plt.text(i + width/2, q2_vals[i] + 500, f'+{delta:.0f}', ha='center', color='red', fontsize=10)
        
    plt.savefig(os.path.join(figures_dir, 'f12_q1_vs_q2_cost.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # F13 Emissions Comparison
    # Q1 vs Q2 distance and emissions
    metrics = ['Total km']
    q1_m = [sum_q1['total_km']]
    q2_m = [sum_q2['total_km']]
    
    plt.figure(figsize=(6, 5))
    x = np.arange(len(metrics))
    plt.bar(x - width/2, q1_m, width, label='Q1', color='#4C72B0', edgecolor='#333333')
    plt.bar(x + width/2, q2_m, width, label='Q2', color='#C44E52', edgecolor='#333333')
    plt.xticks(x, metrics)
    plt.ylabel('Value')
    plt.title('Figure F13: Q1 vs Q2 Distance')
    plt.legend()
    plt.savefig(os.path.join(figures_dir, 'f13_q1_vs_q2_emissions.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    summary_q2 = run_q2()
    
    # Load Q1 summary
    base_dir = Path(__file__).resolve().parent.parent.parent
    with open(os.path.join(base_dir, 'results', 'q1', 'summary.json'), 'r') as f:
        summary_q1 = json.load(f)
        
    plot_q1_q2_comparison(summary_q1, summary_q2, base_dir / 'figures')
    
    # Output comparison JSON
    comp = {
        'total_cost': {'q1': summary_q1['total_cost'], 'q2': summary_q2['total_cost'], 'delta': summary_q2['total_cost'] - summary_q1['total_cost']},
        'dist': {'q1': summary_q1['total_km'], 'q2': summary_q2['total_km'], 'delta': summary_q2['total_km'] - summary_q1['total_km']},
        'penalty': {'q1': summary_q1['penalty_cost'], 'q2': summary_q2['penalty_cost'], 'delta': summary_q2['penalty_cost'] - summary_q1['penalty_cost']}
    }
    with open(os.path.join(base_dir, 'results', 'comparison_q1_q2.json'), 'w') as f:
        json.dump(comp, f, indent=2)
        
    print("Phase 7 (Q2 Execution & Reporting) completed successfully.")
