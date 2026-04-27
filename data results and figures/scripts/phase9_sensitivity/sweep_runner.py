import sys
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from phase0_setup.constants import PRICE_FUEL, CO2_COST
from phase0_setup.datatypes import Task
from phase1_ingest.ingest import ingest_data
from phase3_core.evaluator import RouteEvaluator
from phase4_heuristic.heuristic import greedy_insertion, local_search
from phase6_q1.q1_report import generate_summary

def get_base_environment():
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / 'data'
    precomp_dir = base_dir / 'results' / 'precomputed'
    
    coords, D, df_orders, tw, customers = ingest_data(data_dir)
    T_lookup = np.load(precomp_dir / 'T_lookup.npy')
    G_ij = np.load(precomp_dir / 'G_ij.npy')
    detour = np.load(precomp_dir / 'detour.npy')
    
    with open(precomp_dir / 'packets.json', 'r') as f:
        packet_dicts = json.load(f)
    packets = [Task(**d) for d in packet_dicts]
    time_grid = np.arange(420, 420 + 180 * 5, 5)
    
    return T_lookup, G_ij, detour, time_grid, packets, customers

def run_evaluation(evaluator, packets, v_counts=None):
    sol, _ = greedy_insertion(evaluator, packets, green_ban=True, v_counts_override=v_counts)
    sol = local_search(evaluator, sol, max_iters=5, green_ban=True) # Reduced iters for speed
    
    # Calculate CO2 cost directly from summary (since we need exact carbon)
    # The summary groups energy_cost. We need carbon alone for F16/F18.
    # We can approximate CO2 cost by knowing PRICE_FUEL vs CO2_COST ratio for F-3000,
    # or just tracking it. But we just need Total Cost and CO2 amount.
    summary = generate_summary(sol.routes)
    return summary

def run_sweeps():
    base_dir = Path(__file__).resolve().parent.parent.parent
    sens_dir = os.path.join(base_dir, 'results', 'sensitivity')
    sens_dir.mkdir(parents=True, exist_ok=True)
    
    T_lookup, G_ij, detour, time_grid, packets, customers = get_base_environment()
    
    # Base configuration
    base_v_counts = {
        'E-3000': 10,
        'F-3000': 120,
        'F-1500': 50,
        'E-1250': 15,
        'F-1250': 50
    }
    
    print("--- 1. Sweeping E-Fleet Size ---")
    efleet_vals = [10, 15, 20, 25, 30]
    efleet_results = []
    for val in efleet_vals:
        print(f"Running E-Fleet = {val}...")
        evaluator = RouteEvaluator(T_lookup, G_ij, detour, time_grid, packets, customers, green_ban=True)
        v_c = base_v_counts.copy()
        v_c['E-3000'] = val
        summary = run_evaluation(evaluator, packets, v_counts=v_c)
        efleet_results.append({'E-3000': val, 'total_cost': summary['total_cost']})
        
    pd.DataFrame(efleet_results).to_csv(sens_dir / 'e_fleet.csv', index=False)
    
    print("--- 2. Sweeping Fuel Price ---")
    fuel_multipliers = [0.9, 0.95, 1.0, 1.05, 1.1]
    fuel_results = []
    for mult in fuel_multipliers:
        new_price = PRICE_FUEL * mult
        print(f"Running Fuel Price = {new_price:.2f} ({mult*100:.0f}%)...")
        evaluator = RouteEvaluator(T_lookup, G_ij, detour, time_grid, packets, customers, green_ban=True)
        evaluator.PRICE_FUEL = new_price
        summary = run_evaluation(evaluator, packets, v_counts=base_v_counts)
        fuel_results.append({'mult': mult, 'price': new_price, 'total_cost': summary['total_cost']})
        
    pd.DataFrame(fuel_results).to_csv(sens_dir / 'fuel_price.csv', index=False)
    
    print("--- 3. Sweeping Carbon Price ---")
    carbon_multipliers = [0.8, 0.9, 1.0, 1.1, 1.2]
    carbon_results = []
    for mult in carbon_multipliers:
        new_price = CO2_COST * mult
        print(f"Running Carbon Price = {new_price:.3f} ({mult*100:.0f}%)...")
        evaluator = RouteEvaluator(T_lookup, G_ij, detour, time_grid, packets, customers, green_ban=True)
        evaluator.CO2_COST = new_price
        summary = run_evaluation(evaluator, packets, v_counts=base_v_counts)
        carbon_results.append({'mult': mult, 'price': new_price, 'total_cost': summary['total_cost']})
        
    pd.DataFrame(carbon_results).to_csv(sens_dir / 'carbon_price.csv', index=False)
    
    print("Sensitivity analysis complete.")

if __name__ == "__main__":
    run_sweeps()
