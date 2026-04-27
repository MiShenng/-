import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import phase0_setup
sys.path.append(str(Path(__file__).resolve().parent.parent))
from phase0_setup.constants import R_GREEN
from phase0_setup.datatypes import Customer

def parse_time(time_str):
    if pd.isna(time_str):
        return 0
    parts = time_str.split(':')
    return int(parts[0]) * 60 + int(parts[1])

def ingest_data(data_dir):
    # 1. Load coordinate CSV
    coord_file = data_dir / 'Customer contact information.csv'
    df_coords = pd.read_csv(coord_file)
    coords = {}
    for _, row in df_coords.iterrows():
        coords[int(row['ID'])] = (float(row['X (km)']), float(row['Y (km)']))
        
    # 2. Load distance matrix CSV
    dist_file = data_dir / 'Distance matrix.csv'
    df_dist = pd.read_csv(dist_file, index_col=0)
    D = df_dist.values
    
    # 3. Load orders CSV
    order_file = data_dir / 'Order Information.csv'
    df_orders = pd.read_csv(order_file)
    # The weight column header has a newline in it: 'Weight\n'
    weight_col = [c for c in df_orders.columns if 'Weight' in c][0]
    df_orders[weight_col] = pd.to_numeric(df_orders[weight_col], errors='coerce').fillna(0.0)
    df_orders['Volume'] = pd.to_numeric(df_orders['Volume'], errors='coerce').fillna(0.0)
    
    orders_agg = df_orders.groupby('Target Customer ID').agg({
        weight_col: 'sum',
        'Volume': 'sum'
    }).reset_index()
    orders_agg.columns = ['customer_id', 'total_weight', 'total_volume']
    
    # 4. Load time windows CSV
    tw_file = data_dir / 'Time Window.csv'
    df_tw = pd.read_csv(tw_file)
    df_tw['tw_start'] = df_tw['Start time'].apply(parse_time)
    df_tw['tw_end'] = df_tw['End Time'].apply(parse_time)
    
    tw = {}
    for _, row in df_tw.iterrows():
        tw[int(row['Customer ID'])] = (row['tw_start'], row['tw_end'])
        
    # 5. Merge into unified dictionary
    customers = {}
    # DC is ID 0
    customers[0] = Customer(
        id=0, x=coords[0][0], y=coords[0][1],
        demand_w=0.0, demand_v=0.0,
        tw_start=0.0, tw_end=24*60.0
    )
    
    for i in range(1, 99):
        x, y = coords[i]
        # Get demand
        demand_w = 0.0
        demand_v = 0.0
        order_match = orders_agg[orders_agg['customer_id'] == i]
        if not order_match.empty:
            demand_w = order_match['total_weight'].values[0]
            demand_v = order_match['total_volume'].values[0]
            
        # Get TW (if missing, use [0, 24*60])
        tw_start, tw_end = tw.get(i, (0.0, 24*60.0))
        
        customers[i] = Customer(
            id=i, x=x, y=y,
            demand_w=demand_w, demand_v=demand_v,
            tw_start=tw_start, tw_end=tw_end
        )
        
    return coords, D, df_orders, tw, customers

def plot_layout(customers, figures_dir):
    plt.figure(figsize=(10, 10))
    
    x_green = []
    y_green = []
    x_reg = []
    y_reg = []
    
    for i in range(1, 99):
        c = customers[i]
        # In phase 1, green zone logic isn't strictly executed, but let's color it here for the plot.
        dist_to_center = np.sqrt(c.x**2 + c.y**2)
        if dist_to_center <= R_GREEN:
            x_green.append(c.x)
            y_green.append(c.y)
        else:
            x_reg.append(c.x)
            y_reg.append(c.y)
            
    plt.scatter(x_reg, y_reg, c='blue', label='Regular Customer', alpha=0.6)
    plt.scatter(x_green, y_green, c='green', label='Green Zone Customer', alpha=0.8)
    
    # DC
    dc = customers[0]
    plt.scatter(dc.x, dc.y, c='red', marker='*', s=200, label='DC')
    
    # Green zone circle
    circle = plt.Circle((0, 0), R_GREEN, color='green', fill=False, linestyle='--', label='Green Zone Boundary')
    plt.gca().add_patch(circle)
    
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)
    plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Figure F1: Customer Layout')
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    
    plt.savefig(os.path.join(figures_dir, 'f01_customer_layout.png'), dpi=300, bbox_inches='tight')
    plt.close()

def audit_data(D, customers, results_dir):
    audit_path = results_dir / 'data_audit.txt'
    with open(audit_path, 'w') as f:
        f.write("--- Data Quality Audit ---\n")
        f.write(f"Total customers (excluding DC): {len(customers)-1}\n")
        
        # Customers with zero demand
        zero_demand = [c.id for c in customers.values() if c.id != 0 and c.demand_w == 0 and c.demand_v == 0]
        f.write(f"Customers with zero demand: {len(zero_demand)} (IDs: {zero_demand})\n")
        
        # Distance matrix checks
        n = D.shape[0]
        is_sym = np.allclose(D, D.T, atol=1e-5)
        f.write(f"Distance matrix symmetric: {is_sym}\n")
        
        green_count = sum(1 for c in customers.values() if c.id != 0 and np.sqrt(c.x**2 + c.y**2) <= R_GREEN)
        f.write(f"Green zone customers: {green_count}\n")
        
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / 'data'
    results_dir = base_dir / 'results'
    figures_dir = base_dir / 'figures'
    
    coords, D, df_orders, tw, customers = ingest_data(data_dir)
    
    plot_layout(customers, figures_dir)
    audit_data(D, customers, results_dir)
    print("Phase 1 executed successfully.")
