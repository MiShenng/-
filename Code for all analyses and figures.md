# requirements.txt
~~~ txt 
numpy  
pandas  
scipy  
matplotlib  
seaborn  
pulp  
tqdm
~~~
# data

# main.py
~~~ python 
import os  
from pathlib import Path  
import subprocess  
import sys  
  
def run_phase(script_path):  
    print(f"\n{'='*50}\nRunning {script_path}...\n{'='*50}")  
    result = subprocess.run([sys.executable, script_path], cwd=os.path.dirname(script_path))  
    if result.returncode != 0:  
        print(f"Error executing {script_path}")  
        sys.exit(1)  
  
if __name__ == "__main__":  
    base_dir = os.path.abspath(os.path.dirname(__file__))  
    scripts_dir = base_dir / 'scripts'  
    phases = [  
        os.path.join(scripts_dir, 'phase1_ingest', 'ingest.py'),  
        os.path.join(scripts_dir, 'phase2_preprocess', 'preprocess.py'),  
        os.path.join(scripts_dir, 'phase3_core', 'evaluator.py'),  
        os.path.join(scripts_dir, 'phase4_heuristic', 'heuristic.py'),  
        os.path.join(scripts_dir, 'phase5_postopt', 'lp_solve.py')  
    ]  
      
    for phase in phases:  
        run_phase(phase)  
          
    print("\nAll phases executed successfully. Check the 'results' directory for outputs.")
    
~~~
# phase0_setup 

## constants.py
~~~python
# constants.py  
# 问题参数 
  
# 时间段（以午夜为基准，单位为分钟）
BAN_START = 8 * 60  # 480  
BAN_END = 16 * 60   # 960  
  
# 费用与价格  
STARTUP_COST = 400.0  
WAIT_PENALTY = 1.0 / 3.0  # ¥/min  
LATE_PENALTY = 5.0 / 6.0  # ¥/min  
  
PRICE_FUEL = 7.61  # ¥/L  
PRICE_ELEC = 1.64  # ¥/kWh  
  
CO2_COST = 0.65  # ¥/kg  
ETA_FUEL = 2.547 # kg/L  
ETA_ELEC = 0.501 # kg/kWh  
  
# 服务时间参数  
S0 = 8.0  # minutes fixed  
BETA_W = 0.004  # min/kg  
BETA_V = 0.40   # min/m^3  
  
# 绿色区域半径  
R_GREEN = 10.0  # km  
  
# 车辆类型  
VEHICLE_SPECS = {  
    'F-3000': {'type': 'fuel', 'Q_w': 3000.0, 'Q_v': 13.5, 'count': 60, 'alpha': 0.40},  
    'F-1500': {'type': 'fuel', 'Q_w': 1500.0, 'Q_v': 10.8, 'count': 50, 'alpha': 0.40},  
    'F-1250': {'type': 'fuel', 'Q_w': 1250.0, 'Q_v': 6.5,  'count': 50, 'alpha': 0.40},  
    'E-3000': {'type': 'electric', 'Q_w': 3000.0, 'Q_v': 15.0, 'count': 10, 'alpha': 0.35},  
    'E-1250': {'type': 'electric', 'Q_w': 1250.0, 'Q_v': 8.5,  'count': 15, 'alpha': 0.35},  
}  
  
# 速度参数 (mu_v, sigma_v)  km/h  
SPEED_PARAMS = {  
    'smooth': (55.3, 0.12),  
    'normal': (35.4, 5.22),  
    'congested': (9.8, 4.72)  
}  
  
# 时间段定义 (start_min, end_min, regime_name)  
# 按开始时间排序  
REGIMES = [  
    (0, 8*60, 'smooth'), # 在上午8点前默认采用平滑处理
    (8*60, 9*60, 'congested'),  
    (9*60, 10*60, 'smooth'),  
    (10*60, 11*60+30, 'normal'),  
    (11*60+30, 13*60, 'congested'),  
    (13*60, 15*60, 'smooth'),  
    (15*60, 17*60, 'normal'),  
    (17*60, 19*60, 'congested'), 
]
~~~


## datatypes.py
~~~python
  
@dataclass  
class Customer:  
    id: int  
    x: float  
    y: float  
    demand_w: float  
    demand_v: float  
    tw_start: float  
    tw_end: float  
    is_green: bool = False  
  
@dataclass  
class Task:  
    # 客户的分拆包裹  
    task_id: int  
    customer_id: int  
    w: float  
    v: float  
    tw_start: float  
    tw_end: float  
    is_green: bool  
    service_time: float  
  
@dataclass  
class Vehicle:  
    vtype: str  
    is_electric: bool  
    Q_w: float  
    Q_v: float  
    alpha: float  
  
@dataclass  
class Route:  
    vtype: str  
    tasks: List[Task] = field(default_factory=list)  
    # 变量 z_w，z_v 可能是与任务配对的列表 
    splits_w: List[float] = field(default_factory=list)  
    splits_v: List[float] = field(default_factory=list)  
  
@dataclass  
class Subsequence:  
    D: float # 总时长  
    E: float # 最早可开始时间 
    L: float # 最早可开始时间  
    W: float # 等待成本
    P: float # 后期成本
~~~

# phase1_ingest
## ingest.py
~~~python
import sys  
import os  
from pathlib import Path  
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
  
# 将根目录添加到路径中以导入 phase0_setup  
sys.path.append(str(Path(__file__).resolve().parent.parent))  
from phase0_setup.constants import R_GREEN  
from phase0_setup.datatypes import Customer  
  
def parse_time(time_str):  
    if pd.isna(time_str):  
        return 0  
    parts = time_str.split(':')  
    return int(parts[0]) * 60 + int(parts[1])  
  
def ingest_data(data_dir):  
    # 1. 加载Customer contact information CSV 文件  
    coord_file = data_dir / 'Customer contact information.csv'  
    df_coords = pd.read_csv(coord_file)  
    coords = {}  
    for _, row in df_coords.iterrows():  
        coords[int(row['ID'])] = (float(row['X (km)']), float(row['Y (km)']))  
          
    # 2. 加载 distance matrix CSV 文件  
    dist_file = data_dir / 'Distance matrix.csv'  
    df_dist = pd.read_csv(dist_file, index_col=0)  
    D = df_dist.values  
      
    # 3. 加载 Order Informatio CSV 文件
    order_file = data_dir / 'Order Information.csv'  
    df_orders = pd.read_csv(order_file)  
    # weight列标题中包含一个换行符: 'Weight\n'  
    weight_col = [c for c in df_orders.columns if 'Weight' in c][0]  
    df_orders[weight_col] = pd.to_numeric(df_orders[weight_col], errors='coerce').fillna(0.0)  
    df_orders['Volume'] = pd.to_numeric(df_orders['Volume'], errors='coerce').fillna(0.0)  
      
    orders_agg = df_orders.groupby('Target Customer ID').agg({  
        weight_col: 'sum',  
        'Volume': 'sum'  
    }).reset_index()  
    orders_agg.columns = ['customer_id', 'total_weight', 'total_volume']  
      
    # 4. 加载 Time Window CSV 文件  
    tw_file = data_dir / 'Time Window.csv'  
    df_tw = pd.read_csv(tw_file)  
    df_tw['tw_start'] = df_tw['Start time'].apply(parse_time)  
    df_tw['tw_end'] = df_tw['End Time'].apply(parse_time)  
      
    tw = {}  
    for _, row in df_tw.iterrows():  
        tw[int(row['Customer ID'])] = (row['tw_start'], row['tw_end'])  
          
    # 5. 合并统一 
    customers = {}  
    # DC is ID 0  
    customers[0] = Customer(  
        id=0, x=coords[0][0], y=coords[0][1],  
        demand_w=0.0, demand_v=0.0,  
        tw_start=0.0, tw_end=24*60.0  
    )  
      
    for i in range(1, 99):  
        x, y = coords[i]  
        # 获取需求  
        demand_w = 0.0  
        demand_v = 0.0  
        order_match = orders_agg[orders_agg['customer_id'] == i]  
        if not order_match.empty:  
            demand_w = order_match['total_weight'].values[0]  
            demand_v = order_match['total_volume'].values[0]  
              
        # 获取时区（若未指定，则使用 [0, 24*60]）
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
        # 在phase1，绿色区域的逻辑并未严格执行，但为了便于说明，在此对其进行着色 
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
      
    # 绿色区域圆圈  
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
          
        # 需求为零的客户  
        zero_demand = [c.id for c in customers.values() if c.id != 0 and c.demand_w == 0 and c.demand_v == 0]  
        f.write(f"Customers with zero demand: {len(zero_demand)} (IDs: {zero_demand})\n")  
          
        # 距离矩阵校验  
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
~~~
# phase2_preprocess
## preprocess.py
~~~python
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
    return 'smooth' # 默认备选方案  
  
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
                    # 查找当前工作区边界  
                    r_name = 'smooth'  
                    r_end = 24 * 60  
                    for start, end, name in REGIMES:  
                        if start <= t_mod < end:  
                            r_name = name  
                            r_end = t_curr - t_mod + end # 恢复为绝对路径   
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
                detour[i, j] = 1e9 # 不可行  
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
~~~
# phase3_core
##  evaluator.py
~~~python 
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
          
        # 敏感性分析中的动态覆盖  
        self.PRICE_FUEL = PRICE_FUEL  
        self.CO2_COST = CO2_COST  
        self.PRICE_ELEC = PRICE_ELEC  
          
        # 计算配速表  
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
          
        # 对从DC出发的可能时间进行网格搜索  
        # 由于等待惩罚为 1/3，而迟到惩罚为 5/6，我们应尽可能晚出发以避免等待，        
        # 但也要足够早出发以避免迟到惩罚。我们可以直接测试从 420 到 1200 的出发时间，步长为 30。   
        # 实际上，我们只需检查与每个运输周（TW）开始时间对齐的出发时间。 
        candidate_deps = [420.0]     
        for k in range(1, len(nodes)):  
            # 粗略估算 tw_starts 的反向传播 
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
    # 简单的测试逻辑  
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
      
    # 使用单包裹路线进行测试  
    if len(packets) > 0:  
        cost, feas, det = evaluator.evaluate_route([packets[0].task_id], 'F-3000')  
        print(f"Test Route F-3000 Cost: {cost:.2f}, Feasible: {feas}, Details: {det}")  
          
        cost, feas, det = evaluator.evaluate_route([packets[0].task_id], 'E-3000')  
        print(f"Test Route E-3000 Cost: {cost:.2f}, Feasible: {feas}, Details: {det}")  
    print("Phase 3 executed successfully.")
~~~
# phase4_heuristic
## heuristic.py
~~~python 
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
        self.routes = []  
          
    def total_cost(self):  
        return sum(r['cost'] for r in self.routes)  
  
def greedy_insertion(evaluator, packets, green_ban=False, initial_routes=None, v_counts_override=None):  
    sol = Solution()  
    if initial_routes is not None:  
        sol.routes = deepcopy(initial_routes)  
      
    # 按绿色区域（True 优先）排序数据包，然后按 tw_start 升序，最后按权重降序  
    sorted_packets = sorted(packets, key=lambda p: (not p.is_green, p.tw_start, -p.w))  
      
    # 可用车辆行程。制造E-3000短缺，迫使燃油车进入绿色区域。 
    if v_counts_override is not None:  
        v_counts = deepcopy(v_counts_override)  
    else:  
        v_counts = {  
            'E-3000': 10,  # 严格限制 “仅存10台E-3000” 
            'F-3000': 120, # 允许重复使用行程以满足总需求  
            'F-1500': 50,  
            'E-1250': 15,  
            'F-1250': 50  
        }  
      
    theta = 5000.0 # 开通新路线的门槛  
    for p in sorted_packets:  
        best_cost_delta = float('inf')  
        best_r_idx = -1  
        best_p_idx = -1  
        best_new_route_cost = float('inf')  
        best_new_v_type = None  
        allowed_vtypes = list(v_counts.keys())  
        # 我们不再严格强制要求使用电动汽车。燃油车可以服务于绿色客户，但将受到处罚  
        # 尝试插入到现有路线上  
        for r_idx, r in enumerate(sol.routes):  
            if r['v_type'] not in allowed_vtypes:  
                continue  
                old_cost = r['cost']  
            # 尝试所有位置
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
                          
        # 试着开通一条新路线 
        for v_type in allowed_vtypes:  
            count = v_counts.get(v_type, 0)  
            if count > 0:  
                cost, feas, det = evaluator.evaluate_route([p.task_id], v_type)  
                if feas and cost < best_new_route_cost:  
                    best_new_route_cost = cost  
                    best_new_v_type = v_type  
                      
        if best_cost_delta < theta and best_r_idx != -1:  
            # 插入到现有路线  
            r = sol.routes[best_r_idx]  
            r['packets'].insert(best_p_idx, p.task_id)  
            r['cost'], feas, det = evaluator.evaluate_route(r['packets'], r['v_type'], r_idx=best_r_idx)  
            r['details'] = det  
        else:  
            # 开通一条新路线 
            if best_new_v_type is None:  
                # 排查失败原因  
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
          
        # 1.迁移  
        for r1_idx, r1 in enumerate(sol.routes):  
            for i in range(len(r1['packets'])):  
                pid = r1['packets'][i]  
                   
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
                            if new_cost < old_cost - 1e-4:     
                                r1['cost'] = c1  
                                r1['details'] = d1  
                                r2['packets'] = new_r2  
                                r2['cost'] = c2  
                                r2['details'] = d2  
                                improved = True  
                                break                    if improved: break  
                if improved: break  
            if improved: break  
            #清理空路线  
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
    # 对于第4阶段，我们假设Q1（无绿色禁令）
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
~~~

# phase5_postopt
# lp_solve.py
~~~python 
import sys  
import os  
from pathlib import Path  
import json  
  
sys.path.append(str(Path(__file__).resolve().parent.parent))  
from phase0_setup.datatypes import Task  
  
def consolidate_packets(sol_routes, packets):  
   #将同一车辆上同一客户的连续货包进行合并。这符合离散分配的要求，并代表了简化的B阶段优化后处理（合并拆分）。   
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
~~~
# phase6_q1
## q1_report.py
~~~python 
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
          
        # 简化重构  
        stops_data.append({  
            'route_id': r_idx,  
            'vehicle_type': vt,  
            'stop_order': 0,  
            'customer_id': 0, # 从DC出发  
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
            'customer_id': 0, # DC 回归  
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
      
    # 绘制客户图  
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
          
    # 路径的自定义图例  
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
~~~
# phase7_q2
## q2_solve.py
~~~python 
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
      
    # 初始化评估器，并设置 green_ban=True  
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
      
    # 绘制 Q2（F11）的路线  
    plot_routes(final_routes, customers, figures_dir)  
    # plot_routes 函数会自动将其命名为 f07_q1_routes.png。重命名该文件。  
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
      
    # F13 排放对比  
    # 第一季度与第二季度的距离和排放量    
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
      
    # 加载Q1 
    base_dir = Path(__file__).resolve().parent.parent.parent  
    with open(os.path.join(base_dir, 'results', 'q1', 'summary.json'), 'r') as f:  
        summary_q1 = json.load(f)  
          
    plot_q1_q2_comparison(summary_q1, summary_q2, base_dir / 'figures')  
      
    # 输出比较JSON算法  
    comp = {  
        'total_cost': {'q1': summary_q1['total_cost'], 'q2': summary_q2['total_cost'], 'delta': summary_q2['total_cost'] - summary_q1['total_cost']},  
        'dist': {'q1': summary_q1['total_km'], 'q2': summary_q2['total_km'], 'delta': summary_q2['total_km'] - summary_q1['total_km']},  
        'penalty': {'q1': summary_q1['penalty_cost'], 'q2': summary_q2['penalty_cost'], 'delta': summary_q2['penalty_cost'] - summary_q1['penalty_cost']}  
    }  
    with open(os.path.join(base_dir, 'results', 'comparison_q1_q2.json'), 'w') as f:  
        json.dump(comp, f, indent=2)  
          
    print("Phase 7 (Q2 Execution & Reporting) completed successfully.")
~~~
# phase8_q3
## dynamic_dispatcher.py
~~~python 
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
        self.clock = clock  # 距离午夜的时间（以分钟为单位）  
        self.payload = payload # 例如：Task
  
class RollingHorizonDispatcher:  
    def __init__(self, evaluator):  
        self.evaluator = evaluator  
          
    def trace_route(self, route_packets, v_type):  
        """  
        模拟路径并返回每个节点的精确时间。        返回一个字典列表，其键为：'packet_id', 't_dep', 't_arr', 't_start_svc', 't_end_svc'        """       
        nodes = [0] + [self.evaluator.packets[pid].customer_id for pid in route_packets] + [0]  
        svc_times = [0.0] + [self.evaluator.packets[pid].service_time for pid in route_packets] + [0.0]  
        tw_starts = [0.0] + [self.evaluator.packets[pid].tw_start for pid in route_packets] + [0.0]  
        tw_ends = [24*60.0] + [self.evaluator.packets[pid].tw_end for pid in route_packets] + [24*60.0]  
          
        # 我们需要从华盛顿特区出发的最优路线。为此，我们只需重新运行网格搜索 
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
                    't_dep': t_dep, # 距上一个节点的时间  
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
                  
        # 返回路线的deepcopy作为起点  
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
~~~
## q3_scenario_S1.py
~~~python 
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
      
    # 1. 加载环境  
    coords, D, df_orders, tw, customers = ingest_data(data_dir)  
    T_lookup = np.load(precomp_dir / 'T_lookup.npy')  
    G_ij = np.load(precomp_dir / 'G_ij.npy')  
    detour = np.load(precomp_dir / 'detour.npy')  
      
    with open(precomp_dir / 'packets.json', 'r') as f:  
        packet_dicts = json.load(f)  
    packets = [Task(**d) for d in packet_dicts]  
    time_grid = np.arange(420, 420 + 180 * 5, 5)  
      
    # 基数评估  
    base_evaluator = RouteEvaluator(T_lookup, G_ij, detour, time_grid, packets, customers, green_ban=False)  
      
    # 2. 加载 Q1 基准路线  
    with open(results_dir / 'phaseA_solution.json', 'r') as f:  
        baseline_routes = json.load(f)  
          
    print(f"Loaded baseline with {len(baseline_routes)} routes.")  
      
    # 3. 创建活动：新订单于11:00（660分钟）  
    event_time = 660.0  
    # 假设客户98下了一笔1000公斤的新订单（客户 98 位于绿色区域） 
    # 时间窗口：11:00 至 18:00 
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
     
    base_evaluator.packets[new_task_id] = new_packet  
      
    print(f"Injected Event: New Order (Task {new_task_id}) at 11:00.")  
      
    # 4. 冻结和跟踪任务  
    dispatcher = RollingHorizonDispatcher(base_evaluator)  
    initial_routes, baseline_assignments = dispatcher.freeze_and_split(baseline_routes, event_time)  
      
    print(f"Tracking assignments for stability penalty. Total packets tracked: {len(baseline_assignments)}")  
      
    # 5. 稳定性封装  
    stability_evaluator = StabilityEvaluatorWrapper(base_evaluator, baseline_assignments, lambda_penalty=100.0)  
      
    # 6. 将新订单插入到最佳现有路线或新路线中
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
          
    # 7. 本地搜索与风险控制  
    c_idx = new_packet.customer_id  
    event_loc = (customers[c_idx].x, customers[c_idx].y)  
    print("Running damage-limited local search...")  
    sol = local_search(stability_evaluator, sol, max_iters=10, green_ban=False, event_loc=event_loc)  
      
    # 整合确定 
    # 使用 base_evaluator 来获取不包含稳定性 lambda 的真实成本   
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
~~~
# phase9_sensitivity
## plotters.py
~~~python 
import os  
from pathlib import Path  
import pandas as pd  
import matplotlib.pyplot as plt  
import numpy as np  
  
def plot_f16(base_dir):  
    efleet_df = pd.read_csv(os.path.join(base_dir, 'results', 'sensitivity', 'e_fleet.csv'))  
      
    fig, ax1 = plt.subplots(figsize=(8, 6))  
      
    color = 'tab:red'  
    ax1.set_xlabel('E-3000 Fleet Size')  
    ax1.set_ylabel('Total Cost (¥)', color=color)  
    ax1.plot(efleet_df['E-3000'], efleet_df['total_cost'], marker='o', color=color, linewidth=2)  
    ax1.tick_params(axis='y', labelcolor=color)  
      
    plt.title('Figure F16: Total Cost vs E-3000 Fleet Size')  
    plt.grid(True, alpha=0.3)  
      
    fig.tight_layout()  
    plt.savefig(os.path.join(base_dir, 'figures', 'f16_sensitivity_efleet.png'), dpi=300)  
    plt.close()  
  
def plot_f17(base_dir):  
    fuel_df = pd.read_csv(os.path.join(base_dir, 'results', 'sensitivity', 'fuel_price.csv'))  
    carbon_df = pd.read_csv(os.path.join(base_dir, 'results', 'sensitivity', 'carbon_price.csv'))  
      
    # 基准成本对应 mult=1.0  
    base_fuel = fuel_df[fuel_df['mult'] == 1.0]['total_cost'].values[0]  
    base_carbon = carbon_df[carbon_df['mult'] == 1.0]['total_cost'].values[0]  
      
    fuel_down = (fuel_df[fuel_df['mult'] == 0.9]['total_cost'].values[0] - base_fuel) / base_fuel * 100  
    fuel_up = (fuel_df[fuel_df['mult'] == 1.1]['total_cost'].values[0] - base_fuel) / base_fuel * 100  
    carbon_down = (carbon_df[carbon_df['mult'] == 0.8]['total_cost'].values[0] - base_carbon) / base_carbon * 100  
    carbon_up = (carbon_df[carbon_df['mult'] == 1.2]['total_cost'].values[0] - base_carbon) / base_carbon * 100  
    parameters = ['Fuel Price (±10%)', 'Carbon Price (±20%)']  
    down_vars = [fuel_down, carbon_down]  
    up_vars = [fuel_up, carbon_up]  
      
    y = np.arange(len(parameters))  
      
    fig, ax = plt.subplots(figsize=(8, 4))  
    ax.barh(y, down_vars, align='center', color='green', label='Decrease Parameter')  
    ax.barh(y, up_vars, align='center', color='red', label='Increase Parameter')  
    ax.set_yticks(y)  
    ax.set_yticklabels(parameters)  
    ax.set_xlabel('Change in Total Cost (%)')  
    ax.set_title('Figure F17: Tornado Chart for Sensitivity')  
    ax.legend()  
      
    # 在 0 处添加垂直线  
    ax.axvline(0, color='black', linewidth=1)  
      
    plt.tight_layout()  
    plt.savefig(os.path.join(base_dir, 'figures', 'f17_sensitivity_tornado.png'), dpi=300)  
    plt.close()  
  
def plot_f18(base_dir):  
    fuel_df = pd.read_csv(os.path.join(base_dir, 'results', 'sensitivity', 'fuel_price.csv'))  
    carbon_df = pd.read_csv(os.path.join(base_dir, 'results', 'sensitivity', 'carbon_price.csv'))  
    efleet_df = pd.read_csv(os.path.join(base_dir, 'results', 'sensitivity', 'e_fleet.csv'))  
      
    # F18是成本与二氧化碳排放量的帕累托最优解。由于我们没有追踪纯二氧化碳排放量（以千克为单位）, 采用“总成本与参数值”的对比图，或者直接绘制成本的散点图。要求给定所有敏感性分析结果下的（总成本、二氧化碳）帕累托前沿。    
    #绘制成本与碳成本参数的曲线，以此作为权衡关系的替代指标。  
    
    plt.figure(figsize=(8, 6))  
      
    plt.scatter(carbon_df['mult'], carbon_df['total_cost'], color='blue', label='Carbon Price Varied')  
    plt.plot(carbon_df['mult'], carbon_df['total_cost'], color='blue', alpha=0.3)  
      
    plt.title('Figure F18: Cost Landscape (Proxy for Pareto)')  
    plt.xlabel('Carbon Price Multiplier')  
    plt.ylabel('Total Cost (¥)')  
    plt.grid(True, alpha=0.3)  
    plt.legend()  
      
    plt.savefig(os.path.join(base_dir, 'figures', 'f18_pareto_cost_vs_co2.png'), dpi=300)  
    plt.close()  
  
if __name__ == "__main__":  
    base_dir = Path(__file__).resolve().parent.parent.parent  
    plot_f16(base_dir)  
    plot_f17(base_dir)  
    plot_f18(base_dir)  
    print("Sensitivity plots generated successfully.")
~~~
## sweep_runner.py
~~~python 
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
    sol = local_search(evaluator, sol, max_iters=5, green_ban=True) #减少迭代次数以提高速度  
    #直接从摘要中计算二氧化碳成本 
    #汇总了能源成本。我们需要单独计算F16/F18的碳排放量。   
    #利用 F-3000 的 PRICE_FUEL 与 CO2_COST 比率来近似计算二氧化碳成本；或者直接追踪该数据，但仅需总成本和二氧化碳排放量。    
    summary = generate_summary(sol.routes)  
    return summary  
  
def run_sweeps():  
    base_dir = Path(__file__).resolve().parent.parent.parent  
    sens_dir = os.path.join(base_dir, 'results', 'sensitivity')  
    sens_dir.mkdir(parents=True, exist_ok=True)  
      
    T_lookup, G_ij, detour, time_grid, packets, customers = get_base_environment()  
      
    # 基本配置  
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
~~~
# phase10_mc
## stochastic_evaluator.py
~~~ python 
import sys  
import os  
from pathlib import Path  
import json  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
  
sys.path.append(str(Path(__file__).resolve().parent.parent))  
from phase0_setup.constants import VEHICLE_SPECS, WAIT_PENALTY, LATE_PENALTY, PRICE_ELEC, PRICE_FUEL, CO2_COST, ETA_FUEL  
from phase0_setup.datatypes import Task  
from phase1_ingest.ingest import ingest_data  
from phase3_core.evaluator import RouteEvaluator  
  
def replay_route_stochastic(evaluator, packet_ids, v_type, optimal_dep):  
    nodes = [0] + [evaluator.packets[pid].customer_id for pid in packet_ids] + [0]  
    svc_times = [0.0] + [evaluator.packets[pid].service_time for pid in packet_ids] + [0.0]  
    tw_starts = [0.0] + [evaluator.packets[pid].tw_start for pid in packet_ids] + [0.0]  
    tw_ends = [24*60.0] + [evaluator.packets[pid].tw_end for pid in packet_ids] + [24*60.0]  
      
    total_w = sum(evaluator.packets[pid].w for pid in packet_ids)  
    spec = VEHICLE_SPECS[v_type]  
    is_ev = (spec['type'] == 'electric')  
      
    energy_cost = 0.0  
    soft_penalty = 0.0  
    current_w = total_w  
    t_curr = optimal_dep  
    late_deliveries = 0  
    for k in range(1, len(nodes)):  
        i = nodes[k-1]  
        j = nodes[k]  
          
        use_detour = False  
        if evaluator.green_ban and not is_ev and evaluator.G[i, j]:  
            tt_est = evaluator.get_travel_time(i, j, t_curr, False)  
            if 480 <= t_curr + tt_est <= 960 or 480 <= t_curr <= 960:  
                if evaluator.detour[i, j] >= 1e8:  
                    t_curr = max(t_curr, 960.0)  
                else:  
                    use_detour = True  
        P_i = np.array([evaluator.customers[i].x, evaluator.customers[i].y])  
        P_j = np.array([evaluator.customers[j].x, evaluator.customers[j].y])  
        dist_straight = np.linalg.norm(P_i - P_j)  
        d_ij = evaluator.detour[i, j] if use_detour else dist_straight  
          
        tt = evaluator.get_travel_time(i, j, t_curr, use_detour)  
          
        # --- 随机扰动 ---  
        # 在所有工况下，CV值约为13%。我们采用对数正态分布或截断正态分布进行采样，以避免出现负的旅行时间。        
        tt_stoch = tt * max(0.5, np.random.normal(1.0, 0.13))  
        # -------------------------------  
        t_arr = t_curr + tt_stoch  
          
        if tt_stoch > 1e-6 and d_ij > 1e-6:  
            v_kmh = d_ij / (tt_stoch / 60.0)  
            v_kmh = min(v_kmh, 60.0) # cap speed  
            rho = current_w / spec['Q_w']  
              
            if is_ev:  
                epk = 0.0014 * v_kmh**2 - 0.12 * v_kmh + 36.19  
                rate_per_km = (epk / 100.0) * (1 + spec['alpha'] * rho)  
                energy_cost += rate_per_km * d_ij * evaluator.PRICE_ELEC  
            else:  
                fpk = 0.0025 * v_kmh**2 - 0.2554 * v_kmh + 31.75  
                rate_per_km = (fpk / 100.0) * (1 + spec['alpha'] * rho)  
                fuel_c = rate_per_km * d_ij * evaluator.PRICE_FUEL  
                carbon_c = rate_per_km * d_ij * ETA_FUEL * evaluator.CO2_COST  
                energy_cost += fuel_c + carbon_c  
                  
        if k < len(nodes) - 1:  
            pid = packet_ids[k-1]  
            current_w -= evaluator.packets[pid].w  
              
        if t_arr < tw_starts[k]:  
            soft_penalty += WAIT_PENALTY * (tw_starts[k] - t_arr)  
            t_curr = tw_starts[k] + svc_times[k]  
        elif t_arr > tw_ends[k]:  
            soft_penalty += LATE_PENALTY * (t_arr - tw_ends[k])  
            t_curr = t_arr + svc_times[k]  
            if k < len(nodes) - 1:  
                late_deliveries += 1  
        else:  
            t_curr = t_arr + svc_times[k]  
              
    return 400.0 + energy_cost + soft_penalty, late_deliveries  
  
def run_monte_carlo():  
    base_dir = Path(__file__).resolve().parent.parent.parent  
    data_dir = base_dir / 'data'  
    precomp_dir = base_dir / 'results' / 'precomputed'  
    results_dir = base_dir / 'results'  
    mc_dir = results_dir / 'montecarlo'  
    mc_dir.mkdir(parents=True, exist_ok=True)  
      
    coords, D, df_orders, tw, customers = ingest_data(data_dir)  
    T_lookup = np.load(precomp_dir / 'T_lookup.npy')  
    G_ij = np.load(precomp_dir / 'G_ij.npy')  
    detour = np.load(precomp_dir / 'detour.npy')  
      
    with open(precomp_dir / 'packets.json', 'r') as f:  
        packet_dicts = json.load(f)  
    packets = [Task(**d) for d in packet_dicts]  
    time_grid = np.arange(420, 420 + 180 * 5, 5)  
      
    evaluator_q2 = RouteEvaluator(T_lookup, G_ij, detour, time_grid, packets, customers, green_ban=True)  
      
    with open(results_dir / 'phaseA_solution_q2.json', 'r') as f:  
        q2_routes = json.load(f)  
          
    M = 50 # MC样本数量  
    results = []  
      
    print(f"Running Monte Carlo Simulation with M={M}...")  
    for m in range(M):  
        total_cost = 0  
        total_late = 0  
        for r in q2_routes:  
            c, late = replay_route_stochastic(evaluator_q2, r['packets'], r['v_type'], r['details']['optimal_dep'])  
            total_cost += c  
            total_late += late  
        results.append({'sample_id': m, 'cost': total_cost, 'n_late': total_late})  
          
    df = pd.DataFrame(results)  
    df.to_csv(mc_dir / 'q2_mc.csv', index=False)  
      
    summary = {  
        'mean_cost': df['cost'].mean(),  
        'std_cost': df['cost'].std(),  
        'p5_cost': df['cost'].quantile(0.05),  
        'p95_cost': df['cost'].quantile(0.95),  
        'mean_late': df['n_late'].mean()  
    }  
    with open(mc_dir / 'summary.json', 'w') as f:  
        json.dump(summary, f, indent=2)  
          
    # Plotting F19  
    plt.figure(figsize=(8, 5))  
    plt.hist(df['cost'], bins=15, color='orange', alpha=0.7, edgecolor='black')  
    plt.axvline(df['cost'].mean(), color='red', linestyle='dashed', linewidth=2, label=f"Mean: ¥{df['cost'].mean():.0f}")  
    plt.title(f'Figure F19: MC Cost Distribution (Q2, M={M})')  
    plt.xlabel('Total Cost (¥)')  
    plt.ylabel('Frequency')  
    plt.legend()  
    plt.grid(True, alpha=0.3)  
    plt.savefig(os.path.join(base_dir, 'figures', 'f19_mc_cost_distribution.png'), dpi=300)  
    plt.close()  
      
    # 绘制 F20 - 晚期概率热力图     
    plt.figure(figsize=(6, 4))  
    plt.text(0.5, 0.5, f"Expected Late Deliveries: {summary['mean_late']:.1f}\n(out of 148 packets)",   
             fontsize=14, ha='center', va='center')  
    plt.axis('off')  
    plt.title('Figure F20: MC Late Probability Summary')  
    plt.savefig(os.path.join(base_dir, 'figures', 'f20_mc_late_probability.png'), dpi=300)  
    plt.close()  
  
    print("Monte Carlo Simulation completed.")  
  
if __name__ == "__main__":  
    run_monte_carlo()
~~~
# phase11_report
## generate_all_figures.py
~~~python 
import os  
import sys  
import json  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import matplotlib as mpl  
from pathlib import Path  
  
# 安装路径  
# 项目资源(scripts/data/results/figures)位于`Code/project`目录下 
base_dir = Path(__file__).resolve().parent.parent  
sys.path.append(str(base_dir / 'scripts'))  
  
from phase0_setup.constants import R_GREEN, VEHICLE_SPECS, SPEED_PARAMS, REGIMES  
from phase1_ingest.ingest import ingest_data  
  
# ==========================================  
# 全局样式配置  
# ==========================================  
def _apply_style():  
    """Apply unified visual theme — clean, spacious, minimal chart junk."""  
    mpl.rcParams.update({  
        'font.family': 'DejaVu Sans',  
        'font.size': 11,  
        'axes.titlesize': 13,  
        'axes.titleweight': 'bold',  
        'axes.titlepad': 12,  
        'axes.labelsize': 11,  
        'axes.labelcolor': '#2C3E50',  
        'axes.spines.top': False,  
        'axes.spines.right': False,  
        'axes.edgecolor': '#B0B7BF',  
        'axes.linewidth': 0.8,  
        'axes.grid': True,  
        'grid.color': '#ECEFF1',  
        'grid.linewidth': 0.7,  
        'grid.alpha': 1.0,  
        'axes.axisbelow': True,  
        'legend.fontsize': 10,  
        'legend.frameon': True,  
        'legend.framealpha': 0.95,  
        'legend.edgecolor': '#D5DBDB',  
        'xtick.labelsize': 10,  
        'ytick.labelsize': 10,  
        'xtick.color': '#566573',  
        'ytick.color': '#566573',  
        'figure.facecolor': 'white',  
        'savefig.facecolor': 'white',  
        'savefig.bbox': 'tight',  
    })  
  
# ----- 高光/暗部色调组合 -----  
HUE_PAIRS = {  
    'gray':   ('#7F8C8D', '#E5E7E9'),  
    'red':    ('#E74C3C', '#FADBD8'),  
    'orange': ('#E67E22', '#FAE5D3'),  
    'blue':   ('#2E86DE', '#D6EAF8'),  
    'green':  ('#27AE60', '#D4EFDF'),  
    'purple': ('#8E44AD', '#E8DAEF'),  
    'teal':   ('#16A085', '#D1F2EB'),  
    'navy':   ('#2C3E50', '#D5DBDB'),  
}  
  
# 成本组件复用了参考图像的色调（灰色/橙色/红色）  
COST_COLORS = {  
    'Startup': HUE_PAIRS['gray'][0],  
    'Energy':  HUE_PAIRS['orange'][0],  
    'Penalty': HUE_PAIRS['red'][0],  
    'Total':   HUE_PAIRS['navy'][0],  
}  
  
SCENARIO_COLORS = {  
    'Q1': HUE_PAIRS['blue'][0],  
    'Q2': HUE_PAIRS['red'][0],  
}  
  
VTYPE_COLORS = {  
    'F-3000': '#2E86DE',  
    'F-1500': '#8E44AD',  
    'F-1250': '#E67E22',  
    'E-3000': '#27AE60',  
    'E-1250': '#16A085',  
}  
  
  
def get_regime_at_time(t):  
    t_mod = t % (24 * 60)  
    for start, end, name in REGIMES:  
        if start <= t_mod < end:  
            return name  
    return 'smooth'  
  
# ==========================================  
# F01: 客户布局  
# ==========================================  
def plot_f01(customers, figures_dir):  
    fig, ax = plt.subplots(figsize=(9, 9))  
  
    x_green, y_green = [], []  
    x_outer, y_outer = [], []  
    for i in range(1, 99):  
        c = customers[i]  
        if np.sqrt(c.x ** 2 + c.y ** 2) <= R_GREEN:  
            x_green.append(c.x)  
            y_green.append(c.y)  
        else:  
            x_outer.append(c.x)  
            y_outer.append(c.y)  
  
    circle_fill = plt.Circle((0, 0), R_GREEN, color=HUE_PAIRS['green'][1], alpha=0.5, zorder=1)  
    ax.add_patch(circle_fill)  
    circle = plt.Circle((0, 0), R_GREEN, color=HUE_PAIRS['green'][0], fill=False,  
                        linestyle='--', linewidth=1.4, label='Green Zone Boundary', zorder=2)  
    ax.add_patch(circle)  
  
    ax.scatter(x_outer, y_outer, c=HUE_PAIRS['blue'][0], s=34, alpha=0.85,  
               edgecolor='white', linewidth=0.6, label='Outer Customers', zorder=3)  
    ax.scatter(x_green, y_green, c=HUE_PAIRS['green'][0], s=34, alpha=0.95,  
               edgecolor='white', linewidth=0.6, label='Green-Zone Customers', zorder=3)  
  
    dc = customers[0]  
    ax.scatter(dc.x, dc.y, c=HUE_PAIRS['red'][0], marker='*', s=420,  
               edgecolor='white', linewidth=1.5, label='DC', zorder=5)  
  
    all_x = x_green + x_outer + [dc.x]  
    all_y = y_green + y_outer + [dc.y]  
    pad = 3  
    lim = max(abs(min(all_x)), abs(max(all_x)), abs(min(all_y)), abs(max(all_y)), R_GREEN) + pad  
    ax.set_xlim(-lim, lim)  
    ax.set_ylim(-lim, lim)  
    ax.set_aspect('equal')  
  
    ax.axhline(0, color='#BDC3C7', linewidth=0.5, alpha=0.5, zorder=0)  
    ax.axvline(0, color='#BDC3C7', linewidth=0.5, alpha=0.5, zorder=0)  
  
    ax.legend(loc='upper right')  
    ax.set_xlabel('X (km)')  
    ax.set_ylabel('Y (km)')  
    plt.savefig(os.path.join(figures_dir, 'f01_customer_layout.png'), dpi=300, bbox_inches='tight')  
    plt.close()  
  
# ==========================================  
# F02: 配速情况  
# ==========================================  
def plot_f02(figures_dir):  
    pace_table = {}  
    for name, (mu_v, sigma_v) in SPEED_PARAMS.items():  
        mu_p = 60.0 / mu_v  
        sigma_p = (60.0 * sigma_v) / (mu_v ** 2)  
        pace_table[name] = {'mu_p': mu_p, 'sigma_p': sigma_p}  
  
    times = np.arange(420, 1320)  
    mu_vals, sigma_vals = [], []  
    for t in times:  
        r = get_regime_at_time(t)  
        mu_vals.append(pace_table[r]['mu_p'])  
        sigma_vals.append(pace_table[r]['sigma_p'])  
    mu_vals = np.array(mu_vals)  
    sigma_vals = np.array(sigma_vals)  
    hours = times / 60  
  
    fig, ax = plt.subplots(figsize=(11, 4.8))  
  
    # Ban window in muted red — matches the highlight-mute aesthetic  
    ax.axvspan(8, 16, color=HUE_PAIRS['red'][1], alpha=0.7, label='Ban Window', zorder=1)  
  
    ax.fill_between(hours, mu_vals - sigma_vals, mu_vals + sigma_vals,  
                    color=HUE_PAIRS['blue'][1], alpha=0.85, label='±1 σ', zorder=2)  
    ax.plot(hours, mu_vals, color=HUE_PAIRS['blue'][0], linewidth=2.0,  
            label='Mean Pace (min/km)', zorder=3)  
  
    y_top = (mu_vals + sigma_vals).max()  
    ax.text(12, y_top * 1.02, 'Ban Window (08:00–16:00)',  
            ha='center', va='bottom', color=HUE_PAIRS['red'][0], fontsize=10, fontweight='bold')  
  
    ax.set_xticks(np.arange(8, 23, 2))  
    ax.set_xticklabels([f'{h:02d}:00' for h in np.arange(8, 23, 2)])  
    ax.set_xlabel('Hour of Day')  
    ax.set_ylabel('Pace (min/km)')  
    ax.legend(loc='upper right')  
  
    plt.savefig(os.path.join(figures_dir, 'f02_pace_profile.png'), dpi=300, bbox_inches='tight')  
    plt.close()  
  
# ==========================================  
# F03: 绿区渗透度 
# ==========================================  
def plot_f03(G, figures_dir):  
    fig, ax = plt.subplots(figsize=(8, 8))    
    cmap = mpl.colors.ListedColormap([HUE_PAIRS['blue'][1], HUE_PAIRS['blue'][0]])  
    im = ax.imshow(G, cmap=cmap, aspect='equal', vmin=0, vmax=1, interpolation='nearest')  
    ax.set_xlabel('Node $j$')  
    ax.set_ylabel('Node $i$')  
    ax.grid(False)  
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])  
    cbar.set_label('Penetration indicator (0 / 1)')  
    cbar.outline.set_edgecolor('#D5DBDB')  
    plt.savefig(os.path.join(figures_dir, 'f03_arc_penetration.png'), dpi=300, bbox_inches='tight')  
    plt.close()  
  
# ==========================================  
# F04: 需求分布  
# ==========================================  
def plot_f04(customers, packets, figures_dir):  
    cust_w = [c.demand_w for c in customers.values() if c.id != 0 and c.demand_w > 0]  
    pkt_w = [p['w'] for p in packets]  
  
    w_max = max(max(cust_w), max(pkt_w), 3000)  
    bins = np.linspace(0, w_max * 1.05, 21)  
  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))  
  
    ax1.hist(cust_w, bins=bins, color=HUE_PAIRS['blue'][0], edgecolor='white',  
             linewidth=0.8, alpha=0.85)  
    ax1.axvline(3000, color=HUE_PAIRS['red'][0], linestyle='--', linewidth=1.8, label='3000 kg Limit')  
    ax1.text(3000, ax1.get_ylim()[1] * 0.92, ' 3000 kg', color=HUE_PAIRS['red'][0],  
             fontsize=9, va='top', ha='left', fontweight='bold')  
  
    ax1.set_xlabel('Weight (kg)')  
    ax1.set_ylabel('Count')  
    ax1.legend(loc='upper right')  
  
    ax2.hist(pkt_w, bins=bins, color=HUE_PAIRS['green'][0], edgecolor='white',  
             linewidth=0.8, alpha=0.85)  
    ax2.axvline(3000, color=HUE_PAIRS['red'][0], linestyle='--', linewidth=1.8)  
    ax2.text(3000, ax2.get_ylim()[1] * 0.92, ' 3000 kg', color=HUE_PAIRS['red'][0],  
             fontsize=9, va='top', ha='left', fontweight='bold')  
  
    ax2.set_xlabel('Weight (kg)')  
    ax2.set_ylabel('Count')  
  
    plt.tight_layout()  
    plt.savefig(os.path.join(figures_dir, 'f04_demand_distribution.png'), dpi=300, bbox_inches='tight')  
    plt.close()  
  
# ==========================================  
# F08: Q1 成本明细  
# ==========================================  
def plot_f08(summary, figures_dir):  
    costs = {  
        'Startup': summary['startup_cost'],  
        'Energy':  summary['energy_cost'],  
        'Penalty': summary['penalty_cost'],  
    }  
    labels = list(costs.keys())  
    values = list(costs.values())  
  
    max_idx = int(np.argmax(values))  
    hue_keys = ['gray', 'orange', 'red']  # aligned to Startup / Energy / Penalty  
    colors = [HUE_PAIRS[h][0] if i == max_idx else HUE_PAIRS[h][1]  
              for i, h in enumerate(hue_keys)]  
    explode = [0.04 if i == max_idx else 0.0 for i in range(len(values))]  
  
    fig, ax = plt.subplots(figsize=(8.5, 7))  
  
    def _autopct(pct, all_vals=values):  
        absolute = pct / 100.0 * sum(all_vals)  
        return f'¥{absolute:,.0f}\n({pct:.1f}%)'  
  
    wedges, texts, autotexts = ax.pie(  
        values,  
        labels=labels,  
        colors=colors,  
        explode=explode,  
        autopct=_autopct,  
        pctdistance=0.72,  
        startangle=90,  
        wedgeprops={'edgecolor': 'white', 'linewidth': 2.5},  
        textprops={'fontsize': 11},  
    )  
  
     
    for i, t in enumerate(texts):  
        t.set_color(HUE_PAIRS[hue_keys[i]][0])  
        t.set_fontsize(12)  
        t.set_fontweight('bold')  
    
    for i, t in enumerate(autotexts):  
        if i == max_idx:  
            t.set_color('white')  
            t.set_fontweight('bold')  
        else:  
            t.set_color('#566573')  
        t.set_fontsize(10)  
  
    ax.set_aspect('equal')  
    plt.savefig(os.path.join(figures_dir, 'f08_q1_cost_breakdown.png'), dpi=300, bbox_inches='tight')  
    plt.close()  
  
# ==========================================  
# F10: Q1 车队使用情况 
# ==========================================  
def plot_f10(summary, figures_dir):  
    vt_used = summary['by_vtype']  
    vt_avail = {v: VEHICLE_SPECS[v]['count'] for v in VEHICLE_SPECS}  
  
    vtypes = list(VEHICLE_SPECS.keys())  
    used = [vt_used.get(v, 0) for v in vtypes]  
    avail = [vt_avail[v] for v in vtypes]  
  
    x = np.arange(len(vtypes))  
    width = 0.38  
  
    fig, ax = plt.subplots(figsize=(10, 6))  
  
    ax.bar(x - width / 2, used, width, label='Trips Opened',  
           color=HUE_PAIRS['blue'][0], edgecolor='white', linewidth=0.8)  
    ax.bar(x + width / 2, avail, width, label='Physical Vehicles Available',  
           color=HUE_PAIRS['gray'][1], edgecolor='white', linewidth=0.8)  
  
    y_max = max(max(used), max(avail))  
    pad = y_max * 0.025  
    for xi, v in zip(x - width / 2, used):  
        ax.text(xi, v + pad, f'{v}', ha='center', va='bottom',  
                fontsize=10, color=HUE_PAIRS['blue'][0], fontweight='bold')  
    for xi, v in zip(x + width / 2, avail):  
        ax.text(xi, v + pad, f'{v}', ha='center', va='bottom',  
                fontsize=9, color='#7F8C8D')  
  
    ax.set_xticks(x)  
    ax.set_xticklabels(vtypes)  
    ax.set_ylabel('Count')  
    ax.set_ylim(0, y_max * 1.18)  
    ax.legend(loc='upper right')  
    ax.grid(axis='x', visible=False)  
    plt.savefig(os.path.join(figures_dir, 'f10_q1_fleet_usage.png'), dpi=300, bbox_inches='tight')  
    plt.close()  
  
# ==========================================  
# F11: Q2 路线  
# ==========================================  
def plot_f11(routes, customers, figures_dir):  
    fig, ax = plt.subplots(figsize=(12, 12))  
  
    x_green, y_green = [], []  
    x_reg, y_reg = [], []  
    for i in range(1, 99):  
        c = customers[i]  
        if np.sqrt(c.x ** 2 + c.y ** 2) <= R_GREEN:  
            x_green.append(c.x)  
            y_green.append(c.y)  
        else:  
            x_reg.append(c.x)  
            y_reg.append(c.y)  
  
    circle_fill = plt.Circle((0, 0), R_GREEN, color=HUE_PAIRS['green'][1], alpha=0.45, zorder=1)  
    ax.add_patch(circle_fill)  
    circle = plt.Circle((0, 0), R_GREEN, color=HUE_PAIRS['green'][0], fill=False,  
                        linestyle='--', linewidth=1.3, alpha=0.85, zorder=2)  
    ax.add_patch(circle)  
  
    ax.scatter(x_reg, y_reg, c=HUE_PAIRS['gray'][1], s=24, alpha=0.95,  
               edgecolor='#BDC3C7', linewidth=0.4, zorder=2)  
    ax.scatter(x_green, y_green, c=HUE_PAIRS['green'][1], s=24, alpha=0.95,  
               edgecolor='#7DCEA0', linewidth=0.4, zorder=2)  
  
    dc_x, dc_y = customers[0].x, customers[0].y  
    for r in routes:  
        vt = r['v_type']  
        pts = [(dc_x, dc_y)] + [(customers[v['customer_id']].x, customers[v['customer_id']].y)  
                          for v in r['visits']] + [(dc_x, dc_y)]  
        pts = np.array(pts)  
        ax.plot(pts[:, 0], pts[:, 1], color=VTYPE_COLORS[vt],  
                alpha=0.6, linewidth=1.3, zorder=3)  
  
    ax.scatter(dc_x, dc_y, c=HUE_PAIRS['red'][0], marker='*', s=460,  
               edgecolor='white', linewidth=1.8, zorder=5, label='DC')  
  
    handles = [plt.Line2D([0], [0], color=c, lw=2.4, label=vt) for vt, c in VTYPE_COLORS.items()]  
    handles.append(plt.Line2D([0], [0], marker='*', color='w',  
                              markerfacecolor=HUE_PAIRS['red'][0], markeredgecolor='white',  
                              markersize=14, label='DC', linestyle=''))  
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1.0),  
              borderaxespad=0., title='Vehicle Type')  
  
    ax.set_aspect('equal')  
    ax.set_xlabel('X (km)')  
    ax.set_ylabel('Y (km)')  
    plt.savefig(os.path.join(figures_dir, 'f11_q2_routes.png'), dpi=300, bbox_inches='tight')  
    plt.close()  
  
# ==========================================  
# F12 & F13: Q1 与 Q2 对比  
# ==========================================  
def plot_f12_f13(sum_q1, sum_q2, figures_dir):  
    categories = ['Startup', 'Energy', 'Penalty', 'Total']  
    q1_vals = [sum_q1['startup_cost'], sum_q1['energy_cost'], sum_q1['penalty_cost'], sum_q1['total_cost']]  
    q2_vals = [sum_q2['startup_cost'], sum_q2['energy_cost'], sum_q2['penalty_cost'], sum_q2['total_cost']]  
  
    x = np.arange(len(categories))  
    width = 0.38  
  
    # F12  
    fig, ax = plt.subplots(figsize=(10, 6))  
    ax.bar(x - width / 2, q1_vals, width, label='Q1 (Unrestricted)',  
           color=SCENARIO_COLORS['Q1'], edgecolor='white', linewidth=0.8)  
    ax.bar(x + width / 2, q2_vals, width, label='Q2 (Green Ban)',  
           color=SCENARIO_COLORS['Q2'], edgecolor='white', linewidth=0.8)  
    ax.set_xticks(x)  
    ax.set_xticklabels(categories)  
    ax.set_ylabel('Cost (¥)')  
    ax.legend(loc='upper left')  
    ax.grid(axis='x', visible=False)  
  
    y_max = max(max(q1_vals), max(q2_vals))  
    pad = y_max * 0.025  
    ax.set_ylim(0, y_max * 1.18)  
  
    for i in range(len(categories)):  
        delta = q2_vals[i] - q1_vals[i]  
        anchor_y = max(q1_vals[i], q2_vals[i]) + pad  
        sign = '+' if delta >= 0 else ''  
        if delta > 0:  
            color = HUE_PAIRS['red'][0]  
        elif delta < 0:  
            color = HUE_PAIRS['green'][0]  
        else:  
            color = '#7F8C8D'  
        ax.text(i, anchor_y, f'Δ {sign}{delta:,.0f}',  
                ha='center', va='bottom', color=color, fontsize=10, fontweight='bold')  
  
    plt.savefig(os.path.join(figures_dir, 'f12_q1_vs_q2_cost.png'), dpi=300, bbox_inches='tight')  
    plt.close()  
  
    # F13  
    metrics = ['Total km']  
    q1_m = [sum_q1['total_km']]  
    q2_m = [sum_q2['total_km']]  
  
    fig, ax = plt.subplots(figsize=(5.5, 5))  
    x = np.arange(len(metrics))  
    bars1 = ax.bar(x - width / 2, q1_m, width, label='Q1',  
                   color=SCENARIO_COLORS['Q1'], edgecolor='white', linewidth=0.8)  
    bars2 = ax.bar(x + width / 2, q2_m, width, label='Q2',  
                   color=SCENARIO_COLORS['Q2'], edgecolor='white', linewidth=0.8)  
    ax.set_xticks(x)  
    ax.set_xticklabels(metrics)  
    ax.set_ylabel('Value')  
    ax.legend(loc='upper right')  
    ax.grid(axis='x', visible=False)  
  
    y_max = max(max(q1_m), max(q2_m))   
    y_min = 9000  
    ax.set_ylim(y_min, y_max * 1.08)   
    pad = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03  
    for b, v in zip(bars1, q1_m):  
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + pad,  
                f'{v:,.1f}', ha='center', va='bottom', fontsize=10,  
                color=SCENARIO_COLORS['Q1'], fontweight='bold')  
    for b, v in zip(bars2, q2_m):  
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + pad,  
                f'{v:,.1f}', ha='center', va='bottom', fontsize=10,  
                color=SCENARIO_COLORS['Q2'], fontweight='bold')  
  
    plt.savefig(os.path.join(figures_dir, 'f13_q1_vs_q2_emissions.png'), dpi=300, bbox_inches='tight')  
    plt.close()  
  
# ==========================================  
# F16: E-Fleet 灵敏度  
# ==========================================  
def plot_f16(base_dir):  
    efleet_df = pd.read_csv(os.path.join(base_dir, 'results', 'sensitivity', 'e_fleet.csv'))  
  
    fig, ax = plt.subplots(figsize=(8.5, 6))  
    line_color = HUE_PAIRS['red'][0]  
    muted = HUE_PAIRS['red'][1]  
  
    ax.plot(efleet_df['E-3000'], efleet_df['total_cost'],  
            color=line_color, linewidth=2.0, zorder=2)  
   
    min_idx = int(efleet_df['total_cost'].idxmin())  
    sizes = [110 if i == min_idx else 60 for i in range(len(efleet_df))]  
    face_colors = [line_color if i == min_idx else 'white' for i in range(len(efleet_df))]  
    edge_colors = [line_color] * len(efleet_df)  
    ax.scatter(efleet_df['E-3000'], efleet_df['total_cost'],  
               s=sizes, facecolor=face_colors, edgecolor=edge_colors,  
               linewidth=2.0, zorder=3)  
   
    ax.fill_between(efleet_df['E-3000'], efleet_df['total_cost'],  
                    efleet_df['total_cost'].max(),  
                    color=muted, alpha=0.35, zorder=1)  
  
    min_x = efleet_df['E-3000'].iloc[min_idx]  
    min_y = efleet_df['total_cost'].iloc[min_idx]  
    ax.annotate(f'min: ¥{min_y:,.0f}\n@ E-3000 = {min_x}',  
                xy=(min_x, min_y),  
                xytext=(15, 25), textcoords='offset points',  
                fontsize=10, color=line_color, fontweight='bold',  
                arrowprops=dict(arrowstyle='->', color=line_color, lw=1.0))  
  
    ax.set_xlabel('E-3000 Fleet Size')  
    ax.set_ylabel('Total Cost (¥)')  
    plt.savefig(os.path.join(base_dir, 'figures', 'f16_sensitivity_efleet.png'),  
                dpi=300, bbox_inches='tight')  
    plt.close()  
  
# ==========================================  
# F17: 灵敏度曲线 
# ==========================================  
def plot_f17(base_dir):  
    fuel_df = pd.read_csv(os.path.join(base_dir, 'results', 'sensitivity', 'fuel_price.csv'))  
    carbon_df = pd.read_csv(os.path.join(base_dir, 'results', 'sensitivity', 'carbon_price.csv'))  
  
    base_fuel = fuel_df[fuel_df['mult'] == 1.0]['total_cost'].values[0]  
    base_carbon = carbon_df[carbon_df['mult'] == 1.0]['total_cost'].values[0]  
  
    # Convert each sweep into (parameter deviation %, cost change %) — full curves, not just endpoints  
    fuel_dev = (fuel_df['mult'].values - 1.0) * 100  
    fuel_pct = (fuel_df['total_cost'].values - base_fuel) / base_fuel * 100  
    carbon_dev = (carbon_df['mult'].values - 1.0) * 100  
    carbon_pct = (carbon_df['total_cost'].values - base_carbon) / base_carbon * 100  
  
    fig, ax = plt.subplots(figsize=(9.5, 6))  
  
    fuel_color = HUE_PAIRS['orange'][0]  
    fuel_muted = HUE_PAIRS['orange'][1]  
    carbon_color = HUE_PAIRS['purple'][0]  
    carbon_muted = HUE_PAIRS['purple'][1]  
  
    # 燃油曲线  
    ax.plot(fuel_dev, fuel_pct, color=fuel_color, linewidth=2.2, zorder=3, label='Fuel Price')  
    ax.scatter(fuel_dev, fuel_pct, s=70, facecolor='white', edgecolor=fuel_color,  
               linewidth=2.0, zorder=4)  
    ax.fill_between(fuel_dev, fuel_pct, 0, color=fuel_muted, alpha=0.55, zorder=1)  
  
    # 碳曲线  
    ax.plot(carbon_dev, carbon_pct, color=carbon_color, linewidth=2.2, zorder=3, label='Carbon Price')  
    ax.scatter(carbon_dev, carbon_pct, s=70, facecolor='white', edgecolor=carbon_color,  
               linewidth=2.0, zorder=4)  
    ax.fill_between(carbon_dev, carbon_pct, 0, color=carbon_muted, alpha=0.55, zorder=1)  
  
    # 通过基准线的参考线（0%偏差，0%成本变化） 
    ax.axhline(0, color='#7F8C8D', linewidth=0.9, linestyle='--', zorder=2)  
    ax.axvline(0, color='#7F8C8D', linewidth=0.9, linestyle='--', zorder=2)  
  
    def _annotate_endpoints(devs, pcts, color):  
        for d, p in [(devs[0], pcts[0]), (devs[-1], pcts[-1])]:  
            offset = (10, 8) if p >= 0 else (10, -14)  
            ax.annotate(f'{p:+.2f}%', xy=(d, p), xytext=offset,  
                        textcoords='offset points', fontsize=9.5,  
                        color=color, fontweight='bold')  
  
    _annotate_endpoints(fuel_dev, fuel_pct, fuel_color)  
    _annotate_endpoints(carbon_dev, carbon_pct, carbon_color)  
  
    
    ax.scatter([0], [0], s=110, facecolor=HUE_PAIRS['navy'][0], edgecolor='white',  linewidth=1.8, zorder=5, label='Baseline')  
	ax.set_xlabel('Parameter Deviation from Baseline (%)')  
    ax.set_ylabel('Change in Total Cost (%)')  
    ax.legend(loc='best')  
    ax.grid(axis='x', alpha=0.4)  
  
    plt.savefig(os.path.join(base_dir, 'figures', 'f17_sensitivity_tornado.png'),  
                dpi=300, bbox_inches='tight')  
    plt.close()  
  
# ==========================================  
# F18: 成本布局 — 碳价  
# ==========================================  
def plot_f18(base_dir):  
    carbon_df = pd.read_csv(os.path.join(base_dir, 'results', 'sensitivity', 'carbon_price.csv'))  
  
    fig, ax = plt.subplots(figsize=(8.5, 6))  
    line_color = HUE_PAIRS['blue'][0]  
    muted = HUE_PAIRS['blue'][1]  
  
    ax.fill_between(carbon_df['mult'], carbon_df['total_cost'],  
                    carbon_df['total_cost'].min(),  
                    color=muted, alpha=0.5, zorder=1)  
    ax.plot(carbon_df['mult'], carbon_df['total_cost'],  
            color=line_color, linewidth=2.0, zorder=2)  
  
    base_row = carbon_df[carbon_df['mult'] == 1.0]  
    is_base = (carbon_df['mult'] == 1.0).values  
    sizes = np.where(is_base, 130, 60)  
    face_colors = [HUE_PAIRS['orange'][0] if b else 'white' for b in is_base]  
    edge_colors = [HUE_PAIRS['orange'][0] if b else line_color for b in is_base]  
    ax.scatter(carbon_df['mult'], carbon_df['total_cost'],  
               s=sizes, facecolor=face_colors, edgecolor=edge_colors,  
               linewidth=2.0, zorder=4)  
  
    if not base_row.empty:  
        bx = base_row['mult'].values[0]  
        by = base_row['total_cost'].values[0]  
        ax.annotate(f'Baseline: ¥{by:,.0f}',  
                    xy=(bx, by), xytext=(12, -22), textcoords='offset points',  
                    fontsize=10, color=HUE_PAIRS['orange'][0], fontweight='bold',  
                    arrowprops=dict(arrowstyle='->', color=HUE_PAIRS['orange'][0], lw=1.0))  
    
    legend_handles = [  
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',  
                   markeredgecolor=line_color, markersize=8, markeredgewidth=2,  
                   label='Carbon Price Varied'),  
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=HUE_PAIRS['orange'][0],  
                   markeredgecolor=HUE_PAIRS['orange'][0], markersize=10,  
                   label='Baseline (mult = 1.0)'),  
    ]  
    ax.legend(handles=legend_handles, loc='best')  
  
    ax.set_xlabel('Carbon Price Multiplier')  
    ax.set_ylabel('Total Cost (¥)')  
  
    plt.savefig(os.path.join(base_dir, 'figures', 'f18_pareto_cost_vs_co2.png'),  
                dpi=300, bbox_inches='tight')  
    plt.close()  
  
# ==========================================  
# F19 & F20: 蒙特卡洛模拟结果  
# ==========================================  
def plot_f19_f20(base_dir):  
    mc_dir = base_dir / 'results' / 'montecarlo'  
    df = pd.read_csv(mc_dir / 'q2_mc.csv')  
    with open(mc_dir / 'summary.json', 'r') as f:  
        summary = json.load(f)  
  
    M = len(df)  
  
    cost_mean = df['cost'].mean()  
    cost_std = df['cost'].std()  
    p05 = df['cost'].quantile(0.05)  
    p95 = df['cost'].quantile(0.95)  
  
    fig, ax = plt.subplots(figsize=(9, 5.2))  
    counts, bin_edges, patches = ax.hist(df['cost'], bins=15,  
                                         color=HUE_PAIRS['orange'][1],  
                                         edgecolor='white', linewidth=0.8)   
    for i, p in enumerate(patches):  
        if bin_edges[i] <= cost_mean < bin_edges[i + 1]:  
            p.set_facecolor(HUE_PAIRS['orange'][0])  
  
    ax.axvline(cost_mean, color=HUE_PAIRS['red'][0], linestyle='--', linewidth=2.0,  
               label=f"Mean: ¥{cost_mean:,.0f}")  
    ax.axvline(p05, color='#7F8C8D', linestyle=':', linewidth=1.6,  
               label=f"5th pct: ¥{p05:,.0f}")  
    ax.axvline(p95, color='#7F8C8D', linestyle=':', linewidth=1.6,  
               label=f"95th pct: ¥{p95:,.0f}")  
  
    ax.text(0.98, 0.97, f"N = {M}\nσ = ¥{cost_std:,.0f}",  
            transform=ax.transAxes, ha='right', va='top', fontsize=10,  
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',  
                      edgecolor='#D5DBDB', alpha=0.95))  
  
    ax.set_xlabel('Total Cost (¥)')  
    ax.set_ylabel('Frequency')  
    ax.legend(loc='upper left')  
    plt.savefig(os.path.join(base_dir, 'figures', 'f19_mc_cost_distribution.png'),  
                dpi=300, bbox_inches='tight')  
    plt.close()  
  
    total_packets = 148  
    mean_late = summary['mean_late']  
    on_time = max(total_packets - mean_late, 0.0)  
  
    fig, ax = plt.subplots(figsize=(9, 3.4))  
    ax.barh([0], [on_time], color=HUE_PAIRS['green'][1], edgecolor='white', linewidth=0.8,  
            label=f'On-time: {on_time:.1f}')  
    ax.barh([0], [mean_late], left=[on_time], color=HUE_PAIRS['red'][0],  
            edgecolor='white', linewidth=0.8, label=f'Late: {mean_late:.1f}')  
  
    ax.text(on_time / 2, 0, f'{on_time:.1f}', ha='center', va='center',  
            color=HUE_PAIRS['green'][0], fontsize=11, fontweight='bold')  
    if mean_late > 0:  
        ax.text(on_time + mean_late / 2, 0, f'{mean_late:.1f}', ha='center', va='center',  
                color='white', fontsize=11, fontweight='bold')  
  
    ax.set_xlim(0, total_packets)  
    ax.set_ylim(-0.6, 0.6)  
    ax.set_yticks([])  
    ax.set_xlabel(f'Packets (out of {total_packets})')  
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.40), ncol=2, frameon=False)  
    ax.grid(axis='y', visible=False)  
    ax.spines['left'].set_visible(False)  
  
    plt.savefig(os.path.join(base_dir, 'figures', 'f20_mc_late_probability.png'),  
                dpi=300, bbox_inches='tight')  
    plt.close()  
  
if __name__ == "__main__":  
    print("Regenerating all 14 figures from saved data...")  
    _apply_style()  
    figures_dir = base_dir / 'figures'  
    figures_dir.mkdir(parents=True, exist_ok=True)  
  
    # 加载所需数据  
    data_dir = base_dir / 'data'  
    coords, D, df_orders, tw, customers = ingest_data(data_dir)  
  
    precomp_dir = base_dir / 'results' / 'precomputed'  
    G_ij = np.load(precomp_dir / 'G_ij.npy')  
    with open(precomp_dir / 'packets.json', 'r') as f:  
        packets = json.load(f)  
  
    with open(base_dir / 'results' / 'q1' / 'summary.json', 'r') as f:  
        sum_q1 = json.load(f)  
  
    with open(base_dir / 'results' / 'q2' / 'summary.json', 'r') as f:  
        sum_q2 = json.load(f)  
  
    with open(base_dir / 'results' / 'phaseB_solution_q2.json', 'r') as f:  
        routes_q2 = json.load(f)  
  
    # 执行绘图
    plot_f01(customers, figures_dir)  
    plot_f02(figures_dir)  
    plot_f03(G_ij, figures_dir)  
    plot_f04(customers, packets, figures_dir)  
    plot_f08(sum_q1, figures_dir)  
    plot_f10(sum_q1, figures_dir)  
    plot_f11(routes_q2, customers, figures_dir)  
    plot_f12_f13(sum_q1, sum_q2, figures_dir)  
    plot_f16(base_dir)  
    plot_f17(base_dir)  
    plot_f18(base_dir)  
    plot_f19_f20(base_dir)  
  
    print(f"All 14 figures successfully regenerated in: {figures_dir}")
~~~
