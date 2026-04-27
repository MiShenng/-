import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Setup paths
# Project assets (scripts/data/results/figures) live under `Code/project`.
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir / 'scripts'))

from phase0_setup.constants import R_GREEN, VEHICLE_SPECS, SPEED_PARAMS, REGIMES
from phase1_ingest.ingest import ingest_data

# ==========================================
# Global Style Configuration
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

# ----- Highlight / muted hue pairs (matched to reference image) -----
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

# Cost components reuse the reference image's hues (gray/orange/red)
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
# F01: Customer Layout
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

    # Soft green-zone fill, then dashed boundary
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
# F02: Pace Profile
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
# F03: Arc Penetration
# ==========================================
def plot_f03(G, figures_dir):
    fig, ax = plt.subplots(figsize=(8, 8))
    # Two-tone discrete cmap mirroring the highlight-mute pairing (blue)
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
# F04: Demand Distribution
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
# F08: Q1 Cost Breakdown (PIE CHART, highlight-and-mute style)
# ==========================================
def plot_f08(summary, figures_dir):
    costs = {
        'Startup': summary['startup_cost'],
        'Energy':  summary['energy_cost'],
        'Penalty': summary['penalty_cost'],
    }
    labels = list(costs.keys())
    values = list(costs.values())

    # Highlight largest slice, mute the others (one hue per slice)
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

    # Labels in matching saturated hue (echoes the in-chart annotations of the reference)
    for i, t in enumerate(texts):
        t.set_color(HUE_PAIRS[hue_keys[i]][0])
        t.set_fontsize(12)
        t.set_fontweight('bold')

    # Inside-wedge text: white on the saturated slice, dark gray on muted slices
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
# F10: Q1 Fleet Usage
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

    # Saturated for "Trips Opened" (the operational result), muted for capacity reference
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
# F11: Q2 Routes
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
# F12 & F13: Q1 vs Q2 Comparison
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
    # Start y-axis at 9000 for a cleaner, more readable scale.
    y_min = 9000
    # Keep some headroom above the tallest bar so the plot doesn't feel cramped.
    ax.set_ylim(y_min, y_max * 1.08)
    # Pad for value labels based on the visible range (keeps proportions harmonious).
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
# F16: E-Fleet Sensitivity (highlight-and-mute on minimum point)
# ==========================================
def plot_f16(base_dir):
    efleet_df = pd.read_csv(os.path.join(base_dir, 'results', 'sensitivity', 'e_fleet.csv'))

    fig, ax = plt.subplots(figsize=(8.5, 6))
    line_color = HUE_PAIRS['red'][0]
    muted = HUE_PAIRS['red'][1]

    ax.plot(efleet_df['E-3000'], efleet_df['total_cost'],
            color=line_color, linewidth=2.0, zorder=2)

    # Muted markers everywhere, saturated marker on the minimum
    min_idx = int(efleet_df['total_cost'].idxmin())
    sizes = [110 if i == min_idx else 60 for i in range(len(efleet_df))]
    face_colors = [line_color if i == min_idx else 'white' for i in range(len(efleet_df))]
    edge_colors = [line_color] * len(efleet_df)
    ax.scatter(efleet_df['E-3000'], efleet_df['total_cost'],
               s=sizes, facecolor=face_colors, edgecolor=edge_colors,
               linewidth=2.0, zorder=3)

    # Background band echoing the muted hue
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
# F17: Sensitivity Curves (REPLACES tornado chart)
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

    # Fuel curve
    ax.plot(fuel_dev, fuel_pct, color=fuel_color, linewidth=2.2, zorder=3, label='Fuel Price')
    ax.scatter(fuel_dev, fuel_pct, s=70, facecolor='white', edgecolor=fuel_color,
               linewidth=2.0, zorder=4)
    ax.fill_between(fuel_dev, fuel_pct, 0, color=fuel_muted, alpha=0.55, zorder=1)

    # Carbon curve
    ax.plot(carbon_dev, carbon_pct, color=carbon_color, linewidth=2.2, zorder=3, label='Carbon Price')
    ax.scatter(carbon_dev, carbon_pct, s=70, facecolor='white', edgecolor=carbon_color,
               linewidth=2.0, zorder=4)
    ax.fill_between(carbon_dev, carbon_pct, 0, color=carbon_muted, alpha=0.55, zorder=1)

    # Reference lines through baseline (0% deviation, 0% cost change)
    ax.axhline(0, color='#7F8C8D', linewidth=0.9, linestyle='--', zorder=2)
    ax.axvline(0, color='#7F8C8D', linewidth=0.9, linestyle='--', zorder=2)

    # Endpoint annotations in matching hue (echoes the reference image's labelling)
    def _annotate_endpoints(devs, pcts, color):
        for d, p in [(devs[0], pcts[0]), (devs[-1], pcts[-1])]:
            offset = (10, 8) if p >= 0 else (10, -14)
            ax.annotate(f'{p:+.2f}%', xy=(d, p), xytext=offset,
                        textcoords='offset points', fontsize=9.5,
                        color=color, fontweight='bold')

    _annotate_endpoints(fuel_dev, fuel_pct, fuel_color)
    _annotate_endpoints(carbon_dev, carbon_pct, carbon_color)

    # Highlight baseline point
    ax.scatter([0], [0], s=110, facecolor=HUE_PAIRS['navy'][0], edgecolor='white',
               linewidth=1.8, zorder=5, label='Baseline')

    ax.set_xlabel('Parameter Deviation from Baseline (%)')
    ax.set_ylabel('Change in Total Cost (%)')
    ax.legend(loc='best')
    ax.grid(axis='x', alpha=0.4)

    plt.savefig(os.path.join(base_dir, 'figures', 'f17_sensitivity_tornado.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# F18: Cost Landscape — Carbon Price
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

    # Proxy legend (so we can label both states)
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
# F19 & F20: Monte Carlo Results
# ==========================================
def plot_f19_f20(base_dir):
    mc_dir = base_dir / 'results' / 'montecarlo'
    df = pd.read_csv(mc_dir / 'q2_mc.csv')
    with open(mc_dir / 'summary.json', 'r') as f:
        summary = json.load(f)

    M = len(df)

    # ----- F19: highlight-and-mute histogram (mean-bin saturated) -----
    cost_mean = df['cost'].mean()
    cost_std = df['cost'].std()
    p05 = df['cost'].quantile(0.05)
    p95 = df['cost'].quantile(0.95)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    counts, bin_edges, patches = ax.hist(df['cost'], bins=15,
                                         color=HUE_PAIRS['orange'][1],
                                         edgecolor='white', linewidth=0.8)
    # Saturate the bar containing the mean
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

    # ----- F20: proportion bar (on-time muted, late saturated — late is the key metric) -----
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

    # Load required data
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

    # Execute plotters
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
