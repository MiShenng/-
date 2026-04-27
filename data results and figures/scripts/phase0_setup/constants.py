# constants.py
# Problem parameters

# Time Windows (in minutes from midnight)
BAN_START = 8 * 60  # 480
BAN_END = 16 * 60   # 960

# Costs & Prices
STARTUP_COST = 400.0
WAIT_PENALTY = 1.0 / 3.0  # ¥/min
LATE_PENALTY = 5.0 / 6.0  # ¥/min

PRICE_FUEL = 7.61  # ¥/L
PRICE_ELEC = 1.64  # ¥/kWh

CO2_COST = 0.65  # ¥/kg
ETA_FUEL = 2.547 # kg/L
ETA_ELEC = 0.501 # kg/kWh

# Service time parameters
S0 = 8.0  # minutes fixed
BETA_W = 0.004  # min/kg
BETA_V = 0.40   # min/m^3

# Green zone radius
R_GREEN = 10.0  # km

# Vehicle Types
VEHICLE_SPECS = {
    'F-3000': {'type': 'fuel', 'Q_w': 3000.0, 'Q_v': 13.5, 'count': 60, 'alpha': 0.40},
    'F-1500': {'type': 'fuel', 'Q_w': 1500.0, 'Q_v': 10.8, 'count': 50, 'alpha': 0.40},
    'F-1250': {'type': 'fuel', 'Q_w': 1250.0, 'Q_v': 6.5,  'count': 50, 'alpha': 0.40},
    'E-3000': {'type': 'electric', 'Q_w': 3000.0, 'Q_v': 15.0, 'count': 10, 'alpha': 0.35},
    'E-1250': {'type': 'electric', 'Q_w': 1250.0, 'Q_v': 8.5,  'count': 15, 'alpha': 0.35},
}

# Speed parameters (mu_v, sigma_v) in km/h
SPEED_PARAMS = {
    'smooth': (55.3, 0.12),
    'normal': (35.4, 5.22),
    'congested': (9.8, 4.72)
}

# Time regimes definition (start_min, end_min, regime_name)
# Sorted by start time
REGIMES = [
    (0, 8*60, 'smooth'), # implicitly smooth before 8am (not specified, assuming smooth or normal, let's say smooth)
    (8*60, 9*60, 'congested'),
    (9*60, 10*60, 'smooth'),
    (10*60, 11*60+30, 'normal'),
    (11*60+30, 13*60, 'congested'),
    (13*60, 15*60, 'smooth'),
    (15*60, 17*60, 'normal'),
    (17*60, 19*60, 'congested'), # Wait, report says "17:00-19:00 congested". In the original table it says 8:00-9:00, 11:30-13:00.
    # Ah, the original report said "17:00-19:00 congestion peak" in the executive summary!
    (19*60, 24*60, 'smooth')
]
