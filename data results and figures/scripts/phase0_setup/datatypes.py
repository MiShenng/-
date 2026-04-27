from dataclasses import dataclass, field
from typing import List, Optional

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
    # A split packet of a customer
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
    # the variables z_w, z_v might be lists matched with tasks
    splits_w: List[float] = field(default_factory=list)
    splits_v: List[float] = field(default_factory=list)

@dataclass
class Subsequence:
    D: float # total duration
    E: float # earliest feasible start
    L: float # latest feasible start
    W: float # wait cost
    P: float # late cost
