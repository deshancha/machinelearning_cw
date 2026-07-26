from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Project:
    id: int
    name: str      
    profit: float  
    cost: float
    dev_hours: float

@dataclass
class AllocationInstance:
    projects: List[Project]
    budget: float
    dev_hours_limit: float
    mutual_exclusions: List[Tuple[int, int]] 
    dependencies: List[Tuple[int, int]]

@dataclass
class AllocationSolution:
    selection: List[int]  # Binary decision (1 = select, 0 = reject)
    total_profit: float
    total_cost: float
    total_labor: float
    is_feasible: bool
    violations: int  # constraint violation count
