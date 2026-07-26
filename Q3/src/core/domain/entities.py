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
