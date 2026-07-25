from dataclasses import dataclass


@dataclass
class Project:
    id: int
    name: str      
    profit: float  
    cost: float
    dev_hours: float
