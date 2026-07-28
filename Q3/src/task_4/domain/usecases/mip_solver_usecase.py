import os
import time
import pulp
import matplotlib.pyplot as plt
from typing import Tuple
from core.util.logger import ILogger
from core.domain.entities import Project, AllocationInstance, AllocationSolution

class MipSolverUseCase:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def find_best_projects(self, instance: AllocationInstance) -> Tuple[AllocationSolution, float]:
        """
        MIP with PuLP (BILP here)
        """
        # Problem is find Max profit which is the optimization here
        prob = pulp.LpProblem("Project Portfolio Selection to Maximize Profit", pulp.LpMaximize)
        
        # binary decision var to select/not project
        N = len(instance.projects)
        x = pulp.LpVariable.dicts("project_select", range(N), cat=pulp.LpBinary)
        
        # function is to sum up profit
        prob += pulp.lpSum([x[i] * p.profit for i, p in enumerate(instance.projects)])
        
        # Budget limit constraint
        prob += pulp.lpSum([x[i] * p.cost for i, p in enumerate(instance.projects)]) <= instance.budget, "Budget_Limit"
        
        # dev hors limit constraint
        prob += pulp.lpSum([x[i] * p.dev_hours for i, p in enumerate(instance.projects)]) <= instance.dev_hours_limit, "Dev_Hours_Limit"
        
        # Setup logical constraints
        # Mutex
        for idx, (a, b) in enumerate(instance.mutual_exclusions):
            prob += x[a] + x[b] <= 1, f"Mutual_Exclusion_{idx}"
            
        # Dep
        for idx, (a, b) in enumerate(instance.dependencies):
            prob += x[a] <= x[b], f"Dependency_{idx}"
            
        start_time = time.time()
        status = prob.solve()
        solve_time = time.time() - start_time
        
        # extract solution
        selection = [int(pulp.value(x[i])) for i in range(N)]
        
        total_profit = 0.0
        total_cost = 0.0
        total_labor = 0.0
        violations = []
        
        for i, val in enumerate(selection):
            if val == 1:
                p = instance.projects[i]
                total_profit += p.profit
                total_cost += p.cost
                total_labor += p.dev_hours
                
        # Validate feasibility (Double-check with tolerance for floating point summation precision)
        if total_cost > instance.budget + 1e-5:
            violations.append(f"Budget exceeded: {total_cost} > {instance.budget}")
        if total_labor > instance.dev_hours_limit + 1e-5:
            violations.append(f"Dev Hours exceeded: {total_labor} > {instance.dev_hours_limit}")
            
        for a, b in instance.mutual_exclusions:
            if selection[a] == 1 and selection[b] == 1:
                violations.append(f"Mutex : Both Projects {a},{b} selected")
                
        for a, b in instance.dependencies:
            if selection[a] == 1 and selection[b] == 0:
                violations.append(f"Dep : Project {a} chosen, but {b} not")
                
        is_ok = len(violations) == 0 and pulp.LpStatus[status] == "Optimal"
        
        solution = AllocationSolution(
            selection=selection,
            total_profit=total_profit,
            total_cost=total_cost,
            total_labor=total_labor,
            is_feasible=is_ok,
            violations=violations
        )
        
        self.logger.info(f"Time taken: {solve_time:.4f}s. Status: {pulp.LpStatus[status]}, OK: {is_ok}")
        return solution, solve_time

    def plot_results(self, instance: AllocationInstance, solution: AllocationSolution, plot_dir: str = "plots"):
        os.makedirs(plot_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        project_ids = [p.id for p in instance.projects]
        profits = [p.profit for p in instance.projects]
        colors = ['forestgreen' if val == 1 else 'crimson' for val in solution.selection]
        
        bars = ax.bar(project_ids, profits, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_title(f"(Selected: {sum(solution.selection)} / {len(solution.selection)} projects)")
        ax.set_xlabel("ID")
        ax.set_ylabel("Profit ($k)")
        ax.set_xticks(project_ids)
        
        selection_path = os.path.join(plot_dir, "mip_portfolio_selection.png")
        plt.savefig(selection_path, dpi=150)
        plt.close()
