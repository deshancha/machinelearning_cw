import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from core.util.logger import ILogger
from core.domain.entities import AllocationSolution, AllocationInstance
from task_2.domain.usecases.data_preparation_usecase import DataPreparationUseCase
from task_3.domain.usecases.genetic_algorithm_usecase import GeneticAlgorithmUseCase
from task_4.domain.usecases.mip_solver_usecase import MipSolverUseCase

class CompareGAandMIPUseCase:
    def __init__(self, logger: ILogger, data_prep: DataPreparationUseCase, 
                 ga_solver: GeneticAlgorithmUseCase, mip_solver: MipSolverUseCase):
        self.logger = logger
        self.data_prep = data_prep
        self.ga_solver = ga_solver
        self.mip_solver = mip_solver

    def run_scalability_test(self, sizes: List[int] = [10, 20, 50, 100, 150, 300, 500, 600, 700, 800, 900, 1000], plot_dir: str = "plots") -> Dict[str, List]:
        self.logger.info("GA vs BIP scalability comparison")
        
        mip_times = []
        ga_times = []
        mip_profits = []
        ga_profits = []
        
        for size in sizes:
            self.logger.info(f"Testing {size} projects")
            instance = self.data_prep.execute(num_projects=size, seed=42, plot_dir=plot_dir)
            
            # MIP
            mip_sol, mip_time = self.mip_solver.find_best_projects(instance)
            mip_times.append(mip_time)
            mip_profits.append(mip_sol.total_profit if mip_sol.is_feasible else 0.0)
            
            # GA
            start_ga = time.time()
            ga_sol, _ = self.ga_solver.findBestProjects(
                instance, pop_size=80, generations=100, crossover_rate=0.8
            )
            ga_time = time.time() - start_ga
            ga_times.append(ga_time)
            ga_profits.append(ga_sol.total_profit if ga_sol.is_feasible else 0.0)
            
        # Plot comparison
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sizes, mip_times, marker='o', color='crimson', label='BILP', linewidth=2)
        ax.plot(sizes, ga_times, marker='s', color='forestgreen', label='GA', linewidth=2)
        
        ax.set_xlabel("Project Size (N)")
        ax.set_ylabel("Time Taken (S)")
        ax.legend()
        
        plt.tight_layout()
        scalability_path = os.path.join(plot_dir, "ga_bilp_comparison.png")
        plt.savefig(scalability_path, dpi=150)
        plt.close()
        
        return {
            "sizes": sizes,
            "mip_times": mip_times,
            "ga_times": ga_times,
            "mip_profits": mip_profits,
            "ga_profits": ga_profits
        }
