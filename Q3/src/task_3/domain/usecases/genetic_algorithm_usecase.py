import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from core.util.logger import ILogger
from core.domain.entities import Project, AllocationInstance, AllocationSolution

class GeneticAlgorithmUseCase:
    def __init__(self, logger: ILogger):
        self.logger = logger

    # uning_results = ga_solver_usecase.hyperParamTune(instance, pop_sizes=[50, 100, 200, 300], gens=[50, 100, 150, 200])
    def hyperParamTune(self, instance: AllocationInstance, pop_sizes: List[int], gens: List[int], seed: int = 42) -> Dict[str, Any]:
        """
        GA and hyper param tuning with dynamic pop size and generation number
        """
        best_profit = -1.0
        best_params = {}
        
        results = []
        for pop_size in pop_sizes:
            for gen in gens:
                sol, _ = self.findBestProjects(instance, pop_size=pop_size, generations=gen, seed=seed)
                results.append({
                    "pop_size": pop_size,
                    "generations": gen,
                    "profit": sol.total_profit,
                    "feasible": sol.is_feasible
                })
                # feasible and profit is greater -> update best param
                if sol.is_feasible and sol.total_profit > best_profit:
                    best_profit = sol.total_profit
                    best_params = {"pop_size": pop_size, "generations": gen}
                    
        self.logger.info(f"Hyper Param Tuning Done, Best params: {best_params} with profit: {best_profit:.1f}k")
        return {"best_params": best_params, "best_profit": best_profit, "all_results": results}

    def findBestProjects(self, instance: AllocationInstance, pop_size: int = 100, generations: int = 150, crossover_rate: float = 0.8, seed: int = 42) -> Tuple[AllocationSolution, List[float]]:
        if seed is not None:
            np.random.seed(seed)
        N = len(instance.projects)
        mutation_rate = 1.0 / N
        tournament_size = 3
            
        self.logger.info(f"GA: pop_size={pop_size}, generations={generations}, mutation_rate={mutation_rate:.4f}...")
        
        # Multiple chromosomes of solutions
        population = [list(np.random.randint(0, 2, N)) for _ in range(pop_size)]
        
        history_best = []
        
        best_overall_chromosome = None
        best_overall_fitness = -float('inf')
        
        for gen in range(generations):
            # Calculate fitnesses
            evaluated = [self.evaluate_fitness(chrom, instance) for chrom in population]
            fitnesses = [e[0] for e in evaluated]
            
            # generation metrics
            gen_best_idx = np.argmax(fitnesses)
            gen_best_fitness = fitnesses[gen_best_idx]
            
            history_best.append(gen_best_fitness)
            
            if gen_best_fitness > best_overall_fitness:
                best_overall_fitness = gen_best_fitness
                best_overall_chromosome = population[gen_best_idx].copy()
                
            # next generation, (with/without elitism)
            new_population = self.elitism(population, fitnesses) # []
                
            # Crossover & Mutation
            while len(new_population) < pop_size:
                parent1 = self.tournament_selection(population, fitnesses, tournament_size)
                parent2 = self.tournament_selection(population, fitnesses, tournament_size)
                
                if np.random.rand() < crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                    
                child1 = self.mutate(child1, mutation_rate)
                child2 = self.mutate(child2, mutation_rate)
                
                new_population.append(child1)
                if len(new_population) < pop_size:
                    new_population.append(child2)
                    
            population = new_population

        # Get final solution details
        _, profit, cost, labor, violations = self.evaluate_fitness(best_overall_chromosome, instance)

        is_feasible = (
            violations == 0 
            and cost <= instance.budget + 1e-5 
            and labor <= instance.dev_hours_limit + 1e-5
        )
        
        solution = AllocationSolution(
            selection=best_overall_chromosome,
            total_profit=profit,
            total_cost=cost,
            total_labor=labor,
            is_feasible=is_feasible,
            violations=violations
        )
        
        self.logger.info(f"GA Complete. Best Profit: {profit:.1f}k, Feasible: {is_feasible}")
        return solution, history_best


    def evaluate_fitness(self, chromosome: List[int], instance: AllocationInstance) -> Tuple[float, float, float, int, List[str]]:
        total_profit = 0.0
        total_cost = 0.0
        total_labor = 0.0
        violation_count = 0
        
        for idx, val in enumerate(chromosome):
            if val == 1:
                p = instance.projects[idx]
                total_profit += p.profit
                total_cost += p.cost
                total_labor += p.dev_hours

        # mutual exclusion constraints
        for a, b in instance.mutual_exclusions:
            if chromosome[a] == 1 and chromosome[b] == 1:
                violation_count += 1
        
        # dependency constraints
        for a, b in instance.dependencies:
            if chromosome[a] == 1 and chromosome[b] == 0:
                violation_count += 1

        # resource constraints penalties
        if total_cost > instance.budget:
            violation_count += 1
            
        if total_labor > instance.dev_hours_limit:
            violation_count += 1

        # High penalty for logical + resource vialation, 500 - this is greater than the profit of all single project (100 to 250)
        penalty = 500.0 * violation_count

        fitness = total_profit - penalty
        
        return fitness, total_profit, total_cost, total_labor, violation_count

    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """
        mixing 2 parenets -> create 2 children, child inherits best traits. uses uniform corss(coin toss)
        """
        child1 = parent1.copy()
        child2 = parent2.copy()
        for i in range(len(parent1)):
            if np.random.rand() < 0.5:
                child1[i], child2[i] = parent2[i], parent1[i] # crossed
        return child1, child2

    def mutate(self, chromosome: List[int], mutation_rate: float) -> List[int]:
        """
        Small Random changes to prevent algo stuck in local trap
        """
        mutated = chromosome.copy()
        for i in range(len(mutated)):
            if np.random.rand() < mutation_rate:
                mutated[i] = 1 - mutated[i]  # flip the bit (1->0 or 0->1)
        return mutated

    def tournament_selection(self, population: List[List[int]], fitnesses: List[float], tournament_size: int = 3) -> List[int]:
        """
        Select an best individual from the population for crossover
        """
        indices = np.random.choice(len(population), tournament_size, replace=False)
        best_idx = indices[0]
        for idx in indices[1:]:
            if fitnesses[idx] > fitnesses[best_idx]:
                best_idx = idx
        return population[best_idx].copy()

    # next generation, (with/without elitism)
    # new_population = self.elitism(population, fitnesses) # []

    def elitism(self, population: List[List[int]], fitnesses: List[float]) -> List[List[int]]:
        """
        Copying best individual to next generation, without we wont see continous improvement, but up and down
        """
        best_idx = np.argmax(fitnesses)
        return [population[best_idx].copy()]
    

    def plot_results(self, instance: AllocationInstance, solution: AllocationSolution, 
                     history_best: List[float], plot_dir: str = "plots"):
        os.makedirs(plot_dir, exist_ok=True)
        
        # Project Selection 
        fig, ax = plt.subplots(figsize=(10, 5))
        project_ids = [p.id for p in instance.projects]
        profits = [p.profit for p in instance.projects]
        colors = ['forestgreen' if val == 1 else 'crimson' for val in solution.selection]
        
        ax.bar(project_ids, profits, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.set_title(f"Selected {sum(solution.selection)} / {len(solution.selection)} projects)")
        ax.set_xlabel("Proj ID")
        ax.set_ylabel("Profit ($k)")
        ax.set_xticks(project_ids)
        ax.set_xticklabels([f"{i}" for i in project_ids])
        
        plt.tight_layout()
        selection_path = os.path.join(plot_dir, "ga_portfolio_selection.png")
        plt.savefig(selection_path, dpi=150)
        plt.close()
        
        # Best Fitness Curve
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(history_best, label='Best Fitness', color='forestgreen')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Profit (Fitness)')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        conv_path = os.path.join(plot_dir, "ga_best_fit.png")
        plt.savefig(conv_path, dpi=150)
        plt.close()

    