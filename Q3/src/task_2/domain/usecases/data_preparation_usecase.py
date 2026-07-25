import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.util.logger import ILogger
from core.domain.entities import Project, AllocationInstance

class DataPreparationUseCase:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def execute(self, num_projects: int = 30, seed: int = 42, plot_dir: str = "plots") -> AllocationInstance:
        # projects and logical constraints
        path = "data/projects_data.csv" if os.path.exists("data/projects_data.csv") else "Q3/data/projects_data.csv"
        c_path = "data/logical_constraints.csv" if os.path.exists("data/logical_constraints.csv") else "Q3/data/logical_constraints.csv"
        
        df = pd.read_csv(path)
        
        # If requested > CSV Size, make them dynamically (Scalability part need this)
        # Load cost, profit, dev hours, and name
        costs = df['cost'].values
        dev_hours = df['labor'].values
        profits = df['profit'].values
        names = df['name'].values

        if num_projects <= len(df):
            costs = costs[:num_projects]
            dev_hours = dev_hours[:num_projects]
            profits = profits[:num_projects]
            names = names[:num_projects]
        else:
            # additional cost, dev hours, profit, and name
            ext_costs, ext_dev_hours, ext_profits, ext_names = self.generate_additional_projects(num_projects - len(df), seed)
            costs = np.concatenate([costs, ext_costs])
            dev_hours = np.concatenate([dev_hours, ext_dev_hours])
            profits = np.concatenate([profits, ext_profits])
            names = np.concatenate([names, ext_names])

        projects = [Project(id=i, name=names[i], profit=float(profits[i]), cost=float(costs[i]), dev_hours=float(dev_hours[i])) for i in range(num_projects)]
        
        # Set org capacity to 50% of total projects required
        budget = float(np.round(sum(p.cost for p in projects) * 0.50, 1))
        dev_hours_limit = float(np.round(sum(p.dev_hours for p in projects) * 0.50, 1))

        self.logger.info(f"Loaded {num_projects} projects. Budget: {budget}k, dev hours limit: {dev_hours_limit}h")

        # Load constraints
        df_const = pd.read_csv(c_path)
        mutual_exclusions, dependencies = [], []
        
        for _, row in df_const.iterrows():
            a, b = int(row['project_a']), int(row['project_b'])
            # Filter constraint relative to problem size N
            if a < num_projects and b < num_projects:
                # Mutual exclusion: A or B
                if row['constraint_type'] == "mutual_exclusion":
                    mutual_exclusions.append((a, b))
                # Depend: pick B if A is chosen
                elif row['constraint_type'] == "dependency":
                    dependencies.append((a, b))

        self.plot_proj_distribution(costs, profits, dev_hours, num_projects, plot_dir)

        return AllocationInstance(
            projects=projects,
            budget=budget,
            dev_hours_limit=dev_hours_limit,
            mutual_exclusions=mutual_exclusions,
            dependencies=dependencies
        )

    def generate_additional_projects(self, count: int, seed: int):
        """Gen extra projects for scalability"""
        np.random.seed(seed)
        costs = np.round(np.random.uniform(15.0, 150.0, count), 1)
        dev_hours = np.round(np.random.uniform(20.0, 160.0, count), 1)
        profits = np.round(1.3 * costs + 0.6 * dev_hours + np.random.uniform(-15.0, 35.0, count), 1)
        profits = np.maximum(profits, 5.0)
        names = np.array([f"Extra_{i}" for i in range(count)])
        return costs, dev_hours, profits, names

    def plot_proj_distribution(self, costs, profits, dev_hours, num_projects, plot_dir):
        """Cost vs Profit scatter plot
        - X: cost
        - Y: profit
        - Size: Dev hours compare
        """
        os.makedirs(plot_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(costs, profits, s=dev_hours * 3, alpha=0.7)
        ax.set_title("Project (Cost vs Profit)")
        ax.set_xlabel("Capital Cost ($K)")
        ax.set_ylabel("Expected Profit ($K)")
        ax.grid(True, linestyle='--', alpha=0.5)
        # highlight project id
        for i in range(num_projects):
            ax.annotate(f"P{i}", (costs[i], profits[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "project_distribution.png"), dpi=150)
        plt.close()
