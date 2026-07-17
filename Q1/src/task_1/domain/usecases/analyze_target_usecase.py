import os
import pandas as pd
import matplotlib.pyplot as plt
from core.domain.manager.idata_loader import IDataLoader
from core.util.logger import ILogger

class AnalyzeTargetUseCase:
    def __init__(self, data_loader: IDataLoader, logger: ILogger):
        self.data_loader = data_loader
        self.logger = logger

    def execute(self, output_dir: str) -> None:
        self.logger.info("Executing AnalyzeTargetUseCase...")
        df = self.data_loader.load_raw_data()
        
        # Determine income class column
        target_col = df.columns[-1] # Target column is the last column
        class_counts = df[target_col].value_counts()
        
        self.logger.info(f"Class distribution counts:\n{class_counts}")
        
        max_val = class_counts.max()
        min_val = class_counts.min()
        ratio = max_val / min_val
        self.logger.info(f"Class imbalance ratio: {ratio:.2f}:1")
        
        # Plot target distribution
        os.makedirs(output_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        class_counts.plot(kind='bar', color=['#3498db', '#e74c3c'], ax=ax)
        ax.set_title('Income Distribution')
        ax.set_ylabel('Count')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'target_distribution.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        self.logger.info(f"Saved income distribution plot to {plot_path}")
        self.logger.info("AnalyzeTargetUseCase execution complete.")
