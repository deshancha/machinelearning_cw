import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from core.util.logger import ILogger

class GenerateEdaPlotsUseCase:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def execute(self, df_raw: pd.DataFrame, df_clean: pd.DataFrame, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Numerical columns
        numerical_columns = [
            'age',
            'fnlwgt',
            'education-num',
            'capital-gain',
            'capital-loss',
            'hours-per-week'
        ]
        self.logger.info(f"Generating numerical distributions for: {numerical_columns}")
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.flatten()
        for i, col in enumerate(numerical_columns):
            if col in df_raw.columns:
                df_raw[col].hist(bins=40, ax=axes[i], color='#B22222' if i % 2 == 0 else '#4682B4')
                axes[i].set_title(f'{col} Distribution')
                axes[i].set_ylabel('Frequency')
        plt.tight_layout()
        dist_path = os.path.join(output_dir, 'numerical_distributions.png')
        plt.savefig(dist_path, dpi=150)
        plt.close()
        self.logger.info(f"Saved numerical distributions to {dist_path}")

        # Categorical vs Income (using df_clean)
        self.logger.info("Generating categorical variables vs income plots")
        categorical_columns = ['workclass',
            'education',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'hours_category',
            'edu_group'
            ]
        available_cats = [c for c in categorical_columns if c in df_clean.columns]
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        axes = axes.flatten()
        for i, col in enumerate(available_cats[:9]):
            ct = pd.crosstab(df_clean[col], df_clean['income'])
            ct.plot(kind='bar', stacked=True, color=['#3498db', '#e74c3c'], ax=axes[i])
            axes[i].set_ylabel('Count')
            axes[i].tick_params(axis='x', rotation=45)
            # Shorten labels if they are too long
            labels = [item.get_text()[:15] for item in axes[i].get_xticklabels()]
            axes[i].set_xticklabels(labels)
        cat_path = os.path.join(output_dir, 'categorical_vs_income.png')
        plt.savefig(cat_path, dpi=150)
        plt.close()
        
        self.logger.info("GenerateEdaPlotsUseCase execution complete")
