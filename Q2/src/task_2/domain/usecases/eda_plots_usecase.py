from IPython.core import display_functions
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from core.util.logger import ILogger

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class EdaPlotsUseCase:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def execute_plots(self, df: pd.DataFrame, close_col: str, save_dir: str = "plots") -> None:
        os.makedirs(save_dir, exist_ok=True)
        self.logger.info(f"Generating EDA plots in {save_dir}...")
        
        # 1. Close Price Plot
        plt.figure(figsize=(12, 5))
        plt.plot(df['Date'], df[close_col], label='BTC Close Price', color='royalblue', linewidth=1.5)
        plt.title(f'Bitcoin (BTC) Historical Closing Price', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'btc_close_price.png'), dpi=150)
        plt.close()
        
        # 2. Volume Plot
        volume_col = None
        for col in df.columns:
            if 'Volume' in col:
                volume_col = col
                break
                
        if volume_col:
            plt.figure(figsize=(12, 4))
            plt.plot(df['Date'], df[volume_col], label='Trading Volume', color='royalblue', alpha=0.9, linewidth=1.5)
            plt.title('Bitcoin Historical Trading Volume', fontsize=14)
            plt.xlabel('Date')
            plt.ylabel('Volume')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'btc_volume.png'), dpi=150)
            plt.close()
            
        # 3. Simple Returns Plot (Percentage Change)
        if 'Returns' in df.columns:
            plt.figure(figsize=(12, 4))
            plt.plot(df['Date'], df['Returns'], label='Daily Returns', color='royalblue', linewidth=0.8, alpha=0.8)
            plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
            plt.title('Bitcoin Daily Returns - Percentage Change', fontsize=14)
            plt.xlabel('Date')
            plt.ylabel('Daily Return')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'btc_daily_returns.png'), dpi=150)
            plt.close()
            
        self.logger.info(f"Visualizations saved to directory: {save_dir}")


