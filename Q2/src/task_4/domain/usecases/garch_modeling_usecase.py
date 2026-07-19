import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import het_arch
from statsmodels.graphics.tsaplots import plot_acf
from arch import arch_model
from core.util.logger import ILogger
import joblib

sns.set_theme(style="whitegrid")

class GarchModelingUseCase:
    def __init__(self, logger: ILogger):
        self.logger = logger

    def check_arch_effects(self, df: pd.DataFrame, returns_col: str = 'Returns') -> dict:
        self.logger.info("Checking for ARCH effects on returns")
        returns = df[returns_col].dropna()
        
        # Run ARCH-LM test on returns mean 
        _, p_val, _, _ = het_arch(returns - returns.mean(), maxlag=10)
        
        self.logger.info(f"ARCH-LM Test done. p-value: {p_val:.6e}")
        
        # 0.05 is the significance level
        return {
            'p_value': p_val,
            'has_arch_effects': p_val < 0.05
        }

    def fit_garch(self, df: pd.DataFrame, returns_col: str = 'Returns', model_path: str = None) -> dict:
        self.logger.info("Fitting GARCH(1,1) model on scaled returns")
        
        # Scale -> returns x 100 for stability
        returns_scaled = df[returns_col] * 100
        
        # Constant mean GARCH(1,1) with Normal distribution
        model = arch_model(returns_scaled, mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
        res = model.fit()
        
        # Parameters
        alpha = res.params['alpha[1]']
        beta = res.params['beta[1]']
        persistence = alpha + beta
        
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(res, model_path)
        
        self.logger.info(f"GARCH Fit: alpha={alpha:.4f}, beta={beta:.4f}, persistence={persistence:.4f}")
        
        return {
            'alpha': alpha, 'beta': beta, 'persistence': persistence
        }