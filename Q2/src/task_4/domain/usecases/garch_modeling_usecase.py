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

    def check_arch_effects(self, df: pd.DataFrame, returns_col: str = 'Returns', save_dir: str = "plots") -> dict:
        os.makedirs(save_dir, exist_ok=True)
        self.logger.info("Checking for ARCH effects on returns...")
        returns = df[returns_col].dropna()
        
        # 1. Rolling Volatility Plot
        rolling_std = returns.rolling(window=30).std() * np.sqrt(365) * 100
        plt.figure(figsize=(12, 4))
        plt.plot(df['Date'], rolling_std, color='darkred', label='30-day Rolling Volatility (Annualized %)')
        plt.title('Bitcoin 30-day Rolling Volatility Over Time', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Volatility (%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'btc_rolling_volatility.png'), dpi=150)
        plt.close()
        
        # 2. Engle's ARCH test on returns
        mean_adjusted_returns = returns - returns.mean()
        lm_stat, p_val, f_stat, f_pval = het_arch(mean_adjusted_returns, maxlag=10)
        
        self.logger.info("Engle's ARCH-LM Test for Conditional Heteroscedasticity completed.")
        self.logger.info(f"Lag-10 LM Statistic: {lm_stat:.6f} | p-value: {p_val:.6e}")
        
        has_arch_effects = p_val < 0.05
        
        # 3. Autocorrealtion of Squared Returns Plot
        plt.figure(figsize=(12, 4))
        plot_acf(mean_adjusted_returns ** 2, lags=40, ax=plt.gca(), color='purple', title='Autocorrelation of Squared Returns')
        plt.xlabel('Lags')
        plt.ylabel('Autocorrelation')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'btc_squared_returns_acf.png'), dpi=150)
        plt.close()
        
        return {
            'lm_stat': lm_stat,
            'p_value': p_val,
            'has_arch_effects': has_arch_effects
        }

    def fit_garch(self, df: pd.DataFrame, returns_col: str = 'Returns', save_dir: str = "plots", model_path: str = None) -> dict:
        os.makedirs(save_dir, exist_ok=True)
        self.logger.info("Fitting GARCH model on returns")
        
        # Scale returns by 100 to avoid optimization issues
        returns_scaled = df[returns_col] * 100
        
        model = arch_model(returns_scaled, mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
        res = model.fit(disp='off')
        
        omega = res.params['omega']
        alpha = res.params['alpha[1]']
        beta = res.params['beta[1]']
        mu = res.params['mu']
        
        persistence = alpha + beta
        if persistence < 1.0:
            half_life = np.log(0.5) / np.log(persistence)
        else:
            half_life = np.inf
            
        self.logger.info(f"GARCH model results: alpha={alpha:.4f}, beta={beta:.4f}, persistence={persistence:.4f}, half_life={half_life}")
        
        cond_vol = res.conditional_volatility / 100.0
        
        # Volatility Forecast vs Actual returns plot
        plt.figure(figsize=(12, 5))
        plt.plot(df['Date'], df[returns_col] * 100, label='Daily Returns (%)', color='lightskyblue', alpha=0.7, linewidth=0.8)
        plt.plot(df['Date'], res.conditional_volatility, label='GARCH Conditional Volatility Forecast (%)', color='navy', linewidth=1.5)
        plt.plot(df['Date'], -res.conditional_volatility, color='navy', linestyle='--', linewidth=1.0)
        plt.title('Bitcoin Daily Returns vs GARCH Conditional Volatility Bounds', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Returns / Volatility (%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'btc_garch_volatility_fit.png'), dpi=150)
        plt.close()
        
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(res, model_path)
            self.logger.info(f"Saved GARCH model to {model_path}")

        garch_metrics = {
            'omega': omega,
            'alpha': alpha,
            'beta': beta,
            'persistence': persistence,
            'half_life': half_life,
            'aic': res.aic,
            'bic': res.bic,
            'model_res': res,
            'conditional_volatility': cond_vol
        }
        
        return garch_metrics
