
import pandas as pd
from core.domain.manager.idata_loader import IDataLoader
from core.util.logger import ILogger

class FeatureEngineeringUseCase:
    def __init__(self, data_loader: IDataLoader, logger: ILogger):
        self.data_loader = data_loader
        self.logger = logger

    def execute(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        
        # 1. capital_net
        df_clean['capital_net'] = df_clean['capital-gain'] - df_clean['capital-loss']
        self.logger.info("Created feature 'capital_net' = capital-gain - capital-loss")
        
        # 2. has_capital
        df_clean['has_capital'] = ((df_clean['capital-gain'] > 0) | (df_clean['capital-loss'] > 0)).astype(int)
        self.logger.info("Created binary feature 'has_capital' (gain or loss)")
        
        # 3. age_group
        df_clean['age_group'] = pd.cut(df_clean['age'],
                                       bins=[0, 25, 35, 45, 55, 65, 100],
                                       labels=['Young', 'Early-Career', 'Mid-Career',
                                               'Senior', 'Pre-Retire', 'Retired'])
        self.logger.info("Created binned categorical feature 'age_group'")
        
        # 4. hours_category
        df_clean['hours_category'] = pd.cut(df_clean['hours-per-week'],
                                             bins=[0, 34, 40, 100],
                                             labels=['Part-Time', 'Full-Time', 'Overtime'])
        self.logger.info("Created binned categorical feature 'hours_category'")
        
        # 5. edu_group
        edu_map = {
            'Preschool': 'Low', '1st-4th': 'Low', '5th-6th': 'Low', '7th-8th': 'Low',
            '9th': 'Low', '10th': 'Low', '11th': 'Low', '12th': 'Low',
            'HS-grad': 'High-School',
            'Some-college': 'Some-College', 'Assoc-voc': 'Some-College', 'Assoc-acdm': 'Some-College',
            'Bachelors': 'Bachelors',
            'Masters': 'Advanced', 'Prof-school': 'Advanced', 'Doctorate': 'Advanced'
        }
        df_clean['edu_group'] = df_clean['education'].map(edu_map)
        self.logger.info("Created simplified grouped feature 'edu_group'")
        
        # Save processed dataset
        self.data_loader.save_processed_data(df_clean)
        self.logger.info(f"aved to processed location. Shape: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} columns.")
        return df_clean
