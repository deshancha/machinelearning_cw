import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Model classes
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# Imbalance pipeline and oversampler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

from core.domain.manager.idata_loader import IDataLoader
from core.util.logger import ILogger

class TrainAndTuneUseCase:
    def __init__(self, data_loader: IDataLoader, logger: ILogger):
        self.data_loader = data_loader
        self.logger = logger

    def execute(self, model_dir: str, output_summary_path: str) -> pd.DataFrame:
        os.makedirs(model_dir, exist_ok=True)
        
        # Load dataset
        df = self.data_loader.load_processed_data()
        
        # Drop Target columns
        X = df.drop(columns=['income', 'income_binary'])
        y = df['income_binary']
        
        # train-test split (80/20) to define training set shape
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.logger.info(f"Training set shape: {X_train.shape[0]} rows x {X_train.shape[1]} features")
        
        # Numerical and Categorical column types
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        self.logger.info(f"Numerical columns: {num_cols}")
        self.logger.info(f"Categorical columns: {cat_cols}")
        
        # Preprocessor (Standard Scalar and OneHotEncoding)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
            ]
        )
        
        # Dictionary to store metrics for final comparison
        model_results = {}

        # Execute training & tuning for all 5 models
        self._logistic_regression(X_train, y_train, preprocessor, model_dir, model_results)
        self._knn(X_train, y_train, preprocessor, model_dir, model_results)
        self._random_forest(X_train, y_train, preprocessor, model_dir, model_results)
        self._xgboost(X_train, y_train, preprocessor, model_dir, model_results)
        self._ann(X_train, y_train, preprocessor, model_dir, model_results)

        # Create results summary DataFrame (only CV F1-Score)
        results_df = pd.DataFrame(model_results).T
        results_df = results_df[['cv_f1']]
        results_df.columns = ['CV F1-Score']
        
        self.logger.info(f"Summary Table:\n{results_df.round(4).to_string()}")
        
        # Save summary CSV
        os.makedirs(os.path.dirname(output_summary_path), exist_ok=True)
        results_df.to_csv(output_summary_path, index=True)
        self.logger.info(f"Processed data saved successfully {output_summary_path}")
        
        self.logger.info("TrainAndTuneUseCase execution complete.")
        return results_df

    # ---------- Logistic Regression -----------------
    def _logistic_regression(self, X_train, y_train, preprocessor, model_dir, model_results):
        self.logger.info("--- Tuning & Evaluating: Logistic Regression ---")
        lr_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', LogisticRegression(class_weight='balanced', random_state=42))
        ])
        lr_param_dist = {'clf__C': [0.01, 0.1, 1.0, 10.0]}
        
        # n_jobs=-1 use all available cores
        # scoring='f1' for F1 score
        lr_search = RandomizedSearchCV(
            lr_pipeline, param_distributions=lr_param_dist, scoring='f1', n_jobs=-1, random_state=42
        )
        lr_search.fit(X_train, y_train)
        best_lr = lr_search.best_estimator_
        
        self.logger.info(f"Best parameters: {lr_search.best_params_}")
        self.logger.info(f"Best CV F1: {lr_search.best_score_:.4f}")
        
        joblib.dump(best_lr, os.path.join(model_dir, "logistic_regression_model.joblib"))
        model_results['Logistic Regression'] = {
            'cv_f1': lr_search.best_score_
        }

    # ---------- KNN ------------------------------
    def _knn(self, X_train, y_train, preprocessor, model_dir, model_results):
        self.logger.info("--- Tuning & Evaluating: KNN ---")
        knn_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', KNeighborsClassifier())
        ])
        knn_param_dist = {
            'clf__n_neighbors': [5, 9, 15],
            'clf__weights': ['uniform', 'distance']
        }
        
        knn_search = RandomizedSearchCV(
            knn_pipeline, param_distributions=knn_param_dist, scoring='f1', n_jobs=-1, random_state=42
        )
        knn_search.fit(X_train, y_train)
        best_knn = knn_search.best_estimator_
        
        self.logger.info(f"Best parameters: {knn_search.best_params_}")
        self.logger.info(f"Best CV F1: {knn_search.best_score_:.4f}")
        
        joblib.dump(best_knn, os.path.join(model_dir, "knn_model.joblib"))
        model_results['KNN'] = {
            'cv_f1': knn_search.best_score_
        }

    # ---------- Random Forest ---------------------
    def _random_forest(self, X_train, y_train, preprocessor, model_dir, model_results):
        self.logger.info("--- Tuning & Evaluating: Random Forest ---")
        rf_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
        ])
        rf_param_dist = {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [10, 20, None],
            'clf__min_samples_split': [2, 5]
        }
        
        rf_search = RandomizedSearchCV(
            rf_pipeline, param_distributions=rf_param_dist, scoring='f1', n_jobs=-1, random_state=42
        )
        rf_search.fit(X_train, y_train)
        best_rf = rf_search.best_estimator_
        
        self.logger.info(f"Best parameters: {rf_search.best_params_}")
        self.logger.info(f"Best CV F1: {rf_search.best_score_:.4f}")
        
        joblib.dump(best_rf, os.path.join(model_dir, "random_forest_model.joblib"))
        model_results['Random Forest'] = {
            'cv_f1': rf_search.best_score_
        }

    # ---------- XGBoost ---------------------------
    def _xgboost(self, X_train, y_train, preprocessor, model_dir, model_results):
        self.logger.info("--- Tuning & Evaluating: XGBoost ---")
        
        # Class imbalance scale factor for XGBoost
        ratio_scale = (y_train == 0).sum() / (y_train == 1).sum()
        self.logger.info(f"Calculated class ratio scaling weight: {ratio_scale:.3f}")
        
        xgb_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', XGBClassifier(scale_pos_weight=ratio_scale, random_state=42))
        ])
        xgb_param_dist = {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [3, 5, 7],
            'clf__learning_rate': [0.01, 0.1, 0.2]
        }
        
        xgb_search = RandomizedSearchCV(
            xgb_pipeline, param_distributions=xgb_param_dist, scoring='f1', n_jobs=-1, random_state=42
        )
        xgb_search.fit(X_train, y_train)
        best_xgb = xgb_search.best_estimator_
        
        self.logger.info(f"Best parameters: {xgb_search.best_params_}")
        self.logger.info(f"Best CV F1: {xgb_search.best_score_:.4f}")
        
        joblib.dump(best_xgb, os.path.join(model_dir, "xgboost_model.joblib"))
        model_results['XGBoost'] = {
            'cv_f1': xgb_search.best_score_
        }

    # ---------- ANN -------------------------------
    def _ann(self, X_train, y_train, preprocessor, model_dir, model_results):
        self.logger.info("--- Tuning & Evaluating: ANN (MLPClassifier) ---")
        ann_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('sampler', RandomOverSampler(random_state=42)),
            ('clf', MLPClassifier(random_state=42, early_stopping=True))
        ])
        # early_stopping is to save time and stop overfitting
        
        ann_param_dist = {
            'clf__hidden_layer_sizes': [(50,), (100,), (50, 25)],
            'clf__alpha': [0.0001, 0.001, 0.01],
            'clf__activation': ['relu', 'tanh']
        }
        
        ann_search = RandomizedSearchCV(
            ann_pipeline, param_distributions=ann_param_dist, scoring='f1', n_jobs=-1, random_state=42
        )
        ann_search.fit(X_train, y_train)
        best_ann = ann_search.best_estimator_
        
        self.logger.info(f"Best parameters: {ann_search.best_params_}")
        self.logger.info(f"Best CV F1: {ann_search.best_score_:.4f}")
        
        joblib.dump(best_ann, os.path.join(model_dir, "ann_model.joblib"))
        model_results['ANN'] = {
            'cv_f1': ann_search.best_score_
        }
