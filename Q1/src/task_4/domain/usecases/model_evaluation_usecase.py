import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, roc_curve, confusion_matrix
)
from core.domain.manager.idata_loader import IDataLoader
from core.util.logger import ILogger

class ModelEvaluationUseCase:
    def __init__(self, data_loader: IDataLoader, logger: ILogger):
        self.data_loader = data_loader
        self.logger = logger

    def execute(self, model_dir: str, output_dir: str) -> pd.DataFrame:
        self.logger.info("Executing ModelEvaluationUseCase...")
        os.makedirs(output_dir, exist_ok=True)
        
        df = self.data_loader.load_processed_data()
        
        X = df.drop(columns=['income', 'income_binary'])
        y = df['income_binary']
        
        _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.logger.info(f"Test set: {X_val.shape[0]} rows x {X_val.shape[1]} features.")
        
        # Models
        model_files = {
            'Logistic Regression': 'logistic_regression_model.joblib',
            'KNN': 'knn_model.joblib',
            'Random Forest': 'random_forest_model.joblib',
            'XGBoost': 'xgboost_model.joblib',
            'ANN': 'ann_model.joblib'
        }
        
        # Loading models
        models = {}
        for name, filename in model_files.items():
            path = os.path.join(model_dir, filename)
            if not os.path.exists(path):
                self.logger.error(f"Model file not found: {path}!")
                return
            models[name] = joblib.load(path)
            self.logger.info(f"Loaded model: {name}")
            
        # Dictionaries to log metrics
        metrics = []
        roc_data = {}
        pr_data = {}
        confusion_matrices = {}
        
        # Best model predictions placeholder (for XGBoost)
        best_model_name = "XGBoost"
        best_y_pred = None
        
        # We evalauate now
        for name, clf in models.items():
            self.logger.info(f"Evaluating {name}...")
            
            # predict
            y_pred = clf.predict(X_val)
            y_proba = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else y_pred
            
            # calc classification metrics
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred)
            rec = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            
            # ROC metrics
            fpr, tpr, _ = roc_curve(y_val, y_proba)

            # Receiver Operating Characteristic - Area Under the Curve
            roc_auc = roc_auc_score(y_val, y_proba)
            roc_data[name] = (fpr, tpr, roc_auc)
            
            # PR metrics
            precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_proba)
            pr_auc = auc(recall_vals, precision_vals)
            pr_data[name] = (recall_vals, precision_vals, pr_auc)
            
            # Confusion
            cm = confusion_matrix(y_val, y_pred)
            confusion_matrices[name] = cm
            
            metrics.append({
                'Model': name,
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1,
                'ROC-AUC': roc_auc,
                'PR-AUC': pr_auc
            })
            
            if name == best_model_name:
                best_y_pred = y_pred
                
        metrics_df = pd.DataFrame(metrics).set_index('Model')
        self.logger.info(f"Validation Metrics Table:\n{metrics_df.round(4).to_string()}")
        
        # Save metrics comparison CSV
        csv_path = os.path.join(output_dir, "task4_model_comparison.csv")
        metrics_df.to_csv(csv_path)
        
        # Visualizations
        # ROC Curve
        self.logger.info("ROC Curve")
        plt.figure(figsize=(8, 6))
        for name, (fpr, tpr, auc_val) in roc_data.items():
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})")
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity / Recall)")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        
        roc_path = os.path.join(output_dir, "roc_curves.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()
        
        # PR Curve
        self.logger.info("Precision Recall Curves")
        plt.figure(figsize=(8, 6))
        for name, (rec_vals, prec_vals, auc_val) in pr_data.items():
            plt.plot(rec_vals, prec_vals, label=f"{name} (AP = {auc_val:.4f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        pr_path = os.path.join(output_dir, "pr_curves.png")
        plt.savefig(pr_path, dpi=150)
        plt.close()
        
       
        # Error Analysis
        self.logger.info("Error analysis...")
        
        # Profile misclassifications
        fp_mask = (y_val == 0) & (best_y_pred == 1)
        fn_mask = (y_val == 1) & (best_y_pred == 0)
        
        X_val_dem = X_val.copy()
        X_val_dem['actual'] = y_val
        X_val_dem['predicted'] = best_y_pred
        X_val_dem['is_error'] = X_val_dem['actual'] != X_val_dem['predicted']
        
        X_val_dem['error_type'] = 'Correct'
        X_val_dem.loc[fp_mask, 'error_type'] = 'False Positive'
        X_val_dem.loc[fn_mask, 'error_type'] = 'False Negative'
        
        error_summary_lines = []
        demographic_cols = ['sex', 'edu_group']
        
        for col in demographic_cols:
            if col in X_val_dem.columns:
                self.logger.info(f"Profiling error distributions by category: {col}")
                
                # Overall error counts by category
                err_counts = X_val_dem[X_val_dem['is_error'] == True][col].value_counts()
                total_counts = X_val_dem[col].value_counts()
                
                # Calculate category error rates
                err_rate = (err_counts / total_counts * 100).fillna(0)
                
                # Breakdown of error types inside errors
                fp_counts = X_val_dem[X_val_dem['error_type'] == 'False Positive'][col].value_counts()
                fn_counts = X_val_dem[X_val_dem['error_type'] == 'False Negative'][col].value_counts()
                
                summary_df = pd.DataFrame({
                    'Total Instances': total_counts,
                    'Errors Count': err_counts.fillna(0).astype(int),
                    'Error Rate (%)': err_rate,
                    'False Positives': fp_counts.fillna(0).astype(int),
                    'False Negatives': fn_counts.fillna(0).astype(int)
                }).sort_values(by='Error Rate (%)', ascending=False)
                
                # Export demographics logs to file
                profile_path = os.path.join(output_dir, f"error_profile_{col}.csv")
                summary_df.to_csv(profile_path)
                
                summary_str = f"\n--- Error Profile by '{col}' ---\n" + summary_df.round(2).to_string()
                self.logger.info(summary_str)
                error_summary_lines.append(summary_str)
                
        # Write general logs for error analysis
        with open(os.path.join(output_dir, "error_analysis_summary.txt"), "w") as f:
            f.write("\n".join(error_summary_lines))
            
        self.logger.info("ModelEvaluationUseCase execution complete.")
        return metrics_df
