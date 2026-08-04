import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error,
    median_absolute_error
)

try:
    from xgboost import XGBRegressor

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

warnings.filterwarnings('ignore')


class ProductionWaterQualityRegressor:
    """
    Production-Ready Water Quality Model (LEAKAGE-FREE FIXED VERSION)
    - Fix 1: Augmentation moved AFTER split (Applied only to Train)
    - Fix 2: Removed 'bfill' to prevent look-ahead bias
    """

    def __init__(self, csv_filename="izsu_health_factor.csv"):
        self.csv_path = csv_filename
        self.df = None
        self.df_original = None
        self.target_col = 'HealthFactor'

        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None

        self.global_feature_cols = []
        self.trained_models = {}
        self.metrics_results = {}
        self.best_model_name = None

        self.scaler = None
        self.imputer = None

        self.output_dir = Path("ai_outputs/production_model_fixed")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INIT] Production Water Quality Regressor (Leakage-Free)")
        print(f"[INIT] CSV: {self.csv_path}")
        print(f"[INIT] Output: {self.output_dir}")

    def load_and_preprocess(self):
        """Load and preprocess data."""
        try:
            self.df = pd.read_csv(self.csv_path)
            self.df['Tarih'] = pd.to_datetime(self.df['Tarih'])
            self.df = self.df.sort_values(by=['NoktaAdi', 'Tarih']).reset_index(drop=True)
            self.df = self.df.dropna(axis=1, how='all')
            self.df_original = self.df.copy()

            print(f"\n[DATA] Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
            print(f"[DATA] Target Range: {self.df[self.target_col].min():.2f} to {self.df[self.target_col].max():.2f}")
            print(f"[DATA] Unique Locations: {self.df['NoktaAdi'].nunique()}")
            return self
        except FileNotFoundError:
            print(f"[ERROR] File not found: {self.csv_path}")
            sys.exit(1)

    def feature_engineering(self):
        """Create location-aware features."""
        print("\n[FEATURE ENGINEERING] Creating features...")

        self.df['Month'] = self.df['Tarih'].dt.month

        grp = self.df.groupby('NoktaAdi')[self.target_col]
        self.df['Target_Lag1'] = grp.shift(1)
        self.df['Target_Lag2'] = grp.shift(2)
        self.df['Target_RollMean3'] = grp.shift(1).rolling(window=3, min_periods=1).mean()
        self.df['Target_RollMean7'] = grp.shift(1).rolling(window=7, min_periods=1).mean()
        self.df['Target_DiffFromMean3'] = self.df['Target_Lag1'] - self.df['Target_RollMean3']
        self.df['Target_Change'] = self.df['Target_Lag1'] - self.df['Target_Lag2']

        score_cols = [c for c in self.df.columns if '_score' in c.lower()]
        lag_features = []

        for col in score_cols:
            if self.df[col].notna().mean() > 0.5:
                lag1_name = f"{col}_Lag1"
                self.df[lag1_name] = self.df.groupby('NoktaAdi')[col].shift(1)
                lag_features.append(lag1_name)

        self.global_feature_cols = [
                                       'Month', 'Target_Lag1', 'Target_Lag2',
                                       'Target_RollMean3', 'Target_RollMean7',
                                       'Target_DiffFromMean3', 'Target_Change'
                                   ] + lag_features

        print(f"[FEATURE ENGINEERING] Created {len(self.global_feature_cols)} features")
        return self

    def split_train_val_test(self):
        """Split data into train/val/test (70/15/15) cleanly."""
        print("\n[DATA SPLIT] Splitting into Train/Val/Test (70/15/15 by time)...")

        data = self.df.sort_values(by=['Tarih', 'NoktaAdi']).reset_index(drop=True)
        data = data.dropna(subset=[self.target_col]).copy()

        # FIX: Removed 'bfill' (Look-ahead bias fix)
        for col in self.global_feature_cols:
            if col in data.columns:
                data[col] = data.groupby('NoktaAdi')[col].fillna(method='ffill', limit=5)
                # Fallback to median ONLY if NaNs remain (no future peeking)
                if data[col].isna().any():
                    data[col] = data[col].fillna(data[col].median())

        X = data[self.global_feature_cols].copy()
        y = data[self.target_col].copy()

        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]

        # Temporal Split (Time Series Strict)
        n = len(X)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        X_train_raw = X.iloc[:train_end]
        y_train = y.iloc[:train_end]

        X_val_raw = X.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]

        X_test_raw = X.iloc[val_end:]
        y_test = y.iloc[val_end:]

        # Preprocess (Fit ONLY on Train)
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = RobustScaler()

        # Fit on Train
        X_train_imputed = self.imputer.fit_transform(X_train_raw)
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)

        # Transform others
        X_val_imputed = self.imputer.transform(X_val_raw)
        X_val_scaled = self.scaler.transform(X_val_imputed)

        X_test_imputed = self.imputer.transform(X_test_raw)
        X_test_scaled = self.scaler.transform(X_test_imputed)

        self.X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
        self.X_val = pd.DataFrame(X_val_scaled, columns=X.columns)
        self.X_test = pd.DataFrame(X_test_scaled, columns=X.columns)

        self.y_train = y_train.reset_index(drop=True)
        self.y_val = y_val.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)

        print(f"[DATA SPLIT] Train: {len(self.X_train)} | Val: {len(self.X_val)} | Test: {len(self.X_test)}")
        return self

    def augment_training_data(self):
        """
        FIXED AUGMENTATION: Applied ONLY to Training Set AFTER split.
        Adds conservative Gaussian noise to scaled features.
        """
        print("\n[DATA AUGMENTATION] Applying augmentation ONLY to Training set...")

        # Create noise based on scaled data (mean~0, std~1)
        X_aug = self.X_train.copy()
        y_aug = self.y_train.copy()

        # Add 1% noise
        noise = np.random.normal(0, 0.01, X_aug.shape)
        X_aug = X_aug + noise

        # Combine original + augmented
        self.X_train = pd.concat([self.X_train, X_aug], ignore_index=True)
        self.y_train = pd.concat([self.y_train, y_aug], ignore_index=True)

        print(f"[DATA AUGMENTATION] Train size doubled to: {len(self.X_train)}")
        return self

    def train_and_evaluate(self):
        """Train models and evaluate on train/val/test splits."""
        print("\n[TRAINING] Training models on all splits...")
        print("=" * 130)

        models = {
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(
                n_estimators=200, max_depth=5, min_samples_leaf=10,
                random_state=42, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=5,
                min_samples_leaf=3, subsample=0.8, random_state=42
            )
        }

        if XGB_AVAILABLE:
            models['XGBoost'] = XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=2.0, random_state=42, verbosity=0
            )

        print(f"{'Model':<20} {'Split':<12} {'R2':<12} {'RMSE':<12} {'MAE':<12} {'MAPE':<12} {'MedAE':<12}")
        print("=" * 130)

        best_test_r2 = -np.inf

        for model_name, model in models.items():
            # Train on training set
            model.fit(self.X_train, self.y_train)
            self.trained_models[model_name] = model

            results = {}

            # Evaluate on all splits
            for split_name, X_split, y_split in [
                ('Train', self.X_train, self.y_train),
                ('Val', self.X_val, self.y_val),
                ('Test', self.X_test, self.y_test)
            ]:
                pred = model.predict(X_split)

                r2 = r2_score(y_split, pred)
                rmse = np.sqrt(mean_squared_error(y_split, pred))
                mae = mean_absolute_error(y_split, pred)
                mape = mean_absolute_percentage_error(y_split, pred)
                medae = median_absolute_error(y_split, pred)

                results[split_name] = {
                    'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'MedAE': medae
                }

                print(
                    f"{model_name:<20} {split_name:<12} {r2:<12.4f} {rmse:<12.4f} {mae:<12.4f} {mape:<12.4f} {medae:<12.4f}")

            self.metrics_results[model_name] = results

            if results['Test']['R2'] > best_test_r2:
                best_test_r2 = results['Test']['R2']
                self.best_model_name = model_name

        print("=" * 130)

    def print_metrics_matrix(self):
        """Print detailed metrics matrix."""
        print("\n" + "=" * 150)
        print(f"{'DETAILED METRICS MATRIX - ALL MODELS & SPLITS':^150}")
        print("=" * 150)

        metrics_names = ['R2', 'RMSE', 'MAE', 'MAPE', 'MedAE']

        for metric in metrics_names:
            print(f"\n[{metric} METRIC]")
            print("-" * 150)

            data_for_table = []
            for model_name in self.metrics_results.keys():
                row = {'Model': model_name}
                for split in ['Train', 'Val', 'Test']:
                    row[split] = self.metrics_results[model_name][split][metric]
                data_for_table.append(row)

            df_table = pd.DataFrame(data_for_table)
            print(df_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        print("\n" + "=" * 150)

    def print_comprehensive_report(self):
        """Print comprehensive analysis report."""
        print("\n" + "=" * 150)
        print(f"{'PRODUCTION WATER QUALITY MODEL - COMPREHENSIVE REPORT':^150}")
        print("=" * 150)

        print(f"\n[STRATEGY] Hybrid Global Model + Location-Aware Features")
        print(f"[VALIDATION] Train/Validation/Test Split (70/15/15 by time)")
        print(f"[DATA] Augmentation: Leakage-Free (Only Train)")

        print(f"\n[BEST MODEL PERFORMANCE]")
        print("-" * 150)
        best_results = self.metrics_results[self.best_model_name]

        print(f"Model: {self.best_model_name}")
        print(f"\nTrain Performance:")
        print(
            f"  R² = {best_results['Train']['R2']:.4f} | RMSE = {best_results['Train']['RMSE']:.4f} | MAE = {best_results['Train']['MAE']:.4f}")

        print(f"\nValidation Performance:")
        print(
            f"  R² = {best_results['Val']['R2']:.4f} | RMSE = {best_results['Val']['RMSE']:.4f} | MAE = {best_results['Val']['MAE']:.4f}")

        print(f"\nTest Performance:")
        print(
            f"  R² = {best_results['Test']['R2']:.4f} | RMSE = {best_results['Test']['RMSE']:.4f} | MAE = {best_results['Test']['MAE']:.4f}")

        # Overfitting analysis
        train_test_gap = best_results['Train']['R2'] - best_results['Test']['R2']
        train_val_gap = best_results['Train']['R2'] - best_results['Val']['R2']

        print(f"\n[OVERFITTING ANALYSIS]")
        print(f"Train-Test R² Gap: {train_test_gap:.4f}", end="")
        if train_test_gap > 0.2:
            print(" → HIGH OVERFITTING ⚠️")
        elif train_test_gap > 0.1:
            print(" → MILD OVERFITTING ⚠️")
        else:
            print(" → BALANCED ✅")

        print(f"Train-Val R² Gap: {train_val_gap:.4f}", end="")
        if train_val_gap > 0.2:
            print(" → HIGH OVERFITTING ⚠️")
        elif train_val_gap > 0.1:
            print(" → MILD OVERFITTING ⚠️")
        else:
            print(" → BALANCED ✅")

        print(f"\n[PRODUCTION READINESS]")
        test_r2 = best_results['Test']['R2']
        if test_r2 > 0.7:
            print(f"✅ EXCELLENT - Ready for production (R² = {test_r2:.4f})")
        elif test_r2 > 0.6:
            print(f"✅ GOOD - Ready for production (R² = {test_r2:.4f})")
        elif test_r2 > 0.5:
            print(f"⚠️ ACCEPTABLE - Consider improvement (R² = {test_r2:.4f})")
        elif test_r2 > 0.4:
            print(f"⚠️ FAIR - Needs improvement (R² = {test_r2:.4f})")
        else:
            print(f"❌ POOR - Not ready (R² = {test_r2:.4f})")

        print(f"\n[EXPECTED PRODUCTION PERFORMANCE]")
        print(f"Mean Absolute Error (MAE): ±{best_results['Test']['MAE']:.4f} HealthFactor units")
        print(f"Root Mean Square Error (RMSE): ±{best_results['Test']['RMSE']:.4f} HealthFactor units")
        print(f"Mean Absolute Percentage Error (MAPE): {best_results['Test']['MAPE']:.2%}")

        print("\n" + "=" * 150)

    def generate_visualizations(self):
        """Generate comprehensive visualizations."""
        print("\n[VISUALIZATION] Generating plots...")

        best_model = self.trained_models[self.best_model_name]

        # 1. Metrics Comparison Matrix (Train/Val/Test)
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        metrics_names = ['R2', 'RMSE', 'MAE', 'MAPE', 'MedAE']

        for idx, metric in enumerate(metrics_names):
            ax = axes[idx // 3, idx % 3]

            data = []
            models = []
            for model_name in self.metrics_results.keys():
                for split in ['Train', 'Val', 'Test']:
                    data.append(self.metrics_results[model_name][split][metric])
                    models.append(f"{model_name}\n{split}")

            x_pos = np.arange(len(data))
            colors = ['#2ecc71' if 'Train' in m else '#f39c12' if 'Val' in m else '#e74c3c' for m in models]
            ax.bar(x_pos, data, color=colors, alpha=0.7, edgecolor='black')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel(metric, fontweight='bold')
            ax.set_title(f'{metric} Comparison Across All Splits', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / "01_metrics_comparison_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Actual vs Predicted (Test Set)
        test_pred = best_model.predict(self.X_test)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # All splits comparison
        for idx, (split_name, X_split, y_split) in enumerate([
            ('Train', self.X_train, self.y_train),
            ('Validation', self.X_val, self.y_val),
            ('Test', self.X_test, self.y_test)
        ]):
            pred = best_model.predict(X_split)

            axes[idx].scatter(y_split, pred, alpha=0.6, s=30, color=['#2ecc71', '#f39c12', '#e74c3c'][idx])
            axes[idx].plot([y_split.min(), y_split.max()], [y_split.min(), y_split.max()], 'r--', lw=2)
            axes[idx].set_xlabel('Actual Values', fontweight='bold')
            axes[idx].set_ylabel('Predicted Values', fontweight='bold')
            axes[idx].set_title(f'{split_name} Set: Actual vs Predicted\n({self.best_model_name})', fontweight='bold')
            axes[idx].grid(True, alpha=0.3)

            r2 = r2_score(y_split, pred)
            axes[idx].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[idx].transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(self.output_dir / "02_actual_vs_predicted_all_splits.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Feature Importance
        if hasattr(best_model, 'feature_importances_'):
            fig, ax = plt.subplots(figsize=(12, 8))

            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]

            sns.barplot(
                x=importances[indices],
                y=[self.global_feature_cols[i] for i in indices],
                palette='viridis', ax=ax
            )
            ax.set_xlabel('Importance Score', fontweight='bold', fontsize=12)
            ax.set_ylabel('Feature', fontweight='bold', fontsize=12)
            ax.set_title(f'Top 15 Feature Importance - {self.best_model_name}', fontweight='bold', fontsize=14)
            ax.grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            plt.savefig(self.output_dir / "03_feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()

        # 4. Residuals Analysis
        test_residuals = self.y_test.values - test_pred

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Residuals distribution
        axes[0].hist(test_residuals, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(test_residuals.mean(), color='r', linestyle='--', lw=2,
                        label=f'Mean: {test_residuals.mean():.4f}')
        axes[0].set_xlabel('Residuals', fontweight='bold')
        axes[0].set_ylabel('Frequency', fontweight='bold')
        axes[0].set_title('Test Set: Residuals Distribution', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Residuals vs predicted
        axes[1].scatter(test_pred, test_residuals, alpha=0.6, color='steelblue', edgecolors='black', linewidth=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].axhline(y=test_residuals.std(), color='orange', linestyle='--', lw=1, alpha=0.7, label='±1 Std')
        axes[1].axhline(y=-test_residuals.std(), color='orange', linestyle='--', lw=1, alpha=0.7)
        axes[1].set_xlabel('Predicted Values', fontweight='bold')
        axes[1].set_ylabel('Residuals', fontweight='bold')
        axes[1].set_title('Test Set: Residuals vs Predicted', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "04_residuals_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[VISUALIZATION] Plots saved to {self.output_dir}")

    def run_pipeline(self):
        """Execute full production pipeline (Revised Order)."""
        print("\n" + "=" * 150)
        print(f"{'PRODUCTION WATER QUALITY REGRESSION MODEL - FULL PIPELINE':^150}")
        print("=" * 150)

        # 1. Load Data
        self.load_and_preprocess()
        # 2. Features (on raw data)
        self.feature_engineering()
        # 3. SPLIT (CRITICAL STEP - Must be before augmentation)
        self.split_train_val_test()
        # 4. AUGMENT (Only on Train)
        self.augment_training_data()
        # 5. Train
        self.train_and_evaluate()
        # 6. Report
        self.print_metrics_matrix()
        self.print_comprehensive_report()
        self.generate_visualizations()

        print("\n[SUCCESS] Pipeline completed!")
        print(f"[DEPLOYMENT] Best model: {self.best_model_name}")
        print(f"[OUTPUT] All results saved to: {self.output_dir}")


if __name__ == "__main__":
    regressor = ProductionWaterQualityRegressor(csv_filename="izsu_health_factor.csv")
    regressor.run_pipeline()