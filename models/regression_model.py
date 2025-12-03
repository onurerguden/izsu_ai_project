import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import sys
import os

# Sklearn Imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_percentage_error
)

warnings.filterwarnings('ignore')

# --- CONFIGURATION & WEIGHTS ---
WEIGHTS = {
    "E.coli": 0.133, "Koliform Bakteri": 0.133, "C.Perfringens": 0.133,
    "Arsenik": 0.088, "Nitrit": 0.088, "Alüminyum": 0.055, "Demir": 0.055,
    "Amonyum": 0.055, "pH": 0.040, "Klorür": 0.020, "İletkenlik": 0.020,
    "Oksitlenebilirlik": 0.020,
}

LIMITS_CHECK = {
    "Arsenik": 10.0, "Nitrit": 0.5, "Alüminyum": 200.0,
    "Demir": 200.0, "Amonyum": 0.5, "Klorür": 250.0,
    "E.coli": 1.0, "Koliform Bakteri": 1.0
}


class WaterQualityRegressor:
    def __init__(self, csv_filename="izsu_health_factor.csv"):
        self.csv_path = self._find_csv_path(csv_filename)
        self.df = None
        self.models = {}
        self.output_dir = Path("ai_outputs/regression")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n [Regressor Init] Hedef CSV: {self.csv_path}")

    def _find_csv_path(self, filename):
        current_dir = Path(__file__).resolve().parent
        candidates = [current_dir / filename, current_dir.parent / filename, current_dir.parent / "data" / filename]
        for path in candidates:
            if path.exists(): return path
        raise FileNotFoundError(f"'{filename}' bulunamadi.")

    def load_data(self):
        self.df = pd.read_csv(self.csv_path)
        self.df['Tarih'] = pd.to_datetime(self.df['Tarih'])
        print(f"   [Data] {len(self.df)} satır yüklendi.")
        return self

    def _calculate_hf_synthetic(self, row):
        current_score = 0.0
        total_weight = 0.0
        for param, weight in WEIGHTS.items():
            col = f"{param}_score"
            if col in row and pd.notna(row[col]):
                current_score += weight * row[col]
                total_weight += weight
        if total_weight == 0: return np.nan
        return (current_score / total_weight) * 100.0

    def augment_data_synthetic(self, n_samples=3000):
        print(f"   [Augment] {n_samples} sentetik veri üretiliyor...")
        synthetic_rows = []
        for _ in range(n_samples):
            base_row = self.df.sample(1).iloc[0].copy()
            scenario = np.random.choice(['Caution', 'Risk', 'Risk', 'Risk'])

            if scenario == 'Risk':
                targets = np.random.choice(list(LIMITS_CHECK.keys()), size=np.random.randint(2, 5), replace=False)
                for t in targets:
                    if f"{t}_score" in base_row: base_row[f"{t}_score"] = np.random.uniform(0.0, 0.1)
            elif scenario == 'Caution':
                targets = np.random.choice(list(LIMITS_CHECK.keys()), size=np.random.randint(2, 4), replace=False)
                for t in targets:
                    if f"{t}_score" in base_row: base_row[f"{t}_score"] = np.random.uniform(0.4, 0.6)

            new_hf = self._calculate_hf_synthetic(base_row)
            base_row['HealthFactor'] = new_hf

            if new_hf >= 85:
                base_row['RiskClass'] = 'Good'
            elif new_hf >= 60:
                base_row['RiskClass'] = 'Caution'
            else:
                base_row['RiskClass'] = 'Risk'

            base_row['NoktaAdi'] = f"SYNTHETIC_{scenario}"
            synthetic_rows.append(base_row)

        self.df = pd.concat([self.df, pd.DataFrame(synthetic_rows)], ignore_index=True)
        return self

    def preprocess_features(self):
        print("   [Preprocess] Temizlik ve Özellik Mühendisliği...")
        score_cols = [c for c in self.df.columns if '_score' in c]
        empty_cols = [c for c in score_cols if self.df[c].isna().all()]
        if empty_cols: self.df = self.df.drop(columns=empty_cols)

        valid_score_cols = [c for c in self.df.columns if '_score' in c]
        self.df['Month'] = self.df['Tarih'].dt.month
        self.df['DayOfWeek'] = self.df['Tarih'].dt.dayofweek

        self.feature_cols = valid_score_cols + ['Month', 'DayOfWeek']

        for col in self.feature_cols:
            median_val = self.df[col].median()
            self.df[col] = self.df[col].fillna(0.0 if pd.isna(median_val) else median_val)
        return self

    def detect_data_leakage(self):
        forbidden = ['HealthFactor', 'RiskClass', 'FailFast', 'NoktaAdi', 'LocationID']
        leaks = [f for f in self.feature_cols if any(x.lower() in f.lower() for x in forbidden)]
        if leaks:
            print(f" [LEAKAGE] Sızıntı var: {leaks}")
        else:
            print(" [Security] Data Leakage Yok.")

    def force_balance_classes(self):
        print("   [Balance] Sınıflar eşitleniyor (Regression için veri setini stabilize eder)...")
        min_count = self.df['RiskClass'].value_counts().min()
        self.df = self.df.groupby('RiskClass').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(
            drop=True).sort_values('Tarih')
        print(f"   [Balance] Yeni Satır Sayısı: {len(self.df)} (Her sınıf: {min_count})")

    def train_test_split_time_aware(self):
        X = self.df[self.feature_cols]
        y = self.df['HealthFactor']
        split_idx = int(len(self.df) * 0.8)

        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]

        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        print(f"   [Split] Train: {len(self.X_train)}, Test: {len(self.X_test)}")

    def analyze_feature_importance(self):
        print("\n📊 Feature Importance (Regression - Random Forest)...")
        if "Random Forest" in self.models:
            importances = self.models["Random Forest"].feature_importances_
            fi_df = pd.DataFrame({'Feature': self.feature_cols, 'Importance': importances}).sort_values(by='Importance',
                                                                                                        ascending=False)
            print(fi_df.head(5))
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=fi_df.head(10), palette='viridis')
            plt.title('Feature Importance (Regression)')
            plt.savefig(self.output_dir / "feature_importance.png")
            plt.close()

    def run_models(self):
        print("\n🚀 REGRESSION Modelleri Çalışıyor...")
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        }
        results = []
        for name, model in models.items():
            model.fit(self.X_train_scaled, self.y_train)
            train_r2 = r2_score(self.y_train, model.predict(self.X_train_scaled))
            test_preds = model.predict(self.X_test_scaled)
            test_r2 = r2_score(self.y_test, test_preds)
            mape = mean_absolute_percentage_error(self.y_test, test_preds)

            diff = train_r2 - test_r2
            status = "OK" if diff <= 0.10 else "High Variance"

            print(f"--> {name:20} | Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f} | MAPE: {mape:.4f} [{status}]")
            results.append({'Model': name, 'Train R2': train_r2, 'Test R2': test_r2, 'MAPE': mape, 'Status': status})
            self.models[name] = model
        return pd.DataFrame(results)


if __name__ == "__main__":
    reg = WaterQualityRegressor("izsu_health_factor.csv")
    reg.load_data()
    reg.augment_data_synthetic(n_samples=3000)
    reg.preprocess_features()
    reg.detect_data_leakage()
    reg.force_balance_classes()
    reg.train_test_split_time_aware()
    results = reg.run_models()
    reg.analyze_feature_importance()
    print("\n=== Regression Results ===")
    print(results)