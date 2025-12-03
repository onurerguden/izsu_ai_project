import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Sklearn Imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_percentage_error
)

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
WEIGHTS = {
    "E.coli": 0.133, "Koliform Bakteri": 0.133, "C.Perfringens": 0.133,
    "Arsenik": 0.088, "Nitrit": 0.088, "Alüminyum": 0.055, "Demir": 0.055,
    "Amonyum": 0.055, "pH": 0.040, "Klorür": 0.020, "İletkenlik": 0.020,
    "Oksitlenebilirlik": 0.020,
}


class WaterQualityRegressor:
    def __init__(self, csv_filename="izsu_health_factor.csv"):
        self.csv_path = self._find_csv_path(csv_filename)
        self.df = None
        self.models = {}
        self.output_dir = Path("ai_outputs/regression")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n🟢 [Regressor Init] Hedef CSV: {self.csv_path}")

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

    def augment_data_synthetic(self, n_samples=3000):
        # Regresyon için augmentation yaparken dikkatli olmalıyız.
        # Zaman serisi yapısını bozmamak için sentetik verileri "geçmiş tarihlere" değil
        # "mevcut tarihlerin varyasyonları" olarak eklemek daha doğrudur.
        # Ancak basitlik için standart augmentation'ı koruyoruz ama Tarih'i sıralı tutacağız.
        print(f"   [Augment] {n_samples} sentetik veri üretiliyor...")

        synthetic_rows = []
        last_date = self.df['Tarih'].max()

        for i in range(n_samples):
            base_row = self.df.sample(1).iloc[0].copy()

            # Parametreleri hafifçe boz (Regresyon için süreklilik önemli)
            cols_to_perturb = [c for c in self.df.columns if '_score' in c]
            for c in cols_to_perturb:
                if pd.notna(base_row[c]):
                    noise = np.random.normal(0, 0.05)  # Küçük gürültü
                    base_row[c] = np.clip(base_row[c] + noise, 0.0, 1.0)

            # HF'yi yeniden hesapla
            current_score = 0.0
            total_weight = 0.0
            for param, weight in WEIGHTS.items():
                col = f"{param}_score"
                if col in base_row and pd.notna(base_row[col]):
                    current_score += weight * base_row[col]
                    total_weight += weight

            if total_weight > 0:
                base_row['HealthFactor'] = (current_score / total_weight) * 100.0

            # Tarih: Sentetik verileri geleceğe ekle (Time Series Forecasting mantığı)
            base_row['Tarih'] = last_date + timedelta(days=int(i / 100) + 1)
            base_row['NoktaAdi'] = f"SYNTHETIC_LOC_{np.random.randint(1, 5)}"

            synthetic_rows.append(base_row)

        self.df = pd.concat([self.df, pd.DataFrame(synthetic_rows)], ignore_index=True)
        self.df = self.df.sort_values(['Tarih', 'NoktaAdi']).reset_index(drop=True)
        return self

    def preprocess_features(self):
        print("   [Preprocess] Time-Series & Spatial Feature Engineering...")

        # 1. Temizlik
        score_cols = [c for c in self.df.columns if '_score' in c]
        valid_score_cols = [c for c in score_cols if not self.df[c].isna().all()]

        # 2. NaN Doldurma
        for col in valid_score_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())

        # 3. Zaman Özellikleri
        self.df['Month'] = self.df['Tarih'].dt.month
        self.df['DayOfWeek'] = self.df['Tarih'].dt.dayofweek
        self.df['DayOfYear'] = self.df['Tarih'].dt.dayofyear

        # 4. Spatial (Konum) Özellikleri - Target Encoding
        # Bölgenin "genel kirlilik seviyesini" modele öğretelim
        loc_means = self.df.groupby('NoktaAdi')['HealthFactor'].transform('mean')
        self.df['Location_Avg_HF'] = loc_means

        # 5. Lag Features (Gecikmeli Veriler - Time Series için KRİTİK)
        # "Dünkü HF değeri bugünü tahmin etmede en büyük ipucudur"
        # Not: Nokta bazında shift etmeliyiz
        self.df['HF_Lag1'] = self.df.groupby('NoktaAdi')['HealthFactor'].shift(1)
        self.df['HF_Lag7'] = self.df.groupby('NoktaAdi')['HealthFactor'].shift(7)

        # Lag'lerden oluşan boşlukları doldur
        self.df['HF_Lag1'] = self.df['HF_Lag1'].fillna(method='bfill')
        self.df['HF_Lag7'] = self.df['HF_Lag7'].fillna(method='bfill')

        # 6. Final Feature Listesi
        self.feature_cols = valid_score_cols + ['Month', 'DayOfWeek', 'Location_Avg_HF', 'HF_Lag1', 'HF_Lag7']

        # Son temizlik
        self.df = self.df.dropna(subset=self.feature_cols)
        print(f"   [Preprocess] Final Features: {len(self.feature_cols)} (Added: Lags & Spatial Avg)")
        return self

    def train_test_split_time_aware(self):
        # Time-series split (Geleceği geçmişle tahmin et)
        self.df = self.df.sort_values('Tarih')

        X = self.df[self.feature_cols]
        y = self.df['HealthFactor']

        # Son %20'yi test yap
        split_idx = int(len(self.df) * 0.8)

        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]

        # Scaling
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        print(f"   [Split] Train: {len(self.X_train)}, Test: {len(self.X_test)}")

    def run_models(self):
        print("\n🚀 REGRESSION Modelleri Çalışıyor (Time-Series Optimized)...")
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        }
        results = []
        for name, model in models.items():
            model.fit(self.X_train_scaled, self.y_train)

            train_pred = model.predict(self.X_train_scaled)
            test_pred = model.predict(self.X_test_scaled)

            train_r2 = r2_score(self.y_train, train_pred)
            test_r2 = r2_score(self.y_test, test_pred)
            mape = mean_absolute_percentage_error(self.y_test, test_pred)

            # Time Series Cross Validation (Ekstra Güvenlik)
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=tscv, scoring='r2')
            cv_mean = cv_scores.mean()

            print(f"--> {name:20} | Test R2: {test_r2:.4f} | MAPE: {mape:.4f} | CV R2: {cv_mean:.4f}")
            results.append({
                'Model': name,
                'Train R2': train_r2,
                'Test R2': test_r2,
                'CV R2': cv_mean,
                'MAPE': mape
            })
            self.models[name] = model

        return pd.DataFrame(results)

    def analyze_feature_importance(self):
        print("\n📊 Feature Importance (XGBoost)...")
        if "XGBoost" in self.models:
            importances = self.models["XGBoost"].feature_importances_
            fi_df = pd.DataFrame({'Feature': self.feature_cols, 'Importance': importances}).sort_values(by='Importance',
                                                                                                        ascending=False)
            print(fi_df.head(5))
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=fi_df.head(10), palette='viridis')
            plt.title('Feature Importance (Time-Series Regression)')
            plt.savefig(self.output_dir / "feature_importance_reg.png")
            plt.close()


if __name__ == "__main__":
    reg = WaterQualityRegressor("izsu_health_factor.csv")
    reg.load_data()
    reg.augment_data_synthetic(n_samples=2000)  # Time series için makul bir sayı
    reg.preprocess_features()
    reg.train_test_split_time_aware()
    results = reg.run_models()
    reg.analyze_feature_importance()
    print("\n=== Regression Results ===")
    print(results)