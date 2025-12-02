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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    confusion_matrix,
    balanced_accuracy_score
)

# Uyarıları kapat
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


class WaterQualityAI:
    def __init__(self, csv_filename="izsu_health_factor.csv"):
        self.csv_path = self._find_csv_path(csv_filename)
        self.df = None
        self.models = {}
        self.scalers = {}

        self.output_dir = Path("ai_outputs")
        self.output_dir.mkdir(exist_ok=True)

        print(f"[Init] WaterQualityAI initialized.")
        print(f"[Init] Target CSV Path: {self.csv_path}")

    def _find_csv_path(self, filename):
        current_dir = Path(__file__).resolve().parent
        candidates = [
            current_dir / filename,
            current_dir.parent / filename,
            current_dir.parent / "data" / filename,
            current_dir / "data" / filename
        ]
        for path in candidates:
            if path.exists(): return path
        raise FileNotFoundError(f"'{filename}' bulunamadi.")

    def load_data(self):
        self.df = pd.read_csv(self.csv_path)
        self.df['Tarih'] = pd.to_datetime(self.df['Tarih'])
        print(f"[Data] Loaded {len(self.df)} rows.")
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
        print(f"[Augment] Generating {n_samples} synthetic samples...")
        synthetic_rows = []

        for _ in range(n_samples):
            base_row = self.df.sample(1).iloc[0].copy()
            scenario = np.random.choice(['Caution', 'Risk', 'Risk', 'Risk'])

            if scenario == 'Risk':
                targets = np.random.choice(list(LIMITS_CHECK.keys()), size=np.random.randint(2, 5), replace=False)
                for t in targets:
                    col = f"{t}_score"
                    if col in base_row:
                        base_row[col] = np.random.uniform(0.0, 0.1)
            elif scenario == 'Caution':
                targets = np.random.choice(list(LIMITS_CHECK.keys()), size=np.random.randint(2, 4), replace=False)
                for t in targets:
                    col = f"{t}_score"
                    if col in base_row:
                        base_row[col] = np.random.uniform(0.4, 0.6)

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

        df_syn = pd.DataFrame(synthetic_rows)
        self.df = pd.concat([self.df, df_syn], ignore_index=True)
        return self

    def preprocess_features(self):
        print("[Preprocess] Cleaning and Feature Engineering...")
        score_cols = [c for c in self.df.columns if '_score' in c]
        empty_cols = [c for c in score_cols if self.df[c].isna().all()]
        if empty_cols:
            self.df = self.df.drop(columns=empty_cols)

        valid_score_cols = [c for c in self.df.columns if '_score' in c]

        self.df['Month'] = self.df['Tarih'].dt.month
        self.df['DayOfWeek'] = self.df['Tarih'].dt.dayofweek

        # LocationID çıkarıldı (Best Practice)
        self.feature_cols = valid_score_cols + ['Month', 'DayOfWeek']

        for col in self.feature_cols:
            if col in self.df.columns:
                median_val = self.df[col].median()
                if pd.isna(median_val): median_val = 0.0
                self.df[col] = self.df[col].fillna(median_val)

        print(f"[Preprocess] Final Feature Count: {len(self.feature_cols)}")
        return self

    def force_balance_classes(self):
        print("[Balance] Balancing dataset classes strictly...")
        class_counts = self.df['RiskClass'].value_counts()
        min_count = class_counts.min()
        print(f"    - Min Class Count: {min_count}")
        print(f"    - Original Distribution:\n{class_counts}")

        balanced_df = self.df.groupby('RiskClass').apply(
            lambda x: x.sample(min_count, random_state=42)
        ).reset_index(drop=True)

        self.df = balanced_df.sort_values('Tarih')

        print(f"[Balance] Dataset Balanced! Total Rows: {len(self.df)}")
        print(f"    - New Distribution:\n{self.df['RiskClass'].value_counts()}")

    def train_test_split_time_aware(self):
        # Time-series olduğu için shuffle=False
        X = self.df[self.feature_cols]
        y_class = self.df['RiskClass']
        y_reg = self.df['HealthFactor']

        split_idx = int(len(self.df) * 0.8)

        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_class_train = y_class.iloc[:split_idx]
        self.y_class_test = y_class.iloc[split_idx:]
        self.y_reg_train = y_reg.iloc[:split_idx]
        self.y_reg_test = y_reg.iloc[split_idx:]

        print(f"[Split] Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")

        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        self.scalers['standard'] = scaler
        return self

    # --- YENİ EKLENEN GÜVENLİK MEKANİZMALARI BAŞLANGICI ---
    def detect_data_leakage(self):
        """Feature listesinde hedef değişkenlerin (Target Leakage) olup olmadığını kontrol eder."""
        print("\n=== SECURITY CHECK: DATA LEAKAGE DETECTION ===")
        forbidden_terms = ['HealthFactor', 'RiskClass', 'FailFast', 'NoktaAdi', 'LocationID']
        leaks_found = []

        for feature in self.feature_cols:
            for forbidden in forbidden_terms:
                if forbidden.lower() in feature.lower():
                    leaks_found.append(feature)

        if leaks_found:
            print(f"[KRİTİK UYARI] Data Leakage Tespit Edildi! Bu sütunlar silinmeli: {leaks_found}")
        else:
            print(f"[TEMİZ] Feature listesinde hedef değişken (Target) sızıntısı yok.")

        # Korelasyon Kontrolü (Aşırı yüksek korelasyon var mı?)
        # Not: HF için score'lar yüksek korelasyonlu olabilir, bu normaldir.
        # Ama bir feature target ile %100 aynıysa şüphelidir.
        try:
            # Sadece numeric sütunlar
            numeric_df = self.df[self.feature_cols].select_dtypes(include=np.number)
            # HealthFactor ile korelasyon
            if 'HealthFactor' in self.df.columns:
                corrs = numeric_df.corrwith(self.df['HealthFactor']).abs()
                suspicious = corrs[corrs > 0.995]
                if not suspicious.empty:
                    print(f"[DİKKAT] HealthFactor ile %99.5+ korelasyonlu featurelar: {suspicious.index.tolist()}")
                    print(
                        "   -> Bu regresyon için normal olabilir (formül gereği) ama Classification için kontrol edin.")
        except Exception as e:
            print(f"[Info] Korelasyon kontrolü yapılamadı: {e}")

    def run_cross_validation_check(self, model, X, y, task_name="CLS"):
        """Time Series Split ile Cross Validation yaparak şans faktörünü eler."""
        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy' if task_name == "CLS" else 'r2')
        print(f"    [Cross-Val] 5-Fold TimeSeries Scores: {scores}")
        print(f"    [Cross-Val] Mean Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    # --- GÜVENLİK MEKANİZMALARI BİTİŞİ ---

    def _plot_confusion_matrix(self, y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=sorted(set(y_true)),
                    yticklabels=sorted(set(y_true)))
        plt.title(f'Confusion Matrix - {model_name}')
        filename = self.output_dir / f"confusion_matrix_{model_name.replace(' ', '_')}.png"
        plt.savefig(filename)
        plt.close()

    def analyze_feature_importance(self):
        print("\n=== FEATURE IMPORTANCE ANALIZI ===")
        model_name = "REG_Random Forest Reg"
        model = self.models.get(model_name)

        if model is None:
            model_name = "CLS_Random Forest"
            model = self.models.get(model_name)

        if model is None:
            print("[Uyarı] Analiz için eğitilmiş bir Tree-based model bulunamadı.")
            return

        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            'Feature': self.feature_cols,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        print(fi_df.head(10))

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=fi_df.head(15), palette='viridis')
        plt.title(f'Feature Importance ({model_name})')
        plt.xlabel('Importance')
        save_path = self.output_dir / "feature_importance.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"[Analiz] Grafik kaydedildi: {save_path}")

    def run_classification_models(self):
        print("\n=== CLASSIFICATION TASK (Overfitting Check Enabled) ===")
        models = {
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "SVM": SVC(kernel='rbf', random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
            "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5),
        }
        results = []
        for name, model in models.items():
            model.fit(self.X_train_scaled, self.y_class_train)

            # --- OVERFITTING CHECK ---
            train_preds = model.predict(self.X_train_scaled)
            test_preds = model.predict(self.X_test_scaled)

            train_acc = accuracy_score(self.y_class_train, train_preds)
            test_acc = accuracy_score(self.y_class_test, test_preds)

            bal_acc = balanced_accuracy_score(self.y_class_test, test_preds)

            # Fark analizi
            diff = train_acc - test_acc
            status = "OK"
            if diff > 0.15:
                status = "OVERFITTING"
            elif diff > 0.07:
                status = "Slight Overfit"

            print(
                f"--> {name:15} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Diff: {diff:.4f} [{status}]")

            self._plot_confusion_matrix(self.y_class_test, test_preds, name)
            results.append({'Model': name, 'Train Acc': train_acc, 'Test Acc': test_acc, 'Status': status})
            self.models[f"CLS_{name}"] = model

        return pd.DataFrame(results)

    def run_regression_models(self):
        print("\n=== REGRESSION TASK (Overfitting Check Enabled) ===")
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Reg": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        }
        results = []
        for name, model in models.items():
            model.fit(self.X_train_scaled, self.y_reg_train)

            # --- OVERFITTING CHECK ---
            train_preds = model.predict(self.X_train_scaled)
            test_preds = model.predict(self.X_test_scaled)

            train_r2 = r2_score(self.y_reg_train, train_preds)
            test_r2 = r2_score(self.y_reg_test, test_preds)
            mape = mean_absolute_percentage_error(self.y_reg_test, test_preds)

            diff = train_r2 - test_r2
            status = "OK"
            if diff > 0.10: status = "High Variance"

            print(f"--> {name:20} | Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f} | MAPE: {mape:.4f} [{status}]")
            results.append({'Model': name, 'Train R2': train_r2, 'Test R2': test_r2, 'Status': status})
            self.models[f"REG_{name}"] = model
        return pd.DataFrame(results)


if __name__ == "__main__":
    ai = WaterQualityAI("izsu_health_factor.csv")
    ai.load_data()

    # 1. Bol bol sentetik veri üret
    ai.augment_data_synthetic(n_samples=3000)

    ai.preprocess_features()

    # 2. GÜVENLİK KONTROLÜ 1: Leakage Detection
    ai.detect_data_leakage()

    # 3. Veri setini KESİN olarak eşitle
    ai.force_balance_classes()

    ai.train_test_split_time_aware()

    cls = ai.run_classification_models()
    reg = ai.run_regression_models()

    ai.analyze_feature_importance()

    print("\n=== FINAL SUMMARY ===")
    print(cls)
    print(reg)