import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime


class RiskClassifierDeployer:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {self.model_path}")

        print(f"[Deployer] Model yükleniyor: {self.model_path}")
        bundle = joblib.load(self.model_path)

        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_cols = bundle["feature_cols"]
        self.model_name = bundle.get("model_name", "UnknownModel")

        print(f"[Deployer] Model adı: {self.model_name}")
        print(f"[Deployer] Feature sayısı: {len(self.feature_cols)}")

    def _prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        df = df.copy()

        if 'Tarih' not in df.columns:
            raise ValueError("Girdi CSV'de 'Tarih' kolonu yok. Health factor pipeline ile aynı formatta olmalı.")

        df['Tarih'] = pd.to_datetime(df['Tarih'])
        df['Month'] = df['Tarih'].dt.month
        df['DayOfWeek'] = df['Tarih'].dt.dayofweek

        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0

            median_val = df[col].median()
            df[col] = df[col].fillna(0.0 if pd.isna(median_val) else median_val)

        X = df[self.feature_cols]
        X_scaled = self.scaler.transform(X)

        return df, X_scaled

    def predict_csv(self, input_csv, output_csv=None, add_proba=False) -> pd.DataFrame:
        input_csv = Path(input_csv)
        print(f"\n[Predict] Girdi CSV: {input_csv}")

        if not input_csv.exists():
            raise FileNotFoundError(f"Girdi CSV bulunamadı: {input_csv}")

        df_new = pd.read_csv(input_csv)
        df_prepared, X_scaled = self._prepare_features(df_new)

        preds = self.model.predict(X_scaled)
        df_prepared['PredictedRiskClass'] = preds

        if add_proba and hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_scaled)
            class_labels = self.model.classes_
            for i, cls in enumerate(class_labels):
                df_prepared[f"Prob_{cls}"] = proba[:, i]

        if output_csv is not None:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df_prepared.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"[Predict] Tahminli CSV kaydedildi: {output_csv}")

        cols_to_show = ['Tarih', 'PredictedRiskClass']
        if 'NoktaAdi' in df_prepared.columns:
            cols_to_show.insert(1, 'NoktaAdi')

        print("\n[Predict] Örnek sonuçlar:")
        print(df_prepared[cols_to_show].head())

        return df_prepared


if __name__ == "__main__":
    model_file = "ai_outputs/classification/best_model_SVM_20251203_185302.joblib"
    input_file = "izsu_health_factor.csv"
    output_file = "izsu_health_factor_class_predicted.csv"
    try:
        deployer = RiskClassifierDeployer(model_file)
        deployer.predict_csv(input_file, output_file, add_proba=False)
    except Exception as e:
        print(f"[ERROR] {e}")
