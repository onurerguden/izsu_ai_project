import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score, precision_score, recall_score
)

warnings.filterwarnings("ignore")

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


class WaterQualityClassifier:
    def __init__(self, csv_filename="izsu_health_factor.csv"):
        self.csv_path = self._find_csv_path(csv_filename)
        self.df = None

        self.output_dir = Path("ai_outputs/classification")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.models = {}

        self.scaler = None
        self.best_model = None
        self.best_model_name = None
        self.feature_cols = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        self.X_train_scaled = None
        self.X_val_scaled = None
        self.X_test_scaled = None

        print(f"\n [Classifier Init] Hedef CSV: {self.csv_path}")

    def _find_csv_path(self, filename):
        current_dir = Path(__file__).resolve().parent
        candidates = [
            current_dir / filename,
            current_dir.parent / filename,
            current_dir.parent / "data" / filename
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"'{filename}' bulunamadi.")

    def load_data(self):
        self.df = pd.read_csv(self.csv_path)
        self.df["Tarih"] = pd.to_datetime(self.df["Tarih"])
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
        if total_weight == 0:
            return np.nan
        return (current_score / total_weight) * 100.0

    def augment_data_synthetic(self, n_samples=3000):
        print(f"   [Augment] {n_samples} sentetik veri üretiliyor...")
        synthetic_rows = []
        for _ in range(n_samples):
            base_row = self.df.sample(1).iloc[0].copy()
            scenario = np.random.choice(["Caution", "Risk", "Risk", "Risk"])

            if scenario == "Risk":
                targets = np.random.choice(
                    list(LIMITS_CHECK.keys()),
                    size=np.random.randint(2, 5),
                    replace=False
                )
                for t in targets:
                    if f"{t}_score" in base_row:
                        base_row[f"{t}_score"] = np.random.uniform(0.0, 0.1)

            elif scenario == "Caution":
                targets = np.random.choice(
                    list(LIMITS_CHECK.keys()),
                    size=np.random.randint(2, 4),
                    replace=False
                )
                for t in targets:
                    if f"{t}_score" in base_row:
                        base_row[f"{t}_score"] = np.random.uniform(0.4, 0.6)

            new_hf = self._calculate_hf_synthetic(base_row)
            base_row["HealthFactor"] = new_hf

            if new_hf >= 85:
                base_row["RiskClass"] = "Good"
            elif new_hf >= 60:
                base_row["RiskClass"] = "Caution"
            else:
                base_row["RiskClass"] = "Risk"

            base_row["NoktaAdi"] = f"SYNTHETIC_{scenario}"
            synthetic_rows.append(base_row)

        self.df = pd.concat([self.df, pd.DataFrame(synthetic_rows)], ignore_index=True)
        return self

    def preprocess_features(self):
        print("   [Preprocess] Temizlik ve Özellik Mühendisliği...")
        score_cols = [c for c in self.df.columns if "_score" in c]
        empty_cols = [c for c in score_cols if self.df[c].isna().all()]
        if empty_cols:
            self.df = self.df.drop(columns=empty_cols)

        valid_score_cols = [c for c in self.df.columns if "_score" in c]
        self.df["Month"] = self.df["Tarih"].dt.month
        self.df["DayOfWeek"] = self.df["Tarih"].dt.dayofweek

        self.feature_cols = valid_score_cols + ["Month", "DayOfWeek"]

        for col in self.feature_cols:
            median_val = self.df[col].median()
            self.df[col] = self.df[col].fillna(0.0 if pd.isna(median_val) else median_val)
        return self

    def detect_data_leakage(self):
        forbidden = ["HealthFactor", "RiskClass", "FailFast", "NoktaAdi", "LocationID"]
        leaks = [f for f in self.feature_cols if any(x.lower() in f.lower() for x in forbidden)]
        if leaks:
            print(f" [LEAKAGE] Sızıntı var: {leaks}")
        else:
            print(" [Security] Data Leakage Yok.")

    def force_balance_classes(self):
        print("   [Balance] Sınıflar eşitleniyor...")
        min_count = self.df["RiskClass"].value_counts().min()
        self.df = (
            self.df.groupby("RiskClass")
            .apply(lambda x: x.sample(min_count, random_state=42))
            .reset_index(drop=True)
            .sort_values("Tarih")
        )
        print(f"   [Balance] Yeni Satır Sayısı: {len(self.df)} (Her sınıf: {min_count})")

    def train_val_test_split_time_aware(self, train_ratio=0.70, val_ratio=0.15):
        X = self.df[self.feature_cols]
        y = self.df["RiskClass"]

        n = len(self.df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        self.X_train = X.iloc[:train_end]
        self.y_train = y.iloc[:train_end]

        self.X_val = X.iloc[train_end:val_end]
        self.y_val = y.iloc[train_end:val_end]

        self.X_test = X.iloc[val_end:]
        self.y_test = y.iloc[val_end:]

        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"   [Split] Train: {len(self.X_train)}, Val: {len(self.X_val)}, Test: {len(self.X_test)}")

    def _plot_confusion_matrix(self, y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        labels = sorted(set(list(y_true) + list(y_pred)))

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("True")
        plt.xlabel("Predicted")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.output_dir / f"cm_{model_name.replace(' ', '_')}.png", bbox_inches="tight")
        plt.close()

    def _quality_metrics(self, y_true, y_pred):
        return {
            "Acc": accuracy_score(y_true, y_pred),
            "BalAcc": balanced_accuracy_score(y_true, y_pred),
            "MacroF1": f1_score(y_true, y_pred, average="macro"),
            "WeightedF1": f1_score(y_true, y_pred, average="weighted"),
            "MacroPrecision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "MacroRecall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        }

    def _short_params(self, name, model):
        p = model.get_params()
        if name == "KNN":
            return {"n_neighbors": p.get("n_neighbors")}
        if name == "SVM":
            return {"kernel": p.get("kernel"), "C": p.get("C"), "gamma": p.get("gamma")}
        if name == "Decision Tree":
            return {"max_depth": p.get("max_depth"), "min_samples_split": p.get("min_samples_split")}
        if name == "Random Forest":
            return {"n_estimators": p.get("n_estimators"), "max_depth": p.get("max_depth")}
        return {}

    def run_models(self):
        print("\nCLASSIFICATION Modelleri Çalışıyor... (Train/Val/Test + Quality Metrics)")

        models = {
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "SVM": SVC(kernel="rbf", random_state=42),  # proba lazım olursa: probability=True
            "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
            "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5),
        }

        results = []
        best_score = -1.0
        best_name = None
        best_model = None

        for name, model in models.items():
            model.fit(self.X_train_scaled, self.y_train)

            train_preds = model.predict(self.X_train_scaled)
            val_preds = model.predict(self.X_val_scaled)

            train_q = self._quality_metrics(self.y_train, train_preds)
            val_q = self._quality_metrics(self.y_val, val_preds)

            diff = train_q["Acc"] - val_q["Acc"]
            status = "OK" if diff <= 0.15 else "OVERFITTING"

            print(
                f"--> {name:15} | "
                f"Train Acc: {train_q['Acc']:.4f} | Val Acc: {val_q['Acc']:.4f} | "
                f"Val MacroF1: {val_q['MacroF1']:.4f} | Val BalAcc: {val_q['BalAcc']:.4f} | [{status}]"
            )

            self._plot_confusion_matrix(self.y_val, val_preds, f"{name}_VAL")
            self.models[name] = model

            results.append({
                "Model": name,
                "Params": self._short_params(name, model),

                "Train Acc": train_q["Acc"],
                "Train BalAcc": train_q["BalAcc"],
                "Train MacroF1": train_q["MacroF1"],
                "Train WeightedF1": train_q["WeightedF1"],
                "Train MacroPrecision": train_q["MacroPrecision"],
                "Train MacroRecall": train_q["MacroRecall"],

                "Val Acc": val_q["Acc"],
                "Val BalAcc": val_q["BalAcc"],
                "Val MacroF1": val_q["MacroF1"],
                "Val WeightedF1": val_q["WeightedF1"],
                "Val MacroPrecision": val_q["MacroPrecision"],
                "Val MacroRecall": val_q["MacroRecall"],

                "Status": status
            })

            if val_q["MacroF1"] > best_score:
                best_score = val_q["MacroF1"]
                best_name = name
                best_model = model

        self.best_model_name = best_name
        self.best_model = best_model

        df_results = pd.DataFrame(results)

        print(f"\n 🌟 En iyi model (Validation MacroF1): {best_name} (Val MacroF1: {best_score:.4f})")

        print("\n🧪 TEST metrikleri hesaplanıyor (tüm modeller)...")
        for i, row in df_results.iterrows():
            name = row["Model"]
            model = self.models[name]

            test_preds = model.predict(self.X_test_scaled)
            test_q = self._quality_metrics(self.y_test, test_preds)

            df_results.loc[i, "Test Acc"] = test_q["Acc"]
            df_results.loc[i, "Test BalAcc"] = test_q["BalAcc"]
            df_results.loc[i, "Test MacroF1"] = test_q["MacroF1"]
            df_results.loc[i, "Test WeightedF1"] = test_q["WeightedF1"]
            df_results.loc[i, "Test MacroPrecision"] = test_q["MacroPrecision"]
            df_results.loc[i, "Test MacroRecall"] = test_q["MacroRecall"]

        best_test_preds = self.best_model.predict(self.X_test_scaled)
        best_test_q = self._quality_metrics(self.y_test, best_test_preds)

        print(
            f"\nFinal TEST (best={best_name}) | "
            f"Acc: {best_test_q['Acc']:.4f} | BalAcc: {best_test_q['BalAcc']:.4f} | "
            f"MacroF1: {best_test_q['MacroF1']:.4f} | WeightedF1: {best_test_q['WeightedF1']:.4f}"
        )

        print("\n[TEST] Classification Report (best model):\n", classification_report(self.y_test, best_test_preds))
        self._plot_confusion_matrix(self.y_test, best_test_preds, f"{best_name}_TEST")

        out_compare = self.output_dir / "model_comparison_train_val_test.csv"
        df_results.to_csv(out_compare, index=False, encoding="utf-8-sig", float_format="%.4f")
        print(f"\nModel karşılaştırma tablosu (Train/Val/Test) kaydedildi: {out_compare}")

        return df_results

    def save_quality_table(self):
        if self.best_model is None:
            raise RuntimeError("save_quality_table çağırmadan önce run_models çalışmalı (best_model yok).")

        rows = []

        train_preds = self.best_model.predict(self.X_train_scaled)
        train_q = self._quality_metrics(self.y_train, train_preds)
        rows.append({"Split": "Train", **train_q})

        val_preds = self.best_model.predict(self.X_val_scaled)
        val_q = self._quality_metrics(self.y_val, val_preds)
        rows.append({"Split": "Validation", **val_q})

        test_preds = self.best_model.predict(self.X_test_scaled)
        test_q = self._quality_metrics(self.y_test, test_preds)
        rows.append({"Split": "Test", **test_q})

        df_quality = pd.DataFrame(rows)[
            ["Split", "Acc", "BalAcc", "MacroF1", "WeightedF1", "MacroPrecision", "MacroRecall"]
        ]
        out_path = self.output_dir / f"quality_metrics_{self.best_model_name}.csv"
        df_quality.to_csv(out_path, index=False, encoding="utf-8-sig", float_format="%.4f")

        print(f"\nQuality metrics tablosu kaydedildi: {out_path}")
        print(df_quality)
        return df_quality

    def save_best_model(self, filename=None):
        if self.best_model is None or self.scaler is None or self.feature_cols is None:
            raise RuntimeError("Önce run_models ve train_val_test_split_time_aware çağrılmalı (best_model/scaler/feature_cols yok).")

        bundle = {
            "model": self.best_model,
            "model_name": self.best_model_name,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
        }

        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.output_dir / f"best_model_{self.best_model_name}_{ts}.joblib"
        else:
            filename = Path(filename)

        joblib.dump(bundle, filename)
        print(f"\n [Model Save] En iyi model kaydedildi: {filename}")
        return filename

    def save_model_comparison_png(self, df_results: pd.DataFrame):
        df = df_results.copy()
        if "Params" in df.columns:
            df = df.drop(columns=["Params"])

        wanted_cols = [
            "Model",

            "Train Acc", "Train BalAcc", "Train MacroF1", "Train WeightedF1",
            "Train MacroPrecision", "Train MacroRecall",

            "Val Acc", "Val BalAcc", "Val MacroF1", "Val WeightedF1",
            "Val MacroPrecision", "Val MacroRecall",

            "Test Acc", "Test BalAcc", "Test MacroF1", "Test WeightedF1",
            "Test MacroPrecision", "Test MacroRecall",

            "Status"
        ]

        cols = [c for c in wanted_cols if c in df.columns]
        df = df[cols]

        for c in df.columns:
            if c in ["Model", "Status"]:
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce").map(lambda v: f"{v:.4f}" if pd.notna(v) else "")

        nrows, ncols = df.shape
        fig_w = max(18, ncols * 1.15)
        fig_h = max(3.0, nrows * 0.8)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc="center",
            loc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.7)

        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_text_props(weight="bold")

        out_path = self.output_dir / "model_comparison_full_quality.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\nFull quality model karşılaştırma PNG kaydedildi: {out_path}")
        return out_path


def predict_risk_classes(model_path, csv_path, output_path):
    model_path = Path(model_path)
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    print(f"\n[Predict] Model: {model_path}")
    print(f"[Predict] Girdi CSV: {csv_path}")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]

    df_new = pd.read_csv(csv_path)
    df_new["Tarih"] = pd.to_datetime(df_new["Tarih"])
    df_new["Month"] = df_new["Tarih"].dt.month
    df_new["DayOfWeek"] = df_new["Tarih"].dt.dayofweek

    for col in feature_cols:
        if col not in df_new.columns:
            df_new[col] = 0.0
        median_val = df_new[col].median()
        df_new[col] = df_new[col].fillna(0.0 if pd.isna(median_val) else median_val)

    X_new = df_new[feature_cols]
    X_new_scaled = scaler.transform(X_new)

    preds = model.predict(X_new_scaled)
    df_new["PredictedRiskClass"] = preds

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_new.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[Predict] {len(df_new)} satır için tahmin yapıldı. Çıkış: {output_path}")

    show_cols = ["Tarih", "PredictedRiskClass"]
    if "NoktaAdi" in df_new.columns:
        show_cols.insert(1, "NoktaAdi")
    print(df_new[show_cols].head())

    return df_new


if __name__ == "__main__":
    clf = WaterQualityClassifier("izsu_health_factor.csv")
    clf.load_data()
    clf.augment_data_synthetic(n_samples=3000)
    clf.preprocess_features()
    clf.detect_data_leakage()
    clf.force_balance_classes()

    clf.train_val_test_split_time_aware(train_ratio=0.70, val_ratio=0.15)

    results = clf.run_models()

    print("\n=== Classification Results (printed DF) ===")
    print(results)

    clf.save_quality_table()
    clf.save_model_comparison_png(results)

    saved_path = clf.save_best_model()
    print(f"\n[INFO] Model dosyası: {saved_path}")
