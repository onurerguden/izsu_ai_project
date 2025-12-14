import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, precision_recall_curve
)
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")


# =====================================================
# CONFIGURATION
# =====================================================
class Config:
    CSV_PATH = "izsu_health_factor.csv"
    TARGET_COL = "HealthFactor"
    DROP_THRESHOLD = -0.05
    RISE_THRESHOLD = 0.05
    OUTPUT_DIR = "ai_outputs_best_practice"
    TARGET_RECALL = 0.80
    RANDOM_STATE = 42


# =====================================================
# FEATURE ENGINEERING (BEST PRACTICE - DATA LEAKAGE FIX)
# =====================================================
def calculate_rsi(series, period=14):
    """Suyun değişim momentumunu (hızını) ölçer."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def prepare_data(csv_path):
    print("[DATA] Veri yükleniyor ve özellikler türetiliyor...")
    df = pd.read_csv(csv_path)
    df["Tarih"] = pd.to_datetime(df["Tarih"])
    df = df.sort_values(["NoktaAdi", "Tarih"]).reset_index(drop=True)

    # 1. Temel Delta ve Hedefler
    df["Next_Val"] = df.groupby("NoktaAdi")[Config.TARGET_COL].shift(-1)
    df["Future_Delta"] = df["Next_Val"] - df[Config.TARGET_COL]

    # Binary Hedefler
    df["Target_Drop"] = (df["Future_Delta"] < Config.DROP_THRESHOLD).astype(int)
    df["Target_Rise"] = (df["Future_Delta"] > Config.RISE_THRESHOLD).astype(int)

    # 2. Geçmiş Özellikler (Lags) - Dün ve önceki günleri kullan
    grp = df.groupby("NoktaAdi")[Config.TARGET_COL]
    df["Lag_1"] = grp.shift(1)  # Dün
    df["Lag_2"] = grp.shift(2)  # Önceki gün
    df["Lag_3"] = grp.shift(3)  # 3 gün önce

    # 3. İstatistiksel Özellikler (Rolling) - shift(1) ile data leakage'ı önle
    df["Mean_3D"] = grp.shift(1).rolling(3).mean()
    df["Std_3D"] = grp.shift(1).rolling(3).std()
    df["Mean_7D"] = grp.shift(1).rolling(7).mean()

    # 4. Momentum (RSI) - FIX: shift(1) ekledik (KRITIK BUG FIX)
    df["RSI_7"] = df.groupby("NoktaAdi")[Config.TARGET_COL].apply(
        lambda x: calculate_rsi(x.shift(1), 7)
    ).reset_index(0, drop=True)

    # 5. Döngüsel Zaman (Cyclical Features)
    df["Month_Sin"] = np.sin(2 * np.pi * df["Tarih"].dt.month / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Tarih"].dt.month / 12)

    # Temizlik
    features = [
        "Lag_1", "Lag_2", "Lag_3",
        "Mean_3D", "Std_3D", "Mean_7D",
        "RSI_7", "Month_Sin", "Month_Cos"
    ]

    df_clean = df.dropna(subset=features + ["Target_Drop", "Target_Rise"]).reset_index(drop=True)

    print(f"[DATA] {len(df_clean)} satır hazırlandı. Özellik sayısı: {len(features)}")

    # YENI: Sınıf dağılımı raporu
    print("\n[CLASS BALANCE] Sınıf Dağılımı:")
    print(f"  Drop Sınıfı (1): {df_clean['Target_Drop'].mean():.3%}")
    print(f"  Rise Sınıfı (1): {df_clean['Target_Rise'].mean():.3%}")

    return df_clean, features


# =====================================================
# TRAINING ENGINE (IMBALANCE + BUG FIX)
# =====================================================
def train_specialist_model(X, y, model_name="Model"):
    """
    Belirli bir hedef için (Drop veya Rise) TimeSeriesSplit ile model eğitir.
    En iyi eşik değerini (Threshold) otomatik bulur.
    SMOTE ile sınıf imbalansını çözer.
    """
    print(f"\n--- {model_name} Eğitiliyor ---")

    tscv = TimeSeriesSplit(n_splits=5)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=Config.RANDOM_STATE,
        n_jobs=-1
    )

    best_thresholds = []
    fold_scores = []
    fold_roc_scores = []

    # Cross-Validation Döngüsü
    for fold_num, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # YENI: SMOTE ile imbalansı çöz
        smote = SMOTE(random_state=Config.RANDOM_STATE)
        try:
            X_train_fold_balanced, y_train_fold_balanced = smote.fit_resample(X_train_fold, y_train_fold)
        except:
            # Fold'da yeterli azınlık sınıf yoksa SMOTE'u atla
            X_train_fold_balanced, y_train_fold_balanced = X_train_fold, y_train_fold

        model.fit(X_train_fold_balanced, y_train_fold_balanced)

        # Olasılıkları al
        probs = model.predict_proba(X_val_fold)[:, 1]

        # En iyi Threshold'u bul (FIXED: Index kontrolü yapıldı)
        precisions, recalls, thresholds = precision_recall_curve(y_val_fold, probs)

        optimal_thresh = 0.5  # Default
        try:
            # Recall >= %80 olan eşikleri bul
            valid_mask = recalls >= Config.TARGET_RECALL
            if np.any(valid_mask):
                # Geçerli indisleri bul
                valid_indices = np.where(valid_mask)[0]
                # Thresholds bir kısımdır, indis kontrolü yap
                valid_indices = valid_indices[valid_indices < len(thresholds)]
                if len(valid_indices) > 0:
                    # En yüksek Precision'ı seç
                    best_idx = valid_indices[np.argmax(precisions[valid_indices])]
                    optimal_thresh = thresholds[best_idx]
        except Exception as e:
            print(f"   Fold {fold_num}: Threshold optimizasyonu başarısız, default 0.5 kullanılıyor")

        best_thresholds.append(optimal_thresh)

        # Skoru kaydet
        preds = (probs >= optimal_thresh).astype(int)
        fold_scores.append(f1_score(y_val_fold, preds))

        # YENI: ROC-AUC skoru
        try:
            roc_auc = roc_auc_score(y_val_fold, probs)
            fold_roc_scores.append(roc_auc)
        except:
            fold_roc_scores.append(0.5)

    # Final Model Eğitimi (Tüm Train Verisiyle)
    avg_threshold = np.mean(best_thresholds)

    # Final model için de SMOTE uygula
    try:
        X_balanced, y_balanced = smote.fit_resample(X, y)
        model.fit(X_balanced, y_balanced)
    except:
        model.fit(X, y)

    print(f"   -> Optimize Edilen Eşik Değeri: {avg_threshold:.4f}")
    print(f"   -> Ortalama F1 Skoru (CV): {np.mean(fold_scores):.4f}")
    print(f"   -> Ortalama ROC-AUC (CV): {np.mean(fold_roc_scores):.4f}")

    return model, avg_threshold


# =====================================================
# MAIN PIPELINE
# =====================================================
def run_advanced_pipeline():
    # 1. Veri Hazırlığı
    df, features = prepare_data(Config.CSV_PATH)

    # Train / Test Split (Son %15 Test)
    split_idx = int(len(df) * 0.85)

    X_train = df.iloc[:split_idx][features]
    X_test = df.iloc[split_idx:][features]

    y_train_drop = df.iloc[:split_idx]["Target_Drop"]
    y_test_drop = df.iloc[split_idx:]["Target_Drop"]

    y_train_rise = df.iloc[:split_idx]["Target_Rise"]
    y_test_rise = df.iloc[split_idx:]["Target_Rise"]

    print(f"\n[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")

    # 2. DROP Modelini Eğit
    drop_model, drop_thresh = train_specialist_model(X_train, y_train_drop, "DÜŞÜŞ (Drop) Uzmanı")

    # 3. RISE Modelini Eğit
    rise_model, rise_thresh = train_specialist_model(X_train, y_train_rise, "YÜKSELİŞ (Rise) Uzmanı")

    # 4. Feature Importance Analizi (YENI)
    print("\n[FEATURE IMPORTANCE] DROP Modeli:")
    drop_importance = sorted(zip(features, drop_model.feature_importances_),
                             key=lambda x: x[1], reverse=True)
    for feat, imp in drop_importance[:5]:
        print(f"  {feat}: {imp:.4f}")

    print("\n[FEATURE IMPORTANCE] RISE Modeli:")
    rise_importance = sorted(zip(features, rise_model.feature_importances_),
                             key=lambda x: x[1], reverse=True)
    for feat, imp in rise_importance[:5]:
        print(f"  {feat}: {imp:.4f}")

    # 5. Test Setinde Final Değerlendirme
    print("\n[TEST] Final Değerlendirme Yapılıyor...")

    # Olasılıklar
    drop_probs = drop_model.predict_proba(X_test)[:, 1]
    rise_probs = rise_model.predict_proba(X_test)[:, 1]

    # Predictionlar
    drop_preds = (drop_probs >= drop_thresh).astype(int)
    rise_preds = (rise_probs >= rise_thresh).astype(int)

    # 6. Raporlama - DROP Modeli (YENI: Ayrı confusion matrix)
    print("\n" + "=" * 60)
    print("DÜŞÜŞ (DROP) MODELI - DETAYLI ANALIZ")
    print("=" * 60)
    print("\nClassification Report:")
    print(classification_report(y_test_drop, drop_preds,
                                target_names=["Normal", "Düşüş"]))

    print(f"\nDetaylı Metrikler:")
    print(f"  Recall (Düşüş Yakalanma): {recall_score(y_test_drop, drop_preds):.4f}")
    print(f"  Precision (Düşüş Doğruluğu): {precision_score(y_test_drop, drop_preds):.4f}")
    print(f"  F1-Score: {f1_score(y_test_drop, drop_preds):.4f}")
    print(f"  ROC-AUC: {roc_auc_score(y_test_drop, drop_probs):.4f}")

    # Confusion Matrix - DROP
    cm_drop = confusion_matrix(y_test_drop, drop_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_drop, annot=True, fmt='d', cmap='Reds', cbar=True,
                xticklabels=["Normal", "Düşüş"],
                yticklabels=["Normal", "Düşüş"])
    plt.title("DÜŞÜŞ (DROP) Modeli - Confusion Matrix", fontsize=14, fontweight='bold')
    plt.ylabel("Gerçek Değer", fontsize=12)
    plt.xlabel("Tahmin", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{Config.OUTPUT_DIR}/confusion_matrix_drop.png", dpi=300)
    print(f"\n[ÇIKTI] DROP Confusion Matrix kaydedildi: {Config.OUTPUT_DIR}/confusion_matrix_drop.png")
    plt.close()

    # 7. Raporlama - RISE Modeli (YENI: Ayrı confusion matrix)
    print("\n" + "=" * 60)
    print("YÜKSELİŞ (RISE) MODELI - DETAYLI ANALIZ")
    print("=" * 60)
    print("\nClassification Report:")
    print(classification_report(y_test_rise, rise_preds,
                                target_names=["Normal", "Yükseliş"]))

    print(f"\nDetaylı Metrikler:")
    print(f"  Recall (Yükseliş Yakalanma): {recall_score(y_test_rise, rise_preds):.4f}")
    print(f"  Precision (Yükseliş Doğruluğu): {precision_score(y_test_rise, rise_preds):.4f}")
    print(f"  F1-Score: {f1_score(y_test_rise, rise_preds):.4f}")
    print(f"  ROC-AUC: {roc_auc_score(y_test_rise, rise_probs):.4f}")

    # Confusion Matrix - RISE
    cm_rise = confusion_matrix(y_test_rise, rise_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_rise, annot=True, fmt='d', cmap='Greens', cbar=True,
                xticklabels=["Normal", "Yükseliş"],
                yticklabels=["Normal", "Yükseliş"])
    plt.title("YÜKSELİŞ (RISE) Modeli - Confusion Matrix", fontsize=14, fontweight='bold')
    plt.ylabel("Gerçek Değer", fontsize=12)
    plt.xlabel("Tahmin", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{Config.OUTPUT_DIR}/confusion_matrix_rise.png", dpi=300)
    print(f"[ÇIKTU] RISE Confusion Matrix kaydedildi: {Config.OUTPUT_DIR}/confusion_matrix_rise.png")
    plt.close()

    # 8. Akıllı Kombinasyon Tahminleri (FIX: Basitleştirilmiş mantık)
    print("\n" + "=" * 60)
    print("KOMBİNE MODEL - FINAL TAHMİNLER (Düşüş/Sabit/Yükseliş)")
    print("=" * 60)

    final_preds = []
    for d_pred, r_pred, d_prob, r_prob in zip(drop_preds, rise_preds, drop_probs, rise_probs):
        if d_pred == 1 and r_pred == 0:
            final_preds.append("DÜŞÜŞ")
        elif r_pred == 1 and d_pred == 0:
            final_preds.append("YÜKSELİŞ")
        elif d_pred == 1 and r_pred == 1:
            # Çatışma: Hangisi daha güvenliyse onu seç
            if d_prob > r_prob:
                final_preds.append("DÜŞÜŞ")
            else:
                final_preds.append("YÜKSELİŞ")
        else:
            final_preds.append("SABİT")

    # Gerçek Değerler
    true_labels = []
    for d, r in zip(y_test_drop, y_test_rise):
        if d == 1:
            true_labels.append("DÜŞÜŞ")
        elif r == 1:
            true_labels.append("YÜKSELİŞ")
        else:
            true_labels.append("SABİT")

    # Kombinasyon Raporu
    print("\nClassification Report (Kombinasyon):")
    print(classification_report(true_labels, final_preds))

    # Kombinasyon Confusion Matrix
    cm_combined = confusion_matrix(true_labels, final_preds,
                                   labels=["DÜŞÜŞ", "SABİT", "YÜKSELİŞ"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_combined, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=["DÜŞÜŞ", "SABİT", "YÜKSELİŞ"],
                yticklabels=["DÜŞÜŞ", "SABİT", "YÜKSELİŞ"])
    plt.title("KOMBİNE MODEL - Confusion Matrix", fontsize=14, fontweight='bold')
    plt.ylabel("Gerçek Değer", fontsize=12)
    plt.xlabel("Tahmin", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{Config.OUTPUT_DIR}/confusion_matrix_combined.png", dpi=300)
    print(f"\n[ÇIKTI] Kombinasyon Confusion Matrix kaydedildi: {Config.OUTPUT_DIR}/confusion_matrix_combined.png")
    plt.close()

    print("\n" + "=" * 60)
    print("✅ TÜÜM ANALIZLER TAMAMLANDI!")
    print("=" * 60)


if __name__ == "__main__":
    import os

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    run_advanced_pipeline()