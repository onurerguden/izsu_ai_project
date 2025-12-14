import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =====================================================
# MODELLER (XGBoost EKLENDİ)
# =====================================================
try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost kütüphanesi eksik. Lütfen 'pip install xgboost' çalıştırın.")
    # Fallback (Hata vermemesi için dummy bir class, ama kurulu varsayıyoruz)
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Metrikler ve Araçlar
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, precision_recall_curve
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
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
    OUTPUT_DIR = "ai_outputs_champion_models"
    TARGET_RECALL = 0.80  # Hedeflenen yakalama oranı (%80)
    RANDOM_STATE = 42


# =====================================================
# YARDIMCI FONKSİYON: DETAYLI METRİK ANLATIMI (EĞİTMEN MODU)
# =====================================================
def explain_metrics(y_true, y_pred, y_probs, label_name="DÜŞÜŞ"):
    """
    Bu fonksiyon standart metrikleri hesaplar ve kullanıcıya
    ne anlama geldiklerini terminalde ders verir gibi anlatır.
    """
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    support_positive = sum(y_true)

    try:
        auc = roc_auc_score(y_true, y_probs)
    except:
        auc = 0.5

    print(f"\n{'=' * 20} {label_name} İÇİN DETAYLI ANALİZ RAPORU {'=' * 20}")
    print(f"Classification Report:\n{classification_report(y_true, y_pred)}")

    print(f"--- METRİK SÖZLÜĞÜ VE YORUMLAR ---")

    print(f"1. RECALL (Duyarlılık/Yakalama Gücü): {recall:.4f}")
    print(f"   > Anlamı: Gerçekte '{label_name}' olan durumların yüzde kaçını yakalayabildik?")
    print(
        f"   > Yorum: Eğer bu sayı düşükse (örn 0.40), tehlikeyi görüp uyaramıyoruz demektir. Kritik sistemlerde en önemli değerdir.")
    print(f"   > Hedefimiz: {Config.TARGET_RECALL} (Yani olayların %{int(Config.TARGET_RECALL * 100)}'ini kaçırmamak).")

    print(f"\n2. PRECISION (Kesinlik/Güvenilirlik): {precision:.4f}")
    print(f"   > Anlamı: Model 'Alarm! {label_name} olacak' dediğinde, ne kadarında haklı çıktı?")
    print(f"   > Yorum: Eğer bu sayı düşükse, model çok fazla 'Yalancı Çoban' (False Alarm) durumuna düşüyor demektir.")

    print(f"\n3. F1-SCORE (Denge Skoru): {f1:.4f}")
    print(f"   > Anlamı: Precision ve Recall'un harmonik ortalamasıdır. (İkisini de dengeleyen tek bir not).")
    print(f"   > Yorum: Modelin genel başarısını tek sayıyla özetler.")

    print(f"\n4. SUPPORT (Destek/Örnek Sayısı): {support_positive} Adet")
    print(f"   > Anlamı: Test verisi içinde gerçekten {label_name} olan kaç adet satır vardı.")
    print(
        f"   > Yorum: Eğer bu sayı çok azsa (örn. 5-10), modelin başarısı şans eseri olabilir. İstatistiksel güven için önemlidir.")

    print(f"\n5. ROC-AUC Skoru: {auc:.4f}")
    print(
        f"   > Anlamı: Modelin 0 ve 1 sınıflarını birbirinden ayırma yeteneği. 0.5 yazı-tura (kötü), 1.0 mükemmel tahmindir.")
    print(f"{'=' * 65}\n")


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
    print("\n[DATA] Veri yükleniyor ve özellikler türetiliyor...")

    # Dosya kontrolü ve Dummy Data (Eğer dosya yoksa)
    if not os.path.exists(csv_path):
        print(f"[UYARI] {csv_path} bulunamadı. Dummy (Rastgele) veri üretiliyor...")
        dates = pd.date_range(start="2023-01-01", periods=1000, freq="D")
        df = pd.DataFrame({
            "Tarih": dates,
            "NoktaAdi": ["Point_A"] * 500 + ["Point_B"] * 500,
            "HealthFactor": np.random.uniform(0, 10, 1000)
        })
    else:
        df = pd.read_csv(csv_path)

    df["Tarih"] = pd.to_datetime(df["Tarih"])
    df = df.sort_values(["NoktaAdi", "Tarih"]).reset_index(drop=True)

    # 1. Temel Delta ve Hedefler
    df["Next_Val"] = df.groupby("NoktaAdi")[Config.TARGET_COL].shift(-1)
    df["Future_Delta"] = df["Next_Val"] - df[Config.TARGET_COL]

    # Binary Hedefler
    df["Target_Drop"] = (df["Future_Delta"] < Config.DROP_THRESHOLD).astype(int)
    df["Target_Rise"] = (df["Future_Delta"] > Config.RISE_THRESHOLD).astype(int)

    # 2. Geçmiş Özellikler (Lags)
    grp = df.groupby("NoktaAdi")[Config.TARGET_COL]
    df["Lag_1"] = grp.shift(1)  # Dün
    df["Lag_2"] = grp.shift(2)  # Önceki gün
    df["Lag_3"] = grp.shift(3)  # 3 gün önce

    # 3. İstatistiksel Özellikler (Rolling) - shift(1) ile Data Leakage önlenir
    df["Mean_3D"] = grp.shift(1).rolling(3).mean()
    df["Std_3D"] = grp.shift(1).rolling(3).std()
    df["Mean_7D"] = grp.shift(1).rolling(7).mean()

    # 4. Momentum (RSI) - shift(1) önemli
    df["RSI_7"] = df.groupby("NoktaAdi")[Config.TARGET_COL].apply(
        lambda x: calculate_rsi(x.shift(1), 7)
    ).reset_index(0, drop=True)

    # 5. Döngüsel Zaman (Cyclical Features)
    df["Month_Sin"] = np.sin(2 * np.pi * df["Tarih"].dt.month / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Tarih"].dt.month / 12)

    # Temizlik (NaN düşür)
    features = [
        "Lag_1", "Lag_2", "Lag_3",
        "Mean_3D", "Std_3D", "Mean_7D",
        "RSI_7", "Month_Sin", "Month_Cos"
    ]

    df_clean = df.dropna(subset=features + ["Target_Drop", "Target_Rise"]).reset_index(drop=True)

    print(f"[DATA] {len(df_clean)} satır hazırlandı. Özellik sayısı: {len(features)}")
    return df_clean, features


# =====================================================
# MODEL COMPETITION ENGINE (XGBOOST EKLENDİ)
# =====================================================
def get_models():
    """
    Yarıştırılacak Tüm Modelleri Döndürür.
    Buraya XGBoost, KNN, SVM, RF eklenmiştir.
    """
    models = {
        # 1. KNN: Basit, mesafe temelli
        "KNN": KNeighborsClassifier(n_neighbors=5),

        # 2. SVM: Karmaşık sınırları çizmekte ustadır
        "SVM": SVC(kernel='rbf', probability=True, random_state=Config.RANDOM_STATE),

        # 3. Random Forest: Klasik, güçlü, ensemble model
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1,
                                                random_state=Config.RANDOM_STATE),

        # 4. XGBoost: Kaggle şampiyonlarının favorisi (Hızlı ve güçlü)
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.05, n_jobs=-1,
                                 random_state=Config.RANDOM_STATE, eval_metric='logloss'),

        # 5. Gradient Boosting (Sklearn versiyonu - kıyas için)
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                        random_state=Config.RANDOM_STATE),

        # 6. Extra Trees: RF'ye benzer ama daha rastgele (bazen daha iyi geneller)
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=Config.RANDOM_STATE)
    }
    return models


def optimize_threshold(y_true, y_probs):
    """Recall >= Hedef olduğu noktada en iyi Precision'ı veren eşiği bulur."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)

    # Hedef Recall'u geçen tüm noktaları bul
    valid_mask = recalls >= Config.TARGET_RECALL
    if not np.any(valid_mask):
        return 0.5  # Hedef tutmazsa standart eşik

    valid_indices = np.where(valid_mask)[0]
    valid_indices = valid_indices[valid_indices < len(thresholds)]

    if len(valid_indices) == 0:
        return 0.5

    best_idx = valid_indices[np.argmax(precisions[valid_indices])]
    return thresholds[best_idx]


def run_model_competition(X_train_full, y_train_full, task_name="Task"):
    """
    Verilen X ve y üzerinde 6 modeli CV ile yarıştırır.
    En iyi modeli, en iyi eşik değeriyle birlikte döndürür.
    """
    print(f"\n{'=' * 60}")
    print(f"🏆 [VALIDATION PHASE] MODEL YARIŞMASI BAŞLIYOR: {task_name}")
    print(f"   > Yarışmacılar: KNN, SVM, Random Forest, XGBoost, GB, Extra Trees")
    print(f"{'=' * 60}")

    models = get_models()
    results = []

    tscv = TimeSeriesSplit(n_splits=3)

    for name, model in models.items():
        fold_f1_scores = []
        fold_thresholds = []

        print(f"   -> {name:18} eğitiliyor...", end=" ")

        for train_idx, val_idx in tscv.split(X_train_full):
            # Split
            X_t, X_v = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
            y_t, y_v = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

            # Scaling (SVM ve KNN için kritik)
            scaler = StandardScaler()
            X_t_scaled = scaler.fit_transform(X_t)
            X_v_scaled = scaler.transform(X_v)

            # SMOTE (Imbalance Fix)
            try:
                smote = SMOTE(random_state=Config.RANDOM_STATE)
                X_t_bal, y_t_bal = smote.fit_resample(X_t_scaled, y_t)
            except:
                X_t_bal, y_t_bal = X_t_scaled, y_t

            # Eğitim
            model.fit(X_t_bal, y_t_bal)

            # Olasılıklar
            probs = model.predict_proba(X_v_scaled)[:, 1]

            # Threshold Optimizasyonu
            best_thresh = optimize_threshold(y_v, probs)
            fold_thresholds.append(best_thresh)

            # Skorlama
            preds = (probs >= best_thresh).astype(int)
            fold_f1_scores.append(f1_score(y_v, preds, zero_division=0))

        avg_f1 = np.mean(fold_f1_scores)
        avg_thresh = np.mean(fold_thresholds)

        print(f"| Ort. F1: {avg_f1:.4f} | Eşik: {avg_thresh:.4f}")

        results.append({
            "name": name,
            "model": model,
            "score": avg_f1,
            "threshold": avg_thresh
        })

    # Şampiyonu Seç
    best_result = max(results, key=lambda x: x["score"])
    print(f"\n🌟 [SONUÇ] KAZANAN MODEL ({task_name}): {best_result['name']}")
    print(f"   > Sebebi: Validation setlerinde en yüksek F1 Skorunu ({best_result['score']:.4f}) verdi.")

    # Şampiyonu TÜM Train verisiyle tekrar eğit (Final Model)
    final_model = best_result["model"]
    final_scaler = StandardScaler()
    X_final_scaled = final_scaler.fit_transform(X_train_full)

    try:
        smote_final = SMOTE(random_state=Config.RANDOM_STATE)
        X_bal, y_bal = smote_final.fit_resample(X_final_scaled, y_train_full)
    except:
        X_bal, y_bal = X_final_scaled, y_train_full

    final_model.fit(X_bal, y_bal)

    return final_model, final_scaler, best_result["threshold"], best_result["name"]


# =====================================================
# MAIN PIPELINE
# =====================================================
def run_advanced_pipeline():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # 1. Veri Hazırlığı
    df, features = prepare_data(Config.CSV_PATH)

    # Train / Test Split
    split_idx = int(len(df) * 0.85)

    X_train = df.iloc[:split_idx][features]
    X_test = df.iloc[split_idx:][features]

    y_train_drop = df.iloc[:split_idx]["Target_Drop"]
    y_test_drop = df.iloc[split_idx:]["Target_Drop"]

    y_train_rise = df.iloc[:split_idx]["Target_Rise"]
    y_test_rise = df.iloc[split_idx:]["Target_Rise"]

    print(f"\n[SPLIT BİLGİSİ]")
    print(f"  > Train Set (Eğitim): {len(X_train)} satır.")
    print(f"  > Test Set (Sınav): {len(X_test)} satır.")

    # 2. DROP Modeli İçin Yarışma ve Seçim
    drop_model, drop_scaler, drop_thresh, drop_name = run_model_competition(X_train, y_train_drop, "DÜŞÜŞ (DROP)")

    # 3. RISE Modeli İçin Yarışma ve Seçim
    rise_model, rise_scaler, rise_thresh, rise_name = run_model_competition(X_train, y_train_rise, "YÜKSELİŞ (RISE)")

    # 4. Test Setinde Final Değerlendirme
    print("\n" + "=" * 60)
    print("🚀 [TEST PHASE] TEST SETİ FİNAL DEĞERLENDİRMESİ")
    print("   > Artık modelleri hiç görmedikleri verilerle sınıyoruz.")
    print("=" * 60)

    # Drop Tahminleri
    X_test_drop_scaled = drop_scaler.transform(X_test)
    drop_probs = drop_model.predict_proba(X_test_drop_scaled)[:, 1]
    drop_preds = (drop_probs >= drop_thresh).astype(int)

    # Rise Tahminleri
    X_test_rise_scaled = rise_scaler.transform(X_test)
    rise_probs = rise_model.predict_proba(X_test_rise_scaled)[:, 1]
    rise_preds = (rise_probs >= rise_thresh).astype(int)

    # --- DROP Raporu ---
    print(f"\n>>> 1. SENARYO: DÜŞÜŞ (DROP) ANALİZİ (Şampiyon Model: {drop_name})")
    explain_metrics(y_test_drop, drop_preds, drop_probs, label_name="DÜŞÜŞ")

    # Drop Grafik
    plt.figure(figsize=(6, 5))
    cm_drop = confusion_matrix(y_test_drop, drop_preds)
    sns.heatmap(cm_drop, annot=True, fmt='d', cmap='Reds', xticklabels=["Normal", "Düşüş"],
                yticklabels=["Normal", "Düşüş"])
    plt.title(f"DÜŞÜŞ - Confusion Matrix ({drop_name})")
    plt.savefig(f"{Config.OUTPUT_DIR}/best_drop_model_{drop_name}.png")
    plt.close()

    # --- RISE Raporu ---
    print(f"\n>>> 2. SENARYO: YÜKSELİŞ (RISE) ANALİZİ (Şampiyon Model: {rise_name})")
    explain_metrics(y_test_rise, rise_preds, rise_probs, label_name="YÜKSELİŞ")

    # Rise Grafik
    plt.figure(figsize=(6, 5))
    cm_rise = confusion_matrix(y_test_rise, rise_preds)
    sns.heatmap(cm_rise, annot=True, fmt='d', cmap='Greens', xticklabels=["Normal", "Yükseliş"],
                yticklabels=["Normal", "Yükseliş"])
    plt.title(f"YÜKSELİŞ - Confusion Matrix ({rise_name})")
    plt.savefig(f"{Config.OUTPUT_DIR}/best_rise_model_{rise_name}.png")
    plt.close()

    # 5. Feature Importance (Ağaç tabanlılar için)
    if hasattr(drop_model, "feature_importances_"):
        print(f"\n[ÖNEMLİ ÖZELLİKLER] {drop_name} (Drop) Modeli Neye Bakıyor?")
        imps = pd.Series(drop_model.feature_importances_, index=features).sort_values(ascending=False).head(5)
        print(imps)

    # 6. Kombine Tahminler
    final_labels = []
    for d_pred, r_pred, d_prob, r_prob in zip(drop_preds, rise_preds, drop_probs, rise_probs):
        if d_pred == 1 and r_pred == 0:
            final_labels.append("DÜŞÜŞ")
        elif r_pred == 1 and d_pred == 0:
            final_labels.append("YÜKSELİŞ")
        elif d_pred == 1 and r_pred == 1:
            # Çakışma durumunda olasılığı yüksek olanı seç
            final_labels.append("DÜŞÜŞ" if d_prob > r_prob else "YÜKSELİŞ")
        else:
            final_labels.append("SABİT")

    # Gerçek Etiketler
    true_labels = []
    for d, r in zip(y_test_drop, y_test_rise):
        if d == 1:
            true_labels.append("DÜŞÜŞ")
        elif r == 1:
            true_labels.append("YÜKSELİŞ")
        else:
            true_labels.append("SABİT")

    # Kombine Matris
    plt.figure(figsize=(8, 6))
    cm_comb = confusion_matrix(true_labels, final_labels, labels=["DÜŞÜŞ", "SABİT", "YÜKSELİŞ"])
    sns.heatmap(cm_comb, annot=True, fmt='d', cmap='Blues',
                xticklabels=["DÜŞÜŞ", "SABİT", "YÜKSELİŞ"],
                yticklabels=["DÜŞÜŞ", "SABİT", "YÜKSELİŞ"])
    plt.title("FİNAL KOMBİNE TAHMİN MATRİSİ")
    plt.ylabel("Gerçek Durum")
    plt.xlabel("Model Tahmini")
    plt.tight_layout()
    plt.savefig(f"{Config.OUTPUT_DIR}/final_combined_matrix.png")
    print(f"\n[INFO] Tüm grafikler '{Config.OUTPUT_DIR}' klasörüne kaydedildi.")


if __name__ == "__main__":
    run_advanced_pipeline()