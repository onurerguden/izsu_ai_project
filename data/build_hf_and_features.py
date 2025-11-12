from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import re

ALPHA = 1.2

# -------------------------------
# Ağırlıklar ve limitler
# -------------------------------
WEIGHTS = {
    "E.coli": 0.133,
    "Koliform Bakteri": 0.133,
    "C.Perfringens": 0.133,
    "Arsenik": 0.088,
    "Nitrit": 0.088,
    "Alüminyum": 0.055,
    "Demir": 0.055,
    "Amonyum": 0.055,
    "pH": 0.040,
    "Klorür": 0.020,
    "İletkenlik": 0.020,
    "Oksitlenebilirlik": 0.020,
    "Bulanıklık": 0.010,
    "Tat": 0.010,
    "Koku": 0.010,
    "Renk": 0.010,
    "Toplam Sertlik": 0.010,
    "Tuzluluk": 0.010,
}

LIMITS = {
    ("Arsenik", "μg/L"): 10.0,
    ("Alüminyum", "μg/L"): 200.0,
    ("Demir", "μg/L"): 200.0,
    ("Nitrit", "mg/L"): 0.5,
    ("Amonyum", "mg/L"): 0.5,
    ("Klorür", "mg/L"): 250.0,
    ("Oksitlenebilirlik", "mg/L O2"): 5.0,
    ("pH", "range"): (6.5, 9.5),
    ("İletkenlik", "µS/cm"): 2500.0,
    ("E.coli", "Sayı/100 ml"): 0.0,
    ("Koliform Bakteri", "Sayı/100 ml"): 0.0,
    ("C.Perfringens", "Sayı/100 ml"): 0.0,
    ("Bulanıklık", "acceptable"): None,
    ("Tat", "acceptable"): None,
    ("Koku", "acceptable"): None,
    ("Renk", "acceptable"): None,
    ("Toplam Sertlik", "mg/L as CaCO3"): None,
    ("Tuzluluk", "ppt"): None,
}

STATE_NAME = "last_hf_success_date.txt"


# -------------------------------
# Yardımcı fonksiyonlar
# -------------------------------
def scoring_param_name(p: str) -> str:
    """Parametre adını WEIGHTS/LIMITS ile uyumlu hale getir."""
    if not isinstance(p, str):
        return ""
    s = p.strip()
    low = s.lower()

    # E. coli varyantları
    if low.replace(" ", "") in {"e.coli", "ecoli", "e-coli"}:
        return "E.coli"

    if low == "koliform bakteri":
        return "Koliform Bakteri"
    if low == "c.perfringens":
        return "C.Perfringens"

    return s


def standardize_to(value: float, unit: str, target: str):
    """Birim dönüşümü (şimdilik µg/L <-> mg/L)."""
    if pd.isna(value) or not isinstance(unit, str):
        return np.nan
    u = unit.strip().replace("µ", "μ")
    t = target.strip().replace("µ", "μ")
    if u == t:
        return float(value)
    if (u, t) == ("μg/L", "mg/L"):
        return float(value) / 1000.0
    if (u, t) == ("mg/L", "μg/L"):
        return float(value) * 1000.0
    return np.nan


def score_linear_limit(value: float, limit_value: float) -> float:
    if pd.isna(value) or pd.isna(limit_value) or limit_value <= 0:
        return np.nan
    r = float(np.clip(value / limit_value, 0.0, 1.0))
    bad = r ** ALPHA
    return 1.0 - bad


def score_pH(ph: float, lo: float, hi: float) -> float:
    if pd.isna(ph):
        return np.nan
    if lo <= ph <= hi:
        return 1.0
    center = (lo + hi) / 2.0
    halfwidth = (hi - lo) / 2.0
    dev = abs(ph - center)
    r = np.clip(dev / halfwidth, 0.0, 1.0)
    return 1.0 - (r ** ALPHA)


def score_acceptable(raw_value: str) -> float:
    if not isinstance(raw_value, str):
        return np.nan
    s = raw_value.strip().lower()
    if re.fullmatch(r"(uygun|geçerli|gecerli|0|yok|nd|-|—)", s):
        return 1.0
    return np.nan


def fail_fast_trigger(param: str, value: float, unit: str) -> bool:
    """HF ne olursa olsun otomatik Risk yapan durumlar."""
    p = scoring_param_name(param)
    if p == "E.coli":
        return (not pd.isna(value)) and value > 0
    if p == "Arsenik":
        v = standardize_to(value, unit, "μg/L")
        return (not pd.isna(v)) and v > 10.0
    if p == "Nitrit":
        v = standardize_to(value, unit, "mg/L")
        return (not pd.isna(v)) and v > 0.5
    return False


def classify_hf(hf: float, fail_fast: bool) -> str:
    if fail_fast:
        return "Risk"
    if pd.isna(hf):
        return "Unknown"
    if hf >= 85:
        return "Good"
    if hf >= 60:
        return "Caution"
    return "Risk"


def find_clean_yeni_csv(script_path: Path) -> tuple[Path, Path]:
    """izsu_data_cleaned_yeni.csv dosyasını bul."""
    script_dir = script_path.parent
    candidates = [
        script_dir / "izsu_data_cleaned.csv",
        script_dir / "data" / "izsu_data_cleaned.csv",
        script_dir.parent / "data" / "izsu_data_cleaned.csv",
    ]
    for p in candidates:
        if p.exists():
            return p.parent, p
    # Hiçbiri yoksa script ile aynı klasörde varsay
    return script_dir, script_dir / "izsu_data_cleaned.csv"


def read_state_date(path: Path):
    if not path.exists():
        return None
    txt = path.read_text(encoding="utf-8").strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(txt, fmt).date()
        except ValueError:
            continue
    return None


def compute_scores_for_row(group_df: pd.DataFrame) -> dict:
    """Tek bir (Tarih, Nokta) grubu için tüm parametre skorlarını ve HF'yi hesapla."""
    out_scores, ff = {}, False
    by_param = group_df.groupby("ParametreAdi", dropna=False)

    for param_raw, sub in by_param:
        p = scoring_param_name(param_raw)

        # Deger_Num zaten numeric, ama gene de to_numeric ile güvence al
        v = pd.to_numeric(sub["Deger"], errors="coerce").astype(float).mean(skipna=True)

        unit_series = sub["Birim"].dropna().astype(str)
        unit = None if unit_series.empty else unit_series.iloc[0].replace("µ", "μ")

        raw_series = sub["DegerRaw"].dropna().astype(str)
        raw_sample = None if raw_series.empty else raw_series.iloc[0]

        # Fail-fast tetikleyici
        if fail_fast_trigger(p, v, unit or ""):
            ff = True

        score = np.nan

        # Sayı/100 ml (E.coli, Koliform, C.perfringens)
        if (p, "Sayı/100 ml") in LIMITS:
            if pd.isna(v):
                score = np.nan
            else:
                score = 1.0 if float(v) == 0.0 else 0.0

        # pH aralık
        elif (p, "range") in LIMITS:
            lo, hi = LIMITS[(p, "range")]
            score = score_pH(v, lo, hi)

        # İletkenlik
        elif (p, "µS/cm") in LIMITS or (p, "μS/cm") in LIMITS:
            thr = LIMITS.get((p, "µS/cm"), LIMITS.get((p, "μS/cm")))
            vv = standardize_to(v, unit or "µS/cm", "µS/cm")
            score = score_linear_limit(vv, thr)

        # Oksitlenebilirlik
        elif (p, "mg/L O2") in LIMITS:
            thr = LIMITS[(p, "mg/L O2")]
            vv = v
            score = score_linear_limit(vv, thr)

        # mg/L limitleri
        elif (p, "mg/L") in LIMITS:
            thr = LIMITS[(p, "mg/L")]
            vv = standardize_to(v, unit or "mg/L", "mg/L")
            score = score_linear_limit(vv, thr)

        # μg/L limitleri
        elif (p, "μg/L") in LIMITS:
            thr = LIMITS[(p, "μg/L")]
            vv = standardize_to(v, unit or "μg/L", "μg/L")
            score = score_linear_limit(vv, thr)

        # "Uygun / Geçerli" tipi parametreler
        elif (p, "acceptable") in LIMITS:
            score = score_acceptable(str(raw_sample) if raw_sample is not None else "")

        out_scores[p] = score

    # Ağırlıklı ortalama ile HF
    num, den = 0.0, 0.0
    for p, w in WEIGHTS.items():
        s = out_scores.get(p, np.nan)
        if not pd.isna(s):
            num += w * s
            den += w
    hf = np.nan if den == 0 else (100.0 * num / den)
    return {"scores": out_scores, "fail_fast": ff, "hf": hf}


# -------------------------------
# main
# -------------------------------
def main():
    script_path = Path(__file__).resolve()
    data_dir, in_path = find_clean_yeni_csv(script_path)

    if not in_path.exists():
        print(f"[!] Girdi yok: {in_path}")
        return

    hf_out_path = data_dir / "izsu_health_factor.csv"
    feat_out_path = data_dir / "izsu_features.csv"
    state = data_dir / STATE_NAME

    # İncremental için son tarih
    last_date = read_state_date(state)
    if hf_out_path.exists():
        try:
            tmp = pd.read_csv(hf_out_path, encoding="utf-8-sig")
            dmax = pd.to_datetime(tmp["Tarih"], errors="coerce").dt.date.max()
            if pd.notna(dmax):
                last_date = max(last_date or dmax, dmax)
        except Exception:
            pass
    if feat_out_path.exists():
        try:
            tmp = pd.read_csv(feat_out_path, encoding="utf-8-sig")
            dmax2 = pd.to_datetime(tmp["Tarih"], errors="coerce").dt.date.max()
            if pd.notna(dmax2):
                last_date = max(last_date or dmax2, dmax2)
        except Exception:
            pass

    print(f"[i] Girdi: {in_path}")
    if last_date:
        print(f"[i] Son HF tarihi (state): {last_date}")

    # Yeni cleaned dosyayı oku
    raw = pd.read_csv(in_path, encoding="utf-8-sig")

    # Çalışma dataframe'i: eski pipe'a benzer kolon isimleri
    work = pd.DataFrame()
    work["Tarih"] = pd.to_datetime(raw["Tarih_Clean"], errors="coerce").dt.date
    work["NoktaAdi"] = raw["NoktaAdi_Clean"]
    work["ParametreAdi"] = raw["ParametreAdi_Clean"]
    work["Birim"] = raw["Birim_Clean"]
    work["Deger"] = raw["Deger_Num"]
    work["DegerRaw"] = raw["DegerRaw"]

    work = work.dropna(subset=["Tarih", "NoktaAdi", "ParametreAdi"])

    # İncremental filtre
    if last_date:
        work = work[work["Tarih"] > last_date]

    if work.empty:
        print("[i] Hesaplanacak yeni kayıt yok. Çıkılıyor.")
        return

    # HF hesapla
    groups = work.groupby(["Tarih", "NoktaAdi"], dropna=False)
    rows = []
    for (dt, pt), g in groups:
        res = compute_scores_for_row(g)
        row = {
            "Tarih": dt,
            "NoktaAdi": pt,
            "HealthFactor": res["hf"],
            "FailFast": res["fail_fast"],
        }
        for p in WEIGHTS.keys():
            row[f"{p}_score"] = res["scores"].get(p, np.nan)
        row["RiskClass"] = classify_hf(row["HealthFactor"], row["FailFast"])
        rows.append(row)

    hf_new = (
        pd.DataFrame(rows)
        .sort_values(["Tarih", "NoktaAdi"])
        .reset_index(drop=True)
    )

    # Geniş (feature) tablo
    wide_vals = (
        work
        .pivot_table(
            index=["Tarih", "NoktaAdi"],
            columns="ParametreAdi",
            values="Deger",
            aggfunc="mean",
        )
        .reset_index()
    )

    features_new = pd.merge(wide_vals, hf_new, on=["Tarih", "NoktaAdi"], how="left")

    # Eski dosyalarla birleştir (incremental)
    if hf_out_path.exists():
        base = pd.read_csv(hf_out_path, encoding="utf-8-sig")
        hf_all = pd.concat([base, hf_new], ignore_index=True)
        hf_all = hf_all.drop_duplicates(subset=["Tarih", "NoktaAdi"])
    else:
        hf_all = hf_new

    if feat_out_path.exists():
        base = pd.read_csv(feat_out_path, encoding="utf-8-sig")
        features_all = pd.concat([base, features_new], ignore_index=True)
        features_all = features_all.drop_duplicates(subset=["Tarih", "NoktaAdi"])
    else:
        features_all = features_new

    hf_all = hf_all.sort_values(["Tarih", "NoktaAdi"]).reset_index(drop=True)
    features_all = features_all.sort_values(["Tarih", "NoktaAdi"]).reset_index(drop=True)

    hf_all.to_csv(hf_out_path, index=False, encoding="utf-8-sig")
    features_all.to_csv(feat_out_path, index=False, encoding="utf-8-sig")

    max_date = pd.to_datetime(hf_all["Tarih"], errors="coerce").dt.date.max()
    if pd.notna(max_date):
        state.write_text(max_date.strftime("%Y-%m-%d"), encoding="utf-8")

    print("--------------------------------------------------")
    print("HF & Features (incremental) tamamlandı ✅")
    print(f"Yeni HF satırı: {len(hf_new)} | Toplam HF: {len(hf_all)}")
    print(f"Yeni features: {len(features_new)} | Toplam features: {len(features_all)}")
    print(f"Son tarih: {max_date}")
    print(f"HF CSV   : {hf_out_path}")
    print(f"Feat CSV : {feat_out_path}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
