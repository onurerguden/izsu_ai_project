from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import re

ALPHA = 1.2

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

def standardize_to(value: float, unit: str, target: str):
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
    if param == "E.coli":
        return (not pd.isna(value)) and value > 0
    if param == "Arsenik":
        v = standardize_to(value, unit, "μg/L")
        return (not pd.isna(v)) and v > 10.0
    if param == "Nitrit":
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

def find_clean_csv(script_path: Path) -> tuple[Path, Path]:
    script_dir = script_path.parent
    candidates = [
        script_dir / "izsu_data_cleaned.csv",
        script_dir / "data" / "izsu_data_cleaned.csv",
        script_dir.parent / "data" / "izsu_data_cleaned.csv",
    ]
    for p in candidates:
        if p.exists():
            return p.parent, p
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
    out_scores, ff = {}, False
    by_param = group_df.groupby("ParametreAdi", dropna=False)
    for param, sub in by_param:
        v = sub["Deger"].astype(float).mean(skipna=True)
        unit = sub["Birim"].dropna().astype(str).head(1)
        unit = None if unit.empty else unit.iloc[0].replace("µ", "μ")
        raw_sample = sub["DegerRaw"].dropna().astype(str).head(1)
        raw_sample = None if raw_sample.empty else raw_sample.iloc[0]

        if fail_fast_trigger(param, v, unit or ""):
            ff = True

        score = np.nan
        if (param, "Sayı/100 ml") in LIMITS:
            score = 1.0 if (not pd.isna(v) and v == 0) else (0.0 if not pd.isna(v) else np.nan)
        elif (param, "range") in LIMITS:
            lo, hi = LIMITS[(param, "range")]
            score = score_pH(v, lo, hi)
        elif (param, "µS/cm") in LIMITS:
            thr = LIMITS[(param, "µS/cm")]
            vv = standardize_to(v, unit or "µS/cm", "µS/cm")
            score = score_linear_limit(vv, thr)
        elif (param, "mg/L O2") in LIMITS:
            thr = LIMITS[(param, "mg/L O2")]
            vv = v
            score = score_linear_limit(vv, thr)
        elif (param, "mg/L") in LIMITS:
            thr = LIMITS[(param, "mg/L")]
            vv = standardize_to(v, unit or "mg/L", "mg/L")
            score = score_linear_limit(vv, thr)
        elif (param, "μg/L") in LIMITS:
            thr = LIMITS[(param, "μg/L")]
            vv = standardize_to(v, unit or "μg/L", "μg/L")
            score = score_linear_limit(vv, thr)
        elif (param, "acceptable") in LIMITS:
            score = score_acceptable(str(raw_sample) if raw_sample is not None else "")
        out_scores[param] = score

    num, den = 0.0, 0.0
    for p, w in WEIGHTS.items():
        s = out_scores.get(p, np.nan)
        if not pd.isna(s):
            num += w * s
            den += w
    hf = np.nan if den == 0 else (100.0 * num / den)
    return {"scores": out_scores, "fail_fast": ff, "hf": hf}

def main():
    script_path = Path(__file__).resolve()
    data_dir, in_path = find_clean_csv(script_path)

    if not in_path.exists():
        print(f"[!] Girdi yok: {in_path}")
        return

    hf_out_path = data_dir / "izsu_health_factor.csv"
    feat_out_path = data_dir / "izsu_features.csv"
    state = data_dir / STATE_NAME

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
        print(f"[i] Son HF tarihi: {last_date}")

    df = pd.read_csv(in_path, encoding="utf-8-sig")
    df["__Tarih"] = pd.to_datetime(df["Tarih"], errors="coerce").dt.date
    if last_date:
        df = df[df["__Tarih"] > last_date]
    if df.empty:
        print("[i] Hesaplanacak yeni kayıt yok. Çıkılıyor.")
        return

    groups = df.groupby(["Tarih", "NoktaAdi"], dropna=False)
    rows = []
    for (dt, pt), g in groups:
        res = compute_scores_for_row(g)
        row = {"Tarih": dt, "NoktaAdi": pt, "HealthFactor": res["hf"], "FailFast": res["fail_fast"]}
        for p in WEIGHTS.keys():
            row[f"{p}_score"] = res["scores"].get(p, np.nan)
        row["RiskClass"] = classify_hf(row["HealthFactor"], row["FailFast"])
        rows.append(row)
    hf_new = pd.DataFrame(rows).sort_values(["Tarih", "NoktaAdi"]).reset_index(drop=True)

    wide_vals = (
        df.drop(columns=["__Tarih"])
          .pivot_table(index=["Tarih", "NoktaAdi"], columns="ParametreAdi", values="Deger", aggfunc="mean")
          .reset_index()
    )
    features_new = pd.merge(wide_vals, hf_new, on=["Tarih", "NoktaAdi"], how="left")

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
    print(f"Yeni HF satırı: {len(hf_new)} | Toplam: {len(hf_all)}")
    print(f"Yeni features: {len(features_new)} | Toplam: {len(features_all)}")
    print(f"Son tarih: {max_date}")
    print(f"HF CSV   : {hf_out_path}")
    print(f"Feat CSV : {feat_out_path}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
