from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import re


STATE_NAME = "last_hf_success_date.txt"


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



def scoring_param_name(p: str) -> str:
    if not isinstance(p, str):
        return ""
    s = p.strip()
    low = s.lower()
    if low.replace(" ", "") in {"e.coli", "ecoli", "e-coli"}:
        return "E.coli"
    if low == "koliform bakteri":
        return "Koliform Bakteri"
    if low == "c.perfringens":
        return "C.Perfringens"
    return s


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


def score_acceptable(raw: str):
    if not isinstance(raw, str):
        return np.nan
    s = raw.strip().lower()
    if re.fullmatch(r"(uygun|geçerli|gecerli|0|yok|nd|-|—)", s):
        return 1.0
    return np.nan


def fail_fast_trigger(param, value, unit):
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


def compute_qn_linear(v, S, ideal=0.0):
    if pd.isna(v) or pd.isna(S) or S <= ideal:
        return np.nan
    return abs(v - ideal) / (S - ideal) * 100.0



def compute_qn_ph(ph, lo, hi, ideal=7.0):
    if pd.isna(ph):
        return np.nan
    if ph >= ideal:
        denom = hi - ideal
    else:
        denom = ideal - lo
    if denom <= 0:
        return np.nan
    return abs(ph - ideal) / denom * 100.0


def get_numeric_limit(param: str):
    for (p, u), val in LIMITS.items():
        if p != param:
            continue
        if val is None:
            continue
        if u in ("range", "acceptable", "Sayı/100 ml"):
            continue
        return float(val), u
    return None, None


S_NUMERIC = {}
for (p, u), val in LIMITS.items():
    if val is None:
        continue
    if u == "range":
        if p == "pH":
            lo, hi = val
            S_NUMERIC[p] = float(hi)
        continue
    if u in ("acceptable", "Sayı/100 ml"):
        continue
    S_NUMERIC[p] = float(val)

if S_NUMERIC:
    K = 1.0 / sum(1.0 / v for v in S_NUMERIC.values())
    W_UNIT = {p: K / v for p, v in S_NUMERIC.items()}
else:
    K = 0.0
    W_UNIT = {}


def classify_hf(hf, fail_fast):
    if fail_fast:
        return "Risk"
    if pd.isna(hf):
        return "Unknown"
    if hf >= 85:
        return "Good"
    if hf >= 60:
        return "Caution"
    return "Risk"


def compute_scores_for_row(group_df: pd.DataFrame) -> dict:
    q_values: dict[str, float] = {}
    scores: dict[str, float] = {}
    ff = False

    for param_raw, sub in group_df.groupby("ParametreAdi", dropna=False):
        p = scoring_param_name(param_raw)

        v = pd.to_numeric(sub["Deger"], errors="coerce").astype(float).mean(skipna=True)
        unit_series = sub["Birim"].dropna().astype(str).replace("µ", "μ")
        unit = unit_series.iloc[0] if not unit_series.empty else None
        raw_series = sub["DegerRaw"].dropna().astype(str)
        raw_sample = raw_series.iloc[0] if not raw_series.empty else None

        if fail_fast_trigger(p, v, unit or ""):
            ff = True

        qn = np.nan
        if (p, "Sayı/100 ml") in LIMITS:
            if pd.isna(v):
                qn = np.nan
            else:
                qn = 0.0 if float(v) == 0.0 else 100.0

        elif (p, "range") in LIMITS and p == "pH":
            lo, hi = LIMITS[(p, "range")]
            qn = compute_qn_ph(v, lo, hi, ideal=7.0)

        else:
            limit_val, limit_unit = get_numeric_limit(p)
            if limit_val is not None:
                vv = standardize_to(v, unit or limit_unit, limit_unit)
                qn = compute_qn_linear(vv, limit_val, ideal=0.0)

            elif (p, "acceptable") in LIMITS:
                ok = score_acceptable(raw_sample or "")
                if not pd.isna(ok):
                    qn = 0.0 if ok == 1.0 else 100.0
                else:
                    qn = np.nan
            else:
                qn = np.nan

        q_values[p] = qn
        if not pd.isna(qn):
            scores[p] = max(0.0, 1.0 - max(qn, 0.0) / 100.0)
        else:
            scores[p] = np.nan

    num, den = 0.0, 0.0
    for p, qn in q_values.items():
        if p not in W_UNIT or pd.isna(qn):
            continue
        w = W_UNIT[p]
        num += w * qn
        den += w

    wqi = num / den if den > 0 else np.nan
    if pd.isna(wqi):
        hf = np.nan
    else:
        hf = max(0.0, 100.0 - wqi)

    return {"scores": scores, "fail_fast": ff, "hf": hf}

def main():
    script_path = Path(__file__).resolve()
    data_dir, in_path = find_clean_csv(script_path)

    print("[i] Script klasörü :", script_path.parent)
    print("[i] Data klasörü   :", data_dir)
    print("[i] Girdi dosyası  :", in_path)

    if not in_path.exists():
        print("[!] Girdi yok:", in_path)
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

    if last_date:
        print(f"[i] Son HF tarihi (state/hf/features): {last_date}")

    raw = pd.read_csv(in_path, encoding="utf-8-sig")

    work = pd.DataFrame()
    work["Tarih"] = pd.to_datetime(raw["Tarih_Clean"], errors="coerce").dt.date
    work["NoktaAdi"] = raw["NoktaAdi_Clean"]
    work["ParametreAdi"] = raw["ParametreAdi_Clean"]
    work["Birim"] = raw["Birim_Clean"]
    work["Deger"] = raw["Deger_Num"]
    work["DegerRaw"] = raw["DegerRaw"]

    work = work.dropna(subset=["Tarih", "NoktaAdi", "ParametreAdi"])

    if last_date:
        work = work[work["Tarih"] > last_date]

    if work.empty:
        print("[i] Hesaplanacak yeni kayıt yok. Çıkılıyor.")
        return

    groups = work.groupby(["Tarih", "NoktaAdi"], dropna=False)
    rows = []
    for (dt, pt), g in groups:
        res = compute_scores_for_row(g)
        row = {
            "Tarih": dt,
            "NoktaAdi": pt,
            "HealthFactor": res["hf"],
            "FailFast": res["fail_fast"],
            "RiskClass": classify_hf(res["hf"], res["fail_fast"]),
        }
        for p, s in res["scores"].items():
            row[f"{p}_score"] = s
        rows.append(row)

    hf_new = (
        pd.DataFrame(rows)
        .sort_values(["Tarih", "NoktaAdi"])
        .reset_index(drop=True)
    )

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

    print(f"Yeni HF satırı   : {len(hf_new)} | Toplam HF: {len(hf_all)}")
    print(f"Yeni features satı: {len(features_new)} | Toplam features: {len(features_all)}")
    print(f"Son tarih (state): {max_date}")
    print(f"HF CSV           : {hf_out_path}")
    print(f"Features CSV     : {feat_out_path}")
    print(f"State file       : {state}")

if __name__ == "__main__":
    main()
