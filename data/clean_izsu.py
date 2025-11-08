import re
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

INPUT_NAME  = "izsu_data.csv"
OUTPUT_NAME = "izsu_data_cleaned.csv"
STATE_NAME  = "last_clean_success_date.txt"  # sadece clean aşaması


def normalize_whitespace(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = re.sub(r"\s+", " ", s.strip())
    return s.replace("\u00A0", " ").strip()

def standardize_micro_symbol(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.replace("µ", "μ")
    s = s.replace("ug/L", "μg/L").replace("UG/L", "μg/L").replace("uG/L", "μg/L")
    s = re.sub(r"\bu[gG]/L\b", "μg/L", s)
    return s

def strip_parens(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return re.sub(r"[\(\[].*?[\)\]]", "", text).strip()

def normalize_param_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    raw = normalize_whitespace(name)
    base = strip_parens(raw)
    base_low = base.lower()

    if "nitrat" in base_low or "no3" in base_low:
        return "Nitrat"
    if "arsen" in base_low:
        return "Arsenik"
    if "flor" in base_low or "fluor" in base_low:
        return "Florür"
    if "amon" in base_low or "nh4" in base_low:
        return "Amonyum"
    if "nitrit" in base_low or "no2" in base_low:
        return "Nitrit"
    if base_low.replace("-", "").replace(" ", "") == "ph":
        return "pH"
    if "iletkenlik" in base_low or re.search(r"\bec\b", base_low):
        return "İletkenlik"
    if "sertlik" in base_low:
        return "Toplam Sertlik"
    if "klor" in base_low and "serbest" in base_low:
        return "Serbest Klor"
    if "klorür" in base_low or "klorur" in base_low or "cl-" in base_low:
        return "Klorür"
    if "klor" in base_low:
        return "Klor"
    if "mangan" in base_low:
        return "Mangan"
    if "demir" in base_low or re.search(r"\bfe\b", base_low):
        return "Demir"
    if "sodyum" in base_low or re.search(r"\bna\b", base_low):
        return "Sodyum"
    if "potasyum" in base_low or re.search(r"\bk\b", base_low):
        return "Potasyum"
    if "kalsiyum" in base_low or re.search(r"\bca\b", base_low):
        return "Kalsiyum"
    if "magnezyum" in base_low or base_low == "mg":
        return "Magnezyum"
    if "sülfat" in base_low or "sulfat" in base_low or "so4" in base_low:
        return "Sülfat"
    if "akm" in base_low or "bulan" in base_low or "turb" in base_low:
        return "Bulanıklık"
    return base.title()

def normalize_point_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = normalize_whitespace(name).strip(" .")
    return " ".join([w if (w.isupper() and len(w) > 2) else w.capitalize() for w in s.split()])

def parse_value(raw):
    import math
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return (math.nan, None, math.nan)
    s = str(raw).strip()
    s = s.replace(",", ".").replace("≤", "<=").replace("≥", ">=")
    if re.fullmatch(r"(?i)(nd|yok|na|n/a|—|-|yoktur|bos|boş|uygun|geçerli)", s):
        return (float("nan"), "ND", float("nan"))
    m = re.match(r"^\s*(?P<prefix><=|>=|<|>|≈|~)?\s*(?P<num>\d+(\.\d+)?)", s)
    if not m:
        m2 = re.search(r"(\d+(\.\d+)?)", s)
        if not m2:
            return (float("nan"), None, float("nan"))
    else:
        prefix = m.group("prefix")
        num = float(m.group("num"))
        if prefix in ("<", "<="):
            return (num * 0.5, "<", num)
        if prefix in (">", ">="):
            return (num, ">", num)
        if prefix in ("≈", "~"):
            return (num, "~", num)
        return (num, None, num)
    num = float(m2.group(1))
    return (num, None, num)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    def col_like(target):
        for c in df.columns:
            if c.lower().strip() == target:
                return c
        for c in df.columns:
            if target in c.lower():
                return c
        return None

    c_tarih = col_like("tarih") or "Tarih"
    c_nokta = col_like("noktaadi") or "NoktaAdi"
    c_param = col_like("parametreadi") or "ParametreAdi"
    c_birim = col_like("birim") or "Birim"
    c_deger = col_like("deger") or "Deger"

    for req in [c_tarih, c_nokta, c_param, c_birim, c_deger]:
        if req not in df.columns:
            raise ValueError(f"Gerekli sütun eksik: {req}")

    out = pd.DataFrame()
    out["Tarih"] = pd.to_datetime(df[c_tarih], dayfirst=True, errors="coerce")
    out["NoktaAdi"] = df[c_nokta].apply(normalize_point_name)
    out["ParametreAdiRaw"] = df[c_param].astype(str).map(normalize_whitespace)
    out["ParametreAdi"] = df[c_param].apply(normalize_param_name)
    out["Birim"] = df[c_birim].astype(str).map(normalize_whitespace).map(standardize_micro_symbol)
    out["DegerRaw"] = df[c_deger].astype(str).map(normalize_whitespace)

    parsed = out["DegerRaw"].apply(parse_value)
    out["Deger"], out["DegerCensor"], out["DegerLimit"] = zip(*parsed)

    out = out.dropna(subset=["Tarih"])
    out = out[out["NoktaAdi"] != ""]
    out = out[out["ParametreAdi"] != ""]
    out = out.drop_duplicates(subset=["Tarih", "NoktaAdi", "ParametreAdi", "Birim", "Deger"])
    out = out.sort_values(["Tarih", "NoktaAdi", "ParametreAdi"]).reset_index(drop=True)
    return out

def find_data_dir(script_dir: Path) -> Path:
    cand = script_dir / "data"
    return cand if cand.exists() else script_dir

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

def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = find_data_dir(script_dir)
    csv_dir = data_dir

    in_path  = csv_dir / INPUT_NAME
    out_path = csv_dir / OUTPUT_NAME
    state    = csv_dir / STATE_NAME

    if not in_path.exists():
        alt = script_dir / INPUT_NAME
        if alt.exists():
            in_path = alt
            csv_dir = script_dir
            out_path = csv_dir / OUTPUT_NAME
            state = csv_dir / STATE_NAME

    if not in_path.exists():
        print(f"[!] Girdi bulunamadı: {in_path}")
        sys.exit(1)

    last_date = read_state_date(state)
    if out_path.exists():
        try:
            existing = pd.read_csv(out_path, encoding="utf-8-sig")
            ex_max = pd.to_datetime(existing["Tarih"], errors="coerce").dt.date.max()
            if pd.notna(ex_max):
                last_date = max(last_date or ex_max, ex_max)
        except Exception:
            pass

    print(f"[i] Girdi: {in_path}")
    if last_date:
        print(f"[i] Son temizlenen tarih: {last_date}")

    raw_df = pd.read_csv(in_path, encoding="utf-8-sig")
    raw_df["__Tarih"] = pd.to_datetime(raw_df["Tarih"], dayfirst=True, errors="coerce").dt.date

    if last_date:
        raw_df = raw_df[raw_df["__Tarih"] > last_date]

    if raw_df.empty:
        print("[i] Temizlenecek yeni kayıt yok. Çıkılıyor.")
        sys.exit(0)

    cleaned_new = clean_dataframe(raw_df.drop(columns=["__Tarih"]))

    if out_path.exists():
        base = pd.read_csv(out_path, encoding="utf-8-sig")
        combined = pd.concat([base, cleaned_new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Tarih", "NoktaAdi", "ParametreAdi", "Birim", "Deger"])
    else:
        combined = cleaned_new

    combined = combined.sort_values(["Tarih", "NoktaAdi", "ParametreAdi"]).reset_index(drop=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")

    max_date = pd.to_datetime(combined["Tarih"], errors="coerce").dt.date.max()
    if pd.notna(max_date):
        state.write_text(max_date.strftime("%Y-%m-%d"), encoding="utf-8")

    print("--------------------------------------------------")
    print("Temizlik (incremental) tamamlandı ✅")
    print(f"Yeni eklenen satır: {len(cleaned_new)}")
    print(f"Güncel kayıt sayısı: {len(combined)}")
    print(f"Son tarih: {max_date}")
    print(f"Çıktı: {out_path}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
