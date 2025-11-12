# clean_izsu_lossless.py
import re
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

INPUT_NAME  = "izsu_data.csv"
OUTPUT_NAME = "izsu_data_cleaned.csv"   # aynı ad; istersen _lossless yap
STATE_NAME  = "last_clean_success_date.txt"  # lossless olsa bile bırakıyoruz (opsiyonel)

# ---------- helpers ----------
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
    if "nitrat" in base_low or "no3" in base_low: return "Nitrat"
    if "arsen" in base_low: return "Arsenik"
    if "flor" in base_low or "fluor" in base_low: return "Florür"
    if "amon" in base_low or "nh4" in base_low: return "Amonyum"
    if "nitrit" in base_low or "no2" in base_low: return "Nitrit"
    if base_low.replace("-", "").replace(" ", "") == "ph": return "pH"
    if "iletkenlik" in base_low or re.search(r"\bec\b", base_low): return "İletkenlik"
    if "sertlik" in base_low: return "Toplam Sertlik"
    if "klor" in base_low and "serbest" in base_low: return "Serbest Klor"
    if "klorür" in base_low or "klorur" in base_low or "cl-" in base_low: return "Klorür"
    if "klor" in base_low: return "Klor"
    if "mangan" in base_low: return "Mangan"
    if "demir" in base_low or re.search(r"\bfe\b", base_low): return "Demir"
    if "sodyum" in base_low or re.search(r"\bna\b", base_low): return "Sodyum"
    if "potasyum" in base_low or re.search(r"\bk\b", base_low): return "Potasyum"
    if "kalsiyum" in base_low or re.search(r"\bca\b", base_low): return "Kalsiyum"
    if "magnezyum" in base_low or base_low == "mg": return "Magnezyum"
    if "sülfat" in base_low or "sulfat" in base_low or "so4" in base_low: return "Sülfat"
    if "akm" in base_low or "bulan" in base_low or "turb" in base_low: return "Bulanıklık"
    return base.title()

def parse_value(raw):
    import math
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return (math.nan, None, math.nan)
    s = str(raw).strip()
    s = s.replace(",", ".").replace("≤", "<=").replace("≥", ">=")
    # UYARI: lossless modda metinsel “UYGUN/ND” satırlarını da KORUYORUZ (Deger NaN kalabilir)
    if re.fullmatch(r"(?i)(nd|yok|na|n/a|—|-|yoktur|bos|boş|uygun|geçerli)", s):
        return (float("nan"), "TEXT", float("nan"))
    m = re.match(r"^\s*(?P<prefix><=|>=|<|>|≈|~)?\s*(?P<num>\d+(\.\d+)?)", s)
    if m:
        prefix = m.group("prefix")
        num = float(m.group("num"))
        if prefix in ("<", "<="): return (num * 0.5, "<", num)
        if prefix in (">", ">="): return (num, ">", num)
        if prefix in ("≈", "~"):  return (num, "~", num)
        return (num, None, num)
    # Gövde içinde sayı arama (ör. "553 (25 °C)")
    m2 = re.search(r"(\d+(\.\d+)?)", s)
    if m2:
        num = float(m2.group(1))
        return (num, None, num)
    return (float("nan"), None, float("nan"))

def prefer_measure_date(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    s1 = pd.to_datetime(df["SonucTarihi"], dayfirst=True, errors="coerce") if "SonucTarihi" in df.columns else pd.Series([pd.NaT]*n)
    s2 = pd.to_datetime(df["Tarih"],        dayfirst=True, errors="coerce") if "Tarih"        in df.columns else pd.Series([pd.NaT]*n)
    out = s1.combine_first(s2)
    out.index = df.index
    return out

def find_data_dir(script_dir: Path) -> Path:
    cand = script_dir / "data"
    return cand if cand.exists() else script_dir

# ---------- lossless temizleme ----------
def clean_dataframe_lossless(df: pd.DataFrame) -> pd.DataFrame:
    # Orijinal kolonları aynen taşı
    out = df.copy()

    # Normalize edilmiş yardımcı kolonlar (yeni kolonlar ekliyoruz; hiçbir satır silinmeyecek)
    out["Tarih_Clean"] = prefer_measure_date(out)  # NaT olabilir, satır korunur

    # Nokta adı temiz varyant (hem NoktaTanimi hem NoktaAdi varsa tercih sırası)
    pt = None
    if "NoktaTanimi" in out.columns:
        pt = out["NoktaTanimi"].astype(str).map(normalize_whitespace)
    elif "NoktaAdi" in out.columns:
        pt = out["NoktaAdi"].astype(str).map(normalize_whitespace)
    if pt is not None:
        out["NoktaAdi_Clean"] = pt.apply(lambda s: " ".join([w if (w.isupper() and len(w) > 2) else w.capitalize() for w in s.strip(" .").split()]))

    # Parametre adı + birim normalize
    if "ParametreAdi" in out.columns:
        out["ParametreAdiRaw"]  = out["ParametreAdi"].astype(str).map(normalize_whitespace)
        out["ParametreAdi_Clean"] = out["ParametreAdi"].apply(normalize_param_name)
    if "Birim" in out.columns:
        out["Birim_Clean"] = out["Birim"].astype(str).map(normalize_whitespace).map(standardize_micro_symbol)

    # Değer parse – ama satırları ASLA atma/tekilleştirme yapma
    if "Deger" in out.columns:
        out["DegerRaw"] = out["Deger"].astype(str).map(normalize_whitespace)
        parsed = out["DegerRaw"].apply(parse_value)
        out["Deger_Num"], out["Deger_Censor"], out["Deger_Limit"] = zip(*parsed)

    # Sıralama sadece görüntü için; satır sayısını etkilemez
    sort_keys = [k for k in ["Tarih_Clean","NoktaId","ParametreAdi_Clean","Birim_Clean"] if k in out.columns]
    if sort_keys:
        out = out.sort_values(sort_keys, kind="stable").reset_index(drop=True)

    return out

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
    data_dir   = find_data_dir(script_dir)
    in_path  = data_dir / INPUT_NAME
    out_path = data_dir / OUTPUT_NAME
    state    = data_dir / STATE_NAME

    if not in_path.exists():
        alt = script_dir / INPUT_NAME
        if alt.exists():
            in_path = alt
            out_path = script_dir / OUTPUT_NAME
            state = script_dir / STATE_NAME

    if not in_path.exists():
        print(f"[!] Girdi bulunamadı: {in_path}")
        sys.exit(1)

    raw_df = pd.read_csv(in_path, encoding="utf-8-sig")
    before = len(raw_df)

    cleaned = clean_dataframe_lossless(raw_df)
    after = len(cleaned)

    # Artımsal state istersen kullanılabilir, ama lossless modda şart değil.
    # Yine de son "Tarih_Clean" maksimumunu yazalım (opsiyonel)
    if "Tarih_Clean" in cleaned.columns:
        max_date = pd.to_datetime(cleaned["Tarih_Clean"], errors="coerce").dt.date.max()
        if pd.notna(max_date):
            state.write_text(max_date.strftime("%Y-%m-%d"), encoding="utf-8")

    cleaned.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("--------------------------------------------------")
    print("Temizlik (LOSSLESS) tamamlandı ✅")
    print(f"Girdi satır: {before} | Çıktı satır: {after} (aynı olması beklenir)")
    print(f"Çıktı: {out_path}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
