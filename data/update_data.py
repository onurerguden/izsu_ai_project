import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
import json
from typing import Optional

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / "izsu_data.csv"
STATE_PATH = DATA_DIR / "last_success_date.txt"

# --------------------------------------------------
# Ortak HEADER — Cookie YOK, session bunu yönetecek
# --------------------------------------------------
HEADERS = {
    "accept": "*/*",
    "accept-language": "tr,en-US;q=0.9,en;q=0.8,es;q=0.7,de;q=0.6,nl;q=0.5,ru;q=0.4,zh-CN;q=0.3,zh;q=0.2,fr;q=0.1",
    "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
    "origin": "https://www.izsu.gov.tr",
    "referer": "https://www.izsu.gov.tr/tr/haftalikAnalizSonuclari/1",
    "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Opera GX";v="122"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/138.0.0.0 Safari/537.36 OPR/122.0.0.0"
    ),
    "x-requested-with": "XMLHttpRequest",
}

POINTS_URL = "https://www.izsu.gov.tr/HaftalikAnalizSonuclari/GecmisHaftalikAnalizNoktalariGetirJS"
DETAIL_URL = "https://www.izsu.gov.tr/HaftalikAnalizSonuclari/GecmisSonuclariGetirJS"
REFERER_URL = "https://www.izsu.gov.tr/tr/haftalikAnalizSonuclari/1"


# -----------------------------
# Yardımcı fonksiyonlar
# -----------------------------
def read_last_success_date() -> Optional[datetime.date]:
    if not STATE_PATH.exists():
        return None
    txt = STATE_PATH.read_text(encoding="utf-8").strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(txt, fmt).date()
        except ValueError:
            continue
    return None


def write_last_success_date(d: datetime.date) -> None:
    STATE_PATH.write_text(d.strftime("%d.%m.%Y"), encoding="utf-8")


def daterange_weeks(start_date, end_date):
    cur = start_date
    while cur <= end_date:
        yield cur
        cur += timedelta(days=7)


# -----------------------------
# HTTP çağrıları (session ile)
# -----------------------------
def fetch_points_for_date(session, date_str):
    data = {"tarih": date_str}
    r = session.post(POINTS_URL, data=data, timeout=30)
    r.raise_for_status()
    try:
        return r.json()
    except:
        print("JSON decode hatası:", date_str)
        print(r.text[:300])
        return []


def fetch_analysis_for_point(session, testid, date_str):
    data = {"testid": testid, "tarih": date_str}
    r = session.post(DETAIL_URL, data=data, timeout=30)
    r.raise_for_status()
    try:
        return r.json()
    except:
        print("JSON decode hatası (detail):", date_str, "ID:", testid)
        return None


# -----------------------------
# CSV işlemleri
# -----------------------------
def load_existing_data():
    if CSV_PATH.exists():
        try:
            return pd.read_csv(CSV_PATH, encoding="utf-8-sig")
        except:
            return pd.DataFrame()
    return pd.DataFrame()


def save_data(df):
    tmp = CSV_PATH.with_suffix(".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(CSV_PATH)


# -----------------------------
# MAIN
# -----------------------------
def main():
    session = requests.Session()
    session.headers.update(HEADERS)

    print("[INFO] Referer sayfasına bağlanılıyor...")
    session.get(REFERER_URL, timeout=30)

    last_success = read_last_success_date()
    if last_success:
        start_date = last_success + timedelta(days=7)
    else:
        start_date = datetime(2024, 11, 13).date()

    end_date = datetime.now().date()

    existing_df = load_existing_data()
    all_new = []
    last_valid = None
    empty_streak = 0

    for day in daterange_weeks(start_date, end_date):
        date_str = day.strftime("%d.%m.%Y")
        print("\n=== Tarih:", date_str, "===")

        points = fetch_points_for_date(session, date_str)
        if not points:
            empty_streak += 1
            if empty_streak >= 5:
                print("5 hafta üst üste boş → durduruldu.")
                break
            continue

        empty_streak = 0
        rows = []

        for p in points:
            testid = p.get("NoktaId") or p.get("Id")
            if not testid:
                continue

            detail = fetch_analysis_for_point(session, testid, date_str)
            if not detail:
                continue

            nt = detail.get("NoktaTanimi", "")
            for a in detail.get("analizSonuclari", []):
                rows.append({
                    "Tarih": date_str,
                    "NoktaId": testid,
                    "NoktaTanimi": nt,
                    "ParametreAdi": a.get("ParametreAdi"),
                    "Birim": a.get("Birim"),
                    "Deger": a.get("ParametreDegeri"),
                    "Standart": a.get("Standart"),
                })

            time.sleep(0.20)

        if rows:
            all_new.append(pd.DataFrame(rows))
            last_valid = day

    if not all_new:
        print("Yeni veri yok.")
        return

    new_df = pd.concat(all_new, ignore_index=True)

    if not existing_df.empty:
        full = pd.concat([existing_df, new_df], ignore_index=True)
        full = full.drop_duplicates(
            subset=["Tarih", "NoktaId", "ParametreAdi"], keep="last"
        )
    else:
        full = new_df

    save_data(full)

    if last_valid:
        write_last_success_date(last_valid)

    print("✔ Veri çekme tamamlandı.")


if __name__ == "__main__":
    main()
