import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
import time
import json

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CSV_PATH = DATA_DIR / "izsu_data.csv"


COOKIES = {
    "_ga": "GA1.1.1352645670.1760615720",
    "ASP.NET_SessionId": "5nuocv0vy0grrnvtlfp5kial",
    "_ga_TMNNP5D5PZ": "GS2.1.s1762974809$o9$g1$t1762976978$j60$l0$h0",
}

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
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/138.0.0.0 Safari/537.36 OPR/122.0.0.0",
    "x-requested-with": "XMLHttpRequest",
}

POINTS_URL = "https://www.izsu.gov.tr/HaftalikAnalizSonuclari/GecmisHaftalikAnalizNoktalariGetirJS"
DETAIL_URL = "https://www.izsu.gov.tr/HaftalikAnalizSonuclari/GecmisSonuclariGetirJS"

# -----------------------------
# Belirli tarihleri manuel gir
# -----------------------------
DATES = [
    "05.11.2025", "22.10.2025", "15.10.2025", "08.10.2025",
    "17.09.2025", "10.09.2025", "03.09.2025", "27.08.2025",
    "20.08.2025", "13.08.2025", "06.08.2025", "30.07.2025",
    "23.07.2025", "09.07.2025", "02.07.2025", "25.06.2025",
    "18.06.2025", "11.06.2025", "28.05.2025", "21.05.2025",
    "14.05.2025", "07.05.2025", "24.04.2025", "16.04.2025",
    "09.04.2025", "26.03.2025", "19.03.2025", "12.03.2025",
    "05.03.2025", "26.02.2025", "19.02.2025", "12.02.2025",
    "05.02.2025", "29.01.2025", "22.01.2025", "15.01.2025",
    "08.01.2025", "25.12.2024", "18.12.2024", "11.12.2024",
    "04.12.2024"
]

def fetch_points_for_date(date_str: str):
    data = {"tarih": date_str}
    r = requests.post(POINTS_URL, headers=HEADERS, cookies=COOKIES, data=data, timeout=30)
    r.raise_for_status()
    try:
        points = r.json()
        print(f"[{date_str}] {len(points)} nokta bulundu.")
        return points
    except json.JSONDecodeError:
        print(f"[{date_str}] JSON decode hatası, içerik HTML olabilir.")
        print(r.text[:300])
        return []


def fetch_analysis_for_point(testid: int, date_str: str):
    data = {"testid": testid, "tarih": date_str}
    r = requests.post(DETAIL_URL, headers=HEADERS, cookies=COOKIES, data=data, timeout=30)
    r.raise_for_status()
    try:
        return r.json()
    except json.JSONDecodeError:
        print(f"[{date_str}] Nokta {testid} için JSON decode hatası.")
        return None


def main():
    all_rows = []

    for date_str in DATES:
        print(f"=== {date_str} için veri çekiliyor ===")
        points = fetch_points_for_date(date_str)
        for p in points:
            testid = p.get("NoktaId") or p.get("Id") or p.get("testid")
            if not testid:
                continue

            detail = fetch_analysis_for_point(testid, date_str)
            if not detail:
                continue

            nokta_tanimi = detail.get("NoktaTanimi", "")
            analizler = detail.get("analizSonuclari", [])
            for a in analizler:
                all_rows.append({
                    "Tarih": date_str,
                    "NoktaId": testid,
                    "NoktaTanimi": nokta_tanimi,
                    "ParametreAdi": a.get("ParametreAdi"),
                    "Birim": a.get("Birim"),
                    "Deger": a.get("ParametreDegeri"),
                    "Standart": a.get("Standart"),
                })

            time.sleep(0.2)  # sayfayı yormamak için

    if not all_rows:
        print("Hiç veri çekilemedi.")
        return

    df = pd.DataFrame(all_rows)
    if CSV_PATH.exists():
        base = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
        df = pd.concat([base, df], ignore_index=True).drop_duplicates(
            subset=["Tarih", "NoktaId", "ParametreAdi"], keep="last"
        )
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ {len(df)} satır kaydedildi → {CSV_PATH}")


if __name__ == "__main__":
    main()
