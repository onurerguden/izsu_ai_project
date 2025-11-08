import os
import json
import time
import logging
from typing import Optional
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

BASE_URL = "https://www.izsu.gov.tr"
POINTS_ENDPOINT = f"{BASE_URL}/HaftalikAnalizSonuclari/GecmisHaftalikAnalizNoktalariGetirJS"
DETAIL_ENDPOINT = f"{BASE_URL}/HaftalikAnalizSonuclari/HftSonucGetir"
COOKIE = "AspxAutoDetectCookieSupport=1"

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "izsu_data.csv"
STATE_PATH = DATA_DIR / "last_success_date.txt"

HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Cookie": COOKIE,
    "User-Agent": "Mozilla/5.0"
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def ensure_data_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def read_last_success_date() -> Optional[datetime.date]:
    if not STATE_PATH.exists():
        return None
    content = STATE_PATH.read_text(encoding="utf-8").strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(content, fmt).date()
        except ValueError:
            continue
    logger.warning("Failed to parse last success date from state file.")
    return None


def write_last_success_date(date_obj: datetime.date) -> None:
    STATE_PATH.write_text(date_obj.strftime("%d.%m.%Y"), encoding="utf-8")


def daterange_weeks(start_date: datetime.date, end_date: datetime.date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=7)


def sanitize(text: str | None) -> str:
    if not text:
        return ""
    return str(text).replace("\n", " ").replace("\r", " ").strip()


def fetch_points_for_date(date_str: str) -> list[int]:
    try:
        response = requests.post(
            POINTS_ENDPOINT,
            headers=HEADERS,
            data={"tarih": date_str},
            timeout=20
        )
        response.raise_for_status()
        content = response.text.strip()
        if not content.startswith("["):
            logger.warning(f"{date_str}: Unexpected response format for points.")
            return []
        points = response.json()
        ids = [item.get("Id") or item.get("testid") for item in points if item.get("Id") or item.get("testid")]
        logger.info(f"{date_str}: Found {len(ids)} point IDs.")
        return ids
    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.error(f"{date_str}: Error fetching points - {e}")
        return []


def fetch_analysis_for_point(point_id: int) -> list[dict]:
    try:
        response = requests.post(
            DETAIL_ENDPOINT,
            headers=HEADERS,
            data={"testid": point_id},
            timeout=20
        )
        response.raise_for_status()
        content = response.text.strip()
        if not content.startswith("{"):
            return []
        data = response.json()
        nokta = sanitize(data.get("NoktaTanimi", "UNKNOWN"))
        analyses = data.get("analizSonuclari", [])
        rows = []
        for analysis in analyses:
            rows.append({
                "Tarih": None,
                "NoktaAdi": nokta,
                "ParametreAdi": sanitize(analysis.get("ParametreAdi")),
                "Birim": sanitize(analysis.get("Birim")),
                "Deger": sanitize(analysis.get("ParametreDegeri")),
                "Standart": sanitize(analysis.get("Standart"))
            })
        return rows
    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.error(f"Point ID {point_id}: Error fetching analysis - {e}")
        return []


def fetch_week_data(date_obj: datetime.date) -> list[dict]:
    date_str = date_obj.strftime("%d.%m.%Y")
    point_ids = fetch_points_for_date(date_str)
    if not point_ids:
        return []

    all_rows = []
    for pid in point_ids:
        analysis_rows = fetch_analysis_for_point(pid)
        for row in analysis_rows:
            row["Tarih"] = date_str
        all_rows.extend(analysis_rows)
        time.sleep(0.1)  # polite delay

    return all_rows


def load_existing_data() -> pd.DataFrame:
    if CSV_PATH.exists():
        try:
            return pd.read_csv(CSV_PATH, encoding="utf-8-sig")
        except Exception as e:
            logger.warning(f"Failed to read existing CSV data: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def save_data(df: pd.DataFrame) -> None:
    tmp_path = CSV_PATH.with_suffix(".tmp")
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    tmp_path.replace(CSV_PATH)
    logger.info(f"Data saved to {CSV_PATH}")


def main():
    ensure_data_directory()

    last_success = read_last_success_date()
    start_date = (last_success + timedelta(days=7)) if last_success else datetime(2024, 10, 16).date()
    end_date = datetime.now().date()

    if start_date > end_date:
        logger.info("No new data to fetch. Last success date is up to date.")
        return

    logger.info(f"Fetching data from {start_date} to {end_date}")

    existing_df = load_existing_data()
    new_rows = []
    consecutive_empty_weeks = 0
    last_valid_date = None

    for current_date in daterange_weeks(start_date, end_date):
        week_data = fetch_week_data(current_date)
        if not week_data:
            consecutive_empty_weeks += 1
            logger.info(f"{current_date.strftime('%d.%m.%Y')}: No data found ({consecutive_empty_weeks}/5)")
            if consecutive_empty_weeks >= 5:
                logger.info("No data for 5 consecutive weeks, stopping fetch.")
                break
            continue

        consecutive_empty_weeks = 0
        new_rows.append(pd.DataFrame(week_data))
        last_valid_date = current_date
        logger.info(f"{current_date.strftime('%d.%m.%Y')}: Retrieved {len(week_data)} records.")

    if new_rows:
        combined_df = pd.concat([existing_df] + new_rows, ignore_index=True)
        combined_df.drop_duplicates(subset=["Tarih", "NoktaAdi", "ParametreAdi", "Birim"], inplace=True)
        combined_df.sort_values(by=["Tarih", "NoktaAdi"], inplace=True)
        save_data(combined_df)
    else:
        logger.info("No new data to update.")

    if last_valid_date:
        write_last_success_date(last_valid_date)
        logger.info(f"Updated last success date to {last_valid_date.strftime('%d.%m.%Y')}")
    else:
        logger.info("Last success date not updated due to no new valid data.")

    logger.info("Data update process completed.")


if __name__ == "__main__":
    main()