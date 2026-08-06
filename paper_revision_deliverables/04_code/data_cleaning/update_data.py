from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "izsu_data.csv"
STATE_PATH = DATA_DIR / "last_success_date.txt"
DEBUG_RESPONSE_PATH = DATA_DIR / "last_unparsed_response.json"

API_URL = "https://izsu.gov.tr/api/proxy/Analysis/WeeklyAnalysisResult"
REFERER_URL = (
    "https://izsu.gov.tr/bilgi-merkezi/analiz-sonuclari/"
    "haftalik-analiz-sonuclari"
)

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "tr,en-US;q=0.9,en;q=0.8",
    "appname": "webApp",
    "referer": REFERER_URL,
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/149.0.0.0 Safari/537.36"
    ),
}

POINT_ID_KEYS = (
    "pointId",
    "id",
    "analysisPointId",
    "weeklyAnalysisPointId",
    "noktaId",
    "NoktaId",
)

POINT_NAME_KEYS = (
    "pointName",
    "name",
    "analysisPointName",
    "noktaAdi",
    "NoktaAdi",
)

POINT_ADDRESS_KEYS = (
    "pointAddress",
    "pointDescription",
    "address",
    "noktaTanimi",
    "NoktaTanimi",
)

LATITUDE_KEYS = ("latitude", "lat", "enlem", "Latitude")
LONGITUDE_KEYS = ("longitude", "lng", "lon", "boylam", "Longitude")

MEASUREMENT_LIST_KEYS = (
    "results",
    "analysisResults",
    "weeklyAnalysisResults",
    "analyses",
    "analysisList",
    "measurements",
    "parameters",
    "parameterResults",
    "values",
    "details",
    "detail",
    "analizSonuclari",
)

PARAMETER_NAME_KEYS = (
    "parameterName",
    "paramName",
    "analysisName",
    "parameter",
    "name",
    "parametreAdi",
    "ParametreAdi",
)

PARAMETER_CODE_KEYS = (
    "parameterCode",
    "paramCode",
    "ParametreKodu",
)

UNIT_KEYS = (
    "parameterUnit",
    "unit",
    "unitName",
    "measurementUnit",
    "birim",
    "Birim",
)

VALUE_KEYS = (
    "measurement",
    "value",
    "resultValue",
    "parameterValue",
    "analysisResult",
    "result",
    "measurementValue",
    "deger",
    "Deger",
    "ParametreDegeri",
)

STANDARD_KEYS = (
    "parameterStandard",
    "standard",
    "standardValue",
    "limit",
    "referenceValue",
    "referenceRange",
    "standart",
    "Standart",
)

MEASUREMENT_DATE_KEYS = (
    "date",
    "measurementDate",
    "analysisDate",
    "tarih",
    "Tarih",
)


def first_value(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Sözlükte bulunan ilk boş olmayan alanı döndürür."""
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def format_date(value: Any, fallback: date) -> str:
    """
    API tarihini eski CSV ile uyumlu DD.MM.YYYY biçimine çevirir.

    Tarih okunamazsa sorgulanan tarih kullanılır.
    """
    if value not in (None, ""):
        text = str(value).strip()

        for fmt in (
            "%Y-%m-%d",
            "%d.%m.%Y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
        ):
            try:
                return datetime.strptime(text, fmt).strftime("%d.%m.%Y")
            except ValueError:
                continue

        # ISO tarih saatlerinde saat dilimi bulunması ihtimalini destekle.
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).strftime(
                "%d.%m.%Y"
            )
        except ValueError:
            pass

    return fallback.strftime("%d.%m.%Y")


def read_last_success_date() -> Optional[date]:
    if not STATE_PATH.exists():
        return None

    text = STATE_PATH.read_text(encoding="utf-8").strip()

    for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    print(f"[UYARI] State dosyasındaki tarih okunamadı: {text!r}")
    return None


def write_last_success_date(success_date: date) -> None:
    STATE_PATH.write_text(
        success_date.strftime("%d.%m.%Y"),
        encoding="utf-8",
    )


def daterange_days(start_date: date, end_date: date):
    """Başlangıç ve bitiş tarihleri arasında gün gün ilerler."""
    current = start_date

    while current <= end_date:
        yield current
        current += timedelta(days=1)


def load_existing_data() -> pd.DataFrame:
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    except (pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as exc:
        print(f"[UYARI] Mevcut CSV okunamadı: {exc}")
        return pd.DataFrame()


def save_data(dataframe: pd.DataFrame) -> None:
    temp_path = CSV_PATH.with_suffix(".tmp")
    dataframe.to_csv(temp_path, index=False, encoding="utf-8-sig")
    temp_path.replace(CSV_PATH)


def save_debug_response(payload: Any) -> None:
    DEBUG_RESPONSE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def fetch_weekly_result(
    session: requests.Session,
    query_date: date,
) -> Any:
    """Yeni İZSU API'sinden seçilen haftanın verisini alır."""
    response = session.get(
        API_URL,
        params={"date": query_date.strftime("%Y-%m-%d")},
        timeout=45,
    )
    response.raise_for_status()

    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise RuntimeError(
            "API JSON yerine farklı bir içerik döndürdü: "
            f"{response.text[:300]}"
        ) from exc

    if isinstance(payload, dict):
        ok = payload.get("ok")
        status_code = payload.get("statusCode")
        error_message = payload.get("errorMessage")

        if ok is False or status_code not in (None, 200):
            raise RuntimeError(
                f"İZSU API hatası. statusCode={status_code}, "
                f"errorMessage={error_message!r}"
            )

    return payload


def extract_points(payload: Any) -> list[dict[str, Any]]:
    """API yanıtından örnekleme noktalarını çıkarır."""
    data = payload.get("data") if isinstance(payload, dict) else payload

    if isinstance(data, list):
        return [
            item
            for item in data
            if isinstance(item, dict)
        ]

    if isinstance(data, dict):
        for key in (
            "items",
            "results",
            "points",
            "analysisPoints",
            "list",
        ):
            value = data.get(key)

            if isinstance(value, list):
                return [
                    item
                    for item in value
                    if isinstance(item, dict)
                ]

    return []


def looks_like_measurement(item: dict[str, Any]) -> bool:
    has_parameter = first_value(item, PARAMETER_NAME_KEYS) is not None
    has_value = first_value(item, VALUE_KEYS) is not None
    return has_parameter and has_value


def find_measurements(
    value: Any,
    depth: int = 0,
) -> list[dict[str, Any]]:
    """
    Nokta nesnesi içindeki analiz kayıtlarını bulur.

    Yeni API'de kayıtlar doğrudan ``results`` listesindedir.
    Sınırlı özyinelemeli arama, olası küçük API değişiklikleri için korunur.
    """
    if depth > 5:
        return []

    if isinstance(value, dict):
        for key in MEASUREMENT_LIST_KEYS:
            candidate = value.get(key)

            if not isinstance(candidate, list):
                continue

            dict_items = [
                item
                for item in candidate
                if isinstance(item, dict)
            ]

            if dict_items and any(
                looks_like_measurement(item)
                for item in dict_items
            ):
                return dict_items

        for child in value.values():
            found = find_measurements(child, depth + 1)

            if found:
                return found

    elif isinstance(value, list):
        dict_items = [
            item
            for item in value
            if isinstance(item, dict)
        ]

        if dict_items and any(
            looks_like_measurement(item)
            for item in dict_items
        ):
            return dict_items

        for child in value:
            found = find_measurements(child, depth + 1)

            if found:
                return found

    return []


def point_to_rows(
    point: dict[str, Any],
    query_date: date,
) -> list[dict[str, Any]]:
    point_name = first_value(point, POINT_NAME_KEYS)
    point_address = first_value(point, POINT_ADDRESS_KEYS)
    point_id = first_value(point, POINT_ID_KEYS)
    latitude = first_value(point, LATITUDE_KEYS)
    longitude = first_value(point, LONGITUDE_KEYS)

    if point_id is None:
        point_id = point_name or point_address

    measurements = find_measurements(point)
    rows: list[dict[str, Any]] = []

    for measurement in measurements:
        parameter_name = first_value(
            measurement,
            PARAMETER_NAME_KEYS,
        )
        measured_value = first_value(
            measurement,
            VALUE_KEYS,
        )

        if parameter_name is None or measured_value is None:
            continue

        measurement_date = first_value(
            measurement,
            MEASUREMENT_DATE_KEYS,
        )

        rows.append(
            {
                "Tarih": format_date(
                    measurement_date,
                    query_date,
                ),
                "NoktaId": point_id,
                # Eski veri yapısında NoktaTanimi adresi temsil ediyordu.
                "NoktaTanimi": point_address or point_name,
                "NoktaAdi": point_name,
                "Ilce": point.get("district"),
                "IlceKodu": point.get("districtCode"),
                "NoktaKodu": (
                    measurement.get("pointCode")
                    or point.get("pointCode")
                ),
                "Enlem": latitude,
                "Boylam": longitude,
                "ParametreAdi": parameter_name,
                "ParametreKodu": first_value(
                    measurement,
                    PARAMETER_CODE_KEYS,
                ),
                "Birim": first_value(
                    measurement,
                    UNIT_KEYS,
                ),
                "Deger": measured_value,
                "Standart": first_value(
                    measurement,
                    STANDARD_KEYS,
                ),
            }
        )

    return rows


def payload_to_rows(
    payload: Any,
    query_date: date,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for point in extract_points(payload):
        rows.extend(point_to_rows(point, query_date))

    return rows


def all_result_lists_empty(
    points: list[dict[str, Any]],
) -> bool:
    """
    Noktalar dönmesine rağmen seçilen tarihte hiçbir analiz sonucu yok mu?

    Bazı tarihlerde API 80 noktanın tamamını döndürür fakat bütün
    ``results`` listeleri boş olur. Bu durum ayrıştırma hatası değildir.
    """
    if not points:
        return False

    found_results_field = False

    for point in points:
        results = point.get("results")

        if not isinstance(results, list):
            continue

        found_results_field = True

        if results:
            return False

    return found_results_field


def merge_with_existing(
    existing_dataframe: pd.DataFrame,
    new_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    if existing_dataframe.empty:
        full = new_dataframe.copy()
    else:
        full = pd.concat(
            [existing_dataframe, new_dataframe],
            ignore_index=True,
        )

    if "NoktaId" not in full.columns:
        full["NoktaId"] = pd.NA

    if "NoktaTanimi" not in full.columns:
        full["NoktaTanimi"] = pd.NA

    full["_NoktaTekil"] = full["NoktaId"].fillna(
        full["NoktaTanimi"]
    )

    # Eski CSV dosyaları ParametreKodu içermediği için geçmiş ve yeni
    # kayıtların aynı kuralla karşılaştırılabilmesi amacıyla ParametreAdi
    # üzerinden tekilleştirme yapılır.
    dedup_columns = [
        "Tarih",
        "_NoktaTekil",
        "ParametreAdi",
    ]

    full = full.drop_duplicates(
        subset=dedup_columns,
        keep="last",
    )
    full = full.drop(columns=["_NoktaTekil"])

    return full.reset_index(drop=True)


def main() -> None:
    session = requests.Session()
    session.headers.update(HEADERS)

    print("[INFO] İZSU analiz API'si gün gün taranacak.")
    print(f"[INFO] API: {API_URL}")
    print(f"[INFO] Veri klasörü: {DATA_DIR}")

    # Site ileride oturum veya cookie istemeye başlarsa uyumluluk sağlaması
    # için önce referer sayfasına bağlantı kuruluyor.
    try:
        session.get(REFERER_URL, timeout=30)
    except requests.RequestException as exc:
        print(f"[UYARI] Referer sayfasına bağlanılamadı: {exc}")

    last_success = read_last_success_date()

    if last_success:
        start_date = last_success + timedelta(days=7)
    else:
        start_date = date(2024, 11, 13)

    end_date = datetime.now().date()
    existing_dataframe = load_existing_data()

    new_frames = []
    last_processed_date: Optional[date] = None
    empty_day_count = 0

    for query_date in daterange_days(
        start_date,
        end_date,
    ):
        print(
            "\n=== Tarih: "
            f"{query_date.strftime('%Y-%m-%d')} ==="
        )

        try:
            payload = fetch_weekly_result(
                session,
                query_date,
            )
        except requests.RequestException as exc:
            print(
                f"[ATLANDI] {query_date.strftime('%Y-%m-%d')} "
                f"tarihinde API yanıtı alınamadı: {exc}"
            )
            last_processed_date = query_date
            write_last_success_date(query_date)
            time.sleep(0.20)
            continue

        except RuntimeError as exc:
            print(
                f"[ATLANDI] {query_date.strftime('%Y-%m-%d')} "
                f"tarihindeki API yanıtı kullanılamadı: {exc}"
            )
            last_processed_date = query_date
            write_last_success_date(query_date)
            time.sleep(0.20)
            continue

        points = extract_points(payload)
        rows = payload_to_rows(
            payload,
            query_date,
        )

        if not points:
            print(
                f"[ATLANDI] {query_date.strftime('%Y-%m-%d')}: "
                "API yanıtında örnekleme noktası bulunamadı."
            )
            empty_day_count += 1
            last_processed_date = query_date
            write_last_success_date(query_date)
            time.sleep(0.20)
            continue

        if not rows and all_result_lists_empty(points):
            print(
                f"[ATLANDI] {query_date.strftime('%Y-%m-%d')}: "
                f"{len(points)} noktanın tüm results listeleri boş."
            )
            empty_day_count += 1
            last_processed_date = query_date
            write_last_success_date(query_date)
            time.sleep(0.20)
            continue

        if not rows:
            save_debug_response(payload)
            print(
                f"[ATLANDI] {query_date.strftime('%Y-%m-%d')}: "
                "Dolu results bulundu fakat alanlar ayrıştırılamadı."
            )
            print(
                f"[BİLGİ] Ham yanıt inceleme için kaydedildi: "
                f"{DEBUG_RESPONSE_PATH}"
            )
            last_processed_date = query_date
            write_last_success_date(query_date)
            time.sleep(0.20)
            continue

        points_with_results = sum(
            1
            for point in points
            if isinstance(point.get("results"), list)
            and bool(point["results"])
        )

        empty_point_count = len(points) - points_with_results

        print(
            f"[YAZILDI] {query_date.strftime('%Y-%m-%d')}: "
            f"{len(points)} toplam nokta, "
            f"{points_with_results} dolu nokta, "
            f"{empty_point_count} boş nokta atlandı, "
            f"{len(rows)} parametre kaydı alındı."
        )

        new_frames.append(pd.DataFrame(rows))
        last_processed_date = query_date

        # Her başarılı API yanıtından sonra checkpoint hemen güncellenir.
        # Program sonradan kesilirse kaldığı yerden güvenle devam eder.
        write_last_success_date(query_date)
        time.sleep(0.20)

    if not new_frames:
        print(
            "\nYeni analiz kaydı bulunamadı. API bağlantısı "
            "çalıştı ancak taranan tarihlerde kaydedilecek "
            "bir results verisi yoktu."
        )
        print(
            f"Boş/atlanan gün sayısı: "
            f"{empty_day_count}"
        )
        return

    new_dataframe = pd.concat(
        new_frames,
        ignore_index=True,
    )

    full_dataframe = merge_with_existing(
        existing_dataframe,
        new_dataframe,
    )

    save_data(full_dataframe)

    print("\nVeri çekme tamamlandı ✅")
    print(
        f"Yeni parametre kaydı: "
        f"{len(new_dataframe)}"
    )
    print(
        f"CSV toplam satır sayısı: "
        f"{len(full_dataframe)}"
    )
    print(f"Çıktı: {CSV_PATH}")

    if last_processed_date:
        print(
            "Son işlenen tarih: "
            f"{last_processed_date.strftime('%d.%m.%Y')}"
        )


if __name__ == "__main__":
    main()