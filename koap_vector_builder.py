from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from koap_analyzer_new_structure.text_feature_extractor import TextFeatureExtractor
from legal_regulation import (
    REGULATION_SOURCE,
    REGULATION_VERSION,
    REGULATORY_FEATURE_COLUMNS,
    regulatory_features,
)

warnings.filterwarnings("ignore")
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

_CITY_INDEX: Dict[str, int] = {
    "сургут": 1, "нижневартовск": 2, "ханты-мансийск": 3, "нефтеюганск": 4,
    "нягань": 5, "когалым": 6, "мегион": 7, "радужный": 8, "лангепас": 9,
    "покачи": 10, "урай": 11, "югорск": 12, "белоярский": 13,
    "советский": 14, "пыть-ях": 15, "гатчина": 16,
}

_REGION_INDEX: Dict[str, int] = {
    "хмао": 86,
    "ханты-мансийск": 86,
    "югры": 86,
    "ленинградская": 47,
    "санкт-петербург": 78,
    "москва": 77,
}

_REGION_NAME_BY_CODE: Dict[int, str] = {
    86: "Ханты-Мансийский автономный округ - Югра",
    47: "Ленинградская область",
    77: "Москва",
    78: "Санкт-Петербург",
}

_COURT_TYPE_NAME: Dict[int, str] = {
    1: "мировой судья",
    2: "районный суд",
    3: "городской суд",
    4: "апелляция",
}

_ARTICLE_LABEL_BY_CODE: Dict[int, str] = {
    128: "12.8",
    122: "12.2",
    2641: "264.1",
}


_KOAP_NORM_YEARS: Dict[str, Dict[int, int]] = {
    "12.8": {
        2011: 2011, 2012: 2011, 2013: 2011, 2014: 2014, 2015: 2014,
        2016: 2014, 2017: 2014, 2018: 2018, 2019: 2018, 2020: 2018,
        2021: 2021, 2022: 2021, 2023: 2021, 2024: 2023, 2025: 2023,
    },
    "12.2": {
        2011: 2011, 2014: 2014, 2018: 2018, 2021: 2021, 2024: 2023,
    },
    "264.1": {
        2014: 2014, 2015: 2014, 2018: 2018, 2021: 2021, 2024: 2023,
    },
}


def _get_norm_year(article_name: str, year: int) -> int:
    table = _KOAP_NORM_YEARS.get(article_name, _KOAP_NORM_YEARS["12.8"])
    candidates = [y for y in sorted(table.keys()) if y <= year]
    if candidates:
        return table[candidates[-1]]
    return min(table.values())


def _detect_article_name(article_str: str) -> str:
    s = str(article_str)
    if re.search(r"264\.?1|264,?1", s, re.IGNORECASE):
        return "264.1"
    if re.search(r"12\.?2|12,?2", s, re.IGNORECASE):
        return "12.2"
    return "12.8"


def _article_str_to_code(article_str: str) -> int:
    name = _detect_article_name(article_str)
    return {"12.8": 128, "12.2": 122, "264.1": 2641}[name]


def _article_code_to_label(article_code: int) -> str:
    return _ARTICLE_LABEL_BY_CODE.get(int(article_code), "12.8")


def _license_revocation_max_months(article_name: str, article_part: int, year: int) -> int:
    if article_name != "12.8":
        return 0
    if article_part in (3,):
        return 0
    if article_part in (1, 2, 4, 0):
        return 24
    return 0



@dataclass
class CircumstancesVector:
    """
    Вектор обстоятельств дела.

    Кодировки:
      fine_history    : 1=не назначался, 2=назначался/не выплачен, 3=выплачен
      license_history : 1=не назначалось, 2=права не сданы, 3=сданы
      sobriety_test   : 1=не проводилось, 2=проведено, 3=отказ водителя
      driver_health   : 0=неизвестно, 1=здоров, 2=лёгкие, 3=тяжёлые
      victim_health   : те же коды
      court_type      : 1=мировой, 2=районный, 3=городской, 4=апелляция
      article_code    : 128=12.8, 122=12.2, 2641=264.1
    """

    article_code: int = 128
    article_part: int = 0
    article_year_norm: int = 0
    court_type: int = 1

    year: int = 0
    month: int = 0
    day_of_week: int = 0
    hour: int = -1
    is_night: int = 0

    region_code: int = 0
    city_code: int = 0

    recidivism_count: int = 0
    fine_history: int = 1
    license_history: int = 1

    has_lawyer: int = 0
    is_appeal: int = 0
    guilt_admitted: int = 0
    sobriety_test: int = 1

    has_bac_positive: int = 0
    has_protocol: int = 0
    has_witness_testimony: int = 0
    has_officer_report: int = 0
    has_device_cert: int = 0
    witnesses_count: int = 0

    vehicle_ok: int = -1
    is_work_hours: int = 0
    special_procedure: int = 0

    driver_injured: int = 0
    driver_health: int = 0
    victim_injured: int = 0
    victim_health: int = 0


@dataclass
class DecisionVector:
    """Вектор решения суда."""
    imprisonment_epochs: float = 0.0
    license_revocation_epochs: float = 0.0
    conditional: int = 0
    fine_rub: int = 0
    travel_ban: int = 0
    reconciliation: int = 0
    costs_on_state: int = 0
    costs_amount_rub: int = 0
    self_recusal: int = 0
    decision_year: int = 0
    decision_month: int = 0
    decision_day_of_week: int = 0


@dataclass
class CourtVector:
    """Полный вектор одного постановления."""
    source_case_id: str = ""
    article_label: str = ""
    court_type_name: str = ""
    region_name: str = ""
    city_name: str = ""
    crime_datetime_iso: str = ""
    case_id_datetime_iso: str = ""
    decision_datetime_iso: str = ""
    case_duration_days: float = -1.0
    reg_license_revocation_max_months: int = 0
    circumstances: CircumstancesVector = field(default_factory=CircumstancesVector)
    decision: DecisionVector = field(default_factory=DecisionVector)

    def to_flat_dict(self) -> Dict:
        flat = {
            "source_case_id": self.source_case_id,
            "article_label": self.article_label,
            "court_type_name": self.court_type_name,
            "region_name": self.region_name,
            "city_name": self.city_name,
            "crime_datetime_iso": self.crime_datetime_iso,
            "case_id_datetime_iso": self.case_id_datetime_iso,
            "decision_datetime_iso": self.decision_datetime_iso,
            "case_duration_days": self.case_duration_days,
            "reg_license_revocation_max_months": self.reg_license_revocation_max_months,
        }
        for k, v in asdict(self.circumstances).items():
            flat[f"c_{k}"] = v
        for k, v in asdict(self.decision).items():
            flat[f"d_{k}"] = v
        regulation = regulatory_features(
            self.circumstances.article_code,
            self.circumstances.year,
            self.circumstances.article_part,
            self.circumstances.month,
        )
        flat.update(regulation)
        if regulation["c_reg_norm_effective_year"] > 0:
            flat["c_article_year_norm"] = int(regulation["c_reg_norm_effective_year"])
        flat["reg_license_revocation_max_months"] = int(
            regulation["c_reg_license_revocation_max_months"]
        )
        return flat


class VectorBuilder:
    """
    Преобразует строки full_dataset_lite.csv → CourtVector.

    Поля, требующие full_text (heavy CSV), по умолчанию = 0/-1 и
    перечислены в NOT_AVAILABLE_FIELDS.
    """

    NOT_AVAILABLE_FIELDS: List[str] = []

    def __init__(self) -> None:
        self.text_extractor = TextFeatureExtractor()

    def build_from_csv(self, csv_path: str) -> pd.DataFrame:
        df = self._read_csv(csv_path)
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Building vectors ({Path(csv_path).name})", unit="row"):
            rows.append(self._build_single(row).to_flat_dict())
        return pd.DataFrame(rows)

    def build_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building vectors (in-memory)", unit="row"):
            rows.append(self._build_single(row).to_flat_dict())
        return pd.DataFrame(rows)

    def save(self, vectors_df: pd.DataFrame, output_dir: str = "koap_vectors/overall") -> None:
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(output_dir, "vectors.csv")
        vectors_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  {csv_path} ({len(vectors_df)} векторов)")

        numeric_cols = [c for c in vectors_df.columns if is_numeric_dtype(vectors_df[c])]
        npy_path = os.path.join(output_dir, "vectors.npy")
        np.save(npy_path, vectors_df[numeric_cols].values.astype(np.float32))
        print(f" {npy_path} shape={vectors_df[numeric_cols].shape}")

        id_map = {str(cid): int(i) for i, cid in enumerate(vectors_df["source_case_id"])}
        with open(os.path.join(output_dir, "case_id_index.json"), "w", encoding="utf-8") as fh:
            json.dump(id_map, fh, ensure_ascii=False, indent=2)
        print(f" {output_dir}/case_id_index.json")

        meta = self._build_metadata()
        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        schema = self._build_schema(vectors_df, numeric_cols)
        with open(os.path.join(output_dir, "schema.json"), "w", encoding="utf-8") as fh:
            json.dump(schema, fh, ensure_ascii=False, indent=2)
        print(f" {output_dir}/metadata.json, schema.json")

    def _build_single(self, row: pd.Series) -> CourtVector:
        vec = CourtVector(source_case_id=self._extract_case_id(row))
        c = vec.circumstances
        d = vec.decision
        circumstances_text = self._combine_text_fields(row, ("circumstances", "court_decision"))

        article_str = self._extract_article_text(row)
        article_name = _detect_article_name(article_str)
        c.article_code = _article_str_to_code(article_str)
        vec.article_label = _article_code_to_label(c.article_code)

        if "article_code" in row and not pd.isna(row["article_code"]):
            try:
                c.article_code = int(row["article_code"])
                article_name_map = {128: "12.8", 122: "12.2", 2641: "264.1"}
                article_name = article_name_map.get(c.article_code, "12.8")
                vec.article_label = _article_code_to_label(c.article_code)
            except (ValueError, TypeError):
                pass

        c.article_part = self._extract_article_part(article_str)
        c.year = self._safe_int(row.get("year", 0))
        c.article_year_norm = _get_norm_year(article_name, c.year)
        vec.reg_license_revocation_max_months = _license_revocation_max_months(article_name, c.article_part, c.year)

        court_type_name = str(row.get("court_type_name", "")).strip()
        if "court_type" in row and not pd.isna(row.get("court_type", None)):
            try:
                c.court_type = int(row["court_type"])
            except (ValueError, TypeError):
                c.court_type = 1
        else:
            is_appeal = bool(row.get("is_appeal_case", False))
            c.court_type = 4 if is_appeal else 1
        vec.court_type_name = court_type_name or _COURT_TYPE_NAME.get(c.court_type, "мировой судья")

        crime_datetime_raw = str(row.get("crime_datetime", ""))
        dt = self._parse_any_datetime(crime_datetime_raw)
        if dt:
            c.month = dt.month
            c.day_of_week = dt.weekday()
            c.hour = dt.hour
            c.is_night = int(dt.hour >= 22 or dt.hour < 6)
            c.is_work_hours = int(8 <= dt.hour < 18 and dt.weekday() < 5)
            vec.crime_datetime_iso = dt.isoformat(sep=" ")
        else:
            m_time_only = re.fullmatch(r"(\d{1,2}):(\d{2})", crime_datetime_raw.strip())
            if m_time_only:
                hour = int(m_time_only.group(1))
                c.hour = hour
                c.is_night = int(hour >= 22 or hour < 6)
                c.is_work_hours = int(8 <= hour < 18)
                vec.crime_datetime_iso = crime_datetime_raw.strip()
            else:
                c.hour = -1

        case_dt = self._parse_any_datetime(str(row.get("case_id", "")))
        if case_dt:
            vec.case_id_datetime_iso = case_dt.isoformat(sep=" ")

        district = str(row.get("district", ""))
        c.region_code = self._extract_region_code(district)
        c.city_code = self._extract_city_code(district)
        vec.region_name = self._extract_region_name(district, str(row.get("crime_location", "")), c.region_code)
        vec.city_name = self._extract_city_name(district, str(row.get("crime_location", "")))

        c.recidivism_count = 1 if c.article_part in (3, 4) else 0


        c.has_lawyer = int(self._safe_bool(row.get("has_lawyer", False)))
        c.is_appeal = int(self._safe_bool(row.get("is_appeal_case", False)))
        c.guilt_admitted = int(self._safe_bool(row.get("conviction_confirmed", False)))

        c.witnesses_count = self._safe_int(row.get("witnesses_count", 0))
        c.has_witness_testimony = int(c.witnesses_count > 0)

        text_features = self._extract_text_features(circumstances_text, c.article_part, c.witnesses_count)
        c.fine_history = text_features["fine_history"]
        c.license_history = text_features["license_history"]
        c.sobriety_test = text_features["sobriety_test"]
        c.has_bac_positive = text_features["has_bac_positive"]
        c.has_protocol = text_features["has_protocol"]
        c.has_officer_report = text_features["has_officer_report"]
        c.has_device_cert = text_features["has_device_cert"]
        c.vehicle_ok = text_features["vehicle_ok"]
        c.special_procedure = text_features["special_procedure"]
        c.driver_injured = text_features["driver_injured"]
        c.driver_health = text_features["driver_health"]
        c.victim_injured = text_features["victim_injured"]
        c.victim_health = text_features["victim_health"]
        c.has_witness_testimony = int(c.has_witness_testimony or text_features["has_witness_testimony"])
        c.guilt_admitted = int(c.guilt_admitted or text_features["guilt_admitted"])
        c.recidivism_count = max(c.recidivism_count, text_features["recidivism_count"])
        c.has_lawyer = int(c.has_lawyer or text_features["has_lawyer"])
        c.is_appeal = int(c.is_appeal or text_features["is_appeal"])

        ptype = str(row.get("punishment_type", ""))
        pamount = str(row.get("punishment_amount", ""))
        decision_text = self._combine_text_fields(row, ("court_decision", "punishment_type", "punishment_amount"))

        if ptype == "license_revocation":
            d.license_revocation_epochs = self._duration_to_epochs(pamount)
        if ptype == "administrative_arrest":
            days = self._days_from_amount(pamount)
            d.imprisonment_epochs = round(days / 180, 4)
        if ptype == "fine":
            d.fine_rub = self._parse_fine_rub(pamount)

        decision_features = self._extract_decision_features(decision_text)
        d.conditional = decision_features["conditional"]
        d.travel_ban = decision_features["travel_ban"]
        d.reconciliation = decision_features["reconciliation"]
        d.costs_on_state = decision_features["costs_on_state"]
        d.costs_amount_rub = decision_features["costs_amount_rub"]
        d.self_recusal = decision_features["self_recusal"]

        decision_dt = self._extract_decision_datetime(row)
        if not decision_dt:
            decision_dt = case_dt or dt
        if decision_dt:
            vec.decision_datetime_iso = decision_dt.isoformat(sep=" ")
            d.decision_year = decision_dt.year
            d.decision_month = decision_dt.month
            d.decision_day_of_week = decision_dt.weekday()

        if dt and decision_dt:
            vec.case_duration_days = round((decision_dt - dt).total_seconds() / 86400.0, 4)

        return vec

    @staticmethod
    def _safe_int(val) -> int:
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _safe_bool(val) -> bool:
        if pd.isna(val):
            return False
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, np.integer)):
            return bool(val)
        text = str(val).strip().lower()
        if text in {"true", "1", "yes", "y", "да", "истина"}:
            return True
        if text in {"false", "0", "no", "n", "нет", "ложь", "nan", ""}:
            return False
        return bool(text)

    @staticmethod
    def _parse_any_datetime(dt_str: str) -> Optional[datetime]:
        if not dt_str:
            return None
        text = re.sub(r"\s+", " ", str(dt_str)).strip()
        for fmt in ("%d.%m.%Y %H:%M", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                pass

        russian_months = {
            "января": 1,
            "февраля": 2,
            "марта": 3,
            "апреля": 4,
            "мая": 5,
            "июня": 6,
            "июля": 7,
            "августа": 8,
            "сентября": 9,
            "октября": 10,
            "ноября": 11,
            "декабря": 12,
        }

        patterns = (
            r"(?P<day>\d{1,2})\.(?P<month>\d{1,2})\.(?P<year>\d{4})(?:\s+года?)?(?:.*?(?P<hour>\d{1,2})\s*(?:час(?:\.|а|ов)?|ч)(?:\s*(?P<minute>\d{1,2})\s*(?:мин(?:\.|ут)?|м))?)?",
            r"(?P<day>\d{1,2})\s+(?P<month_name>[а-яё]+)\s+(?P<year>\d{4})(?:\s+года?)?(?:.*?(?P<hour>\d{1,2})\s*(?:час(?:\.|а|ов)?|ч)(?:\s*(?P<minute>\d{1,2})\s*(?:мин(?:\.|ут)?|м))?)?",
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue
            day = int(match.group("day"))
            month_value = match.groupdict().get("month")
            if month_value is not None:
                month_num = int(month_value)
            else:
                month_name = str(match.groupdict().get("month_name", "")).lower()
                month_num = russian_months.get(month_name)
                if not month_num:
                    continue
            year = int(match.group("year"))
            hour = int(match.group("hour")) if match.groupdict().get("hour") else 0
            minute = int(match.group("minute")) if match.groupdict().get("minute") else 0
            try:
                return datetime(year, month_num, day, hour, minute)
            except ValueError:
                continue
        return None

    @staticmethod
    def _extract_article_part(article: str) -> int:
        m = re.search(r"[чЧ]\s*[\.\s]\s*(\d)", article, re.IGNORECASE)
        return int(m.group(1)) if m else 0

    @staticmethod
    def _extract_region_code(district: str) -> int:
        low = district.lower()
        for kw, code in _REGION_INDEX.items():
            if kw in low:
                return code
        return 0

    @staticmethod
    def _extract_region_name(district: str, crime_location: str = "", region_code: int = 0) -> str:
        text = f"{district} {crime_location}".lower()
        normalized = re.sub(r"\bnan\b", "", text).strip()
        if normalized in {"", "none"}:
            return ""
        for kw, code in _REGION_INDEX.items():
            if kw in normalized and code in _REGION_NAME_BY_CODE:
                return _REGION_NAME_BY_CODE[code]
        fallback = re.sub(r"\bnan\b", "", str(district).lower()).strip()
        if fallback in {"", "none"}:
            fallback = ""
        return _REGION_NAME_BY_CODE.get(region_code, fallback)

    @staticmethod
    def _extract_city_code(district: str) -> int:
        low = district.lower()
        for city, code in _CITY_INDEX.items():
            if city in low:
                return code
        return 0

    @staticmethod
    def _extract_city_name(district: str, crime_location: str = "") -> str:
        text = f"{district} {crime_location}".lower()
        normalized = re.sub(r"\bnan\b", "", text).strip()
        if normalized in {"", "none"}:
            return ""
        for city in _CITY_INDEX:
            if city in normalized:
                return city.replace("-", " ").title().replace(" ", "-") if "-" in city else city.title()
        return ""

    @staticmethod
    def _duration_to_epochs(amount: str) -> float:
        years, months = 0, 0
        m_y = re.search(r"(\d+)\s+year", amount, re.IGNORECASE)
        m_m = re.search(r"(\d+)\s+month", amount, re.IGNORECASE)
        if m_y:
            years = int(m_y.group(1))
        if m_m:
            months = int(m_m.group(1))
        return round((years * 12 + months) / 6, 4)

    @staticmethod
    def _days_from_amount(amount: str) -> float:
        m = re.search(r"(\d+)\s+day", amount, re.IGNORECASE)
        return float(m.group(1)) if m else 0.0

    @staticmethod
    def _parse_fine_rub(amount: str) -> int:
        m = re.search(r"(\d+)\s*RUB", amount, re.IGNORECASE)
        return int(m.group(1)) if m else 0

    @staticmethod
    def _parse_russian_date(date_str: str) -> Optional[datetime]:
        if not date_str:
            return None
        text = re.sub(r"\s+", " ", str(date_str)).strip().lower()
        month_map = {
            "января": 1,
            "февраля": 2,
            "марта": 3,
            "апреля": 4,
            "мая": 5,
            "июня": 6,
            "июля": 7,
            "августа": 8,
            "сентября": 9,
            "октября": 10,
            "ноября": 11,
            "декабря": 12,
        }
        match = re.search(r"(?:от\s+)?(\d{1,2})\s+([а-яё]+)\s+(\d{4})", text, re.IGNORECASE)
        if not match:
            return None
        day = int(match.group(1))
        month = month_map.get(match.group(2))
        year = int(match.group(3))
        if not month:
            return None
        try:
            return datetime(year, month, day)
        except ValueError:
            return None

    def _extract_decision_datetime(self, row: pd.Series) -> Optional[datetime]:
        for key in ("case_id", "court_decision", "source_file"):
            dt = self._parse_any_datetime(str(row.get(key, "")))
            if dt:
                return dt
        return None

    @staticmethod
    def _combine_text_fields(row: pd.Series, keys: tuple[str, ...]) -> str:
        parts = []
        for key in keys:
            value = row.get(key, "")
            if pd.isna(value):
                continue
            text = str(value).strip()
            if text and text.lower() != "nan":
                parts.append(text)
        return " \n ".join(parts)

    def _extract_text_features(self, text: str, article_part: int, witnesses_count: int) -> Dict[str, int]:
        extractor = self.text_extractor
        if extractor is None or not text:
            return {
                "fine_history": 2 if article_part in (3, 4) else 1,
                "license_history": 2 if article_part in (3, 4) else 1,
                "sobriety_test": 0,
                "has_bac_positive": 0,
                "has_protocol": 0,
                "has_witness_testimony": int(witnesses_count > 0),
                "has_officer_report": 0,
                "has_device_cert": 0,
                "vehicle_ok": -1,
                "special_procedure": 0,
                "driver_injured": 0,
                "driver_health": 0,
                "victim_injured": 0,
                "victim_health": 0,
                "guilt_admitted": 0,
                "has_lawyer": 0,
                "is_appeal": 0,
                "recidivism_count": 1 if article_part in (3, 4) else 0,
            }

        has_lawyer = int(bool(re.search(r"адвокат|защитник", text, re.IGNORECASE)))
        is_appeal = int(bool(re.search(r"апелляц|жалоб|обжал", text, re.IGNORECASE)))
        driver_health = extractor.extract_driver_health_status(text)
        victim_health = extractor.extract_victim_health_status(text)

        return {
            "fine_history": extractor.extract_previous_fine_status(text, is_repeat_offense=article_part in (3, 4)),
            "license_history": extractor.extract_previous_license_status(text, is_repeat_offense=article_part in (3, 4)),
            "sobriety_test": extractor.extract_medical_examination(text),
            "has_bac_positive": int(extractor.extract_alcohol_level_known(text)),
            "has_protocol": int(extractor.extract_has_removal_protocol(text)),
            "has_witness_testimony": int(extractor.extract_has_witness_explanations(text, witnesses_count)),
            "has_officer_report": int(extractor.extract_has_police_report(text)),
            "has_device_cert": int(extractor.extract_has_device_certificate(text)),
            "vehicle_ok": 1 if extractor.extract_vehicle_operational(text) else -1,
            "special_procedure": int(extractor.extract_special_procedure(text)),
            "driver_injured": int(extractor.extract_driver_injured(text)),
            "driver_health": driver_health,
            "victim_injured": int(extractor.extract_victim_injured(text)),
            "victim_health": victim_health,
            "guilt_admitted": int(extractor.extract_guilt_admitted(text)),
            "has_lawyer": has_lawyer,
            "is_appeal": is_appeal,
            "recidivism_count": extractor.extract_recidivism_count(text, is_repeat_offense=article_part in (3, 4)),
        }

    @staticmethod
    def _parse_costs_amount(text: str) -> int:
        if not text:
            return 0
        match = re.search(r"издержк\w*[^\d]{0,40}(\d[\d\s]{0,12}\d)\s*(?:руб|RUB)", text, re.IGNORECASE)
        if not match:
            return 0
        try:
            return int(re.sub(r"\s+", "", match.group(1)))
        except ValueError:
            return 0

    def _extract_decision_features(self, text: str) -> Dict[str, int]:
        low = (text or "").lower()
        return {
            "conditional": int(bool(re.search(r"условн\w+", low, re.IGNORECASE))),
            "travel_ban": int(bool(re.search(r"подписк\w+\s+о\s+невыезде|невыезде", low, re.IGNORECASE))),
            "reconciliation": int(bool(re.search(r"примирени\w+\s+сторон", low, re.IGNORECASE))),
            "costs_on_state": int(bool(re.search(r"издержк\w+.*за\s+счет\s+государств|за\s+счет\s+государств.*издержк", low, re.IGNORECASE))),
            "costs_amount_rub": self._parse_costs_amount(text),
            "self_recusal": int(bool(re.search(r"самоотвод", low, re.IGNORECASE))),
        }

    def _build_metadata(self) -> Dict:
        return {
            "version": "2.1",
            "epoch_definition": "1 epoch = 6 months",
            "regulation": {
                "source": REGULATION_SOURCE,
                "version": REGULATION_VERSION,
                "feature_columns": list(REGULATORY_FEATURE_COLUMNS),
            },
            "new_in_v2": [
                "article_label / court_type_name / region_name / city_name",
                "crime_datetime_iso / case_id_datetime_iso / decision_datetime_iso / case_duration_days",
                "reg_license_revocation_max_months",
                "source_case_id (case_id_index.json)",
                "text-derived features from circumstances/court_decision",
            ],
            "not_available_from_lite_csv": self.NOT_AVAILABLE_FIELDS,
            "circumstances": {
                "c_article_code": {"desc": "Код статьи: 128=12.8, 122=12.2, 2641=264.1", "type": "categorical", "source": "article"},
                "c_article_part": {"desc": "Часть статьи (1/2/3/4)", "type": "int", "source": "article"},
                "c_article_year_norm": {"desc": "Год актуальной редакции КоАП", "type": "int", "source": "year+article"},
                "c_court_type": {"desc": "Тип суда: 1=мировой, 2=районный, 3=городской, 4=апелляция", "type": "categorical", "source": "court_type / is_appeal_case"},
                "c_year": {"desc": "Год задержания", "type": "int", "source": "year"},
                "c_month": {"desc": "Месяц (1–12)", "type": "int", "source": "crime_datetime"},
                "c_day_of_week": {"desc": "День недели (0=Пн … 6=Вс)", "type": "int", "source": "crime_datetime"},
                "c_hour": {"desc": "Час (0–23, -1=н/д)", "type": "int", "source": "crime_datetime"},
                "c_is_night": {"desc": "1 если 22:00–06:00", "type": "binary", "source": "crime_datetime"},
                "c_region_code": {"desc": "Код субъекта РФ (86=ХМАО, 47=ЛО)", "type": "int", "source": "district"},
                "c_city_code": {"desc": "Индекс города", "type": "int", "source": "district"},
                "c_recidivism_count": {"desc": "Число рецидивов (прокси через ч.3/ч.4)", "type": "int", "source": "article_part"},
                "c_fine_history": {"desc": "1=не назначался, 2=не выплачен, 3=выплачен", "type": "ordinal", "source": "N/A (full_text)"},
                "c_license_history": {"desc": "1=не назначалось, 2=не сдал, 3=сдал", "type": "ordinal", "source": "N/A (full_text)"},
                "c_has_lawyer": {"desc": "Адвокат 0/1", "type": "binary", "source": "has_lawyer"},
                "c_is_appeal": {"desc": "Апелляция 0/1", "type": "binary", "source": "is_appeal_case"},
                "c_guilt_admitted": {"desc": "Признание вины 0/1", "type": "binary", "source": "conviction_confirmed"},
                "c_sobriety_test": {"desc": "1=не проводилось, 2=проведено, 3=отказ", "type": "ordinal", "source": "N/A (full_text)"},
                "c_has_bac_positive": {"desc": "Промилле > 0", "type": "binary", "source": "N/A (full_text)"},
                "c_has_protocol": {"desc": "Протокол об отстранении", "type": "binary", "source": "N/A (full_text)"},
                "c_has_witness_testimony": {"desc": "Показания понятых", "type": "binary", "source": "witnesses_count"},
                "c_has_officer_report": {"desc": "Рапорт ГИБДД", "type": "binary", "source": "N/A (full_text)"},
                "c_has_device_cert": {"desc": "Свидетельство о поверке", "type": "binary", "source": "N/A (full_text)"},
                "c_witnesses_count": {"desc": "Число свидетелей", "type": "int", "source": "witnesses_count"},
                "c_vehicle_ok": {"desc": "1=исправен, 0=нет, -1=н/д", "type": "ordinal", "source": "N/A (full_text)"},
                "c_is_work_hours": {"desc": "Нарушение в рабочее время", "type": "binary", "source": "crime_datetime"},
                "c_special_procedure": {"desc": "Особый порядок", "type": "binary", "source": "N/A (full_text)"},
                "c_driver_injured": {"desc": "Водитель ранен", "type": "binary", "source": "N/A (full_text)"},
                "c_driver_health": {"desc": "Здоровье водителя 0–3", "type": "ordinal", "source": "N/A (full_text)"},
                "c_victim_injured": {"desc": "Жертва ранена", "type": "binary", "source": "N/A (full_text)"},
                "c_victim_health": {"desc": "Здоровье жертвы 0–3", "type": "ordinal", "source": "N/A (full_text)"},
            },
            "top_level": {
                "article_label": {"desc": "Статья с точкой (12.8 / 12.2 / 264.1)", "type": "string", "source": "article"},
                "court_type_name": {"desc": "Человекочитаемый тип суда", "type": "string", "source": "court_type / court_type_name"},
                "region_name": {"desc": "Человекочитаемый регион", "type": "string", "source": "district"},
                "city_name": {"desc": "Человекочитаемый город", "type": "string", "source": "district / crime_location"},
                "crime_datetime_iso": {"desc": "Дата и время события в ISO-формате", "type": "string", "source": "crime_datetime"},
                "case_id_datetime_iso": {"desc": "Дата из case_id в ISO-формате", "type": "string", "source": "case_id"},
                "decision_datetime_iso": {"desc": "Дата решения в ISO-формате", "type": "string", "source": "court_decision / case_id"},
                "case_duration_days": {"desc": "Разница между датой события и датой решения, дней", "type": "float", "source": "crime_datetime + case_id/court_decision"},
                "reg_license_revocation_max_months": {"desc": "Максимальный срок лишения прав по норме", "type": "int", "source": "article/year/part"},
            },
            "decision": {
                "d_imprisonment_epochs": {"desc": "Лишение свободы в эпохах", "type": "float", "source": "punishment_amount (арест)"},
                "d_license_revocation_epochs": {"desc": "Лишение прав в эпохах", "type": "float", "source": "punishment_amount"},
                "d_conditional": {"desc": "Условное 0/1", "type": "binary", "source": "N/A (full_text)"},
                "d_fine_rub": {"desc": "Штраф в рублях", "type": "int", "source": "punishment_amount"},
                "d_travel_ban": {"desc": "Подписка о невыезде", "type": "binary", "source": "N/A (full_text)"},
                "d_reconciliation": {"desc": "Примирение сторон", "type": "binary", "source": "N/A (full_text)"},
                "d_costs_on_state": {"desc": "Издержки на государстве", "type": "binary", "source": "N/A (full_text)"},
                "d_costs_amount_rub": {"desc": "Размер издержек", "type": "int", "source": "N/A (full_text)"},
                "d_self_recusal": {"desc": "Самоотвод", "type": "binary", "source": "N/A (full_text)"},
                "d_decision_year": {"desc": "Год решения", "type": "int", "source": "year (прокси)"},
                "d_decision_month": {"desc": "Месяц решения", "type": "int", "source": "crime_datetime (прокси)"},
                "d_decision_day_of_week": {"desc": "День недели решения", "type": "int", "source": "crime_datetime (прокси)"},
            },
        }

    @staticmethod
    def _build_schema(df: pd.DataFrame, numeric_cols: List[str]) -> Dict:
        schema = {}
        for col in numeric_cols:
            series = df[col].dropna()
            uniq = sorted(series.unique().tolist())
            entry: Dict = {
                "dtype": str(df[col].dtype),
                "min": float(series.min()) if len(series) else None,
                "max": float(series.max()) if len(series) else None,
            }
            if len(uniq) <= 10:
                entry["unique_values"] = uniq
            schema[col] = entry
        return schema

    @staticmethod
    def _read_csv(path: str) -> pd.DataFrame:
        for enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue
        raise IOError(f"Не удалось прочитать {path}")

    @staticmethod
    def _extract_case_id(row: pd.Series) -> str:
        for key in ("source_case_id", "case_id", "№ дела", "case id", "case_number", "id"):
            value = row.get(key, "")
            if pd.isna(value):
                continue
            text = str(value).strip()
            if text and text.lower() != "nan":
                return text
        return ""

    @staticmethod
    def _extract_article_text(row: pd.Series) -> str:
        for key in ("article", "article_name", "article_koap", "статья", "статья коап"):
            value = row.get(key, "")
            if pd.isna(value):
                continue
            text = str(value).strip()
            if text and text.lower() != "nan":
                return text
        article_code = row.get("article_code", "")
        if not pd.isna(article_code):
            try:
                code = int(article_code)
                return {128: "12.8", 122: "12.2", 2641: "264.1"}.get(code, "12.8")
            except (ValueError, TypeError):
                pass
        return ""


def build_all_from_results_dir(results_dir: str, out_dir: str = "koap_vectors") -> None:
    """
    Обходит results_dir рекурсивно и строит векторы для каждого full_dataset_lite.csv.
    Создаёт зеркальную структуру в out_dir.
    """
    results_path = Path(results_dir)
    vb = VectorBuilder()

    print(f"\n{'=' * 80}")
    print(f"ГЕНЕРАТОР ВЕКТОРОВ v2  (12.8 / 12.2 / 264.1)")
    print(f"{'=' * 80}")
    print(f"Источник: {results_path.absolute()}")
    print(f"Выход   : {Path(out_dir).absolute()}\n")

    csv_files = sorted(results_path.rglob("full_dataset_lite.csv"))
    if not csv_files and (results_path / "full_dataset_lite.csv").exists():
        csv_files = [results_path / "full_dataset_lite.csv"]

    for csv_file in tqdm(csv_files, desc="Results folders", unit="file"):
        rel_parent = csv_file.parent.relative_to(results_path)
        out_sub = Path(out_dir) / rel_parent
        print(f"  [{rel_parent.as_posix() or '.'}] ...", end=" ", flush=True)
        try:
            df_vec = vb.build_from_csv(str(csv_file))
            print(f"{len(df_vec)} векторов")
            vb.save(df_vec, str(out_sub))
        except Exception as e:
            print(f"Ошибка: {e}")

    print(f"\n ГЕНЕРАЦИЯ ЗАВЕРШЕНА → {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Генератор векторов КоАП v2 (12.8 / 12.2 / 264.1)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", metavar="CSV",
                       help="Путь к конкретному full_dataset_lite.csv")
    group.add_argument("--results_dir", metavar="DIR",
                       help="Корень koap_results_v2/ — обход всех подпапок")
    parser.add_argument("--out", default="koap_vectors", metavar="DIR",
                        help="Выходная директория (по умолчанию: koap_vectors)")
    args = parser.parse_args()

    if args.results_dir:
        build_all_from_results_dir(args.results_dir, args.out)
    else:
        vb = VectorBuilder()
        print(f"\n{'=' * 80}")
        print("ГЕНЕРАТОР ВЕКТОРОВ КоАП v2")
        print(f"{'=' * 80}")
        print(f"Файл: {args.input}\nВыход: {args.out}\n")
        df_vec = vb.build_from_csv(args.input)
        print(f"Построено: {len(df_vec)} векторов")
        vb.save(df_vec, args.out)
        print("\n ГОТОВО")

main()
