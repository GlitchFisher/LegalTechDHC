from __future__ import annotations

import calendar
from datetime import date
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REGULATION_VERSION = "verified_registry_v2_2026-07-20"
REGULATION_SOURCE = "embedded_verified_registry"


REGULATION_SOURCES: Dict[str, Dict[str, str]] = {
    "koap_195_fz": {
        "title": "КоАП РФ, Федеральный закон от 30.12.2001 № 195-ФЗ",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_34661/",
    },
    "koap_12_8_current": {
        "title": "КоАП РФ, статья 12.8 (действующая редакция)",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_34661/aa69183ecd988ed365aa7b0e5fffb687dc479b71/",
    },
    "koap_1_7": {
        "title": "КоАП РФ, статья 1.7 (действие улучшающего закона во времени)",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_34661/91a591b505fde94f1d7015578d2ae37f3d6e20ac/",
    },
    "fz_82_2000": {
        "title": "Федеральный закон от 19.06.2000 № 82-ФЗ, статья 5 (базовая сумма для штрафов 100 рублей)",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_27572/",
    },
    "fz_93_2004": {
        "title": "Федеральный закон от 28.07.2004 № 93-ФЗ",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_48631/",
    },
    "fz_210_2007": {
        "title": "Федеральный закон от 24.07.2007 № 210-ФЗ",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_52003/",
    },
    "fz_196_2013": {
        "title": "Федеральный закон от 23.07.2013 № 196-ФЗ",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_149643/",
    },
    "fz_515_2014": {
        "title": "Федеральный закон от 31.12.2014 № 515-ФЗ",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_173195/",
    },
    "fz_528_2014": {
        "title": "Федеральный закон от 31.12.2014 № 528-ФЗ",
        "url": "https://publication.pravo.gov.ru/Document/View/0001201412310075",
    },
    "fz_62_2018": {
        "title": "Федеральный закон от 03.04.2018 № 62-ФЗ",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_294729/",
    },
    "fz_65_2019": {
        "title": "Федеральный закон от 23.04.2019 № 65-ФЗ",
        "url": "https://publication.pravo.gov.ru/Document/View/0001201904240002",
    },
    "fz_258_2021": {
        "title": "Федеральный закон от 01.07.2021 № 258-ФЗ",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_388872/",
    },
    "fz_490_2024": {
        "title": "Федеральный закон от 26.12.2024 № 490-ФЗ",
        "url": "https://publication.pravo.gov.ru/Document/View/0001202412260018",
    },
    "uk_264_1_current": {
        "title": "УК РФ, статья 264.1 (действующая редакция)",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_10699/ae5648172402868434a5dd1cb045ba682075fe14/",
    },
    "uk_forced_work_from_2017": {
        "title": "Статья 8 Федерального закона от 07.12.2011 № 420-ФЗ (в ред. № 431-ФЗ): применение принудительных работ с 01.01.2017",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_122864/bdb2754392763f4c0afbdb3bc7ea77ef6a5287c4/",
    },
    "uk_general_part": {
        "title": "УК РФ, статьи 46, 47, 49, 50, 53, 53.1, 56 и 64",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_10699/",
    },
    "uk_46": {
        "title": "УК РФ, статья 46 (штраф)",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_10699/1d89a92f11ffc5fd1f9e111176b14356886d4805/",
    },
    "uk_47": {
        "title": "УК РФ, статья 47 (лишение права занимать должности или заниматься деятельностью)",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_10699/a090fac1812ec1f374f05aa83399aece68b131a7/",
    },
    "uk_49": {
        "title": "УК РФ, статья 49 (обязательные работы)",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_10699/a65dac489bf58afbf78d6fbc5ad58048769b2a93/",
    },
    "uk_50": {
        "title": "УК РФ, статья 50 (исправительные работы)",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_10699/fd830e7cf3c0e1074ca35a580314701483ff611a/",
    },
    "uk_53": {
        "title": "УК РФ, статья 53 (ограничение свободы)",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_10699/412ff33a4344e1fc162cd8ec95285831cec940be/",
    },
    "uk_53_1": {
        "title": "УК РФ, статья 53.1 (принудительные работы)",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_10699/fbf561e8e76ded47846e0b625229d7933bbcc93a/",
    },
    "uk_56": {
        "title": "УК РФ, статья 56 (лишение свободы на определённый срок)",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_10699/57aab780bd6bda63d406bf99082ab3b3afe269ab/",
    },
    "uk_64": {
        "title": "УК РФ, статья 64 (более мягкое наказание и неприменение обязательного дополнительного наказания)",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_10699/9f851250d5e363d361ea33fea95a4daebdd97711/",
    },
    "koap_4_1": {
        "title": "КоАП РФ, статья 4.1, части 2.2–2.3",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_34661/93fe5340f4a47211790c8cfa5ba3ac94e3766a3c/",
    },
}

_UK_GENERAL_SANCTION_SOURCES = ("uk_46", "uk_47", "uk_49", "uk_50", "uk_53", "uk_53_1", "uk_56", "uk_64")

REGULATION_PROFILES: Dict[int, Dict[str, object]] = {
    12802: {"effective_from": "2002-07-01", "sources": ("koap_195_fz", "fz_82_2000")},
    12804: {"effective_from": "2004-08-31", "sources": ("koap_195_fz", "fz_93_2004")},
    12808: {"effective_from": "2008-01-01", "sources": ("fz_93_2004", "fz_210_2007")},
    12813: {"effective_from": "2013-09-01", "sources": ("fz_196_2013",)},
    12814: {"effective_from": "2015-01-11", "sources": ("fz_196_2013", "fz_515_2014")},
    12815: {"effective_from": "2015-07-01", "sources": ("fz_196_2013", "fz_515_2014", "fz_528_2014")},
    12818: {"effective_from": "2018-07-03", "sources": ("fz_196_2013", "fz_515_2014", "fz_528_2014", "fz_62_2018")},
    12825: {"effective_from": "2025-01-01", "sources": ("koap_12_8_current", "fz_490_2024", "fz_515_2014")},
    264115: {"effective_from": "2015-07-01", "sources": ("fz_528_2014", *_UK_GENERAL_SANCTION_SOURCES)},
    264117: {"effective_from": "2017-01-01", "sources": ("fz_528_2014", "uk_forced_work_from_2017", *_UK_GENERAL_SANCTION_SOURCES)},
    264119: {"effective_from": "2019-04-24", "sources": ("fz_528_2014", "fz_65_2019", "uk_forced_work_from_2017", *_UK_GENERAL_SANCTION_SOURCES)},
    264121: {"effective_from": "2021-07-12", "sources": ("fz_258_2021", "uk_forced_work_from_2017", *_UK_GENERAL_SANCTION_SOURCES)},
}

LEGAL_OVERRIDE_DEFAULTS: Dict[str, float] = {
    "min_licence_12.8_1_l": 1.5, "min_licence_12.8_1_s": 1.5,
    "max_licence_12.8_1_l": 2.0, "max_licence_12.8_1_s": 2.0,
    "max_fee_12.8_1_l": 45000.0, "max_fee_12.8_1_s": 181210.5,
    "min_forced_job_264.1_l": 0.0, "min_forced_job_264.1_s": 0.0,
    "max_forced_job_264.1_l": 2.0, "max_forced_job_264.1_s": 2.0,
    "min_jail_264.1_l": 0.0, "min_jail_264.1_s": 0.0,
    "max_jail_264.1_l": 2.0, "max_jail_264.1_s": 2.0,
    "min_licence_264.1_l": 0.0, "min_licence_264.1_s": 0.0,
    "max_licence_264.1_l": 3.0, "max_licence_264.1_s": 3.0,
    "min_fee_264.1_l": 200000.0, "min_fee_264.1_s": 805380.0,
    "max_fee_264.1_l": 300000.0, "max_fee_264.1_s": 1073840.0,
}

LEGAL_OVERRIDE_MAP: Dict[int, Dict[str, Tuple[str, float]]] = {
    128: {
        "min_licence_12.8_1": ("c_reg_license_revocation_min_months", 12.0),
        "max_licence_12.8_1": ("c_reg_license_revocation_max_months", 12.0),
        "max_fee_12.8_1": ("c_reg_fine_max_rub", 1.0),
    },
    2641: {
        "min_forced_job_264.1": ("c_reg_forced_work_min_months", 12.0),
        "max_forced_job_264.1": ("c_reg_forced_work_max_months", 12.0),
        "min_jail_264.1": ("c_reg_imprisonment_min_months", 12.0),
        "max_jail_264.1": ("c_reg_imprisonment_max_months", 12.0),
        "min_licence_264.1": ("c_reg_license_revocation_min_months", 12.0),
        "max_licence_264.1": ("c_reg_license_revocation_max_months", 12.0),
        "min_fee_264.1": ("c_reg_fine_min_rub", 1.0),
        "max_fee_264.1": ("c_reg_fine_max_rub", 1.0),
    },
}

REGULATORY_FEATURE_DEFAULTS: Dict[str, float] = {
    "c_reg_supported": 0.0,
    "c_reg_profile_code": 0.0,
    "c_reg_norm_effective_year": 0.0,
    "c_reg_norm_effective_month": 0.0,
    "c_reg_norm_effective_day": 0.0,
    "c_reg_date_precision": 0.0,
    "c_reg_date_ambiguous": 0.0,
    "c_reg_article_active": 0.0,
    "c_reg_liability_type": 0.0,
    "c_reg_fine_available": 0.0,
    "c_reg_fine_min_rub": 0.0,
    "c_reg_fine_max_rub": 0.0,
    "c_reg_fine_exception_min_rub": 0.0,
    "c_reg_fine_below_min_exception_possible": 0.0,
    "c_reg_fine_requires_arrest_ineligible": 0.0,
    "c_reg_fine_mandatory": 0.0,
    "c_reg_income_fine_available": 0.0,
    "c_reg_income_fine_min_months": 0.0,
    "c_reg_income_fine_max_months": 0.0,
    "c_reg_income_fine_article64_floor_weeks": 0.0,
    "c_reg_license_revocation_min_months": 0.0,
    "c_reg_license_revocation_max_months": 0.0,
    "c_reg_license_revocation_mandatory": 0.0,
    "c_reg_license_omission_exception_possible": 0.0,
    "c_reg_arrest_min_days": 0.0,
    "c_reg_arrest_max_days": 0.0,
    "c_reg_mandatory_work_min_hours": 0.0,
    "c_reg_mandatory_work_max_hours": 0.0,
    "c_reg_forced_work_min_months": 0.0,
    "c_reg_forced_work_max_months": 0.0,
    "c_reg_corrective_work_min_months": 0.0,
    "c_reg_corrective_work_max_months": 0.0,
    "c_reg_restriction_freedom_min_months": 0.0,
    "c_reg_restriction_freedom_max_months": 0.0,
    "c_reg_imprisonment_min_months": 0.0,
    "c_reg_imprisonment_max_months": 0.0,
    "c_reg_breath_threshold_mg_l": 0.0,
    "c_reg_blood_threshold_g_l": 0.0,
    "c_reg_repeat_transferred_to_criminal": 0.0,
}

REGULATORY_FEATURE_COLUMNS: Tuple[str, ...] = tuple(REGULATORY_FEATURE_DEFAULTS)


_PROFILE_BOUNDARIES: Dict[int, Tuple[date, ...]] = {
    128: (
        date(2002, 7, 1),
        date(2004, 8, 31),
        date(2008, 1, 1),
        date(2013, 9, 1),
        date(2015, 1, 11),
        date(2015, 7, 1),
        date(2018, 7, 3),
        date(2025, 1, 1),
    ),
    2641: (date(2015, 7, 1), date(2017, 1, 1), date(2019, 4, 24), date(2021, 7, 12)),
}


def _safe_int(value: object, default: int = 0) -> int:
    """Безопасное преобразование значения в целое число.

    Параметры:
        value: Исходное значение.
        default: Результат при невозможности преобразования.

    Возвращает:
        Целое число или значение по умолчанию.
    """
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_bool(value: object) -> bool:
    """Преобразовать числовой/строковый флаг в bool без исключений."""
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "да", "истина"}
    return bool(value)


def _date_context(
    article_code: int,
    year: int,
    month: int,
    day: int,
) -> Tuple[date | None, int, bool]:
    """Вернуть начало известного интервала даты, точность и неоднозначность.

    Точность: 0 - дата отсутствует, 1 - известен год, 2 - год и месяц,
    3 - полная дата. Для неполной даты выбирается редакция, действовавшая в
    начале известного интервала; если внутри интервала менялась норма,
    ``ambiguous`` равен True.
    """
    if year < 1 or year > 9999:
        return None, 0, False
    if month < 1 or month > 12:
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        precision = 1
    elif day < 1 or day > calendar.monthrange(year, month)[1]:
        start = date(year, month, 1)
        end = date(year, month, calendar.monthrange(year, month)[1])
        precision = 2
    else:
        start = end = date(year, month, day)
        precision = 3
    ambiguous = any(start < boundary <= end for boundary in _PROFILE_BOUNDARIES.get(article_code, ()))
    return start, precision, ambiguous


def _new_features(
    article_code: int,
    profile_code: int,
    effective_from: date | None,
    liability_type: int,
    date_precision: int = 0,
    date_ambiguous: bool = False,
) -> Dict[str, float]:
    """Начальный набор регуляторных признаков для одной редакции нормы.

    Параметры:
        article_code: Числовой код статьи.
        profile_code: Идентификатор временного профиля санкций.
        effective_from: Точная дата начала действия профиля.
        liability_type: Тип ответственности: 1 - административная, 2 - уголовная.

    Возвращает:
        Словарь числовых регуляторных признаков.
    """
    features = dict(REGULATORY_FEATURE_DEFAULTS)
    features.update(
        {
            "c_reg_supported": float(article_code in {128, 2641}),
            "c_reg_profile_code": float(profile_code),
            "c_reg_norm_effective_year": float(effective_from.year if effective_from else 0),
            "c_reg_norm_effective_month": float(effective_from.month if effective_from else 0),
            "c_reg_norm_effective_day": float(effective_from.day if effective_from else 0),
            "c_reg_date_precision": float(date_precision),
            "c_reg_date_ambiguous": float(date_ambiguous),
            "c_reg_liability_type": float(liability_type),
        }
    )
    return features


def _set_fine(features: MutableMapping[str, float], minimum: float, maximum: float, mandatory: bool) -> None:
    """Параметры денежного штрафа.

    Параметры:
        features: Изменяемый словарь регуляторных признаков.
        minimum: Минимальный штраф в рублях.
        maximum: Максимальный штраф в рублях.
        mandatory: Признак обязательности штрафа в данной части статьи.

    Возвращает:
        Нет.
    """
    features.update(
        {
            "c_reg_fine_available": 1.0,
            "c_reg_fine_min_rub": float(minimum),
            "c_reg_fine_max_rub": float(maximum),
            "c_reg_fine_mandatory": float(mandatory),
        }
    )


def _set_income_fine(
    features: MutableMapping[str, float],
    minimum_months: float,
    maximum_months: float,
) -> None:
    """Параметры альтернативного штрафа в размере дохода осуждённого."""
    features.update(
        {
            "c_reg_income_fine_available": 1.0,
            "c_reg_income_fine_min_months": float(minimum_months),
            "c_reg_income_fine_max_months": float(maximum_months),
            "c_reg_income_fine_article64_floor_weeks": 2.0,
        }
    )


def _set_license(
    features: MutableMapping[str, float],
    minimum_months: float,
    maximum_months: float,
    mandatory: bool = False,
    omission_exception_possible: bool = False,
) -> None:
    """Параметры лишения права управления или занятия деятельностью.

    Параметры:
        features: Изменяемый словарь регуляторных признаков.
        minimum_months: Минимальный срок в месяцах.
        maximum_months: Максимальный срок в месяцах.

    Возвращает:
        Нет.
    """
    features.update(
        {
            "c_reg_license_revocation_min_months": float(minimum_months),
            "c_reg_license_revocation_max_months": float(maximum_months),
            "c_reg_license_revocation_mandatory": float(mandatory),
            "c_reg_license_omission_exception_possible": float(omission_exception_possible),
        }
    )


def _set_arrest(features: MutableMapping[str, float], minimum_days: float, maximum_days: float) -> None:
    """Параметры административного ареста.

    Параметры:
        features: Изменяемый словарь регуляторных признаков.
        minimum_days: Минимальный срок в сутках.
        maximum_days: Максимальный срок в сутках.

    Возвращает:
        Нет.
    """
    features.update(
        {
            "c_reg_arrest_min_days": float(minimum_days),
            "c_reg_arrest_max_days": float(maximum_days),
        }
    )


def _set_2641_base_sanctions(
    features: MutableMapping[str, float],
    forced_work_available: bool = True,
) -> None:
    """Базовые санкции статьи 264.1 в редакции 2015 года.

    Параметры:
        features: Изменяемый словарь регуляторных признаков.

    Возвращает:
        Нет.
    """
    _set_fine(features, 200_000, 300_000, mandatory=False)
    _set_income_fine(features, 12, 24)
    features.update(
        {
            "c_reg_fine_exception_min_rub": 5_000.0,
            "c_reg_fine_below_min_exception_possible": 1.0,
        }
    )
    _set_license(features, 0, 36, mandatory=True, omission_exception_possible=True)
    features.update(
        {
            "c_reg_mandatory_work_min_hours": 60.0,
            "c_reg_mandatory_work_max_hours": 480.0,
            "c_reg_forced_work_min_months": 0.0 if forced_work_available else 0.0,
            "c_reg_forced_work_max_months": 24.0 if forced_work_available else 0.0,
            "c_reg_imprisonment_min_months": 0.0,
            "c_reg_imprisonment_max_months": 24.0,
        }
    )


def _enable_koap_fine_exception(
    features: MutableMapping[str, float],
    statutory_minimum: float,
    as_of: date | None,
) -> None:
    """Добавить предел снижения штрафа по частям 2.2–2.3 статьи 4.1 КоАП."""
    if as_of is not None and as_of >= date(2015, 1, 11) and statutory_minimum >= 10_000:
        features.update(
            {
                "c_reg_fine_exception_min_rub": float(statutory_minimum / 2.0),
                "c_reg_fine_below_min_exception_possible": 1.0,
            }
        )


def _article_128_features(year: int, part: int, month: int, day: int) -> Dict[str, float]:
    """Регуляторный профиль статьи 12.8 КоАП.

    Параметры:
        year: Год события.
        part: Часть статьи, ноль означает неизвестную часть.
        month: Месяц события, ноль означает неизвестный месяц.
        day: День события, ноль означает неизвестный день.

    Возвращает:
        Словарь числовых параметров действовавшей редакции.
    """
    as_of, precision, ambiguous = _date_context(128, year, month, day)
    if as_of is None or as_of < date(2002, 7, 1):
        return _new_features(128, 12800, None, liability_type=1, date_precision=precision, date_ambiguous=ambiguous)

    if as_of >= date(2025, 1, 1):
        profile_code, effective_from = 12825, date(2025, 1, 1)
    elif as_of >= date(2018, 7, 3):
        profile_code, effective_from = 12818, date(2018, 7, 3)
    elif as_of >= date(2015, 7, 1):
        profile_code, effective_from = 12815, date(2015, 7, 1)
    elif as_of >= date(2015, 1, 11):
        profile_code, effective_from = 12814, date(2015, 1, 11)
    elif as_of >= date(2013, 9, 1):
        profile_code, effective_from = 12813, date(2013, 9, 1)
    elif as_of >= date(2008, 1, 1):
        profile_code, effective_from = 12808, date(2008, 1, 1)
    elif as_of >= date(2004, 8, 31):
        profile_code, effective_from = 12804, date(2004, 8, 31)
    else:
        profile_code, effective_from = 12802, date(2002, 7, 1)

    features = _new_features(
        128,
        profile_code,
        effective_from,
        liability_type=1,
        date_precision=precision,
        date_ambiguous=ambiguous,
    )
    features["c_reg_article_active"] = 1.0

    if profile_code == 12802:
        if part in {0, 1, 2}:
            _set_fine(features, 1_000, 2_000, mandatory=False)
            _set_license(features, 12, 12, mandatory=False)
        else:
            features["c_reg_article_active"] = 0.0
        return features

    if profile_code == 12804:
        if part in {0, 1, 2}:
            _set_license(features, 18, 24, mandatory=part in {1, 2})
        else:
            features["c_reg_article_active"] = 0.0
        return features

    if profile_code == 12808:
        if part in {1, 2}:
            _set_license(features, 18, 24, mandatory=True)
        elif part == 3:
            _set_fine(features, 5_000, 5_000, mandatory=False)
            features["c_reg_fine_requires_arrest_ineligible"] = 1.0
            _set_arrest(features, 1, 15)
        elif part == 4:
            _set_license(features, 36, 36, mandatory=True)
        elif part == 0:
            _set_fine(features, 5_000, 5_000, mandatory=False)
            _set_license(features, 18, 36)
            _set_arrest(features, 1, 15)
        else:
            features["c_reg_article_active"] = 0.0
        return features

    features.update(
        {
            "c_reg_breath_threshold_mg_l": 0.16,
        }
    )
    if profile_code >= 12818:
        features.update(
            {
                "c_reg_blood_threshold_g_l": 0.3,
            }
        )

    fine = 45_000 if profile_code >= 12825 else 30_000
    if part in {1, 2}:
        _set_fine(features, fine, fine, mandatory=True)
        _enable_koap_fine_exception(features, fine, as_of)
        _set_license(features, 18, 24, mandatory=True)
    elif part == 3:
        _set_fine(features, fine, fine, mandatory=False)
        _enable_koap_fine_exception(features, fine, as_of)
        features["c_reg_fine_requires_arrest_ineligible"] = 1.0
        _set_arrest(features, 10, 15)
    elif part == 4:
        if profile_code >= 12815:
            features["c_reg_article_active"] = 0.0
        else:
            _set_fine(features, 50_000, 50_000, mandatory=True)
            _enable_koap_fine_exception(features, 50_000, as_of)
            _set_license(features, 36, 36, mandatory=True)
    elif part == 0:
        _set_fine(features, fine, fine if profile_code >= 12815 else 50_000, mandatory=False)
        _enable_koap_fine_exception(features, fine, as_of)
        _set_license(features, 18, 24 if profile_code >= 12815 else 36)
        _set_arrest(features, 10, 15)
    else:
        features["c_reg_article_active"] = 0.0

    if profile_code >= 12815:
        features["c_reg_repeat_transferred_to_criminal"] = 1.0
    return features


def _article_2641_features(year: int, part: int, month: int, day: int) -> Dict[str, float]:
    """Регуляторный профиль статьи 264.1 УК.

    Параметры:
        year: Год события.
        part: Часть статьи, ноль означает неизвестную часть.
        month: Месяц события, ноль означает неизвестный месяц.
        day: День события, ноль означает неизвестный день.

    Возвращает:
        Словарь числовых параметров действовавшей редакции.
    """
    as_of, precision, ambiguous = _date_context(2641, year, month, day)
    if as_of is None or as_of < date(2015, 7, 1):
        return _new_features(2641, 264100, None, liability_type=2, date_precision=precision, date_ambiguous=ambiguous)

    if as_of >= date(2021, 7, 12):
        profile_code, effective_from = 264121, date(2021, 7, 12)
    elif as_of >= date(2019, 4, 24):
        profile_code, effective_from = 264119, date(2019, 4, 24)
    elif as_of >= date(2017, 1, 1):
        profile_code, effective_from = 264117, date(2017, 1, 1)
    else:
        profile_code, effective_from = 264115, date(2015, 7, 1)

    features = _new_features(
        2641,
        profile_code,
        effective_from,
        liability_type=2,
        date_precision=precision,
        date_ambiguous=ambiguous,
    )
    features["c_reg_article_active"] = 1.0

    if profile_code < 264121 and part in {0, 1}:
        _set_2641_base_sanctions(features, forced_work_available=profile_code >= 264117)
    elif profile_code < 264121:
        features["c_reg_article_active"] = 0.0
    elif part == 1:
        _set_2641_base_sanctions(features)
    elif part == 2:
        _set_fine(features, 300_000, 500_000, mandatory=False)
        _set_income_fine(features, 24, 36)
        features.update(
            {
                "c_reg_fine_exception_min_rub": 5_000.0,
                "c_reg_fine_below_min_exception_possible": 1.0,
            }
        )
        _set_license(features, 6, 72, mandatory=True, omission_exception_possible=True)
        features.update(
            {
                "c_reg_forced_work_min_months": 2.0,
                "c_reg_forced_work_max_months": 36.0,
                "c_reg_corrective_work_min_months": 2.0,
                "c_reg_corrective_work_max_months": 24.0,
                "c_reg_restriction_freedom_min_months": 2.0,
                "c_reg_restriction_freedom_max_months": 36.0,
                "c_reg_imprisonment_min_months": 2.0,
                "c_reg_imprisonment_max_months": 36.0,
            }
        )
    elif part == 0:
        _set_fine(features, 200_000, 500_000, mandatory=False)
        _set_income_fine(features, 12, 36)
        features.update(
            {
                "c_reg_fine_exception_min_rub": 5_000.0,
                "c_reg_fine_below_min_exception_possible": 1.0,
            }
        )
        _set_license(features, 6, 72, mandatory=True, omission_exception_possible=True)
        features.update(
            {
                "c_reg_mandatory_work_min_hours": 60.0,
                "c_reg_mandatory_work_max_hours": 480.0,
                "c_reg_forced_work_min_months": 2.0,
                "c_reg_forced_work_max_months": 36.0,
                "c_reg_corrective_work_min_months": 2.0,
                "c_reg_corrective_work_max_months": 24.0,
                "c_reg_restriction_freedom_min_months": 2.0,
                "c_reg_restriction_freedom_max_months": 36.0,
                "c_reg_imprisonment_min_months": 2.0,
                "c_reg_imprisonment_max_months": 36.0,
            }
        )
    else:
        features["c_reg_article_active"] = 0.0
    return features


def regulatory_features(
    article_code: object,
    year: object,
    part: object = 0,
    month: object = 0,
    day: object = 0,
) -> Dict[str, float]:
    """Числовые параметры закона для статьи и даты события.

    Параметры:
        article_code: Код статьи: 128 для 12.8 КоАП или 2641 для 264.1 УК.
        year: Год события, определяющий применимую редакцию.
        part: Часть статьи.
        month: Месяц события; ноль означает, что известен только год.
        day: День события; ноль означает, что известны только год и месяц.

    Возвращает:
        Полный словарь признаков ``c_reg_*``.
    """
    code = _safe_int(article_code)
    year_value = _safe_int(year)
    part_value = _safe_int(part)
    month_value = _safe_int(month)
    day_value = _safe_int(day)
    if code == 128:
        return _article_128_features(year_value, part_value, month_value, day_value)
    if code == 2641:
        return _article_2641_features(year_value, part_value, month_value, day_value)
    return dict(REGULATORY_FEATURE_DEFAULTS)


def _event_date_values(values: Mapping[str, object]) -> Tuple[int, int, int]:
    """Извлечь год, месяц и день события из числовых или ISO-полей строки."""
    year = _safe_int(values.get("c_year"))
    month = _safe_int(values.get("c_month"))
    day = _safe_int(values.get("c_day", values.get("c_day_of_month")))
    if day > 0:
        return year, month, day
    raw_iso = values.get("crime_datetime_iso", values.get("c_crime_datetime_iso"))
    if raw_iso is None:
        return year, month, day
    try:
        parsed = date.fromisoformat(str(raw_iso).strip()[:10])
    except (TypeError, ValueError):
        return year, month, day
    if year <= 0:
        year = parsed.year
    if month <= 0:
        month = parsed.month
    if year == parsed.year and month == parsed.month:
        day = parsed.day
    return year, month, day


def enrich_regulatory_features(df: pd.DataFrame, overwrite: bool = True) -> pd.DataFrame:
    """Добавление регуляторных признаков к таблице дел.

    Параметры:
        df: Таблица с ``c_article_code``, ``c_year`` и необязательными частью и месяцем.
        overwrite: Разрешение заменить ранее рассчитанные ``c_reg_*``.

    Возвращает:
        Копию таблицы с числовыми колонками применимой нормы.
    """
    enriched = df.copy()
    keys = []
    for row in enriched.itertuples(index=False):
        values = row._asdict()
        year, month, day = _event_date_values(values)
        keys.append(
            (
                _safe_int(values.get("c_article_code")),
                year,
                _safe_int(values.get("c_article_part")),
                month,
                day,
            )
        )
    cache = {key: regulatory_features(key[0], key[1], key[2], key[3], key[4]) for key in set(keys)}
    regulatory_frame = pd.DataFrame(
        ({feature: cache[key][feature] for feature in REGULATORY_FEATURE_COLUMNS} for key in keys),
        index=enriched.index,
        dtype=np.float32,
    )
    if overwrite:
        enriched = enriched.drop(columns=[feature for feature in REGULATORY_FEATURE_COLUMNS if feature in enriched.columns])
        enriched = pd.concat([enriched, regulatory_frame], axis=1)
    else:
        missing = [feature for feature in REGULATORY_FEATURE_COLUMNS if feature not in enriched.columns]
        enriched = pd.concat([enriched, regulatory_frame.loc[:, missing]], axis=1)

    if "c_article_year_norm" in enriched.columns:
        norm_year = enriched["c_reg_norm_effective_year"].to_numpy(dtype=np.float32)
        original = pd.to_numeric(enriched["c_article_year_norm"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        enriched["c_article_year_norm"] = np.where(norm_year > 0, norm_year, original).astype(np.float32)
    enriched.attrs["regulation_version"] = REGULATION_VERSION
    return enriched


def regulatory_context(row: Mapping[str, object]) -> Dict[str, object]:
    """Контекст применимой нормы для отображения и аудита.

    Параметры:
        row: Значения обстоятельств одного дела.

    Возвращает:
        Словарь идентификаторов дела и числовых параметров нормы.
    """
    article_code = _safe_int(row.get("c_article_code"))
    year, month, day = _event_date_values(row)
    part = _safe_int(row.get("c_article_part"))
    features = regulatory_features(article_code, year, part, month, day)
    profile_code = _safe_int(features.get("c_reg_profile_code"))
    profile = REGULATION_PROFILES.get(profile_code, {})
    source_ids = list(profile.get("sources", ()))
    if article_code == 128:
        source_ids.append("koap_12_8_current")
        explicit_koap_exception = any(
            _safe_bool(row.get(name))
            for name in (
                "c_koap_article_4_1_applied",
                "c_koap_article_4_1_below_min_applied",
                "c_reg_below_min_exception_applied",
            )
        )
        if features["c_reg_fine_below_min_exception_possible"] or explicit_koap_exception:
            source_ids.extend(("fz_515_2014", "koap_4_1"))
        if explicit_koap_exception and not features["c_reg_fine_below_min_exception_possible"]:
            source_ids.append("koap_1_7")
    elif article_code == 2641:
        source_ids.append("uk_264_1_current")
    source_ids = list(dict.fromkeys(source_ids))
    return {
        "source": REGULATION_SOURCE,
        "version": REGULATION_VERSION,
        "article_code": article_code,
        "article": {128: "12.8 КоАП", 2641: "264.1 УК"}.get(article_code, "unsupported"),
        "year": year,
        "month": month,
        "day": day,
        "part": part,
        "date_ambiguous": bool(features["c_reg_date_ambiguous"]),
        "profile": profile,
        "sources": [dict(id=source_id, **REGULATION_SOURCES[source_id]) for source_id in source_ids],
        "features": features,
    }


def _record_adjustment(
    audit: List[Dict[str, object]],
    target: str,
    before: float,
    after: float,
    reason: str,
) -> None:
    """Запись изменения предсказания правовым ограничителем.

    Параметры:
        audit: Список изменений.
        target: Имя целевой переменной.
        before: Значение до ограничения.
        after: Значение после ограничения.
        reason: Основание изменения.

    Возвращает:
        Нет.
    """
    if not np.isclose(before, after, rtol=1e-7, atol=1e-7):
        audit.append({"target": target, "before": before, "after": after, "reason": reason})


def apply_regulation_overrides(
    features: Mapping[str, float],
    article_code: int,
    overrides: Optional[Dict[str, float]],
) -> Dict[str, float]:
    """Применить пользовательские диапазоны санкций к признакам нормы.

    Только там, где санкция уже применима: штраф при ``c_reg_fine_available``,
    сроки при положительном базовом максимуме признака. Пустые ``overrides``
    возвращают копию без изменений (обратная совместимость).
    """
    if not overrides:
        return dict(features)
    mapping = LEGAL_OVERRIDE_MAP.get(article_code)
    if not mapping:
        return dict(features)
    out = dict(features)
    for name, (target, coef) in mapping.items():
        if name not in overrides:
            continue
        try:
            scaled = float(overrides[name]) * coef
        except (TypeError, ValueError):
            continue
        if target.endswith("_rub"):
            if out.get("c_reg_fine_available") != 1.0:
                continue
        else:
            max_feature = target.replace("_min_", "_max_") if "_min_" in target else target
            if out.get(max_feature, 0.0) <= 0:
                continue
        out[target] = float(scaled)
    return out


def constrain_prediction(
    prediction: Mapping[str, object],
    row: Mapping[str, object],
    overrides: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, object], List[Dict[str, object]], Dict[str, object]]:
    """Ограничение предсказания диапазонами применимого закона.

    Параметры:
        prediction: Сырое предсказание модели по одному делу.
        row: Обстоятельства дела с кодом статьи и датой.

    Возвращает:
        Кортеж из скорректированного предсказания, журнала изменений и контекста нормы.
    """
    adjusted = dict(prediction)
    context = regulatory_context(row)
    features = context["features"]
    features = apply_regulation_overrides(features, context["article_code"], overrides)
    audit: List[Dict[str, object]] = []
    if not features["c_reg_supported"] or _safe_int(context.get("year")) <= 0:
        return adjusted, audit, context

    active = bool(features["c_reg_article_active"])
    koap_below_min_applied = any(
        _safe_bool(row.get(name))
        for name in (
            "c_koap_article_4_1_applied",
            "c_koap_article_4_1_below_min_applied",
            "c_reg_below_min_exception_applied",
        )
    )
    uk_article_64_applied = any(
        _safe_bool(row.get(name))
        for name in ("c_uk_article_64_applied", "c_article_64_applied")
    )
    below_min_exception_applied = (
        context["article_code"] == 128 and koap_below_min_applied
    ) or (
        context["article_code"] == 2641 and uk_article_64_applied
    )
    omit_mandatory_license = context["article_code"] == 2641 and any(
        _safe_bool(row.get(name))
        for name in (
            "c_uk_article_64_omit_additional_applied",
            "c_reg_omit_mandatory_additional_applied",
        )
    )
    income_fine_applied = context["article_code"] == 2641 and (
        any(
            _safe_bool(row.get(name))
            for name in (
                "c_uk_income_based_fine_applied",
                "c_income_based_fine_applied",
            )
        )
    )

    if "d_fine_rub" in adjusted:
        before = float(adjusted["d_fine_rub"])
        if not active or not features["c_reg_fine_available"]:
            after = 0.0
            reason = "штраф отсутствует в применимом профиле"
        elif income_fine_applied and features["c_reg_income_fine_available"]:
            after = before
            reason = "рублёвая сумма не ограничивается фиксированным диапазоном: штраф назначен в размере дохода"
        elif features["c_reg_fine_requires_arrest_ineligible"] and any(
            name in row for name in ("c_arrest_ineligible", "c_administrative_arrest_inapplicable")
        ) and not any(
            _safe_bool(row.get(name))
            for name in ("c_arrest_ineligible", "c_administrative_arrest_inapplicable")
        ):
            after = 0.0
            reason = "альтернативный штраф допустим только для лица, к которому административный арест неприменим"
        else:
            minimum = float(features["c_reg_fine_min_rub"])
            maximum = float(features["c_reg_fine_max_rub"])
            explicit_retroactive_koap_exception = (
                context["article_code"] == 128
                and koap_below_min_applied
                and minimum >= 10_000
            )
            if below_min_exception_applied and (
                features["c_reg_fine_below_min_exception_possible"]
                or explicit_retroactive_koap_exception
            ):
                configured_exception_minimum = float(features["c_reg_fine_exception_min_rub"])
                minimum = configured_exception_minimum or minimum / 2.0
                exception_name = "статья 64 УК" if context["article_code"] == 2641 else "части 2.2–2.3 статьи 4.1 КоАП"
                reason = f"исключительный диапазон штрафа [{minimum}, {maximum}] руб. ({exception_name})"
            else:
                reason = f"базовый диапазон штрафа [{minimum}, {maximum}] руб."
            if before <= 0 and not features["c_reg_fine_mandatory"]:
                after = 0.0
            else:
                after = min(max(before, minimum), maximum)
        adjusted["d_fine_rub"] = after
        _record_adjustment(audit, "d_fine_rub", before, after, reason)

    if "d_income_fine_months" in adjusted:
        before = float(adjusted["d_income_fine_months"])
        minimum = float(features["c_reg_income_fine_min_months"])
        maximum = float(features["c_reg_income_fine_max_months"])
        if not active or not features["c_reg_income_fine_available"]:
            after = 0.0
            reason = "штраф в размере дохода отсутствует в применимом профиле"
        elif uk_article_64_applied:
            after = min(before, maximum) if before > 0 else 0.0
            reason = f"верхний предел периода дохода {maximum} месяцев; нижний специальный предел снижен по статье 64 УК"
        else:
            after = 0.0 if before <= 0 else min(max(before, minimum), maximum)
            reason = f"диапазон периода дохода [{minimum}, {maximum}] месяцев"
        adjusted["d_income_fine_months"] = after
        _record_adjustment(audit, "d_income_fine_months", before, after, reason)

    if "d_license_revocation_epochs" in adjusted:
        before = float(adjusted["d_license_revocation_epochs"])
        before_months = before * 6.0
        maximum = float(features["c_reg_license_revocation_max_months"])
        if not active or maximum <= 0:
            after_months = 0.0
            reason = "лишение права отсутствует в применимом профиле"
        else:
            minimum = float(features["c_reg_license_revocation_min_months"])
            if before_months <= 0:
                if features["c_reg_license_revocation_mandatory"] and not (
                    omit_mandatory_license and features["c_reg_license_omission_exception_possible"]
                ):
                    after_months = minimum
                else:
                    after_months = 0.0
            else:
                after_months = max(before_months, minimum)
                if maximum > 0:
                    after_months = min(after_months, maximum)
            if omit_mandatory_license and features["c_reg_license_omission_exception_possible"]:
                reason = f"диапазон лишения права [0, {maximum}] месяцев с учётом статьи 64 УК"
            else:
                reason = f"базовый диапазон лишения права [{minimum}, {maximum}] месяцев"
        after = after_months / 6.0
        adjusted["d_license_revocation_epochs"] = after
        _record_adjustment(audit, "d_license_revocation_epochs", before, after, reason)

    if "d_imprisonment_epochs" in adjusted:
        before = float(adjusted["d_imprisonment_epochs"])
        arrest_maximum = float(features["c_reg_arrest_max_days"])
        imprisonment_maximum = float(features["c_reg_imprisonment_max_months"])
        if active and arrest_maximum > 0:
            before_days = before * 180.0
            minimum = float(features["c_reg_arrest_min_days"])
            maximum = arrest_maximum
            after_days = 0.0 if before_days <= 0 else min(max(before_days, minimum), maximum)
            after = after_days / 180.0
            reason = f"диапазон административного ареста [{minimum}, {maximum}] суток"
        elif active and imprisonment_maximum > 0:
            before_months = before * 6.0
            minimum = float(features["c_reg_imprisonment_min_months"])
            maximum = imprisonment_maximum
            after_months = 0.0 if before_months <= 0 else min(max(before_months, minimum), maximum)
            after = after_months / 6.0
            reason = f"диапазон лишения свободы [{minimum}, {maximum}] месяцев"
        else:
            after = 0.0
            reason = "арест или лишение свободы отсутствует в применимом профиле"
        adjusted["d_imprisonment_epochs"] = after
        _record_adjustment(audit, "d_imprisonment_epochs", before, after, reason)

    optional_durations = (
        (
            "d_mandatory_work_hours",
            "c_reg_mandatory_work_min_hours",
            "c_reg_mandatory_work_max_hours",
            "часов обязательных работ",
        ),
        (
            "d_forced_work_months",
            "c_reg_forced_work_min_months",
            "c_reg_forced_work_max_months",
            "месяцев принудительных работ",
        ),
        (
            "d_corrective_work_months",
            "c_reg_corrective_work_min_months",
            "c_reg_corrective_work_max_months",
            "месяцев исправительных работ",
        ),
        (
            "d_restriction_freedom_months",
            "c_reg_restriction_freedom_min_months",
            "c_reg_restriction_freedom_max_months",
            "месяцев ограничения свободы",
        ),
    )
    for target, minimum_feature, maximum_feature, unit in optional_durations:
        if target not in adjusted:
            continue
        before = float(adjusted[target])
        minimum = float(features[minimum_feature])
        maximum = float(features[maximum_feature])
        if not active or maximum <= 0:
            after = 0.0
            reason = f"санкция в {unit} отсутствует в применимом профиле"
        else:
            after = 0.0 if before <= 0 else min(max(before, minimum), maximum)
            reason = f"допустимый диапазон [{minimum}, {maximum}] {unit}"
        adjusted[target] = after
        _record_adjustment(audit, target, before, after, reason)

    return adjusted, audit, context


def constrain_prediction_arrays(
    predictions: Mapping[str, Sequence[object]],
    df: pd.DataFrame,
    overrides: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Ограничение пакетных предсказаний и расчёт статистики нарушений.

    Параметры:
        predictions: Массивы сырых предсказаний по целям.
        df: Таблица соответствующих обстоятельств.

    Возвращает:
        Скорректированные массивы и агрегированный отчёт ограничителя.
    """
    adjusted = {name: np.asarray(values).copy() for name, values in predictions.items()}
    by_target: Dict[str, int] = {}
    by_reason: Dict[str, int] = {}
    rows_by_article: Dict[str, int] = {}
    adjusted_rows_by_article: Dict[str, int] = {}
    adjustments_by_target_article: Dict[str, Dict[str, int]] = {}
    rows_adjusted = 0
    adjustment_count = 0
    for position, (_, row) in enumerate(df.iterrows()):
        article = str(_safe_int(row.get("c_article_code"), default=-1))
        rows_by_article[article] = rows_by_article.get(article, 0) + 1
        raw_row = {name: np.asarray(values)[position].item() for name, values in predictions.items()}
        constrained, audit, _ = constrain_prediction(raw_row, row.to_dict(), overrides=overrides)
        if audit:
            rows_adjusted += 1
            adjusted_rows_by_article[article] = adjusted_rows_by_article.get(article, 0) + 1
        for item in audit:
            target = str(item["target"])
            reason = str(item["reason"])
            by_target[target] = by_target.get(target, 0) + 1
            by_reason[reason] = by_reason.get(reason, 0) + 1
            target_articles = adjustments_by_target_article.setdefault(target, {})
            target_articles[article] = target_articles.get(article, 0) + 1
            adjustment_count += 1
        for name, value in constrained.items():
            if name in adjusted:
                adjusted[name][position] = value
    total = int(len(df))
    return adjusted, {
        "regulation_version": REGULATION_VERSION,
        "rows": total,
        "rows_adjusted": rows_adjusted,
        "row_adjustment_rate": float(rows_adjusted / total) if total else 0.0,
        "adjustment_count": adjustment_count,
        "adjustments_by_target": by_target,
        "adjustments_by_reason": by_reason,
        "rows_by_article": rows_by_article,
        "adjusted_rows_by_article": adjusted_rows_by_article,
        "adjustments_by_target_article": adjustments_by_target_article,
    }
