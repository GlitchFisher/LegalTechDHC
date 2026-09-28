from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from legal_regulation import (
    REGULATION_SOURCE,
    REGULATION_VERSION,
    REGULATORY_FEATURE_COLUMNS,
    constrain_prediction,
    constrain_prediction_arrays,
    enrich_regulatory_features,
    regulatory_context,
)

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, Ridge
except Exception:
    HistGradientBoostingClassifier = None
    HistGradientBoostingRegressor = None
    TfidfVectorizer = None
    LogisticRegression = None
    Ridge = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    WeightedRandomSampler = None

BaseTorchModule = nn.Module if nn is not None else object


TARGET_COLUMNS: List[str] = [
    "d_imprisonment_epochs",
    "d_license_revocation_epochs",
    "d_mandatory_work_hours",
    "d_forced_work_months",
    "d_corrective_work_months",
    "d_restriction_freedom_months",
    "d_conditional",
    "d_fine_rub",
    "d_travel_ban",
    "d_reconciliation",
    "d_costs_on_state",
    "d_costs_amount_rub",
    "d_self_recusal",
]

TARGET_KIND: Dict[str, str] = {
    "d_imprisonment_epochs": "regression",
    "d_license_revocation_epochs": "regression",
    "d_mandatory_work_hours": "regression",
    "d_forced_work_months": "regression",
    "d_corrective_work_months": "regression",
    "d_restriction_freedom_months": "regression",
    "d_conditional": "binary",
    "d_fine_rub": "regression",
    "d_travel_ban": "binary",
    "d_reconciliation": "binary",
    "d_costs_on_state": "binary",
    "d_costs_amount_rub": "regression",
    "d_self_recusal": "binary",
}

SUBSTANTIVE_TARGETS = [
    "d_imprisonment_epochs",
    "d_license_revocation_epochs",
    "d_mandatory_work_hours",
    "d_forced_work_months",
    "d_corrective_work_months",
    "d_restriction_freedom_months",
    "d_fine_rub",
]

NONNEGATIVE_LOG_TARGETS = {
    "d_imprisonment_epochs",
    "d_license_revocation_epochs",
    "d_mandatory_work_hours",
    "d_forced_work_months",
    "d_corrective_work_months",
    "d_restriction_freedom_months",
    "d_fine_rub",
    "d_costs_amount_rub",
}

# Минимум строк с ненулевой санкцией, при котором двухчастная модель устойчива.
# Ниже него «сколько назначено» оценивать не на чем, и работает обычная регрессия.
MIN_TWO_PART_POSITIVES = 30

EMPTY_PRONE_TARGETS = [
    "d_conditional",
    "d_travel_ban",
    "d_reconciliation",
    "d_costs_on_state",
    "d_costs_amount_rub",
    "d_self_recusal",
]

TEXT_COLUMNS = [
    "circumstances",
]

DEFAULT_ORDER = TARGET_COLUMNS.copy()
REVERSE_ORDER = list(reversed(TARGET_COLUMNS))


def set_seed(seed: int) -> None:
    """Синхронизация зерна для Python, NumPy и доступных генераторов Torch.

    Параметры:
        seed: Целое значение зерна для воспроизводимых экспериментов.
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def accuracy_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Доля совпадений для равных по длине массивов меток NumPy.

    Параметры:
        y_true: Истинные метки.
        y_pred: Предсказанные метки.

    Возвращает:
        Долю полностью совпавших меток или 0.0 для пустого входа.
    """
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def macro_f1_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Макро-F1 для меток, встречающихся в истинных или предсказанных массивах.

    Параметры:
        y_true: Истинные метки классов.
        y_pred: Предсказанные метки классов.

    Возвращает:
        Арифметическое среднее F1-мер по отдельным классам.
    """
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    if not labels:
        return 0.0
    scores: List[float] = []
    for cls in labels:
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Средняя абсолютная ошибка для числовых массивов.

    Параметры:
        y_true: Истинные числовые значения.
        y_pred: Предсказанные числовые значения.

    Возвращает:
        Среднее абсолютное отклонение или 0.0 для пустого входа.
    """
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else 0.0


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Корень из среднеквадратичной ошибки для числовых массивов.

    Параметры:
        y_true: Истинные числовые значения.
        y_pred: Предсказанные числовые значения.

    Возвращает:
        Корень из среднего квадратичного отклонения или 0.0 для пустого входа.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else 0.0


def regression_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Нормированная оценка качества регрессии на основе MAE.

    Параметры:
        y_true: Истинные числовые значения.
        y_pred: Предсказанные числовые значения.

    Возвращает:
        Оценку в интервале [0, 1], где большее значение означает лучшее качество.
    """
    scale = float(np.mean(np.abs(y_true)) + np.std(y_true) + 1.0)
    return float(max(0.0, 1.0 - mae_np(y_true, y_pred) / scale))


def target_metric_score(kind: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Оценка качества одного целевого признака для классификации или регрессии.

    Параметры:
        kind: Тип целевого признака: ``binary``, ``class`` или ``regression``.
        y_true: Истинные значения целевого признака.
        y_pred: Предсказанные значения целевого признака.

    Возвращает:
        Макро-F1 для классификационных целей и нормированную MAE-оценку
        для регрессионных целей.
    """
    if kind in {"binary", "class"}:
        return macro_f1_score_np(y_true.astype(np.int64), np.asarray(y_pred, dtype=np.int64))
    return regression_score(y_true.astype(np.float32), np.asarray(y_pred, dtype=np.float32))


def average_named_scores(scores_by_target: Dict[str, float], names: Sequence[str]) -> Optional[float]:
    """Средняя оценка по именованной группе целевых признаков.

    Параметры:
        scores_by_target: Словарь оценок по именам целевых признаков.
        names: Имена целевых признаков, включенных в групповое среднее.

    Возвращает:
        Округленное арифметическое среднее по доступным целям или ``None``,
        если запрошенные цели отсутствуют.
    """
    values = [scores_by_target[name] for name in names if name in scores_by_target]
    if not values:
        return None
    return round(float(np.mean(values)), 6)


def feature_columns_from_df(df: pd.DataFrame) -> List[str]:
    """Поиск колонок признаков обстоятельств.

    Параметры:
        df: Входная таблица с колонками модели.

    Возвращает:
        Список колонок, имена которых начинаются с ``c_``.

    Исключения:
        ValueError: Колонки признаков обстоятельств не найдены.
    """
    cols = [col for col in df.columns if col.startswith("c_")]
    if not cols:
        raise ValueError("Не найдены признаки c_*. Проверьте vectors.csv.")
    return cols


def target_columns_from_df(df: pd.DataFrame) -> List[str]:
    """Поиск известных целевых колонок судебного решения.

    Параметры:
        df: Входная таблица с колонками модели.

    Возвращает:
        Упорядоченные имена целевых колонок, присутствующих в таблице.
    """
    return [col for col in TARGET_COLUMNS if col in df.columns]


def restrict_bundle_targets(bundle: object, allowed_targets: Sequence[str]) -> None:
    """Ограничение загруженного артефакта текущим списком целевых признаков.

    Параметры:
        bundle: Сохранённое семейство моделей или его вложенный компонент.
        allowed_targets: Имена целей, разрешённых текущей версией системы.

    Возвращает:
        Нет.
    """
    allowed = set(allowed_targets)
    if hasattr(bundle, "target_columns"):
        bundle.target_columns = [target for target in bundle.target_columns if target in allowed]
    if hasattr(bundle, "order"):
        bundle.order = [target for target in bundle.order if target in allowed]

    models = getattr(bundle, "models", None)
    if isinstance(models, dict):
        for target in list(models):
            if target not in allowed:
                del models[target]

    target_info = getattr(bundle, "target_info", None)
    if isinstance(target_info, dict):
        for target in list(target_info):
            if target not in allowed:
                del target_info[target]

    heads = getattr(bundle, "heads", None)
    if heads is not None and hasattr(heads, "keys"):
        for target in list(heads.keys()):
            if target not in allowed:
                del heads[target]

    for pipeline in getattr(bundle, "pipelines", []) or []:
        restrict_bundle_targets(pipeline, allowed_targets)

    nested_model = getattr(bundle, "model", None)
    if nested_model is not None and hasattr(nested_model, "heads"):
        restrict_bundle_targets(nested_model, allowed_targets)


def numeric_matrix(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Преобразование выбранных колонок таблицы в числовую матрицу признаков.

    Параметры:
        df: Исходная таблица.
        columns: Имена колонок для преобразования.

    Возвращает:
        Матрицу NumPy float32, где нечисловые и пропущенные значения заменены на 0.
    """
    data = df.loc[:, columns].copy()
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.fillna(0.0).astype(np.float32).to_numpy(dtype=np.float32, copy=False)


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Статистики стандартизации для числовой матрицы.

    Параметры:
        X: Числовая матрица для оценки средних значений и масштабов колонок.

    Возвращает:
        Пару ``(mean, scale)``, где нулевые масштабы заменены на 1.
    """
    mean = X.mean(axis=0).astype(np.float32)
    scale = X.std(axis=0).astype(np.float32)
    scale[scale == 0] = 1.0
    return mean, scale


def standardize_apply(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Преобразование стандартизации для числовой матрицы.

    Параметры:
        X: Числовая матрица для преобразования.
        mean: Средние значения колонок, полученные из ``standardize_fit``.
        scale: Масштабы колонок, полученные из ``standardize_fit``.

    Возвращает:
        Стандартизированную матрицу float32.
    """
    return ((X - mean) / scale).astype(np.float32)


def to_target_array(values: pd.Series, kind: str) -> np.ndarray:
    """Преобразование целевой колонки с учетом типа целевого признака.

    Параметры:
        values: Исходная серия целевого признака.
        kind: Тип целевого признака: ``binary``, ``class`` или ``regression``.

    Возвращает:
        Массив NumPy с типом данных и ограничениями, подходящими для цели.
    """
    values = pd.to_numeric(values, errors="coerce").fillna(0)
    if kind == "binary":
        return values.astype(int).clip(0, 1).to_numpy(dtype=np.int64)
    if kind == "class":
        return values.astype(int).to_numpy(dtype=np.int64)
    return values.astype(np.float32).to_numpy(dtype=np.float32)


def make_article_key(df: pd.DataFrame) -> np.ndarray:
    """Вектор ключей статей для балансировки на уровне статьи.

    Параметры:
        df: Исходная таблица с необязательной колонкой ``c_article_code``.

    Возвращает:
        Объектный массив ключей с кодами статей.
    """
    article = pd.to_numeric(df.get("c_article_code", pd.Series([128] * len(df))), errors="coerce").fillna(128).astype(int)
    return article.astype(str).to_numpy(dtype=object)


def make_strata_key(df: pd.DataFrame) -> np.ndarray:
    """Вектор ключей стратификации для групп статья-регион-год.

    Профиль наказаний в ключ НЕ входит: попытка добавить его (суффикс
    ``:fine_imposed``) проверялась и отвергнута — см. запись в памяти
    «court-model-planA-profile-strata». Кратко: вес редкого сочетания
    «ст.264.1 со штрафом» вырос в 5.9 раза, полнота по штрафу выросла
    с 3.1% до 26.8%, но калибровка ст.12.8 развалилась (попадание ±10%
    упало с 89.9% до 39.8%), а верхняя граница штрафа ст.264.1 так и не
    начала влиять. Перераспределение веса не добавляет информации.

    Параметры:
        df: Исходная таблица с колонками статьи, региона и года.

    Возвращает:
        Объектный массив ключей в формате ``article:region:year``.
    """
    article = pd.to_numeric(df.get("c_article_code", pd.Series([128] * len(df))), errors="coerce").fillna(128).astype(int)
    region = pd.to_numeric(df.get("c_region_code", pd.Series([0] * len(df))), errors="coerce").fillna(0).astype(int)
    year = pd.to_numeric(df.get("c_year", pd.Series([0] * len(df))), errors="coerce").fillna(0).astype(int)
    return np.asarray([f"{a}:{r}:{y}" for a, r, y in zip(article, region, year)], dtype=object)


def stratified_split_by_key(
    keys: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Стратифицированное разбиение индексов по заранее рассчитанным ключам групп.

    Параметры:
        keys: Ключ группы для каждой строки.
        train_ratio: Доля строк для обучающей выборки.
        val_ratio: Доля строк для валидационной выборки.
        test_ratio: Доля строк для тестовой выборки.
        seed: Зерно случайности для перемешивания внутри групп.

    Возвращает:
        Кортеж массивов индексов обучающей, валидационной и тестовой выборок.

    Исключения:
        ValueError: Сумма долей разбиения не равна 1.
    """
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for key in np.unique(keys):
        idx = np.where(keys == key)[0]
        rng.shuffle(idx)
        n = len(idx)
        if n == 1:
            train_idx.extend(idx.tolist())
            continue
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(0, int(round(n * val_ratio)))
        n_test = max(0, n - n_train - n_val)
        while n_train + n_val + n_test > n:
            n_train = max(1, n_train - 1)
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train : n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val :].tolist())
    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


def sample_weights_from_keys(keys: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Веса объектов, обратно пропорциональные частоте выбранных ключей.

    Параметры:
        keys: Ключ группы для каждой строки полного датасета.
        indices: Выбранные индексы строк, обычно индексы обучающей выборки.

    Возвращает:
        Массив весов float32, нормированный до среднего значения 1.
    """
    selected = keys[indices]
    uniq, counts = np.unique(selected, return_counts=True)
    freq = {key: count for key, count in zip(uniq, counts)}
    weights = np.asarray([1.0 / max(1, freq[key]) for key in selected], dtype=np.float32)
    weights *= len(weights) / max(float(weights.sum()), 1e-12)
    return weights


def oversample_indices(keys: np.ndarray, indices: np.ndarray, seed: int) -> np.ndarray:
    """Индексы бутстрэп-повтора для малочисленных групп ключей.

    Параметры:
        keys: Ключ группы для каждой строки полного датасета.
        indices: Выбранные индексы строк, обычно индексы обучающей выборки.
        seed: Зерно случайности для перемешивания повторенных индексов.

    Возвращает:
        Массив индексов, где каждая выбранная группа повторена до размера
        крупнейшей группы.
    """
    rng = np.random.default_rng(seed)
    selected = keys[indices]
    uniq, counts = np.unique(selected, return_counts=True)
    target_count = int(np.max(counts))
    out: List[int] = []
    for key in uniq:
        key_idx = indices[selected == key]
        reps = int(math.ceil(target_count / len(key_idx)))
        tiled = np.tile(key_idx, reps)[:target_count]
        rng.shuffle(tiled)
        out.extend(tiled.tolist())
    rng.shuffle(out)
    return np.asarray(out, dtype=np.int64)


def apply_balance(df: pd.DataFrame, train_idx: np.ndarray, balance: str, seed: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Применение режима балансировки к обучающим индексам и весам объектов.

    Параметры:
        df: Полная обучающая таблица.
        train_idx: Исходные индексы обучающей выборки.
        balance: Название режима балансировки.
        seed: Зерно случайности для повторной выборки.

    Возвращает:
        Пару ``(indices, sample_weight)``. Значение ``sample_weight`` равно
        ``None`` для невзвешенного режима и режима oversample.

    Исключения:
        ValueError: Указан неизвестный режим балансировки.
    """
    if balance == "none":
        return train_idx, None
    if balance == "article_weights":
        return train_idx, sample_weights_from_keys(make_article_key(df), train_idx)
    strata = make_strata_key(df)
    if balance == "strata_weights":
        return train_idx, sample_weights_from_keys(strata, train_idx)
    if balance == "oversample":
        return oversample_indices(strata, train_idx, seed), None
    raise ValueError("balance must be one of: none, article_weights, strata_weights, oversample")


def build_text_rows(df: pd.DataFrame, feature_columns: Sequence[str]) -> List[str]:
    """Построение псевдотекста из текстовых колонок и токенов признак-значение.

    Параметры:
        df: Исходная таблица.
        feature_columns: Колонки признаков обстоятельств, преобразуемые в токены.

    Возвращает:
        Список псевдотекстовых строк для категориальной базовой модели TF-IDF.
    """
    texts: List[str] = []
    available_text_cols = [col for col in TEXT_COLUMNS if col in df.columns]
    for _, row in df.iterrows():
        parts: List[str] = []
        for col in available_text_cols:
            value = row.get(col)
            if pd.notna(value):
                parts.append(f"{col} {value}")
        for col in feature_columns:
            value = row.get(col)
            if pd.notna(value):
                parts.append(f"{col}_{value}")
        texts.append(" ".join(parts))
    return texts


@dataclass
class TargetEstimator:
    """Артефакт оценивателя для одного выходного признака судебного решения."""

    name: str
    kind: str
    model: object | None = None
    classes: Optional[List[int]] = None
    constant_value: Optional[float] = None
    log_transform: bool = False
    # Двухчастная модель для неотрицательных целей с массой в нуле: «санкция
    # назначена?» и «сколько назначено?» оцениваются отдельно. Артефакты,
    # обученные до её появления, этих полей не содержат — значения по умолчанию
    # (None) сохраняют прежний путь через ``model``.
    impose_model: object | None = None
    amount_model: object | None = None

    def fit(self, X, y: np.ndarray, sample_weight: Optional[np.ndarray], seed: int, family: str) -> "TargetEstimator":
        """Обучение оценивателя для одной цели и одного семейства моделей.

        Параметры:
            X: Матрица признаков для выбранных обучающих строк.
            y: Целевой массив, согласованный с ``X``.
            sample_weight: Необязательные веса объектов, согласованные с ``X``.
            seed: Зерно случайности для инициализации оценивателя.
            family: Название семейства моделей: ``nlp`` или табличный маркер.

        Возвращает:
            Обученный оцениватель целевого признака.
        """
        unique = np.unique(y)
        if len(unique) <= 1:
            self.constant_value = float(unique[0]) if len(unique) else 0.0
            self.classes = unique.astype(int).tolist() if self.kind in {"binary", "class"} else None
            return self
        if family == "nlp":
            if self.kind in {"binary", "class"}:
                model = LogisticRegression(max_iter=1000, class_weight="balanced")
                model.fit(X, y, sample_weight=sample_weight)
            else:
                self.log_transform = self.name in NONNEGATIVE_LOG_TARGETS
                y_fit = np.log1p(np.clip(y, 0.0, None)) if self.log_transform else y
                model = Ridge(alpha=1.0, random_state=seed)
                model.fit(X, y_fit, sample_weight=sample_weight)
        elif self.kind in {"binary", "class"}:
            model = HistGradientBoostingClassifier(
                max_depth=4,
                learning_rate=0.08,
                max_iter=220,
                random_state=seed,
                early_stopping=False,
            )
            model.fit(X, y, sample_weight=sample_weight)
        else:
            positives = y > 0
            if (
                self.name in NONNEGATIVE_LOG_TARGETS
                and int(positives.sum()) >= MIN_TWO_PART_POSITIVES
                and int((~positives).sum()) > 0
            ):
                # Одна лог-регрессия на такой цели предсказывает
                # expm1(E[log1p(y)|x]) — при массе в нуле это почти константа,
                # сильно ниже медианы (ст.12.8: штраф 1289 руб. против медианы
                # 30000). Сырой выход тогда всегда падает ниже законного
                # минимума, clamp вырождается в минимум и верхняя граница
                # наказания перестаёт влиять. Разложение возвращает обе части.
                self.impose_model = HistGradientBoostingClassifier(
                    max_depth=4,
                    learning_rate=0.08,
                    max_iter=220,
                    random_state=seed,
                    early_stopping=False,
                )
                self.impose_model.fit(X, positives.astype(np.int64), sample_weight=sample_weight)
                self.amount_model = HistGradientBoostingRegressor(
                    max_depth=4,
                    learning_rate=0.06,
                    max_iter=260,
                    random_state=seed,
                    early_stopping=False,
                )
                self.amount_model.fit(
                    X[positives],
                    y[positives],
                    sample_weight=None if sample_weight is None else sample_weight[positives],
                )
                return self
            self.log_transform = self.name in NONNEGATIVE_LOG_TARGETS
            y_fit = np.log1p(np.clip(y, 0.0, None)) if self.log_transform else y
            model = HistGradientBoostingRegressor(
                max_depth=4,
                learning_rate=0.06,
                max_iter=260,
                random_state=seed,
                early_stopping=False,
            )
            model.fit(X, y_fit, sample_weight=sample_weight)
        self.model = model
        if self.kind in {"binary", "class"}:
            self.classes = sorted(np.unique(y).astype(int).tolist())
        return self

    def predict(self, X) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Предсказание цели обученным оценивателем одного признака.

        Параметры:
            X: Матрица признаков для предсказания.

        Возвращает:
            Пару ``(prediction, probabilities)``. Вероятности доступны только
            для оценивателей с методом предсказания вероятностей классов.

        Исключения:
            RuntimeError: Оцениватель еще не обучен.
        """
        if self.constant_value is not None:
            if self.kind in {"binary", "class"}:
                pred = np.full(X.shape[0], int(self.constant_value), dtype=np.int64)
                return pred, None
            return np.full(X.shape[0], float(self.constant_value), dtype=np.float32), None
        if self.impose_model is not None and self.amount_model is not None:
            # Решение суда, а не математическое ожидание: рантайм берёт ОДИН
            # вердикт из пула как состоявшееся решение (CourtSystem.sample_verdict),
            # поэтому домножать сумму на вероятность нельзя — иначе теряется
            # частота назначения санкции и минимум приписывается всем строкам.
            imposed = self.impose_model.predict_proba(X)[:, 1] > 0.5
            amount = np.clip(self.amount_model.predict(X), 0.0, None)
            pred = np.where(imposed, amount, 0.0).astype(np.float32)
            return pred, None
        if self.model is None:
            raise RuntimeError(f"Target {self.name} is not fitted")
        if self.kind in {"binary", "class"}:
            pred = self.model.predict(X).astype(np.int64)
            proba = self.model.predict_proba(X).astype(np.float32) if hasattr(self.model, "predict_proba") else None
            return pred, proba
        pred = self.model.predict(X).astype(np.float32)
        if self.log_transform:
            pred = np.expm1(pred)
        return np.clip(pred, 0.0, None), None


class ChainModel:
    """Модель цепочки целей для авторегрессионного многоцелевого предсказания."""

    def __init__(self, order: Sequence[str], seed: int, use_context: bool = True):
        """Инициализация цепочки целевых признаков.

        Параметры:
            order: Порядок целевых признаков внутри цепочки.
            seed: Зерно случайности для целевых оценивателей.
            use_context: Флаг добавления предыдущих предсказаний как
                дополнительных признаков.
        """
        self.order = list(order)
        self.seed = seed
        self.use_context = use_context
        self.models: Dict[str, TargetEstimator] = {}
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []

    def _augment(self, X: np.ndarray, prev: Dict[str, np.ndarray]) -> np.ndarray:
        """Дополнение признаков предыдущими предсказаниями целей.

        Параметры:
            X: Базовая матрица признаков.
            prev: Словарь предыдущих целевых признаков и массивов предсказаний.

        Возвращает:
            Базовую матрицу или матрицу, объединенную с предыдущими предсказаниями.
        """
        if not self.use_context or not prev:
            return X
        extras = [prev[target].reshape(len(X), 1).astype(np.float32) for target in self.order if target in prev]
        return np.concatenate([X] + extras, axis=1) if extras else X

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
        target_columns: Sequence[str],
        train_idx: np.ndarray,
        sample_weight: Optional[np.ndarray],
        desc: str,
    ) -> None:
        """Обучение цепочки целей в заданном порядке.

        Параметры:
            df: Полная таблица с признаками и целями.
            feature_columns: Имена колонок признаков обстоятельств.
            target_columns: Имена доступных целевых колонок.
            train_idx: Индексы строк обучающей выборки.
            sample_weight: Необязательные веса обучающих объектов.
            desc: Описание индикатора прогресса.
        """
        self.feature_columns = list(feature_columns)
        self.target_columns = list(target_columns)
        X_all = numeric_matrix(df, feature_columns)
        X_train = X_all[train_idx]
        prev_train: Dict[str, np.ndarray] = {}
        for pos, target in enumerate(tqdm(self.order, desc=desc, leave=False)):
            if target not in target_columns:
                continue
            kind = TARGET_KIND[target]
            y = to_target_array(df[target], kind)[train_idx]
            sw = sample_weight
            if sw is not None and kind == "binary":
                positives = max(float((y == 1).sum()), 1.0)
                negatives = max(float((y == 0).sum()), 1.0)
                sw = sw * np.where(y == 1, negatives / positives, 1.0).astype(np.float32)
            X_input = self._augment(X_train, prev_train)
            model = TargetEstimator(target, kind).fit(X_input, y, sw, self.seed + pos, family="gbdt")
            self.models[target] = model
            pred, _ = model.predict(X_input)
            prev_train[target] = pred.astype(np.float32)

    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Предсказание цепочки целей для строк таблицы.

        Параметры:
            df: Таблица с колонками признаков обстоятельств.

        Возвращает:
            Словарь имен целей с массивами предсказаний и необязательными
            массивами вероятностей.
        """
        X = numeric_matrix(df, self.feature_columns)
        prev: Dict[str, np.ndarray] = {}
        out: Dict[str, np.ndarray] = {}
        for target in self.order:
            if target not in self.models:
                continue
            pred, proba = self.models[target].predict(self._augment(X, prev))
            out[target] = pred
            if proba is not None:
                out[f"{target}__proba"] = proba
            prev[target] = pred.astype(np.float32)
        return out


class GraphDecisionModel:
    """Ансамбль цепочек целей с несколькими порядками обхода целевых признаков."""

    def __init__(self, seed: int = 42, graph_width: int = 4, method: str = "graph"):
        """Инициализация графового ансамбля цепочек.

        Параметры:
            seed: Зерно случайности для порядков цепочек и оценивателей.
            graph_width: Число порядков цепочек в режиме ``graph``.
            method: Название режима: ``independent``, ``chain`` или ``graph``.
        """
        self.seed = seed
        self.graph_width = graph_width
        self.method = method
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        self.pipelines: List[ChainModel] = []

    def _orders(self) -> List[List[str]]:
        """Генерация порядков целей для режимов independent, chain или graph.

        Возвращает:
            Список порядков целей для создания моделей-цепочек.
        """
        if self.method == "independent":
            return [DEFAULT_ORDER]
        if self.method == "chain":
            return [DEFAULT_ORDER]
        rng = np.random.default_rng(self.seed)
        orders = [DEFAULT_ORDER, REVERSE_ORDER]
        for _ in range(max(0, self.graph_width - 2)):
            order = DEFAULT_ORDER.copy()
            rng.shuffle(order)
            orders.append(order)
        return orders

    def fit(self, df: pd.DataFrame, feature_columns: Sequence[str], target_columns: Sequence[str], train_idx: np.ndarray, balance: str) -> None:
        """Обучение графового режима с необязательной балансировкой.

        Параметры:
            df: Полная таблица с признаками и целями.
            feature_columns: Имена колонок признаков обстоятельств.
            target_columns: Имена доступных целевых колонок.
            train_idx: Индексы строк обучающей выборки.
            balance: Название режима балансировки.

        Исключения:
            RuntimeError: Оцениватели scikit-learn недоступны.
        """
        if HistGradientBoostingClassifier is None or HistGradientBoostingRegressor is None:
            raise RuntimeError("Для методов graph/chain/independent нужен scikit-learn.")
        self.feature_columns = list(feature_columns)
        self.target_columns = list(target_columns)
        balanced_idx, sample_weight = apply_balance(df, train_idx, balance, self.seed)
        self.pipelines = []
        for i, order in enumerate(tqdm(self._orders(), desc=f"{self.method} pipelines")):
            pipe = ChainModel(order, seed=self.seed + i * 17, use_context=self.method != "independent")
            pipe.fit(df, feature_columns, target_columns, balanced_idx, sample_weight, desc=f"{self.method} targets {i + 1}")
            self.pipelines.append(pipe)

    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Предсказание графового ансамбля для строк таблицы.

        Параметры:
            df: Таблица с колонками признаков обстоятельств.

        Возвращает:
            Словарь имен целей с агрегированными массивами предсказаний.
        """
        predictions = [pipe.predict(df) for pipe in self.pipelines]
        out: Dict[str, np.ndarray] = {}
        for target in self.target_columns:
            values = [pred[target] for pred in predictions if target in pred]
            if not values:
                continue
            if TARGET_KIND[target] == "regression":
                out[target] = np.mean(np.stack(values, axis=0), axis=0).astype(np.float32)
            else:
                stacked = np.stack(values, axis=0)
                merged = []
                for col in stacked.T:
                    labels, counts = np.unique(col, return_counts=True)
                    merged.append(int(labels[np.argmax(counts)]))
                out[target] = np.asarray(merged, dtype=np.int64)
        return out


class NlpDecisionModel:
    """Категориальная базовая модель TF-IDF для многоцелевого предсказания решения суда."""

    def __init__(self, seed: int = 42):
        """Инициализация категориальной базовой модели TF-IDF.

        Параметры:
            seed: Зерно случайности для целевых оценивателей.

        Исключения:
            RuntimeError: Необходимые компоненты scikit-learn недоступны.
        """
        if TfidfVectorizer is None or LogisticRegression is None or Ridge is None:
            raise RuntimeError("Для метода nlp нужен scikit-learn.")
        self.seed = seed
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        self.vectorizer = TfidfVectorizer(min_df=1, max_features=60000, ngram_range=(1, 2), sublinear_tf=True)
        self.models: Dict[str, TargetEstimator] = {}

    def fit(self, df: pd.DataFrame, feature_columns: Sequence[str], target_columns: Sequence[str], train_idx: np.ndarray, balance: str) -> None:
        """Обучение категориальной базовой модели TF-IDF только на обучающих строках.

        Параметры:
            df: Полная таблица с признаками и целями.
            feature_columns: Имена колонок признаков обстоятельств.
            target_columns: Имена доступных целевых колонок.
            train_idx: Индексы строк обучающей выборки.
            balance: Название режима балансировки.
        """
        self.feature_columns = list(feature_columns)
        self.target_columns = list(target_columns)
        balanced_idx, sample_weight = apply_balance(df, train_idx, balance, self.seed)
        texts = build_text_rows(df, feature_columns)
        train_texts = [texts[int(idx)] for idx in balanced_idx]
        X_train = self.vectorizer.fit_transform(train_texts)
        for pos, target in enumerate(tqdm(target_columns, desc="nlp targets")):
            kind = TARGET_KIND[target]
            y = to_target_array(df[target], kind)[balanced_idx]
            model = TargetEstimator(target, kind).fit(X_train, y, sample_weight, self.seed + pos, family="nlp")
            self.models[target] = model

    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Предсказание категориальной базовой модели TF-IDF для строк таблицы.

        Параметры:
            df: Таблица с колонками признаков обстоятельств.

        Возвращает:
            Словарь имен целей с массивами предсказаний.
        """
        X = self.vectorizer.transform(build_text_rows(df, self.feature_columns))
        out: Dict[str, np.ndarray] = {}
        for target, model in self.models.items():
            pred, _ = model.predict(X)
            out[target] = pred
        return out


class MultiTaskNet(BaseTorchModule):
    """Модуль с общим стволом признаков и отдельными выходными головами целей."""

    def __init__(self, input_dim: int, target_info: Dict[str, Dict], hidden_dims: Sequence[int], dropout: float):
        """Инициализация модуля с общим стволом.

        Параметры:
            input_dim: Количество входных признаков.
            target_info: Метаданные целей с типами целей и числом классов.
            hidden_dims: Размерности скрытых слоев общего ствола.
            dropout: Вероятность прореживания между скрытыми слоями.
        """
        super().__init__()
        layers: List[nn.Module] = []
        current = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(current, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)])
            current = hidden
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleDict()
        for target, info in target_info.items():
            out_dim = info["num_classes"] if info["kind"] in {"binary", "class"} else 1
            self.heads[target] = nn.Linear(current, out_dim)

    def forward(self, x):
        """Прямой проход через общий ствол и целевые головы.

        Параметры:
            x: Входной тензор признаков.

        Возвращает:
            Словарь имен целей с выходными тензорами.
        """
        shared = self.trunk(x)
        return {target: head(shared) for target, head in self.heads.items()}


class MlpDecisionModel:
    """Многоцелевая модель общего представления для табличных векторов обстоятельств."""

    def __init__(self, seed: int = 42, epochs: int = 35, batch_size: int = 128, lr: float = 1e-3):
        """Инициализация модели общего представления.

        Параметры:
            seed: Зерно случайности для воспроизводимого обучения.
            epochs: Количество эпох обучения.
            batch_size: Размер мини-батча.
            lr: Скорость обучения оптимизатора.

        Исключения:
            RuntimeError: Torch недоступен.
        """
        if torch is None:
            raise RuntimeError("Для метода mlp нужен torch. Установите зависимости из requirements.txt.")
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        self.mean: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None
        self.target_info: Dict[str, Dict] = {}
        self.model: Optional[MultiTaskNet] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prepare_targets(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Подготовка целей для классификационных и регрессионных голов.

        Параметры:
            df: Исходная таблица с целевыми колонками.

        Возвращает:
            Словарь имен целей с закодированными или стандартизированными массивами.
        """
        arrays: Dict[str, np.ndarray] = {}
        self.target_info = {}
        for target in self.target_columns:
            kind = TARGET_KIND[target]
            raw = to_target_array(df[target], kind)
            if kind in {"binary", "class"}:
                classes = sorted(np.unique(raw).astype(int).tolist())
                class_to_idx = {cls: i for i, cls in enumerate(classes)}
                arrays[target] = np.asarray([class_to_idx[int(v)] for v in raw], dtype=np.int64)
                self.target_info[target] = {"kind": kind, "classes": classes, "num_classes": len(classes)}
            else:
                log_transform = target in NONNEGATIVE_LOG_TARGETS
                y = np.log1p(np.clip(raw, 0.0, None)) if log_transform else raw.astype(np.float32)
                mean = float(y.mean())
                scale = float(y.std() or 1.0)
                arrays[target] = ((y - mean) / scale).astype(np.float32)
                self.target_info[target] = {"kind": kind, "mean": mean, "scale": scale, "log_transform": log_transform}
        return arrays

    def fit(self, df: pd.DataFrame, feature_columns: Sequence[str], target_columns: Sequence[str], train_idx: np.ndarray, balance: str) -> None:
        """Обучение модели общего представления с необязательной балансировкой объектов.

        Параметры:
            df: Полная таблица с признаками и целями.
            feature_columns: Имена колонок признаков обстоятельств.
            target_columns: Имена доступных целевых колонок.
            train_idx: Индексы строк обучающей выборки.
            balance: Название режима балансировки.
        """
        set_seed(self.seed)
        self.feature_columns = list(feature_columns)
        self.target_columns = list(target_columns)
        balanced_idx, sample_weight = apply_balance(df, train_idx, balance, self.seed)
        X = numeric_matrix(df, feature_columns)
        self.mean, self.scale = standardize_fit(X[balanced_idx])
        X = standardize_apply(X, self.mean, self.scale)
        targets = self._prepare_targets(df)

        x_tensor = torch.tensor(X[balanced_idx], dtype=torch.float32)
        tensors = [x_tensor]
        for target in self.target_columns:
            dtype = torch.long if self.target_info[target]["kind"] in {"binary", "class"} else torch.float32
            tensors.append(torch.tensor(targets[target][balanced_idx], dtype=dtype))
        dataset = TensorDataset(*tensors)
        if sample_weight is not None:
            weights = torch.tensor(sample_weight, dtype=torch.double)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)
        else:
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = MultiTaskNet(len(feature_columns), self.target_info, hidden_dims=(256, 128, 64), dropout=0.25).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        ce_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        for epoch in tqdm(range(1, self.epochs + 1), desc="mlp epochs"):
            self.model.train()
            total_loss = 0.0
            for batch in loader:
                x_batch = batch[0].to(self.device)
                y_batches = {target: batch[i + 1].to(self.device) for i, target in enumerate(self.target_columns)}
                optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = torch.tensor(0.0, device=self.device)
                for target in self.target_columns:
                    info = self.target_info[target]
                    if info["kind"] in {"binary", "class"}:
                        loss = loss + ce_loss(outputs[target], y_batches[target])
                    else:
                        loss = loss + mse_loss(outputs[target].squeeze(1), y_batches[target])
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
            if epoch == 1 or epoch == self.epochs or epoch % 10 == 0:
                tqdm.write(f"epoch={epoch:03d} loss={total_loss / max(len(loader), 1):.5f}")

    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Предсказание модели общего представления для строк таблицы.

        Параметры:
            df: Таблица с колонками признаков обстоятельств.

        Возвращает:
            Словарь имен целей с массивами предсказаний.

        Исключения:
            RuntimeError: Модель еще не обучена.
        """
        if self.model is None or self.mean is None or self.scale is None:
            raise RuntimeError("MLP model is not fitted")
        X = standardize_apply(numeric_matrix(df, self.feature_columns), self.mean, self.scale)
        loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=self.batch_size)
        buffers: Dict[str, List[np.ndarray]] = {target: [] for target in self.target_columns}
        self.model.eval()
        with torch.no_grad():
            for (x_batch,) in loader:
                outputs = self.model(x_batch.to(self.device))
                for target in self.target_columns:
                    info = self.target_info[target]
                    raw = outputs[target]
                    if info["kind"] in {"binary", "class"}:
                        idx = torch.argmax(raw, dim=1).cpu().numpy()
                        classes = np.asarray(info["classes"], dtype=np.int64)
                        buffers[target].append(classes[idx])
                    else:
                        pred = raw.squeeze(1).cpu().numpy().astype(np.float32) * info["scale"] + info["mean"]
                        if info["log_transform"]:
                            pred = np.expm1(pred)
                        buffers[target].append(np.clip(pred, 0.0, None))
        return {target: np.concatenate(parts, axis=0) for target, parts in buffers.items()}


class DecisionModelSystem:
    """Оркестратор обучения, оценки, сохранения и предсказания."""

    def __init__(self, seed: int = 42):
        """Инициализация системы моделей судебных решений.

        Параметры:
            seed: Зерно случайности для используемых семейств моделей.
        """
        self.seed = seed
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        self.method = "independent"
        self.balance = None
        self.bundle = None
        self.report: Dict = {}

    def _make_bundle(self, method: str, epochs: int) -> object:
        """Фабрика семейства моделей для выбранного метода.

        Параметры:
            method: Название метода: ``independent``, ``chain``, ``graph``,
                ``mlp`` или ``nlp``.
            epochs: Количество эпох, передаваемое в режим общего представления.

        Возвращает:
            Объект модели, реализующий методы ``fit`` и ``predict``.

        Исключения:
            ValueError: Указано неизвестное имя метода.
        """
        if method in {"independent", "chain", "graph"}:
            return GraphDecisionModel(seed=self.seed, method=method, graph_width=4)
        if method == "nlp":
            return NlpDecisionModel(seed=self.seed)
        if method == "mlp":
            return MlpDecisionModel(seed=self.seed, epochs=epochs)
        raise ValueError("method must be one of: independent, chain, graph, mlp, nlp")

    def fit(
        self,
        df: pd.DataFrame,
        out_dir: str,
        method: str,
        balance: str,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        epochs: int = 35,
    ) -> Dict:
        """Полный цикл обучения, оценки, формирования отчета и сохранения.

        Параметры:
            df: Полная таблица с признаками и целями.
            out_dir: Каталог артефактов модели и отчетов.
            method: Название семейства моделей.
            balance: Название режима балансировки.
            train_ratio: Доля строк для обучающей выборки.
            val_ratio: Доля строк для валидационной выборки.
            test_ratio: Доля строк для тестовой выборки.
            epochs: Количество эпох для режима общего представления.

        Возвращает:
            Словарь отчета с размерами выборок, метриками и конфигурацией.

        Исключения:
            ValueError: Целевые колонки или обучающие строки отсутствуют.
        """
        set_seed(self.seed)
        if (
            not set(REGULATORY_FEATURE_COLUMNS).issubset(df.columns)
            or df.attrs.get("regulation_version") != REGULATION_VERSION
        ):
            df = enrich_regulatory_features(df)
        self.method = method
        self.balance = balance
        self.feature_columns = feature_columns_from_df(df)
        self.target_columns = target_columns_from_df(df)
        if not self.target_columns:
            raise ValueError("Не найдены целевые d_* колонки в vectors.csv.")
        keys = make_strata_key(df)
        train_idx, val_idx, test_idx = stratified_split_by_key(keys, train_ratio, val_ratio, test_ratio, self.seed)
        if len(train_idx) == 0:
            raise ValueError("Пустой train split.")
        self.bundle = self._make_bundle(method, epochs)
        start = time.time()
        self.bundle.fit(df, self.feature_columns, self.target_columns, train_idx, balance)
        val_df = df.iloc[val_idx] if len(val_idx) else df.iloc[train_idx]
        test_df = df.iloc[test_idx] if len(test_idx) else df.iloc[train_idx]
        val_metrics = self.evaluate(val_df, "val")
        test_metrics = self.evaluate(test_df, "test")
        self.report = {
            "seed": self.seed,
            "method": method,
            "balance": balance,
            "metric_rationale": (
                "decision_score averages macro-F1 for binary/class targets and normalized-MAE scores "
                "for regression targets; macro-F1 is preferred under article/outcome imbalance."
            ),
            "regulation": {
                "source": REGULATION_SOURCE,
                "version": REGULATION_VERSION,
                "feature_count": len(REGULATORY_FEATURE_COLUMNS),
                "postprocessing": "statutory availability and min/max sanction constraints",
            },
            "feature_count": len(self.feature_columns),
            "target_columns": self.target_columns,
            "rows": {"all": int(len(df)), "train": int(len(train_idx)), "val": int(len(val_df)), "test": int(len(test_df))},
            "article_distribution": article_distribution(df),
            "elapsed_seconds": round(time.time() - start, 3),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
        self.save(out_dir)
        return self.report

    def evaluate(self, df: pd.DataFrame, split_name: str) -> Dict:
        """Отчет оценки для одного разбиения таблицы.

        Параметры:
            df: Таблица разбиения с признаками и целями.
            split_name: Человекочитаемое имя разбиения для отчета.

        Возвращает:
            Словарь метрик с общими, групповыми и целевыми оценками.
        """
        if (
            not set(REGULATORY_FEATURE_COLUMNS).issubset(df.columns)
            or df.attrs.get("regulation_version") != REGULATION_VERSION
        ):
            df = enrich_regulatory_features(df)
        pred = self.bundle.predict(df)
        constrained_pred, legal_constraint_report = constrain_prediction_arrays(pred, df)
        by_target: Dict[str, Dict] = {}
        raw_scores_by_target: Dict[str, float] = {}
        scores: List[float] = []
        constrained_scores: List[float] = []
        for target in self.target_columns:
            kind = TARGET_KIND[target]
            y_true = to_target_array(df[target], kind)
            y_pred = pred[target]
            constrained_scores.append(target_metric_score(kind, y_true, constrained_pred[target]))
            if kind in {"binary", "class"}:
                y_true_i = y_true.astype(np.int64)
                y_pred_i = np.asarray(y_pred, dtype=np.int64)
                metric = macro_f1_score_np(y_true_i, y_pred_i)
                by_target[target] = {
                    "acc": round(accuracy_score_np(y_true_i, y_pred_i), 6),
                    "macro_f1": round(metric, 6),
                    "support": int(len(y_true_i)),
                    "positive_rate": round(float((y_true_i == 1).mean()), 6) if kind == "binary" else None,
                }
                scores.append(metric)
                raw_scores_by_target[target] = metric
            else:
                y_true_f = y_true.astype(np.float32)
                y_pred_f = np.asarray(y_pred, dtype=np.float32)
                metric = regression_score(y_true_f, y_pred_f)
                by_target[target] = {
                    "mae": round(mae_np(y_true_f, y_pred_f), 6),
                    "rmse": round(rmse_np(y_true_f, y_pred_f), 6),
                    "normalized_mae_score": round(metric, 6),
                    "support": int(len(y_true_f)),
                }
                scores.append(metric)
                raw_scores_by_target[target] = metric

        article_scores: Dict[str, Dict] = {}
        balanced_article_score: Optional[float] = None
        article_score_gap: Optional[float] = None
        if "c_article_code" in df.columns:
            articles = pd.to_numeric(df["c_article_code"], errors="coerce").fillna(-1).astype(int).to_numpy()
            article_decision_scores: List[float] = []
            for article in sorted(np.unique(articles).tolist()):
                mask = articles == article
                if int(mask.sum()) == 0:
                    continue
                per_target_scores: Dict[str, float] = {}
                for target in self.target_columns:
                    kind = TARGET_KIND[target]
                    y_true_article = to_target_array(df[target], kind)[mask]
                    y_pred_article = np.asarray(pred[target])[mask]
                    per_target_scores[target] = target_metric_score(kind, y_true_article, y_pred_article)
                decision_score = float(np.mean(list(per_target_scores.values()))) if per_target_scores else 0.0
                article_decision_scores.append(decision_score)
                article_scores[str(article)] = {
                    "support": int(mask.sum()),
                    "decision_score": round(decision_score, 6),
                    "substantive_score": average_named_scores(per_target_scores, SUBSTANTIVE_TARGETS),
                    "empty_prone_score": average_named_scores(per_target_scores, EMPTY_PRONE_TARGETS),
                }
            if article_decision_scores:
                balanced_article_score = round(float(np.mean(article_decision_scores)), 6)
                article_score_gap = round(float(max(article_decision_scores) - min(article_decision_scores)), 6)

        return {
            "split": split_name,
            "decision_score": round(float(np.mean(scores)), 6),
            "legally_constrained_decision_score": round(float(np.mean(constrained_scores)), 6),
            "substantive_score": average_named_scores(raw_scores_by_target, SUBSTANTIVE_TARGETS),
            "empty_prone_score": average_named_scores(raw_scores_by_target, EMPTY_PRONE_TARGETS),
            "balanced_article_score": balanced_article_score,
            "article_score_gap": article_score_gap,
            "by_article": article_scores,
            "by_target": by_target,
            "legal_constraint_report": legal_constraint_report,
        }

    def predict_row(self, row: Dict) -> Dict:
        """Предсказание судебного решения для одной строки признаков.

        Параметры:
            row: Словарь значений признаков обстоятельств.

        Возвращает:
            Словарь имен целей со скалярными предсказаниями.
        """
        return self.predict_row_details(row)["constrained_prediction"]

    def predict_row_details(self, row: Dict) -> Dict[str, object]:
        """Предсказание с правовым контекстом и журналом ограничений.

        Параметры:
            row: Словарь значений признаков обстоятельств.

        Возвращает:
            Сырые и скорректированные цели, регуляторный контекст и журнал изменений.
        """
        complete_row = {feature: 0 for feature in self.feature_columns}
        complete_row.update(row)
        frame = enrich_regulatory_features(pd.DataFrame([complete_row]))
        pred = self.bundle.predict(frame)
        out: Dict[str, object] = {}
        for target in self.target_columns:
            value = pred[target][0]
            out[target] = int(value) if TARGET_KIND[target] in {"binary", "class"} else float(value)
        constrained, audit, context = constrain_prediction(out, frame.iloc[0].to_dict())
        return {
            "raw_prediction": out,
            "constrained_prediction": constrained,
            "legal_context": context,
            "constraint_audit": audit,
            "input_features": frame.iloc[0].to_dict(),
        }

    def save(self, out_dir: str) -> None:
        """Сохранение артефактов обученной модели и метаданных.

        Параметры:
            out_dir: Каталог для записи модели, схемы признаков, схемы целей
                и файлов отчета.

        Исключения:
            RuntimeError: Метод вызван до обучения объекта модели.
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "decision_model.pkl", "wb") as fh:
            pickle.dump(self.bundle, fh)
        (out_path / "feature_columns.json").write_text(json.dumps(self.feature_columns, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_path / "target_columns.json").write_text(json.dumps(self.target_columns, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_path / "report.json").write_text(json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_path / "regulatory_features.json").write_text(
            json.dumps(
                {
                    "source": REGULATION_SOURCE,
                    "version": REGULATION_VERSION,
                    "feature_columns": list(REGULATORY_FEATURE_COLUMNS),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, checkpoint_path: str) -> "DecisionModelSystem":
        """Загрузка артефактов из контрольного каталога или pickle-файла.

        Параметры:
            checkpoint_path: Каталог с ``decision_model.pkl`` или прямой путь
                к pickle-файлу.

        Возвращает:
            Систему моделей судебных решений с загруженной моделью и метаданными.
        """
        path = Path(checkpoint_path)
        bundle_file = path if path.suffix == ".pkl" else path / "decision_model.pkl"
        instance = cls()
        with open(bundle_file, "rb") as fh:
            instance.bundle = pickle.load(fh)
        folder = bundle_file.parent
        if (folder / "feature_columns.json").exists():
            instance.feature_columns = json.loads((folder / "feature_columns.json").read_text(encoding="utf-8"))
        if (folder / "target_columns.json").exists():
            saved_targets = json.loads((folder / "target_columns.json").read_text(encoding="utf-8"))
            instance.target_columns = [target for target in saved_targets if target in TARGET_COLUMNS]
        if (folder / "report.json").exists():
            instance.report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            instance.method = instance.report.get("method", instance.method)
            instance.balance = instance.report.get("balance", instance.balance)
        if instance.target_columns:
            restrict_bundle_targets(instance.bundle, instance.target_columns)
        return instance


def article_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Распределение количества строк по статьям в таблице.

    Параметры:
        df: Исходная таблица с необязательной колонкой ``c_article_code``.

    Возвращает:
        Словарь кодов статей и количества строк.
    """
    if "c_article_code" not in df.columns:
        return {}
    values = pd.to_numeric(df["c_article_code"], errors="coerce").fillna(-1).astype(int)
    return {str(k): int(v) for k, v in values.value_counts().sort_index().items()}


def load_training_data(paths: Sequence[str]) -> pd.DataFrame:
    """Загрузка и объединение обучающих данных из CSV-файлов или папок.

    Параметры:
        paths: Пути к файлам или каталогам, содержащим ``vectors.csv``.

    Возвращает:
        Объединенную таблицу со всеми переданными строками.

    Исключения:
        FileNotFoundError: Переданный файл или CSV из переданной папки отсутствует.
    """
    frames = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            path = path / "vectors.csv"
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    return enrich_regulatory_features(df)


def format_report(report: Dict) -> str:
    """Форматирование JSON-отчета с читаемым Unicode-выводом.

    Параметры:
        report: Словарь отчета.

    Возвращает:
        Отформатированную JSON-строку.
    """
    return json.dumps(report, ensure_ascii=False, indent=2)


def parse_manual_value(value: str) -> object:
    """Разбор ручного значения командной строки для входа режима verdict.

    Параметры:
        value: Сырая строка значения из командной строки.

    Возвращает:
        Распознанное булево-подобное целое, целое число, число с плавающей
        точкой, строку или нулевое значение по умолчанию.
    """
    value = value.strip()
    if value == "":
        return 0
    lowered = value.lower()
    if lowered in {"true", "yes", "y", "да"}:
        return 1
    if lowered in {"false", "no", "n", "нет"}:
        return 0
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_set_args(items: Optional[Sequence[str]]) -> Dict[str, object]:
    """Разбор присваиваний командной строки из элементов ``--set``.

    Параметры:
        items: Последовательность строк в формате ``KEY=VALUE``.

    Возвращает:
        Словарь имен признаков и разобранных значений.

    Исключения:
        ValueError: Присваивание имеет неверный формат.
    """
    values: Dict[str, object] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE for --set, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in --set argument: {item}")
        values[key] = parse_manual_value(value)
    return values


def load_defaults_row(args: argparse.Namespace) -> Dict[str, object]:
    """Загрузка строки значений по умолчанию для режима verdict.

    Параметры:
        args: Разобранные аргументы режима verdict.

    Возвращает:
        Словарь значений признаков по умолчанию из JSON, CSV или пустой
        словарь при отсутствии источника значений.

    Исключения:
        ValueError: CSV со значениями по умолчанию не содержит строк.
    """
    if args.defaults_json:
        return json.loads(Path(args.defaults_json).read_text(encoding="utf-8"))
    if args.defaults_csv:
        df = pd.read_csv(args.defaults_csv)
        if df.empty:
            raise ValueError(f"defaults CSV is empty: {args.defaults_csv}")
        return df.iloc[int(args.defaults_row)].to_dict()
    return {}


def build_manual_row(system: DecisionModelSystem, args: argparse.Namespace) -> Dict[str, object]:
    """Сборка ручной строки предсказания для режима verdict.

    Параметры:
        system: Загруженная система моделей со схемой признаков.
        args: Разобранные аргументы режима verdict.

    Возвращает:
        Строку признаков, собранную из нулей, необязательных значений по
        умолчанию, явных присваиваний и необязательного интерактивного ввода.
    """
    row: Dict[str, object] = {feature: 0 for feature in system.feature_columns}
    row.update(load_defaults_row(args))
    row.update(parse_set_args(args.set))

    if args.interactive:
        print("Enter feature values. Press Enter to keep the current value.")
        for feature in system.feature_columns:
            if feature in REGULATORY_FEATURE_COLUMNS:
                continue
            current = row.get(feature, 0)
            raw = input(f"{feature} [{current}]: ").strip()
            if raw:
                row[feature] = parse_manual_value(raw)
    return row


def format_verdict(prediction: Dict[str, object]) -> Dict[str, object]:
    """Человекочитаемая сводка вердикта из сырых предсказаний целей.

    Параметры:
        prediction: Словарь сырых предсказаний, полученный из ``predict_row``.

    Возвращает:
        Сводный словарь с пересчитанными значениями месяцев и булевыми флагами.
    """
    def epochs_to_months(value: object) -> float:
        """Перевод длительности наказания из эпох в месяцы.

        Параметры:
            value: Значение длительности в эпохах.

        Возвращает:
            Значение длительности, пересчитанное в месяцы.
        """
        return round(float(value) * 6.0, 3)

    return {
        "imprisonment_epochs": prediction.get("d_imprisonment_epochs"),
        "imprisonment_months": epochs_to_months(prediction.get("d_imprisonment_epochs", 0.0)),
        "license_revocation_epochs": prediction.get("d_license_revocation_epochs"),
        "license_revocation_months": epochs_to_months(prediction.get("d_license_revocation_epochs", 0.0)),
        "mandatory_work_hours": prediction.get("d_mandatory_work_hours"),
        "forced_work_months": prediction.get("d_forced_work_months"),
        "corrective_work_months": prediction.get("d_corrective_work_months"),
        "restriction_freedom_months": prediction.get("d_restriction_freedom_months"),
        "conditional_sentence": bool(int(prediction.get("d_conditional", 0))),
        "fine_rub": round(float(prediction.get("d_fine_rub", 0.0)), 2),
        "travel_ban": bool(int(prediction.get("d_travel_ban", 0))),
        "reconciliation": bool(int(prediction.get("d_reconciliation", 0))),
        "costs_on_state": bool(int(prediction.get("d_costs_on_state", 0))),
        "costs_amount_rub": round(float(prediction.get("d_costs_amount_rub", 0.0)), 2),
        "self_recusal": bool(int(prediction.get("d_self_recusal", 0))),
    }


def numeric_summary(values: pd.Series) -> Dict[str, object]:
    """Числовая сводка распределения для серии pandas.

    Параметры:
        values: Исходные значения, приводимые к числовому виду.

    Возвращает:
        Словарь сводки с объемом, долей ненулевых значений, моментами и квантилями.
    """
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    arr = numeric.to_numpy()
    return {
        "support": int(len(arr)),
        "nonzero_count": int((arr != 0).sum()),
        "nonzero_rate": round(float((arr != 0).mean()), 6) if len(arr) else 0.0,
        "unique_count": int(pd.Series(arr).nunique(dropna=False)),
        "mean": round(float(np.mean(arr)), 6) if len(arr) else 0.0,
        "std": round(float(np.std(arr)), 6) if len(arr) else 0.0,
        "min": round(float(np.min(arr)), 6) if len(arr) else 0.0,
        "p50": round(float(np.quantile(arr, 0.50)), 6) if len(arr) else 0.0,
        "p95": round(float(np.quantile(arr, 0.95)), 6) if len(arr) else 0.0,
        "max": round(float(np.max(arr)), 6) if len(arr) else 0.0,
    }


def split_diagnostics(df: pd.DataFrame, seed: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, object]:
    """Диагностика разбиения по пересечениям и распределению статей.

    Параметры:
        df: Полная таблица.
        seed: Зерно случайности для генерации разбиения.
        train_ratio: Доля обучающего разбиения.
        val_ratio: Доля валидационного разбиения.
        test_ratio: Доля тестового разбиения.

    Возвращает:
        Словарь диагностики с размерами разбиений, пересечениями и счетчиками статей.
    """
    keys = make_strata_key(df)
    train_idx, val_idx, test_idx = stratified_split_by_key(keys, train_ratio, val_ratio, test_ratio, seed)
    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())
    test_set = set(test_idx.tolist())

    def article_counts(indices: np.ndarray) -> Dict[str, int]:
        """Распределение количества строк по статьям для выбранных индексов.

        Параметры:
            indices: Индексы строк в исходной таблице.

        Возвращает:
            Словарь кодов статей и количества выбранных строк.
        """
        if "c_article_code" not in df.columns:
            return {}
        values = pd.to_numeric(df.iloc[indices]["c_article_code"], errors="coerce").fillna(-1).astype(int)
        return {str(k): int(v) for k, v in values.value_counts().sort_index().items()}

    return {
        "split_key": "c_article_code:c_region_code:c_year",
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "sizes": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "overlap_counts": {
            "train_val": int(len(train_set & val_set)),
            "train_test": int(len(train_set & test_set)),
            "val_test": int(len(val_set & test_set)),
        },
        "article_distribution": {
            "all": article_distribution(df),
            "train": article_counts(train_idx),
            "val": article_counts(val_idx),
            "test": article_counts(test_idx),
        },
    }


def balance_diagnostics(df: pd.DataFrame, seed: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, object]:
    """Диагностика режимов балансировки по весам и эффектам повторной выборки.

    Параметры:
        df: Полная таблица.
        seed: Зерно случайности для разбиения и повторной выборки.
        train_ratio: Доля обучающего разбиения.
        val_ratio: Доля валидационного разбиения.
        test_ratio: Доля тестового разбиения.

    Возвращает:
        Словарь диагностики для каждого поддерживаемого режима балансировки.
    """
    keys = make_strata_key(df)
    train_idx, _, _ = stratified_split_by_key(keys, train_ratio, val_ratio, test_ratio, seed)
    articles = pd.to_numeric(df["c_article_code"], errors="coerce").fillna(-1).astype(int).to_numpy() if "c_article_code" in df.columns else np.full(len(df), -1)
    out: Dict[str, object] = {}

    for mode in ["none", "article_weights", "strata_weights", "oversample"]:
        balanced_idx, weights = apply_balance(df, train_idx, mode, seed)
        item: Dict[str, object] = {
            "train_rows_before": int(len(train_idx)),
            "train_rows_after": int(len(balanced_idx)),
            "duplicate_factor": round(float(len(balanced_idx) / max(len(train_idx), 1)), 6),
        }
        if weights is not None:
            item["weight_summary"] = numeric_summary(pd.Series(weights))
            by_article: Dict[str, object] = {}
            train_articles = articles[train_idx]
            for article in sorted(np.unique(train_articles).tolist()):
                mask = train_articles == article
                by_article[str(int(article))] = {
                    "support": int(mask.sum()),
                    "mean_weight": round(float(np.mean(weights[mask])), 6),
                    "min_weight": round(float(np.min(weights[mask])), 6),
                    "max_weight": round(float(np.max(weights[mask])), 6),
                }
            item["by_article"] = by_article
        else:
            balanced_articles = articles[balanced_idx]
            item["by_article"] = {str(int(k)): int(v) for k, v in zip(*np.unique(balanced_articles, return_counts=True))}
        out[mode] = item
    return out


def target_distribution_diagnostics(df: pd.DataFrame) -> Dict[str, object]:
    """Диагностика распределения целей в целом и по статьям.

    Параметры:
        df: Полная таблица с целевыми колонками.

    Возвращает:
        Вложенный словарь с числовыми сводками по каждой цели.
    """
    target_columns = target_columns_from_df(df)
    articles = pd.to_numeric(df["c_article_code"], errors="coerce").fillna(-1).astype(int) if "c_article_code" in df.columns else pd.Series([-1] * len(df))
    out: Dict[str, object] = {}
    for target in target_columns:
        item: Dict[str, object] = {"overall": numeric_summary(df[target])}
        by_article: Dict[str, object] = {}
        for article in sorted(articles.unique().tolist()):
            mask = articles == article
            by_article[str(int(article))] = numeric_summary(df.loc[mask, target])
        item["by_article"] = by_article
        out[target] = item
    return out


def regulatory_diagnostics(df: pd.DataFrame) -> Dict[str, object]:
    """Диагностика регуляторных профилей и юридической допустимости целей.

    Параметры:
        df: Таблица дел с автоматически рассчитанными ``c_reg_*``.

    Возвращает:
        Распределение профилей и число целевых значений вне диапазонов закона.
    """
    enriched = enrich_regulatory_features(df)
    profiles = (
        pd.to_numeric(enriched["c_reg_profile_code"], errors="coerce")
        .fillna(0)
        .astype(int)
        .value_counts()
        .sort_index()
    )
    targets = {
        target: to_target_array(enriched[target], TARGET_KIND[target])
        for target in target_columns_from_df(enriched)
    }
    _, target_constraint_report = constrain_prediction_arrays(targets, enriched)
    return {
        "source": REGULATION_SOURCE,
        "version": REGULATION_VERSION,
        "regulatory_feature_count": len(REGULATORY_FEATURE_COLUMNS),
        "profile_distribution": {str(profile): int(count) for profile, count in profiles.items()},
        "target_legal_constraint_report": target_constraint_report,
    }


def comparison_diagnostics(path: Optional[str]) -> Dict[str, object]:
    """Диагностика таблицы сравнения по изменениям относительно режима без балансировки.

    Параметры:
        path: Необязательный путь к файлу ``comparison.csv``.

    Возвращает:
        Словарь диагностики с изменениями по методам, если передана совместимая таблица.
    """
    if not path:
        return {}
    table = pd.read_csv(path)
    out: Dict[str, object] = {"path": path, "rows": int(len(table))}
    if not {"method", "balance", "test_decision_score", "test_balanced_article_score"}.issubset(table.columns):
        return out
    deltas: List[Dict[str, object]] = []
    for method in sorted(table["method"].dropna().unique().tolist()):
        base = table[(table["method"] == method) & (table["balance"] == "none")]
        if base.empty:
            continue
        base_row = base.iloc[0]
        for _, row in table[table["method"] == method].iterrows():
            if row["balance"] == "none":
                continue
            deltas.append(
                {
                    "method": method,
                    "balance": row["balance"],
                    "delta_test_decision_score_vs_none": round(float(row["test_decision_score"] - base_row["test_decision_score"]), 6),
                    "delta_test_balanced_article_score_vs_none": round(float(row["test_balanced_article_score"] - base_row["test_balanced_article_score"]), 6),
                    "delta_test_article_score_gap_vs_none": round(float(row.get("test_article_score_gap", np.nan) - base_row.get("test_article_score_gap", np.nan)), 6),
                }
            )
    out["balance_deltas_vs_none"] = deltas
    return out


def run_diagnose(args: argparse.Namespace) -> None:
    """Выполнение команды диагностики без обучения модели.

    Параметры:
        args: Разобранные аргументы командной строки для режима диагностики.
    """
    df = load_training_data(args.data)
    diagnostics = {
        "rows": int(len(df)),
        "feature_count": len(feature_columns_from_df(df)),
        "target_columns": target_columns_from_df(df),
        "split": split_diagnostics(df, args.seed, args.train_ratio, args.val_ratio, args.test_ratio),
        "balance": balance_diagnostics(df, args.seed, args.train_ratio, args.val_ratio, args.test_ratio),
        "targets": target_distribution_diagnostics(df),
        "regulation": regulatory_diagnostics(df),
        "comparison": comparison_diagnostics(args.comparison),
    }
    text = json.dumps(diagnostics, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Saved diagnostics: {args.out}")
    else:
        print(text)


def run_train(args: argparse.Namespace) -> None:
    """Выполнение команды обучения для одной конфигурации модели.

    Параметры:
        args: Разобранные аргументы командной строки для режима обучения.
    """
    df = load_training_data(args.data)
    system = DecisionModelSystem(seed=args.seed)
    report = system.fit(
        df=df,
        out_dir=args.out,
        method=args.method,
        balance=args.balance,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        epochs=args.epochs,
    )
    print(format_report(report))


def run_compare(args: argparse.Namespace) -> None:
    """Выполнение команды сравнения по сетке моделей и режимов балансировки.

    Параметры:
        args: Разобранные аргументы командной строки для режима сравнения.
    """
    df = load_training_data(args.data)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    for method in args.methods:
        for balance in args.balances:
            run_out = out_root / f"{method}_{balance}"
            print(f"\n=== method={method} balance={balance} ===")
            system = DecisionModelSystem(seed=args.seed)
            try:
                report = system.fit(
                    df=df,
                    out_dir=str(run_out),
                    method=method,
                    balance=balance,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                    epochs=args.epochs,
                )
                rows.append(
                    {
                        "method": method,
                        "balance": balance,
                        "status": "ok",
                        "val_decision_score": report["val_metrics"]["decision_score"],
                        "test_decision_score": report["test_metrics"]["decision_score"],
                        "val_legally_constrained_decision_score": report["val_metrics"].get("legally_constrained_decision_score"),
                        "test_legally_constrained_decision_score": report["test_metrics"].get("legally_constrained_decision_score"),
                        "test_legal_adjustment_rate": report["test_metrics"].get("legal_constraint_report", {}).get("row_adjustment_rate"),
                        "val_balanced_article_score": report["val_metrics"].get("balanced_article_score"),
                        "test_balanced_article_score": report["test_metrics"].get("balanced_article_score"),
                        "test_article_score_gap": report["test_metrics"].get("article_score_gap"),
                        "test_substantive_score": report["test_metrics"].get("substantive_score"),
                        "test_empty_prone_score": report["test_metrics"].get("empty_prone_score"),
                        "elapsed_seconds": report["elapsed_seconds"],
                        "out_dir": str(run_out),
                        "error": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "method": method,
                        "balance": balance,
                        "status": "failed",
                        "val_decision_score": np.nan,
                        "test_decision_score": np.nan,
                        "val_legally_constrained_decision_score": np.nan,
                        "test_legally_constrained_decision_score": np.nan,
                        "test_legal_adjustment_rate": np.nan,
                        "val_balanced_article_score": np.nan,
                        "test_balanced_article_score": np.nan,
                        "test_article_score_gap": np.nan,
                        "test_substantive_score": np.nan,
                        "test_empty_prone_score": np.nan,
                        "elapsed_seconds": np.nan,
                        "out_dir": str(run_out),
                        "error": str(exc),
                    }
                )
                print(f"FAILED: {exc}")
    summary = pd.DataFrame(rows).sort_values(["status", "val_decision_score", "test_decision_score"], ascending=[True, False, False])
    summary_path = out_root / "comparison.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    (out_root / "comparison.json").write_text(summary.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nSaved comparison: {summary_path}")


def run_predict(args: argparse.Namespace) -> None:
    """Выполнение команды предсказания для JSON- или CSV-входа.

    Параметры:
        args: Разобранные аргументы командной строки для режима предсказания.

    Исключения:
        ValueError: Не передан ни JSON-, ни CSV-вход.
    """
    system = DecisionModelSystem.load(args.checkpoint)
    if args.input_json:
        print(json.dumps(system.predict_row_details(json.loads(args.input_json)), ensure_ascii=False, indent=2))
        return
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        out = [system.predict_row_details(row.to_dict()) for _, row in df.iterrows()]
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return
    raise ValueError("Укажите --input-json или --input-csv.")


def run_verdict(args: argparse.Namespace) -> None:
    """Выполнение команды verdict для вручную собранной строки признаков.

    Параметры:
        args: Разобранные аргументы командной строки для режима verdict.
    """
    system = DecisionModelSystem.load(args.checkpoint)
    row = build_manual_row(system, args)
    details = system.predict_row_details(row)
    payload = {
        "raw_prediction": details["raw_prediction"],
        "constrained_prediction": details["constrained_prediction"],
        "verdict": format_verdict(details["constrained_prediction"]),
        "legal_context": details["legal_context"],
        "constraint_audit": details["constraint_audit"],
    }
    if args.show_input:
        payload["input_features"] = details["input_features"]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def run_regulations(args: argparse.Namespace) -> None:
    """Вывод числового регуляторного профиля без запуска модели.

    Параметры:
        args: Аргументы со статьёй, годом, частью и месяцем события.

    Возвращает:
        Нет.
    """
    article_text = str(args.article).strip().lower().replace(" ", "")
    article_code = {"12.8": 128, "128": 128, "264.1": 2641, "2641": 2641}.get(article_text)
    if article_code is None:
        raise ValueError("--article должен быть 12.8, 128, 264.1 или 2641")
    row = {
        "c_article_code": article_code,
        "c_year": int(args.year),
        "c_article_part": int(args.part),
        "c_month": int(args.month),
    }
    print(json.dumps(regulatory_context(row), ensure_ascii=False, indent=2))


def run_smoke(args: argparse.Namespace) -> None:
    """Выполнение быстрого проверочного теста на синтетическом датасете.

    Параметры:
        args: Разобранные аргументы командной строки для режима быстрого теста.
    """
    set_seed(123)
    rows: List[Dict] = []
    for article_code, count in [(128, 180), (2641, 45)]:
        for i in range(count):
            year = 2020 + (i % 5)
            recid = int(article_code == 2641) + (i % 3 == 0)
            fine = 30000.0 if article_code == 128 else float(150000 + (i % 5) * 10000)
            rows.append(
                {
                    "source_case_id": f"case {article_code}-{i}",
                    "c_article_code": article_code,
                    "c_article_part": 1 + (i % 4),
                    "c_article_year_norm": year,
                    "c_court_type": 4 if i % 9 == 0 else 1,
                    "c_year": year,
                    "c_month": 1 + (i % 12),
                    "c_day_of_week": i % 7,
                    "c_hour": i % 24,
                    "c_is_night": int(i % 24 < 6 or i % 24 >= 22),
                    "c_region_code": 86 if i % 2 else 77,
                    "c_city_code": i % 11,
                    "c_recidivism_count": recid,
                    "c_fine_history": 1 + (i % 3),
                    "c_license_history": 1 + ((i + 1) % 3),
                    "c_has_lawyer": int(article_code == 2641 and i % 2 == 0),
                    "c_is_appeal": int(i % 9 == 0),
                    "c_guilt_admitted": int(i % 4 != 0),
                    "c_sobriety_test": 1 + (i % 3),
                    "c_has_bac_positive": int(i % 3 != 0),
                    "c_has_protocol": int(i % 5 != 0),
                    "c_has_witness_testimony": int(i % 2),
                    "c_has_officer_report": int(i % 4 == 0),
                    "c_has_device_cert": int(i % 6 == 0),
                    "c_witnesses_count": i % 4,
                    "c_vehicle_ok": -1 if i % 7 == 0 else 1,
                    "c_is_work_hours": int(8 <= i % 24 < 18),
                    "c_special_procedure": int(i % 10 == 0),
                    "c_driver_injured": int(i % 19 == 0),
                    "c_driver_health": i % 4,
                    "c_victim_injured": int(i % 17 == 0),
                    "c_victim_health": i % 4,
                    "d_imprisonment_epochs": float(max(0, recid - 1) * (1 + i % 3)),
                    "d_license_revocation_epochs": float(3 + recid + (i % 2)),
                    "d_conditional": int(article_code == 2641 and i % 4 == 0),
                    "d_fine_rub": fine,
                    "d_travel_ban": int(article_code == 2641 and i % 5 == 0),
                    "d_reconciliation": int(i % 13 == 0),
                    "d_costs_on_state": int(i % 11 == 0),
                    "d_costs_amount_rub": float(0 if i % 11 else 2500),
                    "d_self_recusal": int(i % 29 == 0),
                }
            )
    out = Path(args.out)
    if out.exists() and args.clean:
        shutil.rmtree(out)
    df = pd.DataFrame(rows)
    system = DecisionModelSystem(seed=123)
    report = system.fit(df, str(out), method=args.method, balance=args.balance, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, epochs=args.epochs)
    print(format_report(report))
    print(json.dumps(system.predict_row(df.iloc[0].to_dict()), ensure_ascii=False, indent=2))


def interactive_ui() -> None:
    """Консольное взаимодействие для базовой настройки обучения.

    Параметры:
        Нет.

    Возвращает:
        Нет.
    """
    print("СУДЕБНАЯ МОДЕЛЬ РЕШЕНИЙ")
    data = input("Путь к vectors.csv или папке [koap_vectors/overall]: ").strip() or "koap_vectors/overall"
    out = input("Папка артефактов [court_decision_artifacts]: ").strip() or "court_decision_artifacts"
    method = input("Метод [graph|chain|independent|mlp|nlp], default graph: ").strip() or "graph"
    balance = input("Баланс [none|article_weights|strata_weights|oversample], default strata_weights: ").strip() or "strata_weights"
    system = DecisionModelSystem(seed=42)
    report = system.fit(load_training_data([data]), out, method, balance, 0.7, 0.15, 0.15, epochs=35)
    print(format_report(report))


def build_parser() -> argparse.ArgumentParser:
    """Построение парсера командной строки для всех поддерживаемых режимов.

    Параметры:
        Нет.

    Возвращает:
        Настроенный парсер аргументов.
    """
    parser = argparse.ArgumentParser(description="KOAP court decision model research MVP")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train one model")
    train.add_argument("--data", nargs="+", default=["koap_vectors/overall/vectors.csv"], help="One or more vectors.csv files or folders with vectors.csv")
    train.add_argument("--out", default="court_decision_artifacts")
    train.add_argument("--method", choices=["independent", "chain", "graph", "mlp", "nlp"], default="graph")
    train.add_argument("--balance", choices=["none", "article_weights", "strata_weights", "oversample"], default="strata_weights")
    train.add_argument("--train-ratio", type=float, default=0.7)
    train.add_argument("--val-ratio", type=float, default=0.15)
    train.add_argument("--test-ratio", type=float, default=0.15)
    train.add_argument("--epochs", type=int, default=35, help="Only used by mlp")
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(func=run_train)

    compare = sub.add_parser("compare", help="Run architecture/balancing grid")
    compare.add_argument("--data", nargs="+", required=True)
    compare.add_argument("--out", default="court_decision_experiments")
    compare.add_argument("--methods", nargs="+", choices=["independent", "chain", "graph", "mlp", "nlp"], default=["graph", "mlp", "nlp"])
    compare.add_argument("--balances", nargs="+", choices=["none", "article_weights", "strata_weights", "oversample"], default=["none", "article_weights", "strata_weights"])
    compare.add_argument("--train-ratio", type=float, default=0.7)
    compare.add_argument("--val-ratio", type=float, default=0.15)
    compare.add_argument("--test-ratio", type=float, default=0.15)
    compare.add_argument("--epochs", type=int, default=20)
    compare.add_argument("--seed", type=int, default=42)
    compare.set_defaults(func=run_compare)

    diagnose = sub.add_parser("diagnose", help="Compute split, balance and target diagnostics without training")
    diagnose.add_argument("--data", nargs="+", required=True)
    diagnose.add_argument("--out", help="Optional JSON output path")
    diagnose.add_argument("--comparison", help="Optional comparison.csv path for balance deltas")
    diagnose.add_argument("--train-ratio", type=float, default=0.7)
    diagnose.add_argument("--val-ratio", type=float, default=0.15)
    diagnose.add_argument("--test-ratio", type=float, default=0.15)
    diagnose.add_argument("--seed", type=int, default=42)
    diagnose.set_defaults(func=run_diagnose)

    predict = sub.add_parser("predict", help="Predict decisions for JSON/CSV circumstances")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--input-json")
    predict.add_argument("--input-csv")
    predict.set_defaults(func=run_predict)

    verdict = sub.add_parser("verdict", help="Predict one case from manually provided feature values")
    verdict.add_argument("--checkpoint", required=True, help="Folder with decision_model.pkl or direct .pkl path")
    verdict.add_argument("--set", nargs="*", help="Feature assignments like c_year=2024 c_article_code=128")
    verdict.add_argument("--defaults-json", help="JSON file with default feature values")
    verdict.add_argument("--defaults-csv", help="CSV file whose row is used as defaults")
    verdict.add_argument("--defaults-row", type=int, default=0, help="Row index for --defaults-csv")
    verdict.add_argument("--interactive", action="store_true", help="Request all model feature values interactively")
    verdict.add_argument("--show-input", action="store_true", help="Include final input feature row in output")
    verdict.set_defaults(func=run_verdict)

    regulations = sub.add_parser("regulations", help="Show numeric legal parameters for article/year/part")
    regulations.add_argument("--article", required=True, help="12.8, 128, 264.1 or 2641")
    regulations.add_argument("--year", required=True, type=int)
    regulations.add_argument("--part", type=int, default=0)
    regulations.add_argument("--month", type=int, default=0)
    regulations.set_defaults(func=run_regulations)

    smoke = sub.add_parser("smoke", help="Fast synthetic smoke test")
    smoke.add_argument("--method", choices=["independent", "chain", "graph", "mlp", "nlp"], default="graph")
    smoke.add_argument("--balance", choices=["none", "article_weights", "strata_weights", "oversample"], default="strata_weights")
    smoke.add_argument("--epochs", type=int, default=5)
    smoke.add_argument("--out", default="court_decision_smoke")
    smoke.add_argument("--clean", action="store_true")
    smoke.set_defaults(func=run_smoke)

    ui = sub.add_parser("ui", help="Minimal interactive mode")
    ui.set_defaults(func=lambda args: interactive_ui())
    return parser


def main() -> None:
    """Точка входа командной строки.
    """
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
