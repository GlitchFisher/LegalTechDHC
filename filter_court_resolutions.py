from __future__ import annotations

import argparse
import html
import json
import logging
import os
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.worksheet.worksheet import Worksheet


# Сюда вводится нейминг соответствующих файлов для фильтрации
#FILES_TO_PROCESS: Dict[str, Dict[str, bool]] = {
#    "12.8 (Мировые суды).xlsx": {"filter_document_type": False},
#    ...
#    },
#}

DEFAULT_MISSING_RESOLUTION_POLICY = "skip"
DEFAULT_CONTEXT_WINDOW = 300
DEFAULT_CHUNK_SIZE = 1_000
OPEN_RETRIES = 4
OPEN_RETRY_DELAY_SECONDS = 2.0

LOGGER = logging.getLogger("court_resolution_filter")


def _spaced_word(word: str) -> str:
    """Регулярное выражение для слова с произвольными пробелами между буквами.

    Параметры:
        word: Слово без пробелов.

    Возвращает:
        Фрагмент регулярного выражения.
    """
    separator = r"[\s.\-_]*"
    return separator.join(re.escape(letter) for letter in word)


_MARKER_WORDS = (
    "постановил",
    "постановила",
    "решил",
    "решила",
    "приговорил",
    "приговорила",
    "определил",
    "определила",
)
RESOLUTION_MARKER_PATTERN = re.compile(
    rf"(?<![а-яёa-z])(?:{_spaced_word('суд')}\s+)?"
    rf"(?:{'|'.join(_spaced_word(word) for word in _MARKER_WORDS)})\s*:?",
    re.IGNORECASE,
)

ARTICLE_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "12.8": re.compile(
        r"(?<!\d)12\s*[\.,]\s*8(?!\s*[\.,]\s*\d)",
        re.IGNORECASE,
    ),
    "264.1": re.compile(
        r"(?<!\d)264\s*[\.,\-]\s*1(?!\s*[\.,]\s*\d)",
        re.IGNORECASE,
    ),
}

GUILTY_PATTERN = re.compile(r"\bвиновн[а-яё]*\b", re.IGNORECASE)
NOT_GUILTY_PATTERN = re.compile(
    r"\b(?:невиновн|оправд|реабилит)[а-яё]*\b",
    re.IGNORECASE,
)
TERMINATION_FOR_ABSENCE_PATTERN = re.compile(
    r"(?:"
    r"\bпрекрат[а-яё]*\b.{0,220}\bотсутств[а-яё]*\b.{0,100}\b(?:состав|событ)[а-яё]*\b"
    r"|"
    r"\bотсутств[а-яё]*\b.{0,100}\b(?:состав|событ)[а-яё]*\b.{0,220}\bпрекрат[а-яё]*\b"
    r")",
    re.IGNORECASE,
)

POSTANOVLENIE_TITLE_PATTERN = re.compile(
    r"^(?:(?:апелляционное|кассационное|надзорное)\s+)?постановлени[ея]\b",
    re.IGNORECASE,
)

COLUMN_ALIASES: Mapping[str, frozenset[str]] = {
    "document_type": frozenset(
        {
            "типдокумента",
            "documenttype",
            "doctype",
            "typeofdocument",
        }
    ),
    "case_id": frozenset(
        {
            "caseid",
            "casenumber",
            "номердела",
            "дело",
        }
    ),
    "judgment_content": frozenset(
        {
            "judgmentcontent",
            "content",
            "text",
            "fulltext",
            "текстрешения",
            "текстпостановления",
            "содержаниерешения",
        }
    ),
    "year": frozenset({"year", "год"}),
}


@dataclass(frozen=True)
class SheetSchema:
    """Распознанная структура одного листа Excel."""

    has_header: bool
    document_type_index: Optional[int]
    case_id_index: Optional[int]
    judgment_content_index: int
    year_index: Optional[int]
    document_type_embedded_in_case_id: bool


@dataclass
class FileSummary:
    """Сводные показатели обработки одного исходного файла."""

    input_file: str
    output_file: str
    status: str = "pending"
    sheets: int = 0
    rows_total: int = 0
    rows_after_document_filter: int = 0
    rows_without_judgment_content: int = 0
    rows_without_resolution_marker: int = 0
    rows_with_resolution: int = 0
    rows_matched_12_8: int = 0
    rows_matched_264_1: int = 0
    rows_with_guilty_indicator: int = 0
    rows_with_not_guilty_indicator: int = 0
    rows_written: int = 0
    rows_filtered_out: int = 0
    elapsed_seconds: float = 0.0
    error: str = ""


@dataclass(frozen=True)
class OutcomeMatch:
    """Найденные статьи и типы исходов в резолютивной части."""

    articles: frozenset[str]
    outcomes: frozenset[str]


def normalize_text(value: object) -> str:
    """Нормализация юридического текста для устойчивого поиска.

    Параметры:
        value: Значение ячейки Excel.

    Возвращает:
        Unicode-строка без HTML-тегов, сущностей и лишних пробелов.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    return re.sub(r"\s+", " ", text).strip()


def normalize_column_name(value: object) -> str:
    """Нормализация заголовка колонки для сопоставления с псевдонимами.

    Параметры:
        value: Исходное имя колонки.

    Возвращает:
        Строка в нижнем регистре, содержащая только буквы и цифры.
    """
    text = normalize_text(value).casefold().replace("ё", "е")
    return re.sub(r"[^0-9a-zа-я]+", "", text)


def is_blank_row(row: Sequence[object]) -> bool:
    """Проверка отсутствия содержательных значений в строке.

    Параметры:
        row: Значения ячеек строки.

    Возвращает:
        ``True``, если все значения пустые.
    """
    return all(normalize_text(value) == "" for value in row)


def _looks_like_year(value: object) -> bool:
    """Проверка значения на четырёхзначный календарный год.

    Параметры:
        value: Значение ячейки.

    Возвращает:
        Результат проверки диапазона 1900–2100.
    """
    try:
        year = int(float(value))
    except (TypeError, ValueError):
        return False
    return 1900 <= year <= 2100


def _find_named_columns(row: Sequence[object]) -> Dict[str, int]:
    """Поиск канонических колонок в предполагаемой строке заголовков.

    Параметры:
        row: Первая непустая строка листа.

    Возвращает:
        Словарь канонических имён и индексов.
    """
    found: Dict[str, int] = {}
    for index, value in enumerate(row):
        normalized = normalize_column_name(value)
        for canonical_name, aliases in COLUMN_ALIASES.items():
            if normalized in aliases and canonical_name not in found:
                found[canonical_name] = index
                break
    return found


def detect_sheet_schema(
    first_row: Sequence[object],
    filter_document_type: bool,
) -> SheetSchema:
    """Определение схемы листа с заголовками или без них.

    Параметры:
        first_row: Первая непустая строка листа.
        filter_document_type: Признак группы файлов, требующей фильтрации.

    Возвращает:
        Индексы служебных и содержательных колонок.

    Исключения:
        ValueError: Не удалось определить колонку полного текста.
    """
    named = _find_named_columns(first_row)
    has_header = (
        "judgment_content" in named
        and len({"case_id", "judgment_content", "year"}.intersection(named)) >= 2
    )

    if has_header:
        document_type_index = named.get("document_type")
        if (
            filter_document_type
            and document_type_index is None
            and first_row
            and normalize_text(first_row[0]) == ""
        ):
            document_type_index = 0
        return SheetSchema(
            has_header=True,
            document_type_index=document_type_index,
            case_id_index=named.get("case_id"),
            judgment_content_index=named["judgment_content"],
            year_index=named.get("year"),
            document_type_embedded_in_case_id=False,
        )

    text_lengths = [len(normalize_text(value)) for value in first_row]
    if not text_lengths or max(text_lengths) == 0:
        raise ValueError("Невозможно определить структуру пустой строки.")
    judgment_index = max(range(len(text_lengths)), key=text_lengths.__getitem__)

    year_index = next(
        (index for index, value in enumerate(first_row) if _looks_like_year(value)),
        None,
    )
    document_type_index = 0 if filter_document_type else None
    if judgment_index >= 2:
        case_id_index = 1
    elif judgment_index == 1:
        case_id_index = 0
    else:
        case_id_index = next(
            (
                index
                for index in range(len(first_row))
                if index != judgment_index and index != year_index
            ),
            None,
        )

    return SheetSchema(
        has_header=False,
        document_type_index=document_type_index,
        case_id_index=case_id_index,
        judgment_content_index=judgment_index,
        year_index=year_index,
        document_type_embedded_in_case_id=(
            filter_document_type
            and document_type_index is not None
            and document_type_index == case_id_index
        ),
    )


def is_postanovlenie(value: object, embedded_in_case_id: bool) -> bool:
    """Проверка типа документа на значение постановление.

    Параметры:
        value: Значение отдельной колонки типа или заголовок дела.
        embedded_in_case_id: Признак объединения типа документа и идентификатора.

    Возвращает:
        ``True`` для постановлений, включая апелляционные и кассационные.
    """
    normalized = normalize_text(value).casefold()
    if normalized == "постановление":
        return True
    if embedded_in_case_id:
        return bool(POSTANOVLENIE_TITLE_PATTERN.search(normalized))
    return bool(POSTANOVLENIE_TITLE_PATTERN.fullmatch(normalized))


def extract_resolution(
    judgment_content: object,
    missing_policy: str,
) -> Tuple[Optional[str], bool]:
    """Извлечение резолютивной части судебного акта.

    Параметры:
        judgment_content: Полный текст судебного акта.
        missing_policy: ``skip`` или ``full_text`` при отсутствии маркера.

    Возвращает:
        Пару ``(текст, маркер_найден)``. При политике ``skip`` текст может
        быть ``None``.

    Примечание:
        По умолчанию строка без маркера пропускается. Использование полного
        документа смешивает описание обвинения и итог суда, что существенно
        увеличивает число ложноположительных совпадений.
    """
    normalized = normalize_text(judgment_content)
    if not normalized:
        return None, False
    marker = RESOLUTION_MARKER_PATTERN.search(normalized)
    if marker:
        return normalized[marker.start() :], True
    if missing_policy == "full_text":
        return normalized, False
    return None, False


def find_article_outcomes(
    resolution: str,
    context_window: int,
) -> Optional[OutcomeMatch]:
    """Поиск исхода дела рядом с упоминанием целевой статьи.

    Параметры:
        resolution: Нормализованная резолютивная часть.
        context_window: Число символов слева и справа от статьи.

    Возвращает:
        Найденные статьи и исходы либо ``None``.
    """
    matched_articles: set[str] = set()
    matched_outcomes: set[str] = set()

    for article, pattern in ARTICLE_PATTERNS.items():
        for article_match in pattern.finditer(resolution):
            start = max(0, article_match.start() - context_window)
            end = min(len(resolution), article_match.end() + context_window)
            context = resolution[start:end]

            has_guilty = bool(GUILTY_PATTERN.search(context))
            has_not_guilty = bool(
                NOT_GUILTY_PATTERN.search(context)
                or TERMINATION_FOR_ABSENCE_PATTERN.search(context)
            )
            if not (has_guilty or has_not_guilty):
                continue

            matched_articles.add(article)
            if has_guilty:
                matched_outcomes.add("guilty")
            if has_not_guilty:
                matched_outcomes.add("not_guilty")
            break

    if not matched_articles:
        return None
    return OutcomeMatch(
        articles=frozenset(matched_articles),
        outcomes=frozenset(matched_outcomes),
    )


def sanitize_excel_value(value: object) -> object:
    """Очистка значения от запрещённых XML-символов Excel.

    Параметры:
        value: Исходное значение ячейки.

    Возвращает:
        Значение, допустимое для записи через openpyxl.
    """
    if isinstance(value, str):
        return ILLEGAL_CHARACTERS_RE.sub("", value)
    return value


def _iter_batches(
    rows: Iterable[Sequence[object]],
    chunk_size: int,
) -> Iterator[List[Tuple[object, ...]]]:
    """Разбиение потока строк на ограниченные по размеру пакеты.

    Параметры:
        rows: Поток строк листа.
        chunk_size: Максимальный размер пакета.

    Возвращает:
        Итератор списков строк.
    """
    batch: List[Tuple[object, ...]] = []
    for row in rows:
        row_tuple = tuple(row)
        if is_blank_row(row_tuple):
            continue
        batch.append(row_tuple)
        if len(batch) >= chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch


def process_batch(
    batch: Sequence[Tuple[object, ...]],
    schema: SheetSchema,
    filter_document_type: bool,
    output_sheet: object,
    summary: FileSummary,
    missing_policy: str,
    context_window: int,
) -> None:
    """Фильтрация и запись одного пакета строк.

    Параметры:
        batch: Исходные строки.
        schema: Распознанная структура листа.
        filter_document_type: Необходимость фильтрации постановлений.
        output_sheet: Потоковый лист openpyxl.
        summary: Изменяемая статистика файла.
        missing_policy: Поведение при отсутствии маркера резолюции.
        context_window: Радиус поиска исхода рядом со статьёй.

    Возвращает:
        Нет.
    """
    frame = pd.DataFrame.from_records(batch)
    summary.rows_total += len(frame)

    document_mask = pd.Series(True, index=frame.index, dtype=bool)
    if filter_document_type and schema.document_type_index is not None:
        document_mask = frame.iloc[:, schema.document_type_index].map(
            lambda value: is_postanovlenie(
                value,
                schema.document_type_embedded_in_case_id,
            )
        )

    selected_indices = frame.index[document_mask].tolist()
    summary.rows_after_document_filter += len(selected_indices)

    for index in selected_indices:
        judgment_content = frame.iat[index, schema.judgment_content_index]
        if normalize_text(judgment_content) == "":
            summary.rows_without_judgment_content += 1
            continue

        resolution, marker_found = extract_resolution(
            judgment_content,
            missing_policy,
        )
        if not marker_found:
            summary.rows_without_resolution_marker += 1
        if resolution is None:
            continue
        summary.rows_with_resolution += 1

        match = find_article_outcomes(resolution, context_window)
        if match is None:
            continue

        if "12.8" in match.articles:
            summary.rows_matched_12_8 += 1
        if "264.1" in match.articles:
            summary.rows_matched_264_1 += 1
        if "guilty" in match.outcomes:
            summary.rows_with_guilty_indicator += 1
        if "not_guilty" in match.outcomes:
            summary.rows_with_not_guilty_indicator += 1

        output_sheet.append(
            [sanitize_excel_value(value) for value in batch[index]]
        )
        summary.rows_written += 1


def _open_workbook_with_retry(path: Path):
    """Открытие Excel-книги с повторами при временной блокировке.

    Параметры:
        path: Путь к книге.

    Возвращает:
        Книга openpyxl в потоковом режиме.

    Исключения:
        Exception: Последняя ошибка после исчерпания повторов.
    """
    last_error: Optional[Exception] = None
    for attempt in range(1, OPEN_RETRIES + 1):
        try:
            return load_workbook(
                path,
                read_only=True,
                data_only=False,
                keep_links=False,
            )
        except PermissionError as exc:
            last_error = exc
            if attempt == OPEN_RETRIES:
                break
            LOGGER.warning(
                "Файл занят, повтор %d/%d через %.1f с: %s",
                attempt,
                OPEN_RETRIES,
                OPEN_RETRY_DELAY_SECONDS,
                path,
            )
            time.sleep(OPEN_RETRY_DELAY_SECONDS)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Не удалось открыть файл: {path}")


def _first_nonempty_row(
    rows: Iterator[Tuple[object, ...]],
) -> Tuple[List[Tuple[object, ...]], Optional[Tuple[object, ...]]]:
    """Поиск первой непустой строки с сохранением пустого префикса.

    Параметры:
        rows: Итератор строк листа.

    Возвращает:
        Пустой префикс и первую содержательную строку.
    """
    prefix: List[Tuple[object, ...]] = []
    for row in rows:
        row_tuple = tuple(row)
        if is_blank_row(row_tuple):
            prefix.append(row_tuple)
            continue
        return prefix, row_tuple
    return prefix, None


def process_sheet(
    input_sheet: Worksheet,
    output_sheet: object,
    filter_document_type: bool,
    summary: FileSummary,
    chunk_size: int,
    missing_policy: str,
    context_window: int,
) -> None:
    """Обработка одного листа с сохранением исходной структуры колонок.

    Параметры:
        input_sheet: Потоковый лист исходной книги.
        output_sheet: Потоковый лист результата.
        filter_document_type: Необходимость отбора постановлений.
        summary: Изменяемая статистика файла.
        chunk_size: Размер пакета pandas.
        missing_policy: Поведение при отсутствии резолютивного маркера.
        context_window: Радиус поиска исхода рядом со статьёй.

    Возвращает:
        Нет.
    """
    row_iterator = input_sheet.iter_rows(values_only=True)
    prefix, first_row = _first_nonempty_row(row_iterator)
    for row in prefix:
        output_sheet.append(list(row))
    if first_row is None:
        return

    schema = detect_sheet_schema(first_row, filter_document_type)
    LOGGER.info(
        "Лист %s: header=%s, doc_type=%s, case_id=%s, judgment=%s, year=%s",
        input_sheet.title,
        schema.has_header,
        schema.document_type_index,
        schema.case_id_index,
        schema.judgment_content_index,
        schema.year_index,
    )

    if schema.has_header:
        output_sheet.append(
            [sanitize_excel_value(value) for value in first_row]
        )
        data_rows: Iterable[Sequence[object]] = row_iterator
    else:
        data_rows = _prepend(first_row, row_iterator)

    for batch in _iter_batches(data_rows, chunk_size):
        process_batch(
            batch=batch,
            schema=schema,
            filter_document_type=filter_document_type,
            output_sheet=output_sheet,
            summary=summary,
            missing_policy=missing_policy,
            context_window=context_window,
        )


def _prepend(
    first: Tuple[object, ...],
    rows: Iterator[Tuple[object, ...]],
) -> Iterator[Tuple[object, ...]]:
    """Добавление первой строки обратно к остальному потоку.

    Параметры:
        first: Уже прочитанная первая строка.
        rows: Оставшийся итератор.

    Возвращает:
        Объединённый итератор.
    """
    yield first
    yield from rows


def process_file(
    input_path: Path,
    output_path: Path,
    filter_document_type: bool,
    chunk_size: int,
    missing_policy: str,
    context_window: int,
    overwrite: bool,
) -> FileSummary:
    """Полная обработка одной Excel-книги.

    Параметры:
        input_path: Исходный файл.
        output_path: Итоговый файл.
        filter_document_type: Необходимость отбора постановлений.
        chunk_size: Размер пакета pandas.
        missing_policy: Поведение при отсутствии резолютивного маркера.
        context_window: Радиус поиска исхода рядом со статьёй.
        overwrite: Разрешение перезаписать существующий результат.

    Возвращает:
        Статистика обработки файла.
    """
    started = time.perf_counter()
    summary = FileSummary(
        input_file=str(input_path),
        output_file=str(output_path),
    )
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{uuid4().hex}.tmp.xlsx"
    )
    input_workbook = None

    try:
        if not input_path.exists():
            raise FileNotFoundError(f"Файл не найден: {input_path}")
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Результат уже существует: {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        input_workbook = _open_workbook_with_retry(input_path)
        output_workbook = Workbook(write_only=True)

        for input_sheet in input_workbook.worksheets:
            output_sheet = output_workbook.create_sheet(title=input_sheet.title)
            summary.sheets += 1
            process_sheet(
                input_sheet=input_sheet,
                output_sheet=output_sheet,
                filter_document_type=filter_document_type,
                summary=summary,
                chunk_size=chunk_size,
                missing_policy=missing_policy,
                context_window=context_window,
            )

        output_workbook.save(temporary_path)
        os.replace(temporary_path, output_path)
        summary.status = "ok"
    except Exception as exc:
        summary.status = "error"
        summary.error = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("Ошибка обработки %s", input_path)
        if temporary_path.exists():
            temporary_path.unlink()
    finally:
        if input_workbook is not None:
            input_workbook.close()
        summary.rows_filtered_out = summary.rows_total - summary.rows_written
        summary.elapsed_seconds = round(time.perf_counter() - started, 3)

    return summary


def resolve_input_path(configured_path: str, input_dir: Path) -> Path:
    """Разрешение относительного или абсолютного пути входного файла.

    Параметры:
        configured_path: Путь из конфигурации или командной строки.
        input_dir: Базовая входная директория.

    Возвращает:
        Разрешённый путь к файлу.
    """
    path = Path(configured_path).expanduser()
    if path.is_absolute():
        return path

    candidates = (
        input_dir / path,
        Path(__file__).resolve().parent / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_processing_plan(
    input_dir: Path,
    explicit_files: Optional[Sequence[str]],
) -> List[Tuple[Path, bool]]:
    """Построение списка файлов и режимов фильтрации.

    Параметры:
        input_dir: Директория файлов из конфигурации.
        explicit_files: Необязательные пути из ``--files``.

    Возвращает:
        Список пар ``(путь, фильтровать_тип_документа)``.
    """
    if explicit_files:
        plan = []
        for configured_path in explicit_files:
            path = resolve_input_path(configured_path, input_dir)
            settings = FILES_TO_PROCESS.get(path.name)
            filter_document_type = (
                bool(settings["filter_document_type"])
                if settings is not None
                else not path.name.startswith("12.8")
            )
            plan.append((path, filter_document_type))
        return plan

    return [
        (
            resolve_input_path(configured_path, input_dir),
            bool(settings["filter_document_type"]),
        )
        for configured_path, settings in FILES_TO_PROCESS.items()
    ]


def output_path_for(input_path: Path, output_dir: Path) -> Path:
    """Формирование имени результирующего Excel-файла.

    Параметры:
        input_path: Исходный путь.
        output_dir: Директория результатов.

    Возвращает:
        Путь с суффиксом ``_filtered``.
    """
    return output_dir / f"{input_path.stem}_filtered{input_path.suffix}"


def save_summary(summaries: Sequence[FileSummary], output_dir: Path) -> None:
    """Сохранение общей сводки в CSV и JSON.

    Параметры:
        summaries: Результаты по файлам.
        output_dir: Директория результатов.

    Возвращает:
        Нет.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [asdict(summary) for summary in summaries]
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(
        output_dir / "processing_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "processing_summary.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_summary(summaries: Sequence[FileSummary]) -> None:
    """Вывод краткой табличной сводки в терминал.

    Параметры:
        summaries: Результаты обработки файлов.

    Возвращает:
        Нет.
    """
    frame = pd.DataFrame.from_records([asdict(item) for item in summaries])
    if frame.empty:
        print("Нет файлов для обработки.")
        return
    columns = [
        "input_file",
        "status",
        "rows_total",
        "rows_after_document_filter",
        "rows_without_resolution_marker",
        "rows_written",
        "rows_filtered_out",
        "elapsed_seconds",
        "error",
    ]
    print("\nСводка обработки:")
    print(frame.loc[:, columns].to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    """Создание парсера аргументов командной строки.

    Возвращает:
        Настроенный парсер.
    """
    script_dir = Path(__file__).resolve().parent
    default_input_dir = script_dir.parent / "файлы для обработки"
    parser = argparse.ArgumentParser(
        description=(
            "Фильтрация Excel-файлов по постановлениям, резолютивной части, "
            "статьям 12.8 и 264.1 и индикаторам исхода дела."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help="Базовая директория входных файлов.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "filtered",
        help="Директория итоговых файлов.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Необязательный явный список относительных или абсолютных путей.",
    )
    parser.add_argument(
        "--missing-resolution-policy",
        choices=("skip", "full_text"),
        default=DEFAULT_MISSING_RESOLUTION_POLICY,
        help="Поведение при отсутствии маркера резолютивной части.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW,
        help="Радиус контекста вокруг статьи в символах.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Число строк в одном пакете pandas.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Не перезаписывать существующие результаты.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Остановиться после первой ошибки файла.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main() -> int:
    """Точка входа пакетной обработки Excel-файлов.

    Возвращает:
        Код завершения: ноль при полном успехе, единица при ошибках файлов.
    """
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.context_window <= 0:
        raise ValueError("--context-window должен быть положительным.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size должен быть положительным.")

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    plan = build_processing_plan(input_dir, args.files)
    summaries: List[FileSummary] = []

    for position, (input_path, filter_document_type) in enumerate(plan, start=1):
        output_path = output_path_for(input_path, output_dir)
        LOGGER.info(
            "[%d/%d] %s -> %s",
            position,
            len(plan),
            input_path,
            output_path,
        )
        summary = process_file(
            input_path=input_path,
            output_path=output_path,
            filter_document_type=filter_document_type,
            chunk_size=args.chunk_size,
            missing_policy=args.missing_resolution_policy,
            context_window=args.context_window,
            overwrite=not args.no_overwrite,
        )
        summaries.append(summary)
        LOGGER.info(
            "Статус=%s, исходных=%d, записано=%d, время=%.3f с",
            summary.status,
            summary.rows_total,
            summary.rows_written,
            summary.elapsed_seconds,
        )
        if summary.status == "error" and args.fail_fast:
            break

    save_summary(summaries, output_dir)
    print_summary(summaries)
    return 1 if any(item.status == "error" for item in summaries) else 0

sys.exit(main())
