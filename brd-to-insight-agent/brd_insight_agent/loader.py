"""Read uploaded spreadsheets into tidy DataFrames."""

from __future__ import annotations

import io
import re
from numbers import Integral
from pathlib import PurePath

import pandas as pd
from pandas.api import types as ptypes

from .profiler import _looks_like_key

_NUMBER = r"[+-]?(?:(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
_ISO_DATE = (
    r"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"(?:[T ](?:[01][0-9]|2[0-3]):[0-5][0-9]"
    r"(?::[0-5][0-9](?:\.[0-9]{1,9})?)?"
    r"(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])?)?"
)


def _extension(filename: str) -> str:
    extension = PurePath(filename).suffix.lower()
    if extension not in {".csv", ".xlsx", ".xlsm", ".xls"}:
        raise ValueError("Unsupported file extension; expected .csv, .xlsx, .xlsm or .xls.")
    return extension


def _header_index(header_row: int) -> int:
    if isinstance(header_row, bool) or not isinstance(header_row, Integral) or header_row < 0:
        raise ValueError("header_row must be a non-negative integer (0-based).")
    return int(header_row)


def _excel_file(file_bytes: bytes, extension: str) -> pd.ExcelFile:
    if extension == ".xls":
        # xlrd reads stored values; it does not evaluate formulas or execute VBA.
        return pd.ExcelFile(io.BytesIO(file_bytes), engine="xlrd")
    # pandas' openpyxl reader already sets read_only=True, data_only=True and
    # keep_links=False. Do not duplicate these keywords on older pandas 2.x.
    return pd.ExcelFile(
        io.BytesIO(file_bytes),
        engine="openpyxl",
        engine_kwargs={"keep_vba": False},
    )


def _csv_rows(file_bytes: bytes, header_row: int, nrows: int | None = None) -> pd.DataFrame:
    if not file_bytes.removeprefix(b"\xef\xbb\xbf").strip():
        raise ValueError("CSV file is empty; no header row was found.")
    try:
        # Do not let inference erase leading zeros, textual NA labels or duplicate
        # headers before clean_frame can make a conservative column-wide decision.
        rows = pd.read_csv(
            io.BytesIO(file_bytes),
            header=None,
            skiprows=header_row,
            dtype=object,
            keep_default_na=False,
            nrows=nrows,
        )
    except pd.errors.EmptyDataError as exc:
        if header_row:
            raise ValueError(f"header_row {header_row} is outside the CSV data.") from exc
        raise ValueError("CSV file is empty; no header row was found.") from exc
    if rows.empty and header_row == 0:
        raise ValueError("CSV file is empty; no header row was found.")
    return rows


def _with_header(rows: pd.DataFrame, header_row: int, sheet_name: str) -> pd.DataFrame:
    if rows.empty:
        if header_row == 0:
            # Preserve load_workbook's empty-worksheet behavior.
            return rows.reset_index(drop=True)
        raise ValueError(f"header_row {header_row} is outside sheet {sheet_name!r}.")
    result = rows.iloc[1:].copy()
    result.columns = rows.iloc[0].tolist()
    # Restore native Excel boolean/numeric/datetime dtypes after removing the
    # header, without parsing strings (particularly textual identifiers).
    return result.reset_index(drop=True).infer_objects()


def _excel_sheet(workbook: pd.ExcelFile, sheet_name: str, header_row: int) -> pd.DataFrame:
    if sheet_name not in workbook.sheet_names:
        raise ValueError(f"Unknown sheet {sheet_name!r}.")
    rows = workbook.parse(
        sheet_name=sheet_name,
        header=None,
        skiprows=header_row,
        dtype=object,
        keep_default_na=False,
    )
    return _with_header(rows, header_row, sheet_name)


def workbook_sheet_names(file_bytes: bytes, filename: str) -> list[str]:
    """List sheets without parsing worksheet tables; CSV has one sheet, 'csv'."""
    extension = _extension(filename)
    if extension == ".csv":
        _csv_rows(file_bytes, 0, nrows=1)
        return ["csv"]
    with _excel_file(file_bytes, extension) as workbook:
        return list(workbook.sheet_names)


def detect_header_row(file_bytes: bytes, filename: str, sheet_name: str, limit: int = 20) -> int:
    """First row that looks like a header, skipping logo/title rows above the table."""
    extension = _extension(filename)
    if extension == ".csv":
        rows = _csv_rows(file_bytes, 0, nrows=limit + 5)
    else:
        with _excel_file(file_bytes, extension) as workbook:
            rows = workbook.parse(sheet_name, header=None, nrows=limit + 5, dtype=object)
    if rows.empty:
        return 0
    widest = max(int(row.notna().sum()) for _, row in rows.iterrows())
    for position, (_, row) in enumerate(rows.iterrows()):
        if position > limit:
            break
        labels = row.dropna()
        if len(labels) < max(2, widest - 1):
            continue
        texts = [str(value).strip() for value in labels]
        if any(not text for text in texts) or len(set(texts)) != len(texts):
            continue
        if all(_header_like(text) for text in texts):
            return position
    return 0


def _header_like(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text)) and len(text) <= 80 and not re.fullmatch(r"-?[\d.,%$ ]+", text)


def load_sheet(
    file_bytes: bytes, filename: str, sheet_name: str, header_row: int = 0
) -> pd.DataFrame:
    """Load only one sheet with its own header, retaining original cell values.

    Apply clean_frame separately for blank/header normalization and conservative
    type inference. Excel formulas use cached values only; VBA is never executed.
    """
    extension = _extension(filename)
    header_row = _header_index(header_row)
    if extension == ".csv":
        if sheet_name != "csv":
            raise ValueError(f"Unknown sheet {sheet_name!r}; CSV has only the 'csv' sheet.")
        return _with_header(_csv_rows(file_bytes, header_row), header_row, "csv")
    with _excel_file(file_bytes, extension) as workbook:
        return _excel_sheet(workbook, sheet_name, header_row)


def load_workbook(file_bytes: bytes, filename: str, header_row: int = 0) -> dict[str, pd.DataFrame]:
    """Return {sheet_name: DataFrame}, using header_row for every sheet as before.

    Use workbook_sheet_names and load_sheet when sheets have different headers.
    """
    extension = _extension(filename)
    header_row = _header_index(header_row)
    if extension == ".csv":
        return {"csv": load_sheet(file_bytes, filename, "csv", header_row)}
    with _excel_file(file_bytes, extension) as workbook:
        return {
            name: _excel_sheet(workbook, name, header_row)
            for name in workbook.sheet_names
        }


def _normalized_headers(columns: pd.Index) -> list[str]:
    labels = []
    for i, col in enumerate(columns):
        label = "" if ptypes.is_scalar(col) and pd.isna(col) else str(col).strip()
        if not label or label.lower().startswith("unnamed:"):
            label = f"column_{i + 1}"
        labels.append(label)

    # Reserve original names so ['a', 'a', 'a_1'] becomes ['a', 'a_2', 'a_1'],
    # rather than generating a name that collides with a later original column.
    reserved = set(labels)
    used: set[str] = set()
    counters: dict[str, int] = {}
    result = []
    for label in labels:
        candidate = label
        suffix = counters.get(label, 1)
        while candidate in used:
            candidate = f"{label}_{suffix}"
            suffix += 1
            while candidate in reserved:
                candidate = f"{label}_{suffix}"
                suffix += 1
        counters[label] = suffix
        used.add(candidate)
        result.append(candidate)
    return result


def _normalize_blanks(series: pd.Series) -> pd.Series:
    blank = series.astype("string").str.fullmatch(r"\s*", na=False)
    return series.mask(series.isna() | blank, pd.NA)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize blanks/headers and coerce only wholly unambiguous columns.

    Every non-empty row is retained. Preview limits belong to presentation code.
    Nonblank values are left untouched if any value prevents safe coercion.
    """
    out = df.copy()
    out.columns = _normalized_headers(out.columns)

    for col in out.columns:
        dtype = out[col].dtype
        if ptypes.is_object_dtype(dtype) or ptypes.is_string_dtype(dtype):
            out[col] = _maybe_coerce(out[col])
        elif isinstance(dtype, pd.CategoricalDtype):
            out[col] = _normalize_blanks(out[col])

    out = out.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return out.reset_index(drop=True)


def _maybe_coerce(series: pd.Series) -> pd.Series:
    series = _normalize_blanks(series)
    text = series.astype("string").str.strip()
    nonblank = text.dropna()
    if nonblank.empty or _looks_like_key(str(series.name)):
        return series

    percent = nonblank.str.endswith("%")
    currency = nonblank.str.contains(r"[$€£]", regex=True)
    if percent.any():
        # A bare 0.5 next to '50%' has no reliable shared unit.
        if not percent.all():
            return series
        numbers = text.str.slice(stop=-1).str.strip()
    elif currency.any():
        symbols = nonblank.str.extract(r"^[+-]?([$€£])", expand=False)
        if symbols.isna().any() or symbols.nunique() != 1:
            return series
        numbers = text.str.replace(r"^([+-]?)[$€£]\s*", r"\1", regex=True)
    else:
        # Leading zeros are evidence of a code even without an ID-like header.
        if nonblank.str.match(r"[+-]?0[0-9]", na=False).any():
            return series
        numbers = text

    if numbers.dropna().str.fullmatch(_NUMBER, na=False).all():
        numeric = pd.to_numeric(numbers.str.replace(",", "", regex=False), errors="coerce")
        if numeric.notna().sum() == len(nonblank) and not numeric.isin([float("inf"), -float("inf")]).any():
            return numeric / 100 if percent.all() else numeric
        return series

    # Explicit ISO calendar syntax only, not locale-dependent dates, years,
    # compact digit IDs, epoch numbers or arbitrary date-like free text.
    if not nonblank.str.fullmatch(_ISO_DATE, na=False).all():
        return series
    zones = nonblank.str.extract(r"(Z|[+-][0-9]{2}:[0-9]{2})$", expand=False).fillna("")
    if zones.nunique() != 1:
        # Do not guess a timezone for naive values or silently normalize offsets.
        return series

    try:
        dates = pd.to_datetime(text, errors="coerce", format="ISO8601")
    except (ValueError, TypeError):
        return series
    if ptypes.is_datetime64_any_dtype(dates.dtype) and dates.notna().sum() == len(nonblank):
        return dates

    return series


def correct_column(df, column, kind, decimal=".", thousands=",", currency="", percent=False, date_format="%Y-%m-%d"):
    if column not in df.columns or kind not in {"numeric", "datetime", "boolean", "text"}:
        raise ValueError("Select an existing column and supported type.")
    source = df[column]
    present = source.notna()
    text = source.astype("string").str.strip()
    try:
        if kind == "text":
            converted = source.astype("string")
        elif kind == "numeric":
            if decimal not in {".", ","} or thousands not in {"", ".", ",", " "} or decimal == thousands:
                raise ValueError("Decimal and thousands separators must differ.")
            if ptypes.is_datetime64_any_dtype(source) or ptypes.is_bool_dtype(source):
                raise ValueError("Dates and booleans are not implicit measures.")
            if ptypes.is_numeric_dtype(source):
                converted = source.copy()
            else:
                if currency:
                    text = text.str.replace(currency, "", regex=False).str.strip()
                if percent:
                    text = text.str.removesuffix("%").str.strip()
                integer = rf"(?:\d+|\d{{1,3}}(?:{re.escape(thousands)}\d{{3}})+)" if thousands else r"\d+"
                numeric_pattern = rf"[+-]?(?:{integer}(?:{re.escape(decimal)}\d*)?|{re.escape(decimal)}\d+)(?:[eE][+-]?\d+)?"
                if not text[present].str.fullmatch(numeric_pattern, na=False).all():
                    raise ValueError("Values do not match the selected number format.")
                if thousands:
                    text = text.str.replace(thousands, "", regex=False)
                text = text.str.replace(decimal, ".", regex=False)
                converted = pd.to_numeric(text, errors="raise")
            if converted.isin([float("inf"), -float("inf")]).any():
                raise ValueError("Nonfinite values are not allowed.")
            if percent:
                converted = converted / 100
        elif kind == "datetime":
            if not date_format or len(date_format) > 80:
                raise ValueError("Specify an exact date format, such as %d/%m/%Y.")
            converted = (source.copy() if ptypes.is_datetime64_any_dtype(source)
                         else pd.to_datetime(text, format=date_format, errors="raise"))
            if not ptypes.is_datetime64_any_dtype(converted):
                raise ValueError("Use one consistent timezone.")
        else:
            mapping = {"true": True, "yes": True, "1": True, "false": False, "no": False, "0": False}
            if not text[present].str.casefold().isin(mapping).all():
                raise ValueError("Use true/false, yes/no or 1/0 values.")
            converted = text.str.casefold().map(mapping).astype("boolean")
        if not converted.notna().equals(present):
            raise ValueError("Conversion would lose nonblank values.")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{column}: conversion rejected; no values changed. {exc}") from exc
    result = df.copy()
    result[column] = converted
    return result
