"""Canonical raw price schema and validator.

Every loader and adapter must emit a long-format table that ``validate``
accepts **before** universe selection or feature engineering.

Required columns
----------------
date : datetime64[ns]
tic : str
open, high, low, close : numeric, strictly positive after adapters
volume : numeric, non-negative

Optional columns
----------------
adj_close, market_cap, sector, permno
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "tic",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "adj_close",
    "market_cap",
    "sector",
    "permno",
)
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


@dataclass
class ValidationIssue:
    """A single schema problem. Validators report; they do not fix."""

    code: str
    message: str
    count: int = 1


@dataclass
class ValidationReport:
    """Result of :func:`validate`.

    Attributes
    ----------
    ok
        True iff ``issues`` is empty.
    issues
        Problems found. Empty when the frame is canonical.
    n_rows, n_tickers, n_dates
        Shape summary (0 when the frame is empty).
    """

    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    n_rows: int = 0
    n_tickers: int = 0
    n_dates: int = 0

    def summary(self) -> str:
        """Human-readable report for notebooks."""

        lines = [
            f"ok={self.ok} rows={self.n_rows} tickers={self.n_tickers} dates={self.n_dates}"
        ]
        if not self.issues:
            lines.append("no issues")
        for issue in self.issues:
            lines.append(f"[{issue.code} n={issue.count}] {issue.message}")
        return "\n".join(lines)

    def raise_if_invalid(self) -> None:
        """Raise ``ValueError`` if the frame is not canonical."""

        if not self.ok:
            raise ValueError("canonical schema validation failed:\n" + self.summary())


def validate(df: pd.DataFrame) -> ValidationReport:
    """Check a long-format price table against the canonical schema.

    Parameters
    ----------
    df
        Columns must include :data:`REQUIRED_COLUMNS`. Extra columns are
        allowed. Shape ``(N, >=7)``.

    Returns
    -------
    ValidationReport
        ``ok`` is False when any issue is recorded. Nothing is mutated.
    """

    issues: list[ValidationIssue] = []
    n_rows = int(len(df))
    n_tickers = int(df["tic"].nunique()) if "tic" in df.columns and n_rows else 0
    n_dates = int(df["date"].nunique()) if "date" in df.columns and n_rows else 0

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        issues.append(
            ValidationIssue(
                "missing_columns",
                f"missing required columns: {missing}",
                count=len(missing),
            )
        )
        return ValidationReport(False, issues, n_rows, n_tickers, n_dates)

    if n_rows == 0:
        issues.append(ValidationIssue("empty", "DataFrame is empty"))
        return ValidationReport(False, issues, 0, 0, 0)

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        issues.append(
            ValidationIssue(
                "date_dtype",
                f"'date' must be datetime64, got {df['date'].dtype}",
            )
        )

    if df["date"].isna().any():
        n = int(df["date"].isna().sum())
        issues.append(ValidationIssue("date_null", "null dates", count=n))

    tic = df["tic"]
    if tic.isna().any():
        n = int(tic.isna().sum())
        issues.append(ValidationIssue("tic_null", "null tickers", count=n))

    dup = df.duplicated(subset=["date", "tic"])
    if dup.any():
        n = int(dup.sum())
        issues.append(
            ValidationIssue(
                "duplicate_date_tic",
                "duplicate (date, tic) rows",
                count=n,
            )
        )

    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        for tic_name, g in df.groupby("tic", sort=False):
            dates = g["date"]
            if not dates.is_monotonic_increasing:
                issues.append(
                    ValidationIssue(
                        "dates_not_monotone",
                        f"dates not monotone increasing for tic={tic_name}",
                    )
                )
                break

    for col in PRICE_COLUMNS:
        s = pd.to_numeric(df[col], errors="coerce")
        bad = (s <= 0) | s.isna()
        if bad.any():
            issues.append(
                ValidationIssue(
                    "non_positive_price",
                    f"non-positive or non-numeric prices in '{col}'",
                    count=int(bad.sum()),
                )
            )

    vol = pd.to_numeric(df["volume"], errors="coerce")
    if (vol < 0).any() or vol.isna().any():
        n = int(((vol < 0) | vol.isna()).sum())
        issues.append(
            ValidationIssue(
                "bad_volume",
                "volume must be numeric and >= 0",
                count=n,
            )
        )

    nan_req = df[list(REQUIRED_COLUMNS)].isna().any(axis=1)
    if nan_req.any():
        issues.append(
            ValidationIssue(
                "required_nan",
                "NaN in required columns",
                count=int(nan_req.sum()),
            )
        )

    return ValidationReport(
        ok=len(issues) == 0,
        issues=issues,
        n_rows=n_rows,
        n_tickers=n_tickers,
        n_dates=n_dates,
    )
