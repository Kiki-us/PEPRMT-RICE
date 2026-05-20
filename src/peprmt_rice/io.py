"""
Input / output helpers for PEPRMT-Rice.

A single CSV per run is the user-facing input contract. Every row is one
calendar day. The set of required columns depends on whether you're running
a forward simulation (drivers only) or MCMC calibration (drivers + observed
fluxes).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnSpec:
    name: str
    unit: str
    description: str
    required: bool = True


#: Driver columns the model always needs.
#:
#: PAR, LAI, and VPD are only needed when the GPP model is run. If the user
#: supplies ``GPP`` and sets ``simulation.use_observed_gpp: true`` in
#: their config, those three columns are not required.
DRIVER_COLUMNS: List[ColumnSpec] = [
    ColumnSpec("Date", "YYYY-MM-DD", "Observation date (one row per day)."),
    ColumnSpec("TA", "degC", "Daily-mean air temperature."),
    ColumnSpec("PAR", "umol m-2 d-1",
               "Daily sum of photosynthetically active radiation. "
               "Optional when use_observed_gpp=true."),
    ColumnSpec("VPD", "hPa",
               "Daily-mean vapor pressure deficit. "
               "Optional when use_observed_gpp=true."),
    ColumnSpec("LAI", "m2 m-2",
               "Daily leaf area index (gap-filled). "
               "Optional when use_observed_gpp=true."),
    ColumnSpec("WT", "cm", "Water-table height (positive = above soil surface)."),
    ColumnSpec("DoP", "YYYY-MM-DD", "Date of planting for the current season."),
    ColumnSpec("Season", "int", "Phenology stage code (1-6; see docs)."),
    ColumnSpec("Harvest", "g m-2", "Harvested biomass on this date (0 if none)."),
]

#: Columns whose presence depends on ``use_observed_gpp``.
GPP_OPTIONAL_DRIVERS = ("PAR", "VPD", "LAI")

#: Observation columns. Whichever ones the user supplies and lists in
#: ``simulation.target_fluxes`` will be fitted in MCMC mode. ``GPP`` also
#: enables the "skip the GPP model" workflow via ``use_observed_gpp=true``.
OBSERVATION_COLUMNS: List[ColumnSpec] = [
    ColumnSpec("GPP", "g C m-2 d-1",
               "Observed GPP (negative = uptake). When provided and "
               "use_observed_gpp=true, the GPP model is skipped and this "
               "column is fed directly to the NEE and CH4 models.",
               required=False),
    ColumnSpec("NEE", "g C m-2 d-1", "Observed NEE (negative = uptake).", required=False),
    ColumnSpec("Reco", "g C m-2 d-1", "Observed ecosystem respiration.", required=False),
    ColumnSpec("CH4", "g C m-2 d-1", "Observed CH4 flux.", required=False),
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class InputValidationError(Exception):
    """Raised when an input CSV does not match the expected schema."""


def validate_input(
    df: pd.DataFrame,
    *,
    require_observations: Iterable[str] = (),
    use_observed_gpp: bool = False,
) -> List[str]:
    """Validate a parsed input DataFrame and return a list of warning messages.

    Hard errors raise :class:`InputValidationError`.

    Parameters
    ----------
    df : DataFrame
        Already-parsed input (e.g. from :func:`load_input`).
    require_observations : iterable of str
        Names of observation columns that must be present and non-empty
        (e.g. ``["NEE", "CH4"]`` for an MCMC run that targets both).
    use_observed_gpp : bool
        If True, the GPP model is being skipped in favour of an observed
        ``GPP`` column. In that mode PAR / LAI / VPD are no longer
        required, but ``GPP`` must be present and non-empty.
    """
    warnings: List[str] = []
    errors: List[str] = []

    if use_observed_gpp:
        optional_now = set(GPP_OPTIONAL_DRIVERS)
        missing = [c.name for c in DRIVER_COLUMNS
                   if c.required and c.name not in df.columns
                   and c.name not in optional_now]
        if "GPP" not in df.columns:
            errors.append(
                "use_observed_gpp=true requires a 'GPP' column in the input CSV."
            )
        elif df["GPP"].dropna().empty:
            errors.append("'GPP' is present but entirely empty.")
        # Soft warn for any GPP drivers that *are* still present — easy to spot
        # a config/data mismatch.
        stray = [c for c in GPP_OPTIONAL_DRIVERS if c in df.columns]
        if stray:
            warnings.append(
                f"use_observed_gpp=true, so {stray} will be ignored by the GPP "
                "model — they're carried through to outputs but not used."
            )
    else:
        missing = [c.name for c in DRIVER_COLUMNS if c.required and c.name not in df.columns]
    if missing:
        errors.append(f"Missing required driver columns: {missing}")

    for col_name in require_observations:
        if col_name not in df.columns:
            errors.append(f"MCMC target '{col_name}' not present in input.")
        elif df[col_name].dropna().empty:
            errors.append(f"MCMC target '{col_name}' is present but entirely empty.")

    # Date column dtype check
    if "Date" in df.columns:
        if not np.issubdtype(df["Date"].dtype, np.datetime64):
            errors.append("Column 'Date' must parse as a date — got dtype %s" % df["Date"].dtype)

    # Physical range warnings (soft)
    if "TA" in df.columns:
        if (df["TA"] < -40).any() or (df["TA"] > 60).any():
            warnings.append("Column 'TA' has values outside [-40, 60] °C — please double-check.")
    if "LAI" in df.columns and (df["LAI"] < 0).any():
        warnings.append("Column 'LAI' contains negative values — please double-check.")
    if "PAR" in df.columns and (df["PAR"] < 0).any():
        warnings.append("Column 'PAR' contains negative values — please double-check.")

    # ---- Sign-convention sniff test -----------------------------------------
    # PEPRMT-Rice uses NEGATIVE GPP = uptake. Most published GPP products use
    # POSITIVE = uptake — feeding one of those in unchanged would produce
    # garbage outputs silently. Catch the most common mistake loudly.
    warnings.extend(_check_gpp_sign(df))

    # NaN audit — skip columns that are optional in the current mode.
    optional_now = set(GPP_OPTIONAL_DRIVERS) if use_observed_gpp else set()
    nan_cols = [c for c in DRIVER_COLUMNS
                if c.required and c.name in df.columns
                and c.name not in optional_now
                and df[c.name].isna().any()]
    if nan_cols:
        warnings.append(
            "Driver column(s) contain NaNs which will be propagated through the "
            f"simulation: {[c.name for c in nan_cols]}. Please gap-fill before running."
        )

    if errors:
        raise InputValidationError("\n  - ".join(["Input validation failed:"] + errors))

    return warnings


# ---------------------------------------------------------------------------
# Sign-convention check
# ---------------------------------------------------------------------------

#: Columns whose canonical sign convention is "negative = uptake".
_NEG_UPTAKE_COLUMNS = ("GPP", "NEE")


def _check_gpp_sign(df: pd.DataFrame) -> List[str]:
    """Return a list of warnings if a flux column looks wrong-signed.

    PEPRMT-Rice convention: **negative = uptake**, positive = release.
    The most common user mistake is feeding a `GPP` column with the
    opposite (positive = uptake) convention, which silently produces
    nonsense outputs.

    Heuristic: for ``GPP``, almost every value should be ≤ 0 for any
    real photosynthesising ecosystem. If the median is positive *and*
    most values are positive, we flag it.

    For ``NEE`` we are gentler — sites can be net sources or sinks,
    so we only flag when nearly every value is positive *and* the median
    is unusually large.
    """
    out: List[str] = []

    if "GPP" in df.columns:
        s = df["GPP"].dropna()
        if not s.empty:
            frac_pos = float((s > 0).mean())
            med = float(s.median())
            if frac_pos > 0.75 and med > 0:
                out.append(
                    "GPP looks like it uses the POSITIVE = uptake convention "
                    f"({frac_pos:.0%} of values are > 0, median = {med:.2f}). "
                    "PEPRMT-Rice expects NEGATIVE = uptake. "
                    "Multiply the column by -1 before running, or use "
                    "`peprmt-rice flip-sign INPUT OUTPUT --columns GPP` "
                    "to convert."
                )

    if "NEE" in df.columns:
        s = df["NEE"].dropna()
        if not s.empty:
            frac_pos = float((s > 0).mean())
            med = float(s.median())
            if frac_pos > 0.90 and med > 1.0:
                out.append(
                    "NEE may be using POSITIVE = uptake "
                    f"({frac_pos:.0%} > 0, median = {med:.2f}). "
                    "PEPRMT-Rice expects NEGATIVE = uptake — please double-check."
                )

    return out


def flip_sign(
    df: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Return a copy of ``df`` with the listed columns multiplied by -1.

    Used by the ``peprmt-rice flip-sign`` CLI subcommand to convert from
    positive-uptake (common in the literature) to PEPRMT-Rice's
    negative-uptake convention. Columns that are missing from the frame
    are silently skipped.
    """
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = -out[col]
    return out


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def load_input(
    path: str | Path,
    *,
    column_map: Optional[Dict[str, str]] = None,
    date_column: str = "Date",
) -> pd.DataFrame:
    """Read a user CSV and return a DataFrame with canonical column names.

    Parameters
    ----------
    path : str or Path
        Path to the user CSV.
    column_map : dict, optional
        Mapping ``{user_csv_name: canonical_name}``. Anything not listed is
        kept verbatim.
    date_column : str
        Name of the date column **after** renaming.
    """
    path = Path(path).expanduser().resolve()
    df = pd.read_csv(path)

    if column_map:
        df = df.rename(columns=column_map)

    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        if df[date_column].isna().any():
            raise InputValidationError(
                f"Some values in '{date_column}' could not be parsed as dates."
            )
        df = df.sort_values(date_column).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def save_results(
    df: pd.DataFrame,
    out_dir: str | Path,
    name: str,
    *,
    fmt: str = "csv",
) -> Path:
    """Write a DataFrame to disk in the chosen format and return the path.

    Creates the directory if needed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
    elif fmt == "parquet":
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {fmt!r}")
    return path
