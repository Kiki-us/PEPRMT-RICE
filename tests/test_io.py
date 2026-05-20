"""Tests for input/output handling."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peprmt_rice.io import load_input, validate_input, flip_sign, InputValidationError


EXAMPLE = Path(__file__).resolve().parents[1] / "data" / "example" / "example_input.csv"


def test_example_input_parses():
    df = load_input(EXAMPLE)
    assert len(df) > 0
    assert "Date" in df.columns
    assert np.issubdtype(df["Date"].dtype, np.datetime64)


def test_example_input_passes_validation():
    df = load_input(EXAMPLE)
    warnings = validate_input(df)
    # Warnings are allowed, but no exception should fire.
    assert isinstance(warnings, list)


def test_missing_required_column_raises(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"Date": ["2022-01-01"], "TA": [10.0]}).to_csv(p, index=False)
    df = load_input(p)
    with pytest.raises(InputValidationError):
        validate_input(df)


def test_use_observed_gpp_relaxes_par_lai_vpd(tmp_path):
    """When use_observed_gpp=true and GPP is provided, PAR/LAI/VPD
    may be omitted from the input without triggering a validation error."""
    df = load_input(EXAMPLE)
    minimal = df[["Date", "TA", "WT", "DoP", "Season", "Harvest", "GPP"]].copy()
    p = tmp_path / "no_par.csv"
    minimal.to_csv(p, index=False)
    df2 = load_input(p)
    warnings = validate_input(df2, use_observed_gpp=True)
    assert isinstance(warnings, list)


def test_use_observed_gpp_requires_gpp_column(tmp_path):
    """use_observed_gpp=true without a GPP column must fail."""
    df = load_input(EXAMPLE)
    no_obs = df.drop(columns=["GPP"])
    p = tmp_path / "no_gppobs.csv"
    no_obs.to_csv(p, index=False)
    df2 = load_input(p)
    with pytest.raises(InputValidationError):
        validate_input(df2, use_observed_gpp=True)


def test_wrong_signed_gpp_triggers_warning(tmp_path):
    """GPP with positive-uptake convention must produce a warning."""
    df = load_input(EXAMPLE).copy()
    df["GPP"] = -df["GPP"]  # flip to positive-uptake on purpose
    p = tmp_path / "wrong_sign.csv"
    df.to_csv(p, index=False)
    df2 = load_input(p)
    warnings = validate_input(df2)
    joined = " ".join(warnings)
    assert "GPP" in joined and "POSITIVE = uptake" in joined


def test_correct_signed_gpp_does_not_trigger_warning():
    """The bundled example uses the model's convention — no sign warning."""
    df = load_input(EXAMPLE)
    warnings = validate_input(df)
    assert not any("POSITIVE = uptake" in w for w in warnings)


def test_flip_sign_helper():
    df = load_input(EXAMPLE)
    flipped = flip_sign(df, ["GPP", "NEE"])
    # Multiplication by -1, element-wise, ignoring NaNs.
    assert np.allclose(flipped["GPP"].dropna(),
                       -df["GPP"].dropna(),
                       equal_nan=False)
    # Untouched columns stay identical.
    assert flipped["TA"].equals(df["TA"])


def test_flip_sign_skips_missing_columns_silently():
    df = load_input(EXAMPLE)
    out = flip_sign(df, ["does_not_exist", "GPP"])
    assert "does_not_exist" not in out.columns
    assert np.allclose(out["GPP"].dropna(), -df["GPP"].dropna())


def test_column_map_renames(tmp_path):
    p = tmp_path / "renamed.csv"
    df = load_input(EXAMPLE)
    df.rename(columns={"TA": "TairAvg"}).to_csv(p, index=False)
    df2 = load_input(p, column_map={"TairAvg": "TA"})
    assert "TA" in df2.columns
    assert "TairAvg" not in df2.columns
