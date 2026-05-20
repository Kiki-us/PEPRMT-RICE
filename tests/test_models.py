"""Smoke tests for the three model blocks.

These check that the public API is callable end-to-end with default
parameters and synthetic inputs, and that outputs have the expected shapes
and signs. They are deliberately permissive — physics correctness is the
job of the full validation suite, not these unit tests.
"""

import numpy as np
import pandas as pd

from peprmt_rice.models import (
    GPPModel, GPPParameters,
    NEEModel, NEEParameters,
    CH4Model, CH4Parameters,
)


def _synthetic_inputs(n_days: int = 60):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-05-01", periods=n_days, freq="D")
    return {
        "dates": dates,
        "TA": 15 + 10 * np.sin(np.linspace(0, 2 * np.pi, n_days)) + rng.normal(0, 0.5, n_days),
        "PAR": 4e7 + rng.normal(0, 2e6, n_days),
        "VPD": np.clip(5 + rng.normal(0, 1, n_days), 0.1, None),
        "LAI": np.clip(np.linspace(0.2, 4.0, n_days), 0.1, None),
        "WT": 5 + rng.normal(0, 2, n_days),
        "Season": np.full(n_days, 2),
        "Harvest": np.zeros(n_days),
        "DoP": dates[0],
    }


def test_gpp_runs_and_is_negative():
    x = _synthetic_inputs()
    gpp, f_T, apar = GPPModel(GPPParameters.defaults()).estimate(
        x["TA"], x["PAR"], x["LAI"],
    )
    assert gpp.shape == (60,)
    assert (gpp <= 0).all(), "GPP should be non-positive (sign convention: negative = uptake)"
    assert (apar >= 0).all()


def test_nee_runs():
    x = _synthetic_inputs()
    gpp, _, _ = GPPModel(GPPParameters.defaults()).estimate(x["TA"], x["PAR"], x["LAI"])
    out = NEEModel(NEEParameters.defaults()).estimate(
        dates=x["dates"], air_temp_C=x["TA"], water_table_cm=x["WT"],
        season=x["Season"], harvest_g_m2=x["Harvest"], gpp=gpp,
    )
    for k in ("NEE", "Reco", "SOC_left", "labile_left",
              "GPP_aboveground", "GPP_belowground"):
        assert k in out
        assert out[k].shape == (60,)
    assert (out["Reco"] >= 0).all()
    assert (out["SOC_left"] >= 0).all()


def test_ch4_runs():
    x = _synthetic_inputs()
    gpp, _, _ = GPPModel(GPPParameters.defaults()).estimate(x["TA"], x["PAR"], x["LAI"])
    nee = NEEModel(NEEParameters.defaults()).estimate(
        dates=x["dates"], air_temp_C=x["TA"], water_table_cm=x["WT"],
        season=x["Season"], harvest_g_m2=x["Harvest"], gpp=gpp,
    )
    out = CH4Model(CH4Parameters.defaults()).estimate(
        dates=x["dates"], air_temp_C=x["TA"], water_table_cm=x["WT"],
        day_of_planting=x["DoP"],
        soc=nee["SOC_total"], labile=nee["labile_total"],
        gpp=gpp, gpp_above=nee["GPP_aboveground"],
        gpp_below=nee["GPP_belowground"],
    )
    for k in ("CH4_total", "Plant_flux", "Hydro_total",
              "CH4_oxidation_total"):
        assert k in out
        assert out[k].shape == (60,)


def test_parameter_from_list_round_trip():
    p = GPPParameters.defaults()
    again = GPPParameters.from_list(p.to_array().tolist())
    assert again == p
