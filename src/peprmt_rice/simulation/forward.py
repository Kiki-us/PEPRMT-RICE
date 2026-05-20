"""
Forward-mode driver: deterministic GPP → NEE → CH4 chain with given
parameters.

Sources of parameters (in priority order):

1. If ``cfg.parameters.source == "file"``, the YAML/CSV at
   ``cfg.parameters.path`` is parsed into ``GPPParameters``,
   ``NEEParameters``, and ``CH4Parameters``.
2. Otherwise the built-in defaults from each model's ``.defaults()`` are used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from peprmt_rice.config import RunConfig
from peprmt_rice.models import (
    GPPModel, GPPParameters,
    NEEModel, NEEParameters,
    CH4Model, CH4Parameters,
)


# ---------------------------------------------------------------------------
# Parameter loading
# ---------------------------------------------------------------------------

def _load_parameter_file(path: Path) -> Dict[str, Dict[str, float]]:
    """Load a parameter YAML or CSV into a nested dict.

    Expected YAML layout::

        gpp:
          LUE_max: 2.5
          k: 0.5
          ...
        nee:
          alpha_soc: 3e3
          ...
        ch4:
          M_alpha1: 0.001
          ...

    CSV layout (long form)::

        model,name,value
        gpp,LUE_max,2.5
        gpp,k,0.5
        ...
    """
    if path.suffix.lower() in (".yaml", ".yml"):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        out: Dict[str, Dict[str, float]] = {}
        for _, row in df.iterrows():
            out.setdefault(row["model"], {})[row["name"]] = float(row["value"])
        return out
    raise ValueError(f"Unsupported parameter file extension: {path.suffix!r}")


def resolve_parameters(cfg: RunConfig) -> tuple[GPPParameters, NEEParameters, CH4Parameters]:
    if cfg.parameters.source == "defaults":
        return GPPParameters.defaults(), NEEParameters.defaults(), CH4Parameters.defaults()

    path = Path(cfg.parameters.path).expanduser()
    if not path.is_absolute() and cfg.source_file is not None:
        path = cfg.source_file.parent / path
    raw = _load_parameter_file(path.resolve())

    gpp = GPPParameters(**raw["gpp"]) if "gpp" in raw else GPPParameters.defaults()
    nee = NEEParameters(**raw["nee"]) if "nee" in raw else NEEParameters.defaults()
    ch4 = CH4Parameters(**raw["ch4"]) if "ch4" in raw else CH4Parameters.defaults()
    return gpp, nee, ch4


# ---------------------------------------------------------------------------
# Forward run
# ---------------------------------------------------------------------------

def run_forward(df: pd.DataFrame, cfg: RunConfig) -> Dict[str, Any]:
    """Run the three-stage forward chain.

    Parameters
    ----------
    df : DataFrame
        Parsed input (output of :func:`peprmt_rice.io.load_input`).
    cfg : RunConfig
        Validated run configuration.

    Returns
    -------
    dict
        Keys:
          - ``timeseries`` (DataFrame): daily GPP, Reco, NEE, CH4, plus the
            phenology partitions and pool diagnostics.
          - ``parameters`` (dict): the parameter set used for the run.
    """
    gpp_params, nee_params, ch4_params = resolve_parameters(cfg)

    # ---- GPP ----------------------------------------------------------------
    # Either run the GPP model (default) or use the user's observed series
    # directly. In the observed-GPP path we still record placeholder f_T /
    # APAR columns of NaN so the output schema is stable across modes.
    if cfg.simulation.use_observed_gpp:
        if "GPP" not in df.columns:
            raise ValueError(
                "use_observed_gpp=true but the input has no 'GPP' column."
            )
        import numpy as np
        gpp = df["GPP"].to_numpy(dtype=float)
        f_T = np.full(len(df), np.nan)
        apar = np.full(len(df), np.nan)
    else:
        gpp_model = GPPModel(gpp_params)
        gpp, f_T, apar = gpp_model.estimate(
            air_temp_C=df["TA"].values,
            par=df["PAR"].values,
            lai=df["LAI"].values,
        )

    # ---- NEE / Reco ---------------------------------------------------------
    nee_model = NEEModel(nee_params)
    nee_out = nee_model.estimate(
        dates=df["Date"].values,
        air_temp_C=df["TA"].values,
        water_table_cm=df["WT"].values,
        season=df["Season"].values,
        harvest_g_m2=df["Harvest"].values,
        gpp=gpp,
    )

    # ---- CH4 ----------------------------------------------------------------
    ch4_model = CH4Model(ch4_params)
    ch4_out = ch4_model.estimate(
        dates=df["Date"].values,
        air_temp_C=df["TA"].values,
        water_table_cm=df["WT"].values,
        day_of_planting=df["DoP"].iloc[0] if "DoP" in df.columns else df["Date"].iloc[0],
        soc=nee_out["SOC_total"],
        labile=nee_out["labile_total"],
        gpp=gpp,
        gpp_above=nee_out["GPP_aboveground"],
        gpp_below=nee_out["GPP_belowground"],
    )

    # ---- Bundle outputs -----------------------------------------------------
    # Echo back whichever drivers we actually have; PAR/LAI may be absent
    # in observed-GPP mode.
    import numpy as np
    nan_col = np.full(len(df), np.nan)
    timeseries = pd.DataFrame({
        "Date": df["Date"].values,
        "TA": df["TA"].values,
        "WT": df["WT"].values,
        "PAR": df["PAR"].values if "PAR" in df.columns else nan_col,
        "LAI": df["LAI"].values if "LAI" in df.columns else nan_col,
        "GPP": gpp,
        "f_T": f_T,
        "APAR": apar,
        "Reco": nee_out["Reco"],
        "Reco_soc": nee_out["Reco_soc"],
        "Reco_labile": nee_out["Reco_labile"],
        "Ra_above": nee_out["Ra_above"],
        "Ra_root": nee_out["Ra_root"],
        "NEE": nee_out["NEE"],
        "SOC_left": nee_out["SOC_left"],
        "labile_left": nee_out["labile_left"],
        "GPP_aboveground": nee_out["GPP_aboveground"],
        "GPP_belowground": nee_out["GPP_belowground"],
        "CH4": ch4_out["CH4_total"],
        "CH4_plant": ch4_out["Plant_flux"],
        "CH4_hydro": ch4_out["Hydro_total"],
        "CH4_oxidation": ch4_out["CH4_oxidation_total"],
    })

    return {
        "timeseries": timeseries,
        "parameters": {
            "gpp": gpp_params.asdict(),
            "nee": nee_params.asdict(),
            "ch4": ch4_params.asdict(),
        },
    }
