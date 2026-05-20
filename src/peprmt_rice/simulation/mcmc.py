"""
MCMC calibration driver.

Wraps :mod:`pymcmcstat` (the same library used by the research codebase)
behind a thin layer that:

1. Reads parameter priors from the YAML pointed to by ``cfg.parameters.path``,
2. Runs the forward model inside the likelihood,
3. Compares simulated fluxes to whatever observation columns the user asked
   for in ``cfg.simulation.target_fluxes``,
4. Returns the chain plus the posterior-mean forward run.

The priors file uses the same nested layout as the forward-mode parameter
file, but each value is a ``[low, high, initial]`` triple instead of a
scalar::

    gpp:
      LUE_max: [0.5, 5.0, 2.5]
      k:       [0.1, 1.5, 0.5]
      ...

Calibration in this initial release is **per-model**: GPP is calibrated
against ``GPP`` (if present) or skipped; Reco/NEE against the chosen
observation columns; CH4 against ``CH4``. The three calibrations are
independent, which matches the structure of the research codebase.

Heavy MCMC machinery is imported lazily so users running forward-only do not
pay the import cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from peprmt_rice.config import RunConfig


# ---------------------------------------------------------------------------
# Prior file
# ---------------------------------------------------------------------------

def _load_priors(cfg: RunConfig) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    if cfg.parameters.source != "file":
        raise ValueError("MCMC mode requires parameters.source='file' with a priors YAML.")
    path = Path(cfg.parameters.path).expanduser()
    if not path.is_absolute() and cfg.source_file is not None:
        path = cfg.source_file.parent / path
    with open(path.resolve()) as f:
        raw = yaml.safe_load(f) or {}
    out: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    for model_key, params in raw.items():
        out[model_key] = {}
        for name, triple in params.items():
            if len(triple) != 3:
                raise ValueError(
                    f"Prior for {model_key}.{name} must be [low, high, initial], "
                    f"got {triple!r}"
                )
            out[model_key][name] = tuple(float(x) for x in triple)
    return out


# ---------------------------------------------------------------------------
# Calibration loop (per model)
# ---------------------------------------------------------------------------

def _make_ssfun(forward_fn, loss_function: str):
    """Build the loss function pymcmcstat will minimise.

    ``loss_function`` selects the residual aggregator:

    * ``"sse"`` — ``nansum((sim - obs)**2)`` (Gaussian likelihood, public default).
    * ``"mae"`` — ``nanmean(|sim - obs|)`` (Laplace likelihood; matches the
      paper's ``PostProcessing.sum_squares_ch4`` in
      ``methane_model/code_scripts/GPP_paper_processing_Tang_v1.py:5507-5541``).
    * ``"rmse"`` — ``sqrt(nanmean((sim - obs)**2))`` (older paper variant).
    """
    def ssfun(theta, data):
        sim = forward_fn(theta)
        residual = sim - data.ydata[0].flatten()
        if loss_function == "sse":
            return float(np.nansum(residual ** 2))
        if loss_function == "mae":
            return float(np.nanmean(np.abs(residual)))
        if loss_function == "rmse":
            return float(np.sqrt(np.nanmean(residual ** 2)))
        raise ValueError(
            f"Unknown loss_function {loss_function!r}; expected one of 'sse', 'mae', 'rmse'."
        )
    return ssfun


def _calibrate_block(
    *,
    forward_fn,
    obs: np.ndarray,
    priors: Dict[str, Tuple[float, float, float]],
    niter: int,
    burnin: int,
    seed: int,
    loss_function: str = "sse",
) -> Dict[str, Any]:
    """Calibrate a single model block.

    ``forward_fn(params_vector) -> simulated`` must return a 1-D numpy array
    of model output aligned to ``obs``.

    ``loss_function`` selects how residuals are aggregated for pymcmcstat;
    see :func:`_make_ssfun` for the supported choices.

    Returns a dict with ``chain``, ``posterior_mean``, ``param_names``,
    ``acceptance_rate``.
    """
    # pymcmcstat 1.9.x (via mcmcplot) does ``from scipy import pi, sin, cos`` —
    # those names were removed from the top-level ``scipy`` namespace in
    # SciPy ≥ 1.14. We restore them on the ``scipy`` module before the
    # import so the package works on modern environments without the user
    # having to know about this. If a future pymcmcstat release fixes it
    # upstream, this shim becomes a harmless no-op.
    import scipy as _scipy  # noqa: PLR0402 (intentional alias for shim)
    if not hasattr(_scipy, "pi"):  # pragma: no cover - depends on SciPy version
        _scipy.pi = np.pi
        _scipy.sin = np.sin
        _scipy.cos = np.cos
        _scipy.exp = np.exp
        _scipy.log = np.log
        _scipy.sqrt = np.sqrt

    try:
        from pymcmcstat.MCMC import MCMC
    except ImportError as e:  # pragma: no cover - import-time error path
        raise ImportError(
            "MCMC mode requires `pymcmcstat`. Install with: pip install pymcmcstat"
        ) from e

    # IMPORTANT: pymcmcstat seeding must go through ``MCMC(rngseed=seed)`` —
    # the previous ``np.random.seed(seed)`` call followed by ``MCMC()`` did
    # NOT produce reproducible chains because ``MCMC.__init__`` internally
    # calls ``np.random.seed(seed=rngseed)`` (defaulting to ``None``), which
    # silently re-seeded the RNG from OS entropy and undid the caller's seed.
    # See pymcmcstat/MCMC.py:68-80.
    names = list(priors.keys())
    ssfun = _make_ssfun(forward_fn, loss_function)

    mcstat = MCMC(rngseed=seed)
    mcstat.data.add_data_set(np.arange(len(obs)), obs.reshape(-1, 1))

    for n in names:
        low, high, init = priors[n]
        mcstat.parameters.add_model_parameter(name=n, theta0=init, minimum=low, maximum=high)

    mcstat.simulation_options.define_simulation_options(
        nsimu=int(niter), updatesigma=True, method="dram",
    )
    mcstat.model_settings.define_model_settings(sos_function=ssfun)
    mcstat.run_simulation()

    results = mcstat.simulation_results.results
    chain = results["chain"]
    posterior_mean = chain[burnin:].mean(axis=0)
    return {
        "chain": chain,
        "posterior_mean": dict(zip(names, posterior_mean.tolist())),
        "param_names": names,
        "acceptance_rate": float(results.get("local_acceptance_rate", np.nan)),
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run_mcmc(df: pd.DataFrame, cfg: RunConfig) -> Dict[str, Any]:
    """Run an MCMC calibration of the model.

    Returns a dict with:
      - ``chains`` per model block,
      - ``parameters`` (posterior means as plain dicts), and
      - ``timeseries`` from a deterministic forward run using the posterior
        means.

    The detailed prior layout is described in the module docstring.
    """
    from peprmt_rice.models import GPPModel, GPPParameters, NEEModel, NEEParameters, CH4Model, CH4Parameters

    priors = _load_priors(cfg)
    sim = cfg.simulation
    out: Dict[str, Any] = {"chains": {}, "parameters": {}}

    # ---- GPP -----------------------------------------------------------------
    # Three cases:
    #   (a) use_observed_gpp=true  → skip the GPP model entirely; feed GPP
    #       to the downstream NEE / CH4 calibrations.
    #   (b) GPP is in target_fluxes and GPP is present → calibrate the
    #       GPP model against the observation, then use its posterior-mean
    #       output downstream.
    #   (c) Otherwise → run the GPP model with default parameters and use
    #       its output downstream.
    if sim.use_observed_gpp:
        if "GPP" not in df.columns:
            raise ValueError(
                "use_observed_gpp=true but the input has no 'GPP' column."
            )
        out["parameters"]["gpp"] = {}  # GPP block not used
        gpp_series = df["GPP"].to_numpy(dtype=float)
    else:
        if "GPP" in sim.target_fluxes and "GPP" in df.columns and "gpp" in priors:
            gpp_priors = priors["gpp"]

            def gpp_forward(theta):
                pars = GPPParameters.from_list(theta)
                gpp, _, _ = GPPModel(pars).estimate(
                    df["TA"].values, df["PAR"].values, df["LAI"].values,
                )
                return gpp

            block = _calibrate_block(
                forward_fn=gpp_forward, obs=df["GPP"].values,
                priors=gpp_priors,
                niter=sim.niter_for("gpp"), burnin=sim.burnin_for("gpp"),
                seed=sim.seed,
                loss_function=sim.loss_function,
            )
            out["chains"]["gpp"] = block["chain"]
            out["parameters"]["gpp"] = block["posterior_mean"]
        else:
            out["parameters"]["gpp"] = GPPParameters.defaults().asdict()

        gpp_params = GPPParameters(**out["parameters"]["gpp"])
        gpp_series, _, _ = GPPModel(gpp_params).estimate(
            df["TA"].values, df["PAR"].values, df["LAI"].values,
        )

    # ---- NEE -----------------------------------------------------------------
    nee_target = "NEE" if "NEE" in sim.target_fluxes and "NEE" in df.columns else None
    reco_target = "Reco" if "Reco" in sim.target_fluxes and "Reco" in df.columns else None
    if (nee_target or reco_target) and "nee" in priors:
        nee_priors = priors["nee"]
        target_col = nee_target or reco_target
        target_key = "NEE" if nee_target else "Reco"

        def nee_forward(theta):
            pars = NEEParameters.from_list(theta)
            res = NEEModel(pars).estimate(
                dates=df["Date"].values, air_temp_C=df["TA"].values,
                water_table_cm=df["WT"].values, season=df["Season"].values,
                harvest_g_m2=df["Harvest"].values, gpp=gpp_series,
            )
            return res[target_key]

        block = _calibrate_block(
            forward_fn=nee_forward, obs=df[target_col].values,
            priors=nee_priors,
            niter=sim.niter_for("nee"), burnin=sim.burnin_for("nee"),
            seed=sim.seed,
            loss_function=sim.loss_function,
        )
        out["chains"]["nee"] = block["chain"]
        out["parameters"]["nee"] = block["posterior_mean"]
    else:
        out["parameters"]["nee"] = NEEParameters.defaults().asdict()

    # ---- CH4 -----------------------------------------------------------------
    if "CH4" in sim.target_fluxes and "CH4" in df.columns and "ch4" in priors:
        ch4_priors = priors["ch4"]
        nee_params = NEEParameters(**out["parameters"]["nee"])
        nee_out = NEEModel(nee_params).estimate(
            dates=df["Date"].values, air_temp_C=df["TA"].values,
            water_table_cm=df["WT"].values, season=df["Season"].values,
            harvest_g_m2=df["Harvest"].values, gpp=gpp_series,
        )

        def ch4_forward(theta):
            pars = CH4Parameters.from_list(theta)
            res = CH4Model(pars).estimate(
                dates=df["Date"].values, air_temp_C=df["TA"].values,
                water_table_cm=df["WT"].values,
                day_of_planting=df["DoP"].iloc[0] if "DoP" in df.columns else df["Date"].iloc[0],
                soc=nee_out["SOC_total"], labile=nee_out["labile_total"],
                gpp=gpp_series, gpp_above=nee_out["GPP_aboveground"],
                gpp_below=nee_out["GPP_belowground"],
            )
            return res["CH4_total"]

        block = _calibrate_block(
            forward_fn=ch4_forward, obs=df["CH4"].values,
            priors=ch4_priors,
            niter=sim.niter_for("ch4"), burnin=sim.burnin_for("ch4"),
            seed=sim.seed,
            loss_function=sim.loss_function,
        )
        out["chains"]["ch4"] = block["chain"]
        out["parameters"]["ch4"] = block["posterior_mean"]
    else:
        out["parameters"]["ch4"] = CH4Parameters.defaults().asdict()

    # ---- Final forward pass with posterior means -----------------------------
    # Build the three parameter dataclasses directly from the in-memory
    # posterior means; running the model in-place avoids a YAML round-trip
    # (and a class of fragile path-resolution bugs).
    import pandas as pd

    nee_params = NEEParameters(**out["parameters"]["nee"])
    ch4_params = CH4Parameters(**out["parameters"]["ch4"])

    if sim.use_observed_gpp:
        gpp_final = df["GPP"].to_numpy(dtype=float)
        import numpy as _np
        f_T_final = _np.full(len(df), _np.nan)
        apar_final = _np.full(len(df), _np.nan)
    else:
        gpp_params = GPPParameters(**out["parameters"]["gpp"])
        gpp_final, f_T_final, apar_final = GPPModel(gpp_params).estimate(
            df["TA"].values, df["PAR"].values, df["LAI"].values,
        )
    nee_final = NEEModel(nee_params).estimate(
        dates=df["Date"].values, air_temp_C=df["TA"].values,
        water_table_cm=df["WT"].values, season=df["Season"].values,
        harvest_g_m2=df["Harvest"].values, gpp=gpp_final,
    )
    ch4_final = CH4Model(ch4_params).estimate(
        dates=df["Date"].values, air_temp_C=df["TA"].values,
        water_table_cm=df["WT"].values,
        day_of_planting=df["DoP"].iloc[0] if "DoP" in df.columns else df["Date"].iloc[0],
        soc=nee_final["SOC_total"], labile=nee_final["labile_total"],
        gpp=gpp_final, gpp_above=nee_final["GPP_aboveground"],
        gpp_below=nee_final["GPP_belowground"],
    )

    # Persist posterior-mean parameters as a YAML next to the run so users can
    # rerun forward mode against them later if they want to.
    out_dir = cfg.output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "posterior_mean_parameters.yaml", "w") as f:
        yaml.safe_dump(out["parameters"], f, sort_keys=False)

    out["timeseries"] = pd.DataFrame({
        "Date": df["Date"].values,
        "TA": df["TA"].values,
        "WT": df["WT"].values,
        "PAR": df["PAR"].values if "PAR" in df.columns else gpp_final * float("nan"),
        "LAI": df["LAI"].values if "LAI" in df.columns else gpp_final * float("nan"),
        "GPP": gpp_final,
        "f_T": f_T_final,
        "APAR": apar_final,
        "Reco": nee_final["Reco"],
        "Reco_soc": nee_final["Reco_soc"],
        "Reco_labile": nee_final["Reco_labile"],
        "Ra_above": nee_final["Ra_above"],
        "Ra_root": nee_final["Ra_root"],
        "NEE": nee_final["NEE"],
        "SOC_left": nee_final["SOC_left"],
        "labile_left": nee_final["labile_left"],
        "GPP_aboveground": nee_final["GPP_aboveground"],
        "GPP_belowground": nee_final["GPP_belowground"],
        "CH4": ch4_final["CH4_total"],
        "CH4_plant": ch4_final["Plant_flux"],
        "CH4_hydro": ch4_final["Hydro_total"],
        "CH4_oxidation": ch4_final["CH4_oxidation_total"],
    })
    return out
