"""
Per-run output helpers.

Turn the in-memory output of a forward or MCMC run into the canonical
file layout under ``outputs/<run>/``::

    outputs/<run>/
    ├── timeseries/fluxes.csv         (or .parquet)
    ├── parameters/{gpp,nee,ch4}.csv  (per-block posterior means)
    └── mcmc_chains/{gpp,nee,ch4}.npy (raw chains; MCMC mode only)

Plus per-block ``ch4_posterior_summary.csv`` etc. from ``summarize_chain``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def summarize_chain(chain: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Return a per-parameter summary (mean, std, 95% CI) for an MCMC chain."""
    return pd.DataFrame({
        "parameter": names,
        "mean": chain.mean(axis=0),
        "std": chain.std(axis=0),
        "ci_lo": np.percentile(chain, 2.5, axis=0),
        "ci_hi": np.percentile(chain, 97.5, axis=0),
    })


def write_run_outputs(
    out_dir: Path,
    timeseries: pd.DataFrame,
    parameters: Dict[str, Dict[str, float]],
    chains: Dict[str, np.ndarray] | None = None,
    *,
    fmt: str = "csv",
) -> None:
    """Persist the standard set of run artefacts under ``out_dir``."""
    out_dir = Path(out_dir)
    (out_dir / "timeseries").mkdir(parents=True, exist_ok=True)
    (out_dir / "parameters").mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        timeseries.to_csv(out_dir / "timeseries" / "fluxes.csv", index=False)
    else:  # parquet
        timeseries.to_parquet(out_dir / "timeseries" / "fluxes.parquet", index=False)

    for model_key, pdict in parameters.items():
        pd.DataFrame({"name": list(pdict.keys()), "value": list(pdict.values())}).to_csv(
            out_dir / "parameters" / f"{model_key}.csv", index=False
        )

    if chains:
        (out_dir / "mcmc_chains").mkdir(parents=True, exist_ok=True)
        for model_key, chain in chains.items():
            np.save(out_dir / "mcmc_chains" / f"{model_key}.npy", chain)
            names = list(parameters[model_key].keys())
            summarize_chain(chain, names).to_csv(
                out_dir / "parameters" / f"{model_key}_posterior_summary.csv",
                index=False,
            )
