"""
Scenario comparison for PEPRMT-Rice.

Runs the model against an input CSV under four configurations, computes
performance metrics against the observations, and writes a CSV table plus
comparison figures.

Scenarios
---------
A. **fwd_all_modeled**     – forward run with defaults: GPP→NEE→CH4 chain.
B. **fwd_observed_gpp**    – forward run: skip GPP model, feed ``GPP`` in.
C. **mcmc_all_modeled**    – MCMC-calibrate GPP, NEE, CH4 against obs,
                             then forward-run with posterior means.
D. **mcmc_observed_gpp**   – skip GPP module; MCMC-calibrate NEE + CH4
                             against obs, then forward-run.

A/B use the package's default (paper-calibrated) parameters as-is.
C/D fit parameters specifically to whatever input CSV you point at.

Run as a module
---------------
    python -m peprmt_rice.postprocessing.scenario_comparison

By default this targets the bundled example input at
``data/example/example_input.csv`` and writes to
``outputs/scenario_comparison/``. Use ``--input`` and ``--output`` to
override.

Outputs land in the chosen output dir::

    metrics.csv                per-scenario performance metrics
    fluxes_<scenario>.csv      timeseries from each run (merged with obs)
    timeseries.png             time-series figure (4 fluxes × scenarios)
    scatter_obs_vs_sim.png     obs-vs-sim scatter (4 fluxes × scenarios)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from peprmt_rice.config import (
    InputOptions, OutputOptions, ParameterOptions, RunConfig, SimulationOptions,
)
from peprmt_rice.io import load_input, validate_input
from peprmt_rice.simulation.forward import run_forward
from peprmt_rice.simulation.mcmc import run_mcmc


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Repo root resolved from this file's path:
#   src/peprmt_rice/postprocessing/scenario_comparison.py  →  parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT_CSV = _REPO_ROOT / "data" / "example" / "example_input.csv"
DEFAULT_PRIORS_YAML = _REPO_ROOT / "configs" / "priors.yaml"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "outputs" / "scenario_comparison"

# Per-block MCMC settings.
#
# CH4 has the largest parameter space and the roughest likelihood surface
# (water-table-driven ebullition/runoff terms cause discontinuities), so it
# converges much more slowly than GPP / NEE. We run CH4 ~5× longer.
MCMC_NITER_DEFAULT = 200
MCMC_BURNIN_DEFAULT = 50
MCMC_NITER_PER_BLOCK = {"gpp": 200, "nee": 200, "ch4": 1000}
MCMC_BURNIN_PER_BLOCK = {"gpp": 50,  "nee": 50,  "ch4": 250}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _drop_pair_na(obs: np.ndarray, sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    return obs[mask], sim[mask]


def metrics(obs: np.ndarray, sim: np.ndarray) -> dict:
    """Per-flux performance metrics: RMSE, MAE, bias, R², NSE."""
    obs, sim = _drop_pair_na(obs, sim)
    if obs.size == 0:
        return {"n": 0, "obs_mean": np.nan, "sim_mean": np.nan,
                "rmse": np.nan, "mae": np.nan, "bias": np.nan,
                "r2": np.nan, "nse": np.nan}
    residual = sim - obs
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    bias = float(np.mean(residual))
    if obs.std() > 0 and sim.std() > 0:
        r2 = float(np.corrcoef(obs, sim)[0, 1] ** 2)
    else:
        r2 = np.nan
    denom = float(np.sum((obs - obs.mean()) ** 2))
    nse = float(1 - np.sum(residual ** 2) / denom) if denom > 0 else np.nan
    return {"n": int(obs.size),
            "obs_mean": float(obs.mean()), "sim_mean": float(sim.mean()),
            "rmse": rmse, "mae": mae, "bias": bias, "r2": r2, "nse": nse}


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def make_config(*, scenario_name: str, mode: str, use_observed_gpp: bool,
                input_csv: Path, priors_yaml: Path, output_dir: Path) -> RunConfig:
    if mode == "forward":
        params = ParameterOptions(source="defaults")
    else:
        params = ParameterOptions(source="file", path=str(priors_yaml))
    return RunConfig(
        simulation=SimulationOptions(
            mode=mode, seed=42, use_observed_gpp=use_observed_gpp,
            niter=MCMC_NITER_DEFAULT, burnin=MCMC_BURNIN_DEFAULT,
            niter_per_block=MCMC_NITER_PER_BLOCK,
            burnin_per_block=MCMC_BURNIN_PER_BLOCK,
            target_fluxes=["GPP", "NEE", "CH4"],
        ),
        input=InputOptions(path=str(input_csv), date_column="Date"),
        parameters=params,
        output=OutputOptions(dir=str(output_dir / scenario_name), save_figures=False),
        source_file=Path(__file__).resolve(),
    )


def run_scenario(name: str, *, mode: str, use_observed_gpp: bool,
                 input_csv: Path, priors_yaml: Path,
                 output_dir: Path) -> pd.DataFrame:
    cfg = make_config(scenario_name=name, mode=mode,
                      use_observed_gpp=use_observed_gpp,
                      input_csv=input_csv, priors_yaml=priors_yaml,
                      output_dir=output_dir)
    df = load_input(cfg.input.path)
    validate_input(df, use_observed_gpp=use_observed_gpp)
    if mode == "forward":
        result = run_forward(df, cfg)
    else:
        result = run_mcmc(df, cfg)

    sim = result["timeseries"].copy()
    sim = sim.rename(columns={
        "GPP": "GPP_sim", "NEE": "NEE_sim",
        "Reco": "Reco_sim", "CH4": "CH4_sim",
    })
    obs = df[["Date", "GPP", "NEE", "Reco", "CH4"]].rename(columns={
        "GPP": "GPP_obs", "NEE": "NEE_obs",
        "Reco": "Reco_obs", "CH4": "CH4_obs",
    })
    merged = sim.merge(obs, on="Date", how="inner")
    merged.to_csv(output_dir / f"fluxes_{name}.csv", index=False)
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenarios", default="all",
        help="Comma-separated subset of scenarios to run (or 'all'). "
             "Options: fwd_all_modeled, fwd_observed_gpp, "
             "mcmc_all_modeled, mcmc_observed_gpp.",
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT_CSV,
        help=f"Input CSV path. Default: {DEFAULT_INPUT_CSV}",
    )
    parser.add_argument(
        "--priors", type=Path, default=DEFAULT_PRIORS_YAML,
        help=f"Priors YAML (used in MCMC scenarios). Default: {DEFAULT_PRIORS_YAML}",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--no-summary", action="store_true",
        help="Skip the final metrics-table + figures step.",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Skip running scenarios; just rebuild metrics + figures from any "
             "fluxes_*.csv files already in the output dir.",
    )
    args = parser.parse_args(argv)

    input_csv: Path = args.input.resolve()
    priors_yaml: Path = args.priors.resolve()
    output_dir: Path = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running PEPRMT-Rice scenario comparison\n")
    print(f"  input : {input_csv}")
    print(f"  priors: {priors_yaml}")
    print(f"  output: {output_dir}\n")

    all_scenarios = [
        ("fwd_all_modeled",   "forward", False),
        ("fwd_observed_gpp",  "forward", True),
        ("mcmc_all_modeled",  "mcmc",    False),
        ("mcmc_observed_gpp", "mcmc",    True),
    ]
    if args.scenarios == "all":
        scenarios = all_scenarios
    else:
        wanted = {s.strip() for s in args.scenarios.split(",") if s.strip()}
        scenarios = [s for s in all_scenarios if s[0] in wanted]
        missing = wanted - {s[0] for s in all_scenarios}
        if missing:
            print(f"Unknown scenario name(s): {sorted(missing)}", file=sys.stderr)
            return 2

    rows = []
    timeseries: dict[str, pd.DataFrame] = {}
    fluxes = ("GPP", "NEE", "Reco", "CH4")

    if not args.summary_only:
        for name, mode, use_obs_gpp in scenarios:
            tag = "MCMC" if mode == "mcmc" else "forward"
            print(f"  ▸ {name}  ({tag}, use_observed_gpp={use_obs_gpp})")
            run_scenario(name, mode=mode, use_observed_gpp=use_obs_gpp,
                         input_csv=input_csv, priors_yaml=priors_yaml,
                         output_dir=output_dir)

    if args.no_summary:
        return 0

    for name, _, _ in all_scenarios:
        path = output_dir / f"fluxes_{name}.csv"
        if path.exists():
            df_ts = pd.read_csv(path, parse_dates=["Date"])
            timeseries[name] = df_ts
            for flux in fluxes:
                m = metrics(df_ts[f"{flux}_obs"].values, df_ts[f"{flux}_sim"].values)
                rows.append({
                    "scenario": name,
                    "pair": f"{flux}_obs vs {flux}_sim",
                    "obs_col": f"{flux}_obs", "sim_col": f"{flux}_sim",
                    "flux": flux, **m,
                })
    if not rows:
        print("No fluxes_*.csv found in output dir; nothing to summarize.")
        return 0

    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df[["scenario", "pair", "obs_col", "sim_col", "flux",
                             "n", "obs_mean", "sim_mean",
                             "rmse", "mae", "bias", "r2", "nse"]]
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    print("\nobs-vs-sim performance, per scenario:\n")
    pivot = metrics_df.pivot_table(index="pair", columns="scenario",
                                   values=["rmse", "bias", "r2", "nse"])
    print(pivot.round(3).to_string())

    _make_figures(timeseries, metrics_df, fluxes, scenarios, output_dir)

    print(f"\nTime-series figure: {output_dir / 'timeseries.png'}")
    print(f"Scatter figure:     {output_dir / 'scatter_obs_vs_sim.png'}")
    print(f"Metrics CSV:        {output_dir / 'metrics.csv'}")
    return 0


def _make_figures(timeseries: dict[str, pd.DataFrame],
                  metrics_df: pd.DataFrame, fluxes: tuple[str, ...],
                  scenarios: list[tuple[str, str, bool]], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n(skipping figures — matplotlib unavailable: {e})")
        return

    scenario_colors = {
        "fwd_all_modeled":   "tab:orange",
        "fwd_observed_gpp":  "tab:blue",
        "mcmc_all_modeled":  "tab:red",
        "mcmc_observed_gpp": "tab:green",
    }

    fig, axes = plt.subplots(len(fluxes), 1, figsize=(11, 12), sharex=True)
    for ax, flux in zip(axes, fluxes):
        ref = next(iter(timeseries.values()))
        ax.plot(ref["Date"], ref[f"{flux}_obs"], color="black", lw=1.5,
                label=f"{flux}_obs", zorder=3)
        for name, df in timeseries.items():
            ax.plot(df["Date"], df[f"{flux}_sim"], lw=1.1, alpha=0.85,
                    color=scenario_colors[name],
                    label=f"{flux}_sim ({name})")
        ax.set_ylabel(f"{flux}\n(g C m$^{{-2}}$ d$^{{-1}}$)")
        ax.grid(alpha=0.3)
        if flux == "NEE":
            ax.axhline(0, color="grey", lw=0.5)
    axes[0].legend(loc="upper left", frameon=False, ncol=3, fontsize=8)
    axes[-1].set_xlabel("Date")
    fig.suptitle("PEPRMT-Rice — obs vs sim time series, 4 scenarios", y=1.0)
    fig.tight_layout()
    fig.savefig(output_dir / "timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    scenario_names = [name for name, _, _ in scenarios]
    fig, axes = plt.subplots(len(fluxes), len(scenario_names),
                             figsize=(14, 11), sharex=False, sharey=False)
    if len(scenario_names) == 1:
        axes = np.array([[ax] for ax in axes])  # normalize to 2-D for indexing
    for i, flux in enumerate(fluxes):
        for j, name in enumerate(scenario_names):
            ax = axes[i, j]
            df = timeseries[name]
            obs = df[f"{flux}_obs"].values
            sim = df[f"{flux}_sim"].values
            ax.scatter(obs, sim, s=14, alpha=0.7,
                       color=scenario_colors[name], edgecolor="none")
            lo = float(np.nanmin([obs, sim]))
            hi = float(np.nanmax([obs, sim]))
            ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8, ls="--", label="1:1")
            row = metrics_df[(metrics_df["scenario"] == name)
                             & (metrics_df["flux"] == flux)].iloc[0]
            text = (f"n = {int(row['n'])}\n"
                    f"R² = {row['r2']:.2f}\n"
                    f"RMSE = {row['rmse']:.2f}\n"
                    f"bias = {row['bias']:+.2f}\n"
                    f"NSE = {row['nse']:.2f}")
            ax.text(0.03, 0.97, text, transform=ax.transAxes,
                    va="top", ha="left", fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
            ax.set_xlabel(f"{flux}_obs")
            ax.set_ylabel(f"{flux}_sim")
            ax.set_title(f"{flux}   ·   {name}", fontsize=9)
            ax.grid(alpha=0.25)
    fig.suptitle("PEPRMT-Rice — obs vs sim, scenarios", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_obs_vs_sim.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
