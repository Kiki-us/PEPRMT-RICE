"""
Run configuration for PEPRMT-Rice.

A run is fully described by a YAML file that is parsed into a :class:`RunConfig`
dataclass. The CLI reads the file with :func:`load_config` and hands the
resulting object to the simulation drivers.

The config has four sections:

1. **simulation**  - which mode to run ("forward" or "mcmc") and any global
   options (random seed, MCMC iterations, etc.).
2. **input**       - path to the user CSV and optional column overrides.
3. **parameters**  - either a path to a YAML/CSV of parameter values
   (forward mode), or prior bounds (MCMC mode), or "defaults" to use the
   built-in starting values.
4. **output**      - where to write results and what artefacts to keep.

A minimal example::

    simulation:
      mode: forward
      seed: 42

    input:
      path: data/example/example_input.csv

    parameters:
      source: defaults

    output:
      dir: outputs/my_run
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SimulationOptions:
    mode: str = "forward"           # "forward" or "mcmc"
    seed: int = 42
    niter: int = 2000               # global MCMC default; can be overridden per block
    burnin: int = 500               # global MCMC burn-in default
    niter_per_block: dict = field(default_factory=dict)
    """Per-block overrides for ``niter``. Keys: ``gpp``, ``nee``, ``ch4``.
    Example: ``niter_per_block: {ch4: 8000}`` keeps GPP/NEE at the global
    ``niter`` but runs CH4 longer — CH4 typically needs more iterations to
    converge because of its larger parameter space and the WT-driven
    discontinuities in its likelihood surface."""
    burnin_per_block: dict = field(default_factory=dict)
    """Per-block overrides for ``burnin``, mirroring ``niter_per_block``."""

    target_fluxes: list = field(default_factory=lambda: ["NEE", "Reco", "CH4"])
    """Which observed columns to fit in MCMC mode."""

    use_observed_gpp: bool = False
    """If True, skip the GPP model entirely and feed the user's GPP
    column directly into the NEE and CH4 models. In this mode the input CSV
    no longer needs PAR / VPD / LAI, and MCMC mode will not calibrate any
    GPP parameters."""

    loss_function: str = "sse"
    """SOS (sum-of-squares) function passed to pymcmcstat. One of:

    * ``"sse"`` (default) — ``sum((sim - obs)**2)``, treats missing values via
      ``np.nansum``. Corresponds to a Gaussian likelihood under
      ``updatesigma=True``.
    * ``"mae"`` — ``mean(|sim - obs|)``, the loss used by the paper's
      ``PostProcessing.sum_squares_ch4`` in
      ``methane_model/code_scripts/GPP_paper_processing_Tang_v1.py``.
      Corresponds to a Laplace likelihood and down-weights outliers linearly
      instead of quadratically. Use this if you want to reproduce the
      paper's posterior chains.
    * ``"rmse"`` — ``sqrt(mean((sim - obs)**2))``, the loss used by the
      older ``sum_squares_ch4`` variant in ``GPP_paper_processing_Tang.py``
      (kept for completeness; the production paper run used ``mae``).
    """

    # ---- helpers ------------------------------------------------------------

    def niter_for(self, block: str) -> int:
        """Return ``niter`` for ``block`` (``gpp`` / ``nee`` / ``ch4``),
        falling back to the global default."""
        return int(self.niter_per_block.get(block, self.niter))

    def burnin_for(self, block: str) -> int:
        """Return ``burnin`` for ``block``, falling back to the global default."""
        return int(self.burnin_per_block.get(block, self.burnin))

    def validate(self) -> None:
        if self.mode not in ("forward", "mcmc"):
            raise ValueError(
                f"simulation.mode must be 'forward' or 'mcmc', got {self.mode!r}"
            )
        if self.niter <= 0:
            raise ValueError("simulation.niter must be positive")
        for block in ("gpp", "nee", "ch4"):
            if self.niter_for(block) <= 0:
                raise ValueError(
                    f"simulation.niter for block {block!r} must be positive"
                )
        if self.loss_function not in ("sse", "mae", "rmse"):
            raise ValueError(
                f"simulation.loss_function must be 'sse', 'mae', or 'rmse', "
                f"got {self.loss_function!r}"
            )


@dataclass
class InputOptions:
    path: str = ""
    date_column: str = "Date"
    column_map: Dict[str, str] = field(default_factory=dict)
    """Optional overrides mapping user-CSV column names → canonical names
    (e.g. ``{TairAvg: TA}``). Anything not in the map is read as-is."""

    def validate(self) -> None:
        if not self.path:
            raise ValueError("input.path is required")


@dataclass
class ParameterOptions:
    source: str = "defaults"
    """One of:
       - ``"defaults"``  – use built-in starting values
       - ``"file"``      – read parameter values / priors from ``path``
    """
    path: Optional[str] = None
    # Used only when source == "file" and the file format is ambiguous.
    format: str = "yaml"

    def validate(self) -> None:
        if self.source not in ("defaults", "file"):
            raise ValueError(
                f"parameters.source must be 'defaults' or 'file', got {self.source!r}"
            )
        if self.source == "file" and not self.path:
            raise ValueError("parameters.path is required when source='file'")


@dataclass
class OutputOptions:
    dir: str = "outputs/run"
    save_chains: bool = True        # only relevant in MCMC mode
    save_figures: bool = True
    format: str = "csv"             # "csv" or "parquet"

    def validate(self) -> None:
        if self.format not in ("csv", "parquet"):
            raise ValueError(
                f"output.format must be 'csv' or 'parquet', got {self.format!r}"
            )


@dataclass
class RunConfig:
    """Top-level run configuration."""
    simulation: SimulationOptions = field(default_factory=SimulationOptions)
    input: InputOptions = field(default_factory=InputOptions)
    parameters: ParameterOptions = field(default_factory=ParameterOptions)
    output: OutputOptions = field(default_factory=OutputOptions)

    # Path of the YAML file we were loaded from (set by :func:`load_config`).
    source_file: Optional[Path] = None

    def validate(self) -> None:
        self.simulation.validate()
        self.input.validate()
        self.parameters.validate()
        self.output.validate()

    def output_dir(self) -> Path:
        """Resolve the output directory relative to the config file (if any)."""
        p = Path(self.output.dir)
        if not p.is_absolute() and self.source_file is not None:
            p = self.source_file.parent / p
        return p


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def _section(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    val = d.get(key, {})
    if not isinstance(val, dict):
        raise ValueError(f"Config section '{key}' must be a mapping, got {type(val).__name__}")
    return val


def load_config(path: str | Path) -> RunConfig:
    """Load a YAML config file and return a validated :class:`RunConfig`."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    cfg = RunConfig(
        simulation=SimulationOptions(**_section(raw, "simulation")),
        input=InputOptions(**_section(raw, "input")),
        parameters=ParameterOptions(**_section(raw, "parameters")),
        output=OutputOptions(**_section(raw, "output")),
        source_file=path,
    )
    cfg.validate()
    return cfg
