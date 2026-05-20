"""
Command-line interface for PEPRMT-Rice.

Subcommands:

* ``peprmt-rice validate-input PATH``
    Parse a CSV against the expected schema and print any warnings / errors
    without running the model. Quick way to sanity-check your data file.
    Includes a sign-convention sniff test on ``GPP`` and ``NEE``.

* ``peprmt-rice run --config CONFIG.yaml``
    Run the full simulation pipeline described by the config. Dispatches to
    forward or MCMC mode based on ``simulation.mode``.

* ``peprmt-rice info``
    Print package version and the canonical input column schema.

* ``peprmt-rice flip-sign INPUT OUTPUT --columns GPP,NEE``
    Convert flux columns from positive-uptake (the common literature
    convention) to negative-uptake (the PEPRMT-Rice convention) by
    multiplying the listed columns by -1.

Run-output goes to ``cfg.output.dir`` (creating it if needed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from peprmt_rice._version import __version__


def _cmd_validate_input(args: argparse.Namespace) -> int:
    from peprmt_rice.io import load_input, validate_input, InputValidationError

    try:
        df = load_input(args.path)
        warnings = validate_input(df)
    except InputValidationError as e:
        print(f"INVALID: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(f"OK: parsed {len(df):,} rows from {args.path}")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("No warnings.")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from peprmt_rice.config import load_config
    from peprmt_rice.io import load_input, validate_input
    from peprmt_rice.simulation.forward import run_forward
    from peprmt_rice.simulation.mcmc import run_mcmc
    from peprmt_rice.postprocessing import write_run_outputs
    from peprmt_rice.plotting import plot_fluxes

    cfg = load_config(args.config)
    print(f"Loaded config: {cfg.source_file}")
    print(f"  mode      : {cfg.simulation.mode}")
    print(f"  input     : {cfg.input.path}")
    print(f"  output dir: {cfg.output_dir()}")

    df = load_input(cfg.input.path, column_map=cfg.input.column_map,
                    date_column=cfg.input.date_column)

    required_obs = []
    if cfg.simulation.mode == "mcmc":
        required_obs = [t for t in cfg.simulation.target_fluxes
                        if t in df.columns]
    warnings = validate_input(
        df,
        require_observations=required_obs,
        use_observed_gpp=cfg.simulation.use_observed_gpp,
    )
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)

    if cfg.simulation.mode == "forward":
        result = run_forward(df, cfg)
        chains = None
    else:
        result = run_mcmc(df, cfg)
        chains = result.get("chains") if cfg.output.save_chains else None

    out_dir = cfg.output_dir()
    write_run_outputs(
        out_dir,
        timeseries=result["timeseries"],
        parameters=result["parameters"],
        chains=chains,
        fmt=cfg.output.format,
    )
    if cfg.output.save_figures:
        plot_fluxes(result["timeseries"], out_dir / "figures" / "fluxes.png")

    print(f"Done. Outputs written to {out_dir}")
    return 0


def _cmd_flip_sign(args: argparse.Namespace) -> int:
    from peprmt_rice.io import flip_sign, load_input, InputValidationError

    try:
        df = load_input(args.input)
    except InputValidationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    cols = [c.strip() for c in args.columns.split(",") if c.strip()]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"WARNING: columns not present in input, skipped: {missing}",
              file=sys.stderr)

    flipped = flip_sign(df, cols)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Date column is parsed to datetime by load_input; serialise as YYYY-MM-DD.
    if "Date" in flipped.columns:
        flipped = flipped.copy()
        flipped["Date"] = flipped["Date"].dt.strftime("%Y-%m-%d")
    flipped.to_csv(out_path, index=False)
    print(f"OK: flipped {[c for c in cols if c in df.columns]} "
          f"and wrote {len(flipped):,} rows to {out_path}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from peprmt_rice.io import DRIVER_COLUMNS, OBSERVATION_COLUMNS

    print(f"PEPRMT-Rice v{__version__}")
    print()
    print("Sign convention:  NEGATIVE flux = uptake by the ecosystem,")
    print("                  POSITIVE flux = release to the atmosphere.")
    print("This applies to GPP, NEE, Reco, and CH4 — input AND output.")
    print("If your observed columns use the opposite convention, run")
    print("`peprmt-rice flip-sign IN OUT --columns GPP,NEE` first.")
    print()
    print("Required driver columns:")
    for c in DRIVER_COLUMNS:
        print(f"  - {c.name:<8s} ({c.unit:<14s}) {c.description}")
    print()
    print("Optional observation columns (needed for MCMC fitting):")
    for c in OBSERVATION_COLUMNS:
        print(f"  - {c.name:<8s} ({c.unit:<14s}) {c.description}")
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="peprmt-rice",
        description="PEPRMT-Rice command-line interface.",
    )
    p.add_argument("--version", action="version", version=f"peprmt-rice {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser(
        "validate-input",
        help="Parse a CSV against the expected schema without running the model.",
    )
    p_validate.add_argument("path", type=Path, help="Path to user input CSV.")
    p_validate.set_defaults(func=_cmd_validate_input)

    p_run = sub.add_parser(
        "run",
        help="Run a forward or MCMC simulation described by a YAML config.",
    )
    p_run.add_argument("--config", required=True, type=Path,
                       help="Path to the run YAML config.")
    p_run.set_defaults(func=_cmd_run)

    p_info = sub.add_parser(
        "info",
        help="Print package version and input column schema.",
    )
    p_info.set_defaults(func=_cmd_info)

    p_flip = sub.add_parser(
        "flip-sign",
        help="Multiply the listed columns by -1 (convert positive-uptake → "
             "negative-uptake) and write to a new CSV.",
    )
    p_flip.add_argument("input", type=Path, help="Path to user input CSV.")
    p_flip.add_argument("output", type=Path, help="Path to write the converted CSV.")
    p_flip.add_argument(
        "--columns",
        default="GPP",
        help="Comma-separated list of columns to flip (default: GPP). "
             "Common: 'GPP,NEE,Reco'.",
    )
    p_flip.set_defaults(func=_cmd_flip_sign)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
