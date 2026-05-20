# PEPRMT-Rice

A site-level biogeochemistry model that simulates rice-paddy carbon fluxes —
gross primary productivity (GPP), net ecosystem exchange / ecosystem
respiration (NEE / Reco), and methane (CH4) emissions — from daily
flux-tower or equivalent meteorological inputs.

PEPRMT-Rice runs in two simulation modes:

1. **Forward mode** — supply parameters (defaults, your own, or from a previous
   calibration) and produce a deterministic time series of GPP, Reco, NEE and
   CH4. Fast and lightweight.
2. **MCMC calibration mode** — fit the model parameters to observed fluxes
   using Markov Chain Monte Carlo, then propagate parameter uncertainty
   through to predicted fluxes.

Both modes share the same input format and the same physics. The only
difference is whether parameters are *given* or *inferred*.

---

## Quickstart

```bash
# 1. Install
pip install -e .

# 2. Validate your input CSV against the expected schema
peprmt-rice validate-input data/example/example_input.csv

# 3. Run a forward simulation with default parameters
peprmt-rice run --config configs/example_forward.yaml

# 4. Or calibrate parameters to your data with MCMC
peprmt-rice run --config configs/example_mcmc.yaml
```

Outputs land under `outputs/` in the run's working directory:

```
outputs/
├── timeseries/        # daily GPP, Reco, NEE, CH4 (CSV)
├── parameters/        # posterior summaries (MCMC mode only)
├── mcmc_chains/       # raw MCMC chains as .pkl (MCMC mode only)
└── figures/           # diagnostic plots
```

---

## Input data

PEPRMT-Rice needs **one CSV** per run with daily values for the columns
listed in [`docs/input_format.md`](docs/input_format.md). At minimum:

| Column    | Unit         | Description                                        |
| --------- | ------------ | -------------------------------------------------- |
| `Date`    | YYYY-MM-DD   | Observation date                                   |
| `TA`      | °C           | Daily-mean air temperature                         |
| `PAR`     | umol m-2 d-1 | Daily sum of photosynthetically active radiation   |
| `VPD`     | hPa          | Daily-mean vapor pressure deficit                  |
| `LAI`     | m2 m-2       | Daily leaf area index (gap-filled)                 |
| `WT`      | cm           | Water-table height (positive = above soil surface) |
| `DoP`     | YYYY-MM-DD   | Date of planting for the season                    |
| `Season`  | int          | Phenology stage code (see docs)                    |
| `Harvest` | g m-2        | Harvested biomass on this date (0 if none)         |

For calibration runs, also include observed fluxes you want to fit (any of
`GPP`, `NEE`, `Reco`, `CH4`).

> ⚠️  **Sign convention**: PEPRMT-Rice uses **negative = uptake** for every
> carbon flux (`GPP`, `NEE`, `Reco`, `CH4`) — for both inputs *and* outputs.
> Most public GPP/NEE products use the opposite convention. The validator
> will flag a wrong-signed `GPP` automatically, and
> `peprmt-rice flip-sign INPUT OUTPUT --columns GPP,NEE` will
> convert the columns for you. See
> [`docs/input_format.md`](docs/input_format.md) for the full discussion.

If you already have a partitioned GPP time series and want PEPRMT-Rice to
use it directly (skipping its own GPP module), set
`simulation.use_observed_gpp: true` in the config and provide a `GPP`
column. In that mode `PAR`, `VPD`, and `LAI` are no longer required.
See [`docs/input_format.md`](docs/input_format.md) for details.

A worked example lives at `data/example/example_input.csv`.

---

## Project layout

```
PEPRMT-Rice/
├── src/peprmt_rice/             # the package
│   ├── __main__.py         # enables `python -m peprmt_rice`
│   ├── _version.py         # package version string
│   ├── cli.py              # entry point for `peprmt-rice` command
│   ├── config.py           # YAML-backed run configuration
│   ├── io.py               # CSV reader and schema validator
│   ├── models/             # core physics
│   │   ├── gpp.py          # light-use-efficiency GPP model
│   │   ├── nee.py          # DAMM-based respiration / NEE model
│   │   └── ch4.py          # rice-paddy CH4 production / transport / oxidation
│   ├── simulation/
│   │   ├── forward.py      # deterministic run with fixed parameters
│   │   └── mcmc.py         # MCMC calibration driver
│   ├── postprocessing/     # turn raw output into deliverables
│   │   ├── _run_outputs.py        # write_run_outputs + summarize_chain
│   │   └── scenario_comparison.py # 4-scenario sweep + metrics + figures
│   │                              # → python -m peprmt_rice.postprocessing.scenario_comparison
│   └── plotting.py         # diagnostic figures
├── configs/                # YAML run configurations
│   ├── example_forward.yaml       # forward-mode example
│   ├── example_mcmc.yaml          # MCMC calibration example
│   ├── example_observed_gpp.yaml  # forward run using an observed GPP column
│   └── priors.yaml                # default prior distributions
├── data/example/           # small synthetic example input
├── docs/                   # quickstart, installation, input format, outputs
└── tests/                  # unit + integration tests
```

------

## Citing

If you use PEPRMT-Rice in published work, please cite Tang et. al., (2026), AFM and Oikawa et al. (2017), JGR-B —the "PEPRMT" architecture for rice-paddy adaptation by the authors.

---

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
