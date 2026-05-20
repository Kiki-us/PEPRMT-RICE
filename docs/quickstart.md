# Quickstart

> ⚠️  **Sign convention**: PEPRMT-Rice uses **negative = uptake** for all
> carbon fluxes. If your observed GPP / NEE come from FLUXNET, REddyProc,
> or any product that uses positive = uptake, run
> `peprmt-rice flip-sign INPUT OUTPUT --columns GPP,NEE` once
> before you start. The validator warns automatically if it spots
> wrong-signed `GPP`. Full details in
> [`docs/input_format.md`](input_format.md).

After [installing](installation.md), running the model is three commands:

```bash
# 1. Sanity-check that your CSV matches the expected schema.
peprmt-rice validate-input data/example/example_input.csv

# 2. Run the deterministic forward chain with built-in default parameters.
peprmt-rice run --config configs/example_forward.yaml

# 3. Inspect the outputs.
ls outputs/example_forward/
#   parameters/  timeseries/  figures/  (and mcmc_chains/ for MCMC runs)
```

## Skipping the GPP model when you have observed GPP

If you already have partitioned GPP from REddyProc (or similar), you can
feed it directly to the NEE and CH4 models and skip the GPP module:

```bash
peprmt-rice run --config configs/example_observed_gpp.yaml
```

In this mode the CSV must include a `GPP` column (g C m⁻² d⁻¹,
negative = uptake) and the `PAR`, `VPD`, `LAI` columns become optional.
See [`docs/input_format.md`](input_format.md) for the full discussion.

## Forward run on your own data

1. Format your CSV to match [`docs/input_format.md`](input_format.md), or
   write a `column_map` in the config that renames your columns.

2. Copy `configs/example_forward.yaml` and edit:

   ```yaml
   input:
     path: /absolute/or/relative/path/to/my_site.csv

   output:
     dir: outputs/my_site
   ```

3. Run:

   ```bash
   peprmt-rice run --config configs/my_run.yaml
   ```

Forward mode finishes in seconds for a typical site-year.

## MCMC calibration on your own data

1. Same input format, but the CSV must also contain at least one of
   `NEE`, `Reco`, `CH4`.

2. Copy `configs/example_mcmc.yaml` and edit `target_fluxes` to match what
   your CSV contains. Adjust priors in `configs/priors.yaml` if the defaults
   are too wide for your site.

3. Run:

   ```bash
   peprmt-rice run --config configs/my_mcmc.yaml
   ```

MCMC writes:
- `mcmc_chains/<model>.npy` — raw chains
- `parameters/<model>_posterior_summary.csv` — mean / std / 95% CI per parameter
- `parameters/posterior_mean_parameters.yaml` — single best parameter set
- `timeseries/fluxes.csv` — forward run with the posterior-mean parameters

MCMC runtime scales linearly in `simulation.niter`. 5,000 iterations per
block is typically enough for a smoke test; 20,000+ for a publication.
