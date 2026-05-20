# Outputs

Every run writes a directory under `cfg.output.dir` with the following
structure:

```
<output dir>/
├── timeseries/
│   └── fluxes.csv            # daily GPP, Reco, NEE, CH4 + diagnostics
├── parameters/
│   ├── gpp.csv               # parameter values used / posterior means
│   ├── nee.csv
│   ├── ch4.csv
│   └── *_posterior_summary.csv   # MCMC mode only: mean, std, 95% CI
├── mcmc_chains/              # MCMC mode only
│   ├── gpp.npy               # raw chain array, shape (niter, n_params)
│   ├── nee.npy
│   └── ch4.npy
└── figures/
    └── fluxes.png            # 4-panel sanity-check plot
```

## `timeseries/fluxes.csv`

One row per input day. Columns:

| Column | Unit | Description |
|---|---|---|
| `Date` | YYYY-MM-DD | Same date as the input row. |
| `TA`, `WT`, `PAR`, `LAI` | (input units) | Echoed from input for self-contained plotting. |
| `GPP` | g C m-2 d-1 | Negative = uptake. |
| `f_T` | unitless | Temperature scaler used in the GPP model. |
| `APAR` | MJ m-2 d-1 | Absorbed PAR. |
| `Reco` | g C m-2 d-1 | Total ecosystem respiration. |
| `Reco_soc`, `Reco_labile` | g C m-2 d-1 | Heterotrophic split by pool. |
| `Ra_above`, `Ra_root` | g C m-2 d-1 | Autotrophic split by tissue. |
| `NEE` | g C m-2 d-1 | `GPP + Reco`. |
| `SOC_left`, `labile_left` | g C m-2 | Remaining pool sizes at end of day. |
| `GPP_aboveground`, `GPP_belowground` | g C m-2 d-1 | Allocation partitions. |
| `CH4` | g C m-2 d-1 | Total CH4 emission flux. |
| `CH4_plant`, `CH4_hydro` | g C m-2 d-1 | Transport-pathway split. |
| `CH4_oxidation` | g C m-2 d-1 | Total CH4 oxidation removed before emission. |

## `parameters/`

One CSV per model block (`gpp`, `nee`, `ch4`) with two columns: `name` and
`value`. In MCMC mode, `value` is the posterior mean; a separate
`*_posterior_summary.csv` lists mean, std, and 95% credible interval per
parameter.

## `mcmc_chains/`

NumPy `.npy` files, shape `(niter, n_params)`. Load with:

```python
import numpy as np
chain = np.load("outputs/my_mcmc/mcmc_chains/ch4.npy")
print(chain.shape)
```

The order of columns matches the order of parameters in
`parameters/ch4.csv`.
