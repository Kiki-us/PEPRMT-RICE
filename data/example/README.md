# Example input data

`example_input.csv` contains 90 days (April–July 2015) of preprocessed
flux-tower and ancillary data from the `US-HRA` AmeriFlux rice site in
Arkansas, included here purely so users can verify their installation:

```bash
peprmt-rice validate-input data/example/example_input.csv
peprmt-rice run --config configs/example_forward.yaml
```

The file is a small slice of a longer, published dataset and is **not** the
authoritative version of the HRA record. For real scientific use, download
the original AmeriFlux data from <https://ameriflux.lbl.gov/sites/siteinfo/US-HRA>.

Columns follow the canonical PEPRMT-Rice schema (`docs/input_format.md`):

| Column | Description |
|---|---|
| `Date` | Observation date |
| `TA` | Daily-mean air temperature (°C) |
| `PAR` | Daily sum of photosynthetically active radiation (umol m-2 d-1) |
| `VPD` | Daily-mean vapor pressure deficit (hPa) |
| `LAI` | Leaf area index (m2 m-2) |
| `WT` | Water-table height (cm, positive above soil surface) |
| `DoP` | Date of planting for the season |
| `Season` | Phenology stage code (1–6) |
| `Harvest` | Harvested biomass on this date (g m-2) |
| `NEE` | Observed NEE (g C m-2 d-1) — used by MCMC mode |
| `Reco` | Observed Reco (g C m-2 d-1) — used by MCMC mode |
| `CH4` | Observed CH4 flux (g C m-2 d-1) — used by MCMC mode |
