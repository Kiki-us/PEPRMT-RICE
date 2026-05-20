# Input format

PEPRMT-Rice takes a single CSV per run, one row per day. The set of required
columns is fixed; you can keep extra columns in the file (they will be
ignored), and you can map your CSV's column names to the canonical ones in
the run config.

## Sign convention (read this first)

**All carbon fluxes in PEPRMT-Rice use NEGATIVE = uptake by the ecosystem,
POSITIVE = release to the atmosphere.** This applies to:

- The observation columns you provide: `GPP`, `NEE`, `Reco`, `CH4`.
- The output columns the model writes: `GPP`, `Reco`, `NEE`, `CH4`.

So a healthy growing-season day has **negative** GPP and NEE, **positive**
Reco and CH4.

Most published GPP / NEE products (FLUXNET, AmeriFlux gap-filled, REddyProc,
MODIS MOD17, etc.) use the opposite convention — **positive = uptake** for
GPP. If you feed one of those in without flipping the sign, PEPRMT-Rice will
silently produce garbage outputs.

The CLI catches the most common case automatically:

```bash
peprmt-rice validate-input my_data.csv
# WARNING: GPP looks like it uses the POSITIVE = uptake convention
# (92% of values are > 0, median = 3.42). PEPRMT-Rice expects
# NEGATIVE = uptake. Multiply the column by -1 before running, or use
# `peprmt-rice flip-sign INPUT OUTPUT --columns GPP` to convert.
```

…and a helper subcommand does the conversion for you:

```bash
peprmt-rice flip-sign my_data.csv my_data_flipped.csv --columns GPP,NEE
```

`NEE` is checked more leniently because some sites are genuine net
sources; you should still verify it manually if the warning fires.

## Required driver columns

These must be present and gap-filled before running. Missing values
propagate through the simulation and will produce NaN outputs.

| Column | Unit | Description |
|---|---|---|
| `Date` | YYYY-MM-DD | Observation date — one row per day, sorted ascending. |
| `TA` | °C | Daily-mean air temperature. |
| `PAR` | umol m-2 d-1 | Daily sum of PAR. *Optional when `use_observed_gpp: true`.* |
| `VPD` | hPa | Daily-mean vapor pressure deficit. *Optional when `use_observed_gpp: true`.* |
| `LAI` | m2 m-2 | Daily leaf area index (gap-filled — *not* the raw 4-day MODIS product). *Optional when `use_observed_gpp: true`.* |
| `WT` | cm | Water-table height; positive = above soil surface, negative = below. |
| `DoP` | YYYY-MM-DD | Date of planting for the current season. Same value repeated across the season's rows is fine. |
| `Season` | int | Phenology stage code 1–6 (see below). |
| `Harvest` | g m-2 | Harvested biomass removed on this date (0 elsewhere). |

### Phenology stage codes (`Season`)

| Code | Stage |
|---|---|
| 1 | Pre-planting / fallow |
| 2 | Vegetative |
| 3 | Reproductive |
| 4 | Ripening |
| 5 | Post-harvest / senescence |
| 6 | Winter / dormancy |

The model uses `Season >= 5` to flush remaining labile carbon into the SOC
pool at end of season; everything else is informational.

## Optional observation columns

Required only for MCMC calibration and/or for the "skip the GPP model" mode.
The simulation driver fits whichever of these the user lists in
`simulation.target_fluxes`.

| Column | Unit | Description |
|---|---|---|
| `GPP` | g C m-2 d-1 | Observed GPP (negative = uptake — same sign as model GPP). Enables `use_observed_gpp: true`. |
| `NEE` | g C m-2 d-1 | Observed NEE (negative = uptake). |
| `Reco` | g C m-2 d-1 | Observed ecosystem respiration. |
| `CH4` | g C m-2 d-1 | Observed CH4 flux. |

## Skipping the GPP model with `GPP`

If you already have a partitioned GPP time series (for example from
REddyProc or another flux-partitioning tool), you can ask PEPRMT-Rice to use
it directly and skip its own GPP module. Set in the run config:

```yaml
simulation:
  use_observed_gpp: true
```

When this flag is true:

- The CSV must contain a `GPP` column with daily values in
  g C m⁻² d⁻¹ and the model sign convention (negative = uptake).
- `PAR`, `VPD`, and `LAI` are **not** required — leave them out of the
  CSV entirely if you don't have them.
- The forward driver feeds `GPP` straight into the NEE/Reco and CH4
  models.
- The MCMC driver does **not** calibrate any GPP parameters; only the NEE
  and CH4 blocks are sampled.

Sign convention check: the model uses negative GPP for ecosystem uptake.
If your partitioned product reports GPP as positive (more common in the
literature), multiply by −1 before saving the CSV, or apply the conversion
via a small preprocessing script. PEPRMT-Rice does *not* auto-flip signs.

## Mapping your column names

If your CSV uses different names, list them in the config:

```yaml
input:
  path: my_site_2024.csv
  date_column: Date
  column_map:
    TairAvg: TA
    SWdown: PAR
    VPD_kPa: VPD     # caution: unit check is the user's responsibility
    LAI_smooth: LAI
    water_table: WT
    plant_date: DoP
    phenology: Season
    harvest_kg_ha: Harvest
```

PEPRMT-Rice does **not** convert units automatically. You are responsible
for making sure each column is in the unit listed in the table above
*before* writing the CSV.

## Validation

Use the bundled validator to check a CSV without running the full model:

```bash
peprmt-rice validate-input my_site_2024.csv
```

The command prints any schema errors (missing columns, unparseable dates)
and physical-range warnings (negative LAI, suspiciously hot temperatures,
etc.).
