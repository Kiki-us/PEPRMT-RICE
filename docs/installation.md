# Installation

PEPRMT-Rice is a pure-Python package and works on Linux, macOS, and Windows.

## Requirements

- Python 3.9 or newer
- A C compiler is **not** needed — all dependencies ship as wheels.

## Recommended: clean virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

## Install from source

```bash
git clone https://github.com/<your-org>/peprmt-rice.git
cd peprmt-rice
pip install -e .                   # editable install — useful for development
```

The `peprmt-rice` command becomes available immediately:

```bash
peprmt-rice --version
peprmt-rice info
```

## Development extras

If you plan to run tests or contribute, install the dev extras:

```bash
pip install -e ".[dev]"
pytest
```

## MCMC mode

Forward simulation has no extra dependencies. MCMC calibration uses
[`pymcmcstat`](https://github.com/prmiles/pymcmcstat), which is listed in the
core requirements and installed automatically.

## Verifying the install

A small built-in example is included:

```bash
peprmt-rice validate-input data/example/example_input.csv
peprmt-rice run --config configs/example_forward.yaml
```

The second command should finish in a few seconds and write timeseries, a
parameter dump, and one diagnostic figure under `outputs/example_forward/`.
