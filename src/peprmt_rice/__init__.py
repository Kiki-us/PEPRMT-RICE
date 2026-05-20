"""
PEPRMT-Rice
===========

A site-level biogeochemistry model that simulates rice-paddy carbon and
methane fluxes from daily meteorological inputs.

Top-level public API
--------------------

>>> from peprmt_rice import load_config, load_input, run_forward, run_mcmc
>>> cfg = load_config("configs/example_forward.yaml")
>>> df = load_input(cfg.input_path)
>>> results = run_forward(df, cfg)

For most users the CLI is the recommended entry point:

    peprmt run --config configs/example_forward.yaml
"""

from peprmt_rice._version import __version__

# Lazy/explicit imports keep `import peprmt_rice` cheap even when only the CLI is
# being invoked.
from peprmt_rice.config import RunConfig, load_config
from peprmt_rice.io import load_input, validate_input, save_results
from peprmt_rice.simulation.forward import run_forward
from peprmt_rice.simulation.mcmc import run_mcmc

__all__ = [
    "__version__",
    "RunConfig",
    "load_config",
    "load_input",
    "validate_input",
    "save_results",
    "run_forward",
    "run_mcmc",
]
