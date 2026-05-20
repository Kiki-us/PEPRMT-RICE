"""
Postprocessing — turn raw model output into the user-facing deliverables.

This package supersedes the old single-file ``postprocessing.py`` module.
It now contains:

* :mod:`peprmt_rice.postprocessing._run_outputs` — the canonical
  ``write_run_outputs`` and ``summarize_chain`` helpers used by every
  forward / MCMC run (re-exported at the package level for back-compat).
* :mod:`peprmt_rice.postprocessing.scenario_comparison` — runs the four
  comparison scenarios (forward × MCMC × {all-modeled, observed-GPP})
  on the bundled example input and emits a metrics table plus figures.
  Invoke via ``python -m peprmt_rice.postprocessing.scenario_comparison``.

The legacy import path ``from peprmt_rice.postprocessing import
write_run_outputs`` continues to work — anything that consumed the
single-file module before the split is unaffected.
"""

from peprmt_rice.postprocessing._run_outputs import (
    summarize_chain,
    write_run_outputs,
)

__all__ = ["summarize_chain", "write_run_outputs"]
