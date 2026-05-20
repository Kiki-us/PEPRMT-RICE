"""Simulation drivers.

Two entry points:

- :func:`peprmt_rice.simulation.forward.run_forward` — deterministic run with
  given parameters.
- :func:`peprmt_rice.simulation.mcmc.run_mcmc` — Bayesian calibration via MCMC.

Both consume the same :class:`peprmt_rice.config.RunConfig` and the same parsed
input DataFrame, so switching modes is a one-line config change.
"""
