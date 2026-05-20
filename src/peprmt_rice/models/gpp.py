"""
Gross Primary Productivity (GPP) model — single light-use-efficiency (LUE)
formulation.

The model follows the form

    GPP = -LUE_max * f(T) * APAR

where
    APAR  = fPAR * PAR              # absorbed PAR (MJ m-2 d-1)
    fPAR  = p * (1 - exp(-k*LAI))   # Yuan (2007), AFM. `p` is the asymptotic
                                    # max fPAR (~0.85-0.95, calibratable).
    f(T)  = Arrhenius-style temperature response from Oikawa et al. (2017)
            with optimum temperature Topt and activation/deactivation
            enthalpies Ha and Hd.

By convention GPP is reported as a **negative** flux (carbon uptake by the
ecosystem), matching the sign convention of the rest of the model.

This is the only LUE formulation exposed by the public package. Earlier
research variants ("Patty's APAR", an alternative VPD-and-Topt scaler, etc.)
have been retired in favour of a single, well-documented formulation that
takes the same set of parameters every time.

References
----------
Oikawa, P. Y., et al. (2017). Evaluation of a hierarchy of models reveals
importance of substrate limitation for predicting carbon dioxide and methane
exchange in restored wetlands. JGR-Biogeosciences.

Yuan, W., et al. (2007). Deriving a light use efficiency model from
eddy covariance flux data. Agricultural and Forest Meteorology.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence, Tuple, Union

import numpy as np


# Universal gas constant in kJ mol^-1 K^-1 (units chosen to match Ha, Hd).
GAS_CONSTANT_KJ = 0.00831


@dataclass
class GPPParameters:
    """Calibratable parameters for the GPP model.

    Attributes
    ----------
    LUE_max : float
        Maximum light-use efficiency, g C MJ^-1.
    k : float
        Beer-Lambert light-extinction coefficient (unitless).
    Topt : float
        Optimum temperature for photosynthesis, °C.
    Ha : float
        Rate of exponential increase below Topt, kJ mol^-1.
    Hd : float
        Rate of decrease above Topt, kJ mol^-1.
    p : float
        Asymptotic maximum fPAR as LAI → ∞ (unitless, typically 0.85–0.95).
        Accounts for canopy clumping and incomplete light interception.
    """

    LUE_max: float
    k: float
    Topt: float
    Ha: float
    Hd: float
    p: float

    # ---- convenience constructors ------------------------------------------------

    @classmethod
    def from_list(cls, values: Sequence[float]) -> "GPPParameters":
        """Build from a positional list in the documented order
        ``[LUE_max, k, Topt, Ha, Hd, p]``.

        Useful when interfacing with MCMC libraries that work in plain arrays.
        """
        if len(values) != 6:
            raise ValueError(
                f"GPPParameters.from_list expects 6 values "
                f"(LUE_max, k, Topt, Ha, Hd, p), got {len(values)}."
            )
        return cls(*values)

    @classmethod
    def defaults(cls) -> "GPPParameters":
        """Default parameter values from the calibrated US-HRA 2015 posterior.

        These reproduce a physically reasonable forward run on a temperate
        rice paddy. They are NOT a universal fit — calibrate to your site
        for publication-quality predictions.

        Source: ``PEPRMT-Rice_v0`` MCMC posterior mean for HRA 2015.
        """
        return cls(LUE_max=1.73, k=1.02, Topt=26.7, Ha=118.0, Hd=212.0, p=0.87)

    def to_array(self) -> np.ndarray:
        return np.array([self.LUE_max, self.k, self.Topt, self.Ha, self.Hd, self.p])

    def asdict(self) -> dict:
        return asdict(self)


class GPPModel:
    """Single-formulation LUE GPP model.

    Parameters are provided as a :class:`GPPParameters` (or anything that can
    be converted via :meth:`GPPParameters.from_list`). The model is stateless
    apart from its parameters, so the same instance can be reused for many
    runs.

    Examples
    --------
    >>> params = GPPParameters.defaults()
    >>> model = GPPModel(params)
    >>> gpp, f_T, apar = model.estimate(air_temp_C, par, lai)
    """

    def __init__(self, params: Union[GPPParameters, Sequence[float]]):
        if isinstance(params, GPPParameters):
            self.params = params
        else:
            self.params = GPPParameters.from_list(params)

    # ---- physics components ------------------------------------------------------

    def absorbed_par(self, par: np.ndarray, lai: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute absorbed PAR (APAR) from incoming PAR and LAI.

        Parameters
        ----------
        par : array-like
            Daily sum of incoming photosynthetically active radiation,
            umol m^-2 d^-1.
        lai : array-like
            Daily leaf area index, m^2 m^-2.

        Returns
        -------
        apar : ndarray, MJ m^-2 d^-1
        fpar : ndarray, unitless
        """
        par = np.asarray(par, dtype=float)
        lai = np.asarray(lai, dtype=float)
        # Convert PAR from umol m^-2 d^-1 to MJ m^-2 d^-1
        par_mj = par * 0.0002186 * 0.001
        fpar = self.params.p * (1.0 - np.exp(-self.params.k * lai))
        apar = fpar * par_mj
        return apar, fpar

    def temperature_response(self, air_temp_C: np.ndarray) -> np.ndarray:
        """Arrhenius-style f(T) with optimum.

        Parameters
        ----------
        air_temp_C : array-like
            Daily-mean air temperature, °C.

        Returns
        -------
        f_T : ndarray
            Unitless temperature scaler in roughly [0, 1].
        """
        air_temp_K = np.asarray(air_temp_C, dtype=float) + 273.15
        opt_T_K = self.params.Topt + 273.15
        denom = air_temp_K * GAS_CONSTANT_KJ * opt_T_K

        exp1 = self.params.Ha * (air_temp_K - opt_T_K) / denom
        exp2 = self.params.Hd * (air_temp_K - opt_T_K) / denom

        top = self.params.Hd * np.exp(exp1)
        bot = self.params.Hd - self.params.Ha * (1.0 - np.exp(exp2))
        return top / bot

    # ---- main entry point --------------------------------------------------------

    def estimate(
        self,
        air_temp_C: np.ndarray,
        par: np.ndarray,
        lai: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the GPP model for a daily time series.

        Parameters
        ----------
        air_temp_C : array-like, °C
            Daily-mean air temperature.
        par : array-like, umol m^-2 d^-1
            Daily sum of incoming PAR.
        lai : array-like, m^2 m^-2
            Daily LAI (gap-filled).

        Returns
        -------
        gpp : ndarray, g C m^-2 d^-1
            Negative values denote uptake by the ecosystem (PEPRMT-Rice sign
            convention).
        f_T : ndarray
            Temperature scaler for the day.
        apar : ndarray, MJ m^-2 d^-1
            Absorbed PAR.
        """
        apar, _ = self.absorbed_par(par, lai)
        f_T = self.temperature_response(air_temp_C)
        gpp = -self.params.LUE_max * f_T * apar
        return gpp, f_T, apar

    def __repr__(self) -> str:
        return f"GPPModel({self.params!r})"
