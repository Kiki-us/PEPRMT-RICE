"""
Net Ecosystem Exchange (NEE) and ecosystem respiration (Reco) model.

The model is a DAMM-style decomposition of two soil carbon pools (SOC and
labile) coupled to a phenology-driven autotrophic respiration term for above-
and below-ground biomass. It expects GPP as an *input* — typically the output
of :class:`peprmt_rice.models.gpp.GPPModel`.

By convention all fluxes are in g C m^-2 d^-1, with **negative GPP**
representing uptake by the ecosystem (so NEE = GPP + Reco).

This module exposes a single parameter set (no more "original" /
"modified" / "calibrated" variants). The historical research switches that
existed in the internal codebase have been consolidated into one model with
all parameters explicitly named.

The carbon-allocation step inside :meth:`NEEModel._phenology_partition`
uses the **BESS-Rice** scheme: a normalised-accumulated-GPP curve (what
the code calls ``GPP_acc_ratio``) drives daily fractional allocation to
root, leaf, stem, and grain via a published double-exponential plus three
Gaussian functions.

References
----------
Davidson, E. A., et al. (2012). The Dual Arrhenius and Michaelis-Menten kinetics
model for decomposition of soil organic matter at hourly to seasonal time scales.
*Global Change Biology* 18, 371-384.

Oikawa, P. Y., et al. (2017). Evaluation of a hierarchy of models reveals
importance of substrate limitation for predicting carbon dioxide and methane
exchange in restored wetlands. *JGR-Biogeosciences* 122, 145-167. (PEPRMT
model architecture.)

Jeong, S., Ko, J., Kang, M., Yeom, J., et al. (2018). BESS-Rice: A remote
sensing derived and biophysical process-based rice productivity simulation
model. *Agricultural and Forest Meteorology* 256-257, 11-25. (Source of the
carbon-allocation scheme used here.)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence, Union

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


GAS_CONSTANT = 8.314  # J K^-1 mol^-1; used with Ea in J mol^-1 (kJ * 1000)


@dataclass
class NEEParameters:
    """Calibratable parameters for the NEE/Reco model.

    All activation energies are in **J mol⁻¹** (so they appear in the
    Arrhenius factor as ``exp(-Ea / RT)`` with R in J mol⁻¹ K⁻¹).

    Attributes
    ----------
    alpha_soc, alpha_labile : float
        Pre-exponential factor (g C m^-2 d^-1) for SOC and labile carbon
        pools in the Arrhenius enzyme-velocity term.
    ea_soc, ea_labile : float
        Activation energy (J mol^-1) for the SOC and labile pools.
    km_soc, km_labile : float
        Michaelis-Menten half-saturation constants (g C m^-2) for the
        SOC and labile pools.
    Aa, Ab : float
        Multiplicative scaling for above-ground and below-ground
        autotrophic respiration, respectively.
    Ea_above, Ea_root : float
        Activation energy (J mol^-1) for above-ground and root
        autotrophic respiration.
    percent_avail_soc, percent_avail_labile : float
        Fraction of each pool accessible to enzymatic decomposition on a
        given day (0–1).
    r_mic : float
        Per-cm scaling for the water-table reduction on microbial
        respiration when WT < 0.
    """

    alpha_soc: float
    ea_soc: float
    km_soc: float
    alpha_labile: float
    ea_labile: float
    km_labile: float
    Aa: float
    Ab: float
    Ea_above: float
    Ea_root: float
    percent_avail_soc: float
    percent_avail_labile: float
    r_mic: float

    @classmethod
    def from_list(cls, values: Sequence[float]) -> "NEEParameters":
        """Build from a positional list in the documented order."""
        if len(values) != 13:
            raise ValueError(
                f"NEEParameters.from_list expects 13 values, got {len(values)}."
            )
        return cls(*values)

    @classmethod
    def defaults(cls) -> "NEEParameters":
        """Default parameter values from the calibrated US-HRA 2015 posterior.

        Source: ``PEPRMT-Rice_v0`` MCMC posterior mean for HRA 2015 (the
        ``mod`` parameter set). Substitute your own site-calibrated values
        for publication-quality predictions.
        """
        return cls(
            alpha_soc=2971.9, ea_soc=16881.3, km_soc=0.161,
            alpha_labile=2291.7, ea_labile=51345.4, km_labile=7.53,
            Aa=175.5, Ab=70.97,
            Ea_above=20768.9, Ea_root=30057.0,
            percent_avail_soc=0.058, percent_avail_labile=0.245,
            r_mic=0.0136,
        )

    def to_array(self) -> np.ndarray:
        return np.array(list(self.asdict().values()))

    def asdict(self) -> dict:
        return asdict(self)


class NEEModel:
    """DAMM-based ecosystem respiration + NEE model.

    The model assumes a single canonical parameter set (see
    :class:`NEEParameters`). Water-table suppression of microbial respiration
    can be toggled at construction time via ``water_suppress``.

    Parameters
    ----------
    params : NEEParameters or sequence of 13 floats
        Calibration parameters.
    water_suppress : bool, default True
        If True, microbial respiration is reduced when the water table drops
        below the soil surface (WT < 0).
    burning_rate : float, default 0.9
        Fraction of remaining above-ground biomass that is removed by burning
        at end of season (used in the carbon accounting).

    Notes
    -----
    The model is intentionally stateless apart from its parameters. Multiple
    runs over different years/sites can reuse a single instance.
    """

    def __init__(
        self,
        params: Union[NEEParameters, Sequence[float]],
        *,
        water_suppress: bool = True,
        burning_rate: float = 0.9,
    ):
        if isinstance(params, NEEParameters):
            self.params = params
        else:
            self.params = NEEParameters.from_list(params)
        self.water_suppress = water_suppress
        self.burning_rate = burning_rate
        # Initial pool sizes (g C m^-2); fixed structural defaults.
        self.soc_pool_init = 0.38 * 0.156 * 1e6
        self.labile_pool_init = 0.0

    # ---- helpers -----------------------------------------------------------------

    def _phenology_partition(self, gpp_pos: np.ndarray) -> tuple[np.ndarray, ...]:
        """Smooth GPP and derive root/leaf/stem/grain allocation fractions
        using the BESS-Rice carbon-allocation scheme (Jeong et al. 2018).

        Steps:

        1. Despike and Savitzky-Golay-smooth positive GPP, giving ``smoothed``.
        2. Normalise the smoothed series by its peak to get
           ``gpp_ratio`` ≡ Normalised Accumulated GPP (NAGPP) — the BESS-Rice
           phenology proxy.
        3. Apply the published BESS-Rice partition functions:
             - root  : 0.35 * exp(-2.13 * NAGPP) + 0.003 * exp(3.23 * NAGPP)
             - leaf  : 0.46 * exp(-((NAGPP - 0.02) / 0.74)^2)
             - stem  : 0.46 * exp(-((NAGPP - 0.35) / 0.74)^2)
             - grain : 0.57 * exp(-((NAGPP - 0.98) / 0.35)^2)

        Downstream code shifts the curves to be non-negative and renormalises
        them to sum to 1 — see :meth:`estimate`.

        Reference: Jeong, S., et al. (2018). BESS-Rice. *Agric. For. Meteorol.*
        256-257, 11-25.
        """
        # ---- (1) despike + smooth ------------------------------------------------
        thresh_top = gpp_pos.mean() + 2 * gpp_pos.std()
        thresh_bot = gpp_pos.mean() - 2 * gpp_pos.std()
        cleaned = np.where(
            (gpp_pos > thresh_bot) & (gpp_pos < thresh_top),
            gpp_pos,
            np.nan,
        )
        cleaned = (
            pd.Series(cleaned).infer_objects().interpolate(method="spline", order=3).to_numpy()
        )
        smoothed = savgol_filter(cleaned, 30, 2)

        # ---- (2) Normalised Accumulated GPP (NAGPP) — BESS-Rice phenology proxy --
        gpp_ratio = smoothed / np.nanmax(smoothed)

        # ---- (3) BESS-Rice allocation functions ----------------------------------
        p_root = 0.35 * np.exp(-2.13 * gpp_ratio) + 0.003 * np.exp(3.23 * gpp_ratio)
        p_leaf = 0.46 * np.exp(-((gpp_ratio - 0.02) / 0.74) ** 2)
        p_stem = 0.46 * np.exp(-((gpp_ratio - 0.35) / 0.74) ** 2)
        p_grain = 0.57 * np.exp(-((gpp_ratio - 0.98) / 0.35) ** 2)
        return smoothed, p_root, p_leaf, p_stem, p_grain

    # ---- main entry point --------------------------------------------------------

    def estimate(
        self,
        dates: np.ndarray,
        air_temp_C: np.ndarray,
        water_table_cm: np.ndarray,
        season: np.ndarray,
        harvest_g_m2: np.ndarray,
        gpp: np.ndarray,
    ) -> dict:
        """Run the NEE / Reco model for a daily time series.

        Parameters
        ----------
        dates : array-like of date or datetime64
            Observation dates (one per day).
        air_temp_C : array-like, °C
        water_table_cm : array-like, cm (positive = above soil surface)
        season : array-like, int
            Phenology stage code per day (5 = senescence/post-harvest, etc.).
        harvest_g_m2 : array-like, g biomass m^-2
            Harvest events as biomass removed on the given date (0 elsewhere).
        gpp : array-like, g C m^-2 d^-1
            Daily GPP (negative = uptake) — typically from
            :class:`peprmt_rice.models.gpp.GPPModel`.

        Returns
        -------
        dict
            Keys include ``NEE``, ``Reco``, ``SOC_left``, ``labile_left``,
            ``GPP_aboveground``, ``GPP_belowground``, ``p_root``, ``p_leaf``,
            ``p_stem``, ``p_grain``, and other intermediate diagnostics.
            All time series have the same length as ``dates``.
        """
        air_temp_C = np.asarray(air_temp_C, dtype=float)
        WT = np.asarray(water_table_cm, dtype=float)
        season = np.asarray(season)
        harvest = np.asarray(harvest_g_m2, dtype=float)
        gpp = np.asarray(gpp, dtype=float)
        time_step = len(gpp)

        # convert harvest mass to C
        harvest_C = harvest * 0.87 * 0.40  # g C m^-2 d^-1

        # enzyme reaction velocity maxima — all activation energies in J/mol.
        RT = GAS_CONSTANT * (air_temp_C + 273.15)  # J mol^-1
        vmax_soc = self.params.alpha_soc * np.exp(-self.params.ea_soc / RT)
        vmax_labile = self.params.alpha_labile * np.exp(-self.params.ea_labile / RT)

        # water-table reduction on microbial respiration
        if self.water_suppress:
            red_mic = np.ones(time_step)
            mask_mid = (WT >= -10) & (WT < 0)
            mask_low = WT < -10
            red_mic[mask_mid] = 1 - self.params.r_mic * (-WT[mask_mid])
            red_mic[mask_low] = 1 - self.params.r_mic * 10
        else:
            red_mic = np.ones(time_step)

        # phenology-driven allocation
        gpp_pos = -gpp
        smoothed, p_root, p_leaf, p_stem, p_grain = self._phenology_partition(gpp_pos)

        # adjust grain partition for harvest distribution
        total_harvest = harvest_C.sum()
        if total_harvest > 0:
            p_grain = np.asarray(p_grain * (1 - harvest_C / total_harvest), dtype=float)

        min_p = min(p_root.min(), p_leaf.min(), p_stem.min(), p_grain.min())
        if min_p < 0:
            shift = -min_p + 1e-4
            p_root, p_leaf, p_stem, p_grain = (p + shift for p in (p_root, p_leaf, p_stem, p_grain))
        total = p_root + p_leaf + p_stem + p_grain
        p_root, p_leaf, p_stem, p_grain = (p / total for p in (p_root, p_leaf, p_stem, p_grain))

        grain = smoothed * p_grain
        leaves = smoothed * p_leaf
        stems = smoothed * p_stem
        straw = smoothed * (p_stem + p_leaf)
        root = smoothed * p_root

        burn = np.zeros_like(harvest_C)
        burn[-1] = self.burning_rate

        gpp_above_acc = (grain.cumsum() + straw.cumsum()
                         - harvest_C.cumsum() - straw.cumsum() * burn)
        gpp_below_acc = root.cumsum()
        gpp_above_acc = np.where(gpp_above_acc < 0, 0, gpp_above_acc)

        gpp_above = grain + straw
        gpp_below = root

        # autotrophic respiration
        Ra_root = self.params.Ab * gpp_below * np.exp(-self.params.Ea_root / RT)
        Ra_above = self.params.Aa * gpp_above * np.exp(-self.params.Ea_above / RT)

        npp_above_acc = np.maximum(gpp_above_acc - Ra_above, 0.0)
        npp_below = np.maximum(gpp_below - Ra_root, 0.0)

        # day-by-day pool evolution
        SOC_total = np.zeros(time_step)
        labile_total = np.zeros(time_step)
        SOC_left = np.zeros(time_step)
        labile_left = np.zeros(time_step)
        Reco_soc = np.zeros(time_step)
        Reco_labile = np.zeros(time_step)

        harvest_idx = set(np.where(harvest_C > 0)[0])
        for t in range(time_step):
            if t == 0:
                SOC_total[t] = self.soc_pool_init
                labile_total[t] = self.labile_pool_init
            else:
                SOC_total[t] = SOC_left[t - 1]
                labile_total[t] = labile_left[t - 1]

            if t not in harvest_idx:
                SOC_total[t] *= self.params.percent_avail_soc
                labile_total[t] = (npp_below[t] * 0.4 + labile_total[t]) * self.params.percent_avail_labile
            else:
                SOC_total[t] = ((npp_below[t] * 0.6 + npp_above_acc[t] * 0.9 + SOC_total[t])
                                * self.params.percent_avail_soc)
                labile_total[t] = ((npp_below[t] * 0.4 + npp_above_acc[t] * 0.1 + labile_total[t])
                                   * self.params.percent_avail_labile)

            Reco_soc[t] = max(0.0, vmax_soc[t] * SOC_total[t]
                              / (self.params.km_soc + SOC_total[t])) * red_mic[t]
            Reco_labile[t] = max(0.0, vmax_labile[t] * labile_total[t]
                                 / (self.params.km_labile + labile_total[t])) * red_mic[t]

            if t == 0:
                SOC_left[t] = SOC_total[t] - Reco_soc[t]
                labile_left[t] = labile_total[t] - Reco_labile[t]
            else:
                SOC_left[t] = SOC_left[t - 1] - Reco_soc[t]
                labile_left[t] = labile_left[t - 1] - Reco_labile[t]
            SOC_left[t] = max(0.0, SOC_left[t])
            labile_left[t] = max(0.0, labile_left[t])

            # Senescence transfer of labile C into SOC at end of season.
            if season[t] >= 5:
                SOC_left[t] += labile_left[t]
                labile_left[t] = 0.0

        Reco = Reco_soc + Reco_labile + Ra_above + Ra_root
        NEE = gpp + Reco  # GPP is negative, Reco is positive

        return {
            "NEE": NEE,
            "Reco": Reco,
            "Reco_soc": Reco_soc,
            "Reco_labile": Reco_labile,
            "Ra_above": Ra_above,
            "Ra_root": Ra_root,
            "SOC_left": SOC_left,
            "labile_left": labile_left,
            "SOC_total": SOC_total,
            "labile_total": labile_total,
            "GPP_aboveground": gpp_above,
            "GPP_belowground": gpp_below,
            "p_root": p_root,
            "p_leaf": p_leaf,
            "p_stem": p_stem,
            "p_grain": p_grain,
            "GPP_smoothed": smoothed,
        }

    def __repr__(self) -> str:
        return f"NEEModel({self.params!r}, water_suppress={self.water_suppress})"
