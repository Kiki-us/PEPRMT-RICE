"""
Rice-paddy CH4 emissions model.

Physics summary
---------------
1. Methane is produced in the anaerobic zone of the soil column from two
   substrate pools (SOC and labile carbon) following dual-Arrhenius +
   Michaelis-Menten kinetics.
2. Part of the produced methane partitions into a gas phase that is
   transported to the atmosphere via plant aerenchyma and hydrodynamic
   diffusion; the remainder stays as dissolved CH4 and acetate.
3. Acetate is converted to CH4 via acetoclastic methanogenesis.
4. Methane is oxidised both anaerobically and aerobically through
   plant rhizosphere, with the oxidation rate modulated by GPP and
   temperature.
5. Episodic events — ebullition under slow drying, runoff during drainage,
   and oxidation under irrigation — redistribute CH4 between the aerobic and
   anaerobic zones according to a calibrated skew-normal vertical PDF.

The public API is :class:`CH4Model` with :class:`CH4Parameters`. The internal
helpers are preserved from the research codebase (see
``ch4Estimator_Tang_all4_copy.py`` in the original repository) and are kept
as module-private functions so the physics is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import skewnorm
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler

try:
    from scipy.integrate import simpson as _simps
except ImportError:  # SciPy < 1.6
    from scipy.integrate import simps as _simps  # type: ignore


# ---------------------------------------------------------------------------
# Fixed structural constants (not calibrated)
# ---------------------------------------------------------------------------

_PARA_FIXED = {
    "M_ea1": 65 * 1000,    # J mol^-1, activation energy for SOC methanogenesis
    "M_ea2": 69 * 1000,    # J mol^-1, activation energy for labile methanogenesis
    "M_km1": 19,           # half-saturation constant for SOC pool
    "M_km2": 8.5,          # half-saturation constant for labile pool
    "K_gas": 0.3,          # fraction of produced CH4 entering gas phase
    "K_acetate": 0.7,      # fraction of produced CH4 entering acetate phase
}

# Vertical depth grid (cm) over which the methanogenesis / runoff /
# ebullition integrations are computed.  The default below is the "ark"
# grid ``linspace(-50, 50, 500)`` — matching ``ch4Estimator_Tang_all4_copy.py``
# line 49 in the research codebase that generated the paper's calibration
# outputs, and the grid against which ``CH4Parameters.defaults`` was
# calibrated.
#
# Site-specific override — Twitchell Island and similar deep-water-table sites
# ------------------------------------------------------------------------
# Twitchell-style sites (e.g. US-TW3) routinely see water-table excursions
# beyond the ±50 cm window the ark grid covers, which clips the substrate
# distribution and truncates ebullition integrations.  For those sites pass
# the wider "twt" grid through the constructor:
#
#     model = CH4Model(
#         params=my_calibrated_params,            # NOT CH4Parameters.defaults
#         depth_range=np.linspace(-100, 100, 1000),
#     )
#
# Important — changing the depth grid changes:
#   * ``height_anerobic`` (= ``min(height_all, target_aerobic_soil − depth_range.min())``)
#     — drops from 95 cm (twt) to 45 cm (ark) or vice versa, and
#   * the substrate-normalisation denominator ``np.abs(depth_range).sum()``
#     — 50050 (twt) vs 12500 (ark), a 4× shift.
# Both feed every downstream quantitative result.  Empirically the two
# grids differ by ~35 % on cumulative seasonal CH4 at fixed parameters
# (see ``audit/DIAGNOSTIC_public_reproducibility.md`` Test 1 in the
# project audit notes), so **a twt-grid run with ark-tuned parameters is
# not physically meaningful** — recalibrate parameters on the matching
# grid before reporting results.
DEFAULT_DEPTH_RANGE = np.linspace(-50, 50, 500)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class CH4Parameters:
    """Calibratable parameters for the rice-paddy CH4 model.

    Attributes
    ----------
    M_alpha1, M_alpha2 : float
        Pre-exponential factor for SOC and labile-pool methanogenesis
        (Arrhenius alpha; scaled internally by 1e12).
    K_act2gas : float
        Fraction of dissolved acetate converted to CH4 per day.
    Kr : float
        Rate constant for CH4 oxidation through the plant rhizosphere
        (scales with GPP and temperature).
    Kp : float
        Rate constant for plant-mediated CH4 transport (scales with GPP).
    Kh : float
        Rate constant for hydrodynamic CH4 transport (scales with TA).
    K_Eh : float
        Rate constant for anaerobic CH4 oxidation.
    """
    M_alpha1: float
    M_alpha2: float
    K_act2gas: float
    Kr: float
    Kp: float
    Kh: float
    K_Eh: float

    @classmethod
    def from_list(cls, values: Sequence[float]) -> "CH4Parameters":
        if len(values) != 7:
            raise ValueError(
                f"CH4Parameters.from_list expects 7 values, got {len(values)}."
            )
        return cls(*values)

    @classmethod
    def defaults(cls) -> "CH4Parameters":
        """Verified paper posterior — US-HRA 2016 (true scenario).

        Values taken byte-for-byte from row ``year==2016`` of::

            methane_model/output_0405/outputs_allyear_Twt_final/
              CH4_parameters(paral(7k))/hra_ch4_parameters(true).csv

        That row is the posterior mean of the 7 000-iteration DRAM chain
        the paper run produced for the US-HRA AmeriFlux tower in 2016
        (with the corresponding chain pickle at
        ``CH4_MCMC_v4(paral(7k))/CH4_mcmc_results_hra_2016_true.pkl``).

        These values reproduce paper ``CH4_total_true`` for HRA 2016 to
        machine epsilon when run on the default ark depth grid and the
        paper-converted input (see ``scripts/convert_paper_input.py``).
        See ``tests/test_paper_replication.py`` and
        ``scripts/replicate_paper_hra_2016.sh`` for the end-to-end check.

        For sites other than HRA, recalibrate via
        ``peprmt-rice run --config configs/example_mcmc.yaml`` and load
        the resulting posterior means via ``CH4Parameters(**posterior)``.
        """
        return cls(
            # row hra_ch4_parameters(true).csv, year==2016, full float64 precision
            M_alpha1=0.0176298946581885,
            M_alpha2=0.2609431838966145,
            K_act2gas=0.6156160242203926,
            Kr=0.1259719094787869,
            Kp=0.8170217461963778,
            Kh=0.1299280032790325,
            K_Eh=0.6798033838574522,
        )

    def to_array(self) -> np.ndarray:
        return np.array([self.M_alpha1, self.M_alpha2, self.K_act2gas,
                         self.Kr, self.Kp, self.Kh, self.K_Eh])

    def asdict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal physics helpers (preserved from the research codebase)
# ---------------------------------------------------------------------------

def _preprocess(
    dates, air_temp_C, water_table_cm, day_of_planting,
    soc, labile, gpp, gpp_above, gpp_below,
    depth_range=None,
):
    """Bundle the inputs into the dict used by the inner loop.

    Parameters
    ----------
    depth_range : array-like or None, optional
        Vertical grid (cm) over which the methanogenesis / runoff / ebullition
        integrations are computed.  Defaults to the paper-matching ark grid
        ``np.linspace(-50, 50, 500)`` (see ``DEFAULT_DEPTH_RANGE`` at module
        top).  Pass ``np.linspace(-100, 100, 1000)`` for the wider Twitchell
        ("twt") grid.
    """
    TA = np.asarray(air_temp_C, dtype=float)
    WT = np.asarray(water_table_cm, dtype=float)
    date_disc = pd.to_datetime(dates).date
    doy_disc = pd.to_datetime(date_disc).dayofyear.values
    dop_disc = pd.to_datetime(day_of_planting)
    soc_org = np.asarray(soc, dtype=float).copy()
    labile_org = np.asarray(labile, dtype=float).copy()
    gpp = np.asarray(gpp, dtype=float)
    gpp_aboveground = np.asarray(gpp_above, dtype=float) * (-1)
    gpp_belowground = np.asarray(gpp_below, dtype=float) * (-1)
    gpp_pos = gpp * (-1)
    gpp_acc = np.cumsum(gpp_pos)
    gpp_acc_zscore = MinMaxScaler().fit_transform(gpp_acc.reshape(-1, 1)).flatten()
    gpp_zscore = MinMaxScaler().fit_transform(gpp_pos.reshape(-1, 1)).flatten()
    TA_std = MinMaxScaler().fit_transform(TA.reshape(-1, 1)).flatten()

    depth_range = (
        DEFAULT_DEPTH_RANGE if depth_range is None
        else np.asarray(depth_range, dtype=float)
    )
    target_peak = -10
    target_aerobic_soil = -5

    height_all = WT - depth_range.min()
    height_aerobic_innudated = np.where(
        WT <= target_aerobic_soil,
        np.abs(target_aerobic_soil),
        np.minimum(WT - target_aerobic_soil, np.abs(target_aerobic_soil)),
    )
    height_aerobic_uninnudated = np.abs(target_aerobic_soil) - height_aerobic_innudated
    height_aerobic = [height_aerobic_innudated, height_aerobic_uninnudated]
    height_anerobic = np.minimum(height_all, target_aerobic_soil - depth_range.min())

    return {
        "TA": TA, "TA_std": TA_std, "WT": WT,
        "date_disc": date_disc, "doy_disc": doy_disc, "dop_disc": dop_disc,
        "soc_org": soc_org, "labile_org": labile_org,
        "gpp": gpp, "gpp_pos": gpp_pos,
        "gpp_aboveground": gpp_aboveground, "gpp_belowground": gpp_belowground,
        "gpp_acc": gpp_acc, "gpp_acc_zscore": gpp_acc_zscore,
        "gpp_zscore": gpp_zscore,
        "depth_range": depth_range, "target_peak": target_peak,
        "target_aerobic_soil": target_aerobic_soil,
        "height_all": height_all, "height_aerobic": height_aerobic,
        "height_anerobic": height_anerobic,
    }


def _custom_skewed_pdf(depths, wtd, peak, target_peak,
                        sigma=5, alpha=5, scale_left=1.5, scale_right=0.7):
    decay_factor_right = 0.001
    decay_factor_left = 1
    pdf = skewnorm.pdf(depths, a=alpha, loc=peak, scale=sigma)
    right_decay = np.exp(-decay_factor_right * (depths + peak))
    pdf = np.where(
        (depths > peak) & (depths < 100),
        pdf * scale_left,
        pdf,
    )
    pdf = np.where(
        (-100 <= depths) & (depths <= peak),
        pdf * scale_right * right_decay,
        pdf,
    )
    norm = _simps(pdf, depths)
    return pdf / norm if norm > 0 else pdf


def _calibrate_skewed_pdf(depths, wtd, target_peak,
                          initial_max_depth=-5, initial_sigma=50, initial_alpha=-5,
                          scale_left=1, scale_right=1):
    def objective(params):
        sigma, alpha, peak = params
        pdf = _custom_skewed_pdf(depths, wtd, peak=peak, target_peak=target_peak,
                                 sigma=sigma, alpha=alpha,
                                 scale_left=scale_left, scale_right=scale_right)
        max_depth = depths[np.argmax(pdf)]
        return abs(max_depth - target_peak) + np.std(np.diff(pdf))
    result = minimize(
        objective,
        x0=[initial_sigma, initial_alpha, initial_max_depth],
        bounds=[(1, 100), (-50, 50), (-25, 0)],
        method="Powell",
    )
    sigma_opt, alpha_opt, peak = result.x
    pdf = _custom_skewed_pdf(depths, wtd, peak, target_peak,
                             sigma=sigma_opt, alpha=alpha_opt,
                             scale_left=scale_left, scale_right=scale_right)
    return pdf, sigma_opt, alpha_opt, peak


def _integrate_pdf(pdf, depths, range_min, range_max):
    mask = (depths >= range_min) & (depths <= range_max)
    return _simps(pdf[mask], depths[mask])


def _methanogenesis(S1, S2, p, RT):
    M1 = np.maximum(0, p["M_alpha1"] * 1e12 * np.exp(-p["M_ea1"] / RT) * S1 / (p["M_km1"] + S1))
    M2 = np.maximum(0, p["M_alpha2"] * 1e12 * np.exp(-p["M_ea2"] / RT) * S2 / (p["M_km2"] + S2))
    return M1, M2


def _update_pools(t, soc_org, labile_org, S1sol, S2sol):
    if t == 0:
        return soc_org[t], labile_org[t]
    return S1sol[t] + soc_org[t - 1], S2sol[t] + labile_org[t - 1]


def _transport_rates(p, gpp_zscore, gpp_acc_zscore, TA_std):
    Vplant = p["Kp"] * gpp_zscore * gpp_acc_zscore
    Vhydro = p["Kh"] * TA_std
    Oxi_rate_plant = p["Kr"] * gpp_acc_zscore * TA_std
    return Vplant, Vhydro, Oxi_rate_plant


def _update_ch4_pools(t, M_full, acetate, M_remain, A_remain, p):
    if t == 0:
        M_remain[t] = M_full[t]
        A_remain[t] = acetate[t]
    else:
        M_remain[t] = M_remain[t - 1] + M_full[t]
        A_remain[t] = A_remain[t - 1] + acetate[t]
    delta = p["K_act2gas"] * A_remain[t]
    M_remain[t] += delta
    A_remain[t] = max(A_remain[t] - delta, 0)
    return max(0, M_remain[t]), max(0, A_remain[t]), delta


def _oxidation(t, M_remain, height_all, Oxi_rate_plant, p):
    R_base = M_remain[t] / height_all[t]
    ox_plant = Oxi_rate_plant[t] * (R_base - 0.0002) * height_all[t]
    ox_anaerobic = p["K_Eh"] * (R_base - 0.0002) * height_all[t]
    M_remain[t] = np.maximum(M_remain[t] - ox_plant - ox_anaerobic, 0)
    return M_remain[t], ox_plant, ox_anaerobic


# ---------------------------------------------------------------------------
# Public model class
# ---------------------------------------------------------------------------

class CH4Model:
    """Rice-paddy CH4 emissions model with named inputs.

    Parameters
    ----------
    params : CH4Parameters or sequence of 7 floats
        Calibratable parameters (see :class:`CH4Parameters`).
    depth_range : array-like or None, optional
        Vertical grid (cm) used for methanogenesis / runoff / ebullition
        integrations.  Defaults to the paper-matching ark(arkansas) grid
        ``np.linspace(-50, 50, 500)``.  Pass ``np.linspace(-100, 100, 1000)``
        for the Twitchell Island ("twt") grid.  Changing this grid affects
        every quantitative result and is **not** interchangeable with the
        default parameter set tuned for the ark grid — recalibrate if you
        change it.
    """

    def __init__(
        self,
        params: Union[CH4Parameters, Sequence[float]],
        depth_range: Union[np.ndarray, None] = None,
    ):
        if isinstance(params, CH4Parameters):
            self.params = params
        else:
            self.params = CH4Parameters.from_list(params)
        self.depth_range = (
            None if depth_range is None else np.asarray(depth_range, dtype=float)
        )

    def estimate(
        self,
        dates,
        air_temp_C,
        water_table_cm,
        day_of_planting,
        soc,
        labile,
        gpp,
        gpp_above,
        gpp_below,
    ) -> dict:
        """Run the CH4 model.

        Parameters
        ----------
        dates : array of date-like
            Observation dates.
        air_temp_C : array, °C
            Daily-mean air temperature.
        water_table_cm : array, cm
            Water table height (positive = above soil surface).
        day_of_planting : scalar or array of date-like
            Date of planting for the (current) season.
        soc, labile : array, g C m^-2
            Soil organic carbon and labile carbon pool sizes per day —
            typically from the NEE model's ``SOC_total`` / ``labile_total``.
        gpp : array, g C m^-2 d^-1
            Daily GPP (negative = uptake).
        gpp_above, gpp_below : array, g C m^-2 d^-1
            Aboveground and belowground GPP partitions — typically from the
            NEE model's ``GPP_aboveground`` / ``GPP_belowground``.

        Returns
        -------
        dict
            Dictionary of daily output time series (CH4 total flux, plant
            transport, hydrodynamic transport, oxidation, pool dynamics,
            etc.).
        """
        inputs = _preprocess(
            dates, air_temp_C, water_table_cm, day_of_planting,
            soc, labile, gpp, gpp_above, gpp_below,
            depth_range=self.depth_range,
        )

        TA = inputs["TA"]
        WT = inputs["WT"]
        soc_org = inputs["soc_org"]
        labile_org = inputs["labile_org"]
        TA_std = inputs["TA_std"]
        gpp_pos = inputs["gpp_pos"]
        gpp_acc_zscore = inputs["gpp_acc_zscore"]
        gpp_zscore = inputs["gpp_zscore"]
        depth_range = inputs["depth_range"]
        target_peak = inputs["target_peak"]
        height_all = inputs["height_all"]
        height_aerobic = inputs["height_aerobic"]
        height_anerobic = inputs["height_anerobic"]
        target_aerobic_soil = inputs["target_aerobic_soil"]

        para_dyn = {
            "M_alpha1": self.params.M_alpha1,
            "M_alpha2": self.params.M_alpha2,
            "K_act2gas": self.params.K_act2gas,
            "Kr": self.params.Kr,
            "Kp": self.params.Kp,
            "Kh": self.params.Kh,
            "K_Eh": self.params.K_Eh,
        }
        para_all = {**para_dyn, **_PARA_FIXED}

        N = len(TA)
        RT = 8.314 * (TA + 273.15).astype(float)

        # Allocate outputs
        S1_avail = np.zeros(N); S2_avail = np.zeros(N)
        M1 = np.zeros(N); M2 = np.zeros(N)
        S1sol = np.zeros(N); S2sol = np.zeros(N)
        M_full = np.zeros(N); M_remain = np.zeros(N)
        acetate = np.zeros(N); A_remain = np.zeros(N)
        CH4water_store = np.zeros(N); CH4water_acetate_store = np.zeros(N)
        R_CH4_base = np.zeros(N); R_actate_base = np.zeros(N)
        CH4_oxidation_total = np.zeros(N)
        CH4_oxidation_through_plant = np.zeros(N)
        CH4_Oxidation_anerobic = np.zeros(N)
        CH4_Oxidation_hydrodynamics = np.zeros(N)
        Hydro_flux = np.zeros(N); Plant_flux = np.zeros(N)
        ch4_burst = np.zeros(N); Hydro_total = np.zeros(N)
        initial_production = np.zeros(N)
        run_off_acetate = np.zeros(N); run_off_ch4 = np.zeros(N)
        delta_acetate_to_CH4gas = np.zeros(N)

        Vplant, Vhydro, Oxi_rate_plant = _transport_rates(
            para_all, gpp_zscore, gpp_acc_zscore, TA_std
        )
        pdf_calibrated, *_ = _calibrate_skewed_pdf(depth_range, 15, target_peak)

        for t in range(N):
            soc_org[t], labile_org[t] = _update_pools(t, soc_org, labile_org, S1sol, S2sol)
            S1_avail[t] = soc_org[t] * height_anerobic[t] / np.abs(depth_range).sum()
            S2_avail[t] = labile_org[t] * height_anerobic[t] / np.abs(depth_range).sum()

            M1[t], M2[t] = _methanogenesis(S1_avail[t], S2_avail[t], para_all, RT[t])
            S1sol[t] = max(0, soc_org[t] - M1[t])
            S2sol[t] = max(0, labile_org[t] - M2[t])
            M_full[t] = (M1[t] + M2[t]) * para_all["K_gas"]
            acetate[t] = para_all["K_acetate"] * (M1[t] + M2[t])
            M_remain[t], A_remain[t], delta_acetate_to_CH4gas[t] = _update_ch4_pools(
                t, M_full, acetate, M_remain, A_remain, para_all
            )
            initial_production[t] = M_remain[t]

            if t >= 1:
                WT_delta = WT[t] - WT[t - 1]
                diff_WT_series = pd.Series(WT).diff()[t - 9: t + 1]

                # Slow drying: ebullition in the aerobic zone.
                if (
                    t >= 30
                    and (np.count_nonzero(diff_WT_series < 0) >= 9)
                    and (diff_WT_series.sum() <= -10)
                    and (np.count_nonzero(np.abs(diff_WT_series) <= 2) >= 9)
                ):
                    try:
                        ch4_burst[t] = (
                            _integrate_pdf(pdf_calibrated, depth_range, target_peak, WT[t - 1])
                            * M_remain[t]
                        )
                        M_remain[t] -= ch4_burst[t]
                    except Exception:
                        pass

                # Drainage: runoff + ebullition.
                if WT_delta < -5:
                    run_off_acetate[t] = (
                        _integrate_pdf(pdf_calibrated, depth_range, WT[t], WT[t - 1]) * A_remain[t]
                    )
                    A_remain[t] -= run_off_acetate[t]
                    if WT[t - 1] > target_aerobic_soil:
                        lower = max(WT[t], target_aerobic_soil)
                        upper = WT[t - 1]
                        step = np.mean(np.diff(depth_range))
                        if upper - lower >= step:
                            ch4_burst[t] = (
                                _integrate_pdf(pdf_calibrated, depth_range, lower, upper)
                                * M_remain[t]
                            )
                    if WT_delta <= -10:
                        ch4_burst[t] += (
                            _integrate_pdf(pdf_calibrated, depth_range, WT[t] - 0.5, WT[t])
                            * M_remain[t]
                        )
                    run_off_ch4[t] = (
                        _integrate_pdf(pdf_calibrated, depth_range, WT[t], WT[t - 1])
                        * M_remain[t]
                    )
                    run_off_ch4[t] -= ch4_burst[t]
                    M_remain[t] = M_remain[t] - ch4_burst[t] - run_off_ch4[t]

                # Irrigation: hydrodynamic oxidation.
                if WT_delta > 0 and WT[t] > target_aerobic_soil:
                    try:
                        CH4_Oxidation_hydrodynamics[t] = (
                            _integrate_pdf(
                                pdf_calibrated, depth_range, target_aerobic_soil,
                                min(WT[t], 0),
                            )
                            * M_remain[t]
                        )
                    except Exception:
                        CH4_Oxidation_hydrodynamics[t] = 0
                    M_remain[t] -= CH4_Oxidation_hydrodynamics[t]

                if WT_delta >= 10:
                    ch4_burst[t] = (
                        _integrate_pdf(pdf_calibrated, depth_range, WT[t] - 0.5, WT[t])
                        * M_remain[t]
                    )
                    M_remain[t] -= ch4_burst[t]

            M_remain[t], CH4_oxidation_through_plant[t], CH4_Oxidation_anerobic[t] = _oxidation(
                t, M_remain, height_all, Oxi_rate_plant, para_all
            )

            R_CH4_base[t] = M_remain[t] / height_all[t]
            R_actate_base[t] = A_remain[t] / height_all[t]
            Plant_flux[t] = Vplant[t] * (R_CH4_base[t] - 0.0002) * height_all[t]
            Hydro_flux[t] = Vhydro[t] * (R_CH4_base[t] - 0.0002) * height_all[t]
            M_remain[t] -= Plant_flux[t] + Hydro_flux[t]
            Hydro_total[t] = Hydro_flux[t] + ch4_burst[t]

            CH4_oxidation_total[t] = (
                CH4_Oxidation_anerobic[t]
                + CH4_oxidation_through_plant[t]
                + CH4_Oxidation_hydrodynamics[t]
            )
            CH4water_store[t] = M_remain[t]
            CH4water_acetate_store[t] = A_remain[t]

        pulse_emission_total = Plant_flux + Hydro_total

        return {
            "CH4_total": pulse_emission_total,
            "Plant_flux": Plant_flux,
            "Hydro_total": Hydro_total,
            "CH4_oxidation_total": CH4_oxidation_total,
            "CH4_oxidation_anaerobic": CH4_Oxidation_anerobic,
            "CH4_oxidation_plant": CH4_oxidation_through_plant,
            "CH4_oxidation_hydro": CH4_Oxidation_hydrodynamics,
            "ch4_burst": ch4_burst,
            "M_full": M_full,
            "M1": M1,
            "M2": M2,
            "M_remain": M_remain,
            "CH4water_store": CH4water_store,
            "soc": soc_org,
            "labile": labile_org,
            "S1_avail": S1_avail,
            "S2_avail": S2_avail,
            "run_off_ch4": run_off_ch4,
            "run_off_acetate": run_off_acetate,
            "initial_production": initial_production,
        }

    def __repr__(self) -> str:
        return f"CH4Model({self.params!r})"
