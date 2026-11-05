"""
Real-Fluid Cryogenic Thermophysical Property Tables & Equations of State.

Covers NIST REFPROP-calibrated correlations for:
  - LN2  (Liquid Nitrogen, N2)   — Benchtop & HIL validation cryogen
  - LOX  (Liquid Oxygen, O2)     — Orbital depot oxidizer
  - LCH4 (Liquid Methane, CH4)   — Methalox orbital depot fuel
  - LH2  (Liquid Hydrogen, H2)   — High-Isp upper-stage fuel (with ortho-to-para exothermic conversion)

Includes:
  - Wagner saturation curve P_sat(T) and inverse T_sat(P)
  - Watson temperature-dependent latent heat of vaporization h_fg(T)
  - Rackett saturated liquid density rho_l(T) and real-gas vapor density rho_v(T, P)
  - Van der Waals / Guggenheim surface tension sigma(T)
  - Wood / Jakob two-phase acoustic speed of sound a_tp(alpha, T, P)
  - Joule-Thomson isenthalpic throttling cooling performance
  - Ortho-to-para hydrogen spin-isomer conversion equilibrium & exothermic heat load
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Dict

import numpy as np

R_UNIVERSAL = 8.314462618  # J/(mol*K)
P_ATM = 101325.0           # Pa


class CryoFluidType(str, Enum):
    """Supported aerospace cryogenic fluids."""

    LN2 = "LN2"
    LOX = "LOX"
    LCH4 = "LCH4"
    LH2 = "LH2"


@dataclass(frozen=True)
class CryoFluidProperties:
    """Reference thermophysical constants and correlation coefficients for a cryogen."""

    name: str
    formula: str
    molar_mass_kg_mol: float
    t_triple_k: float
    p_triple_pa: float
    t_nbp_k: float
    t_crit_k: float
    p_crit_pa: float
    rho_crit_kg_m3: float
    acentric_factor: float

    # Reference properties at normal boiling point (101.325 kPa)
    rho_l_nbp: float          # kg/m^3
    rho_v_nbp: float          # kg/m^3
    h_fg_nbp_j_kg: float      # J/kg
    cp_l_nbp_j_kgk: float     # J/(kg*K)
    cp_v_nbp_j_kgk: float     # J/(kg*K)
    mu_l_nbp_pas: float       # Pa*s
    mu_v_nbp_pas: float       # Pa*s
    k_l_nbp_w_mk: float       # W/(m*K)
    k_v_nbp_w_mk: float       # W/(m*K)
    sigma_0_n_m: float        # N/m (Guggenheim prefactor)
    sigma_n: float            # Guggenheim exponent
    sound_speed_l_nbp: float  # m/s

    # Wagner saturation pressure coefficients (a1, a2, a3, a4) for ln(P/Pc) = (Tc/T)*(a1*tau + a2*tau^1.5 + a3*tau^2.5 + a4*tau^5)
    wagner_a1: float
    wagner_a2: float
    wagner_a3: float
    wagner_a4: float

    # Joule-Thomson expansion reference coefficient (K/MPa) near saturation liquid/two-phase throttling
    jt_coeff_k_mpa: float
    ortho_para_heat_j_kg: float = 0.0  # Specific ortho->para conversion heat release (J/kg of ortho converted)


FLUID_DATABASE: Dict[CryoFluidType, CryoFluidProperties] = {
    CryoFluidType.LN2: CryoFluidProperties(
        name="Liquid Nitrogen",
        formula="N2",
        molar_mass_kg_mol=0.0280134,
        t_triple_k=63.151,
        p_triple_pa=12520.0,
        t_nbp_k=77.355,
        t_crit_k=126.192,
        p_crit_pa=3395800.0,
        rho_crit_kg_m3=313.3,
        acentric_factor=0.0372,
        rho_l_nbp=806.1,
        rho_v_nbp=4.614,
        h_fg_nbp_j_kg=199180.0,
        cp_l_nbp_j_kgk=2041.5,
        cp_v_nbp_j_kgk=1123.0,
        mu_l_nbp_pas=1.58e-4,
        mu_v_nbp_pas=5.44e-6,
        k_l_nbp_w_mk=0.1396,
        k_v_nbp_w_mk=0.0072,
        sigma_0_n_m=0.02906,
        sigma_n=1.250,
        sound_speed_l_nbp=852.0,
        wagner_a1=-6.12445,
        wagner_a2=1.26327,
        wagner_a3=-1.35967,
        wagner_a4=-1.75580,
        jt_coeff_k_mpa=2.15,
        ortho_para_heat_j_kg=0.0,
    ),
    CryoFluidType.LOX: CryoFluidProperties(
        name="Liquid Oxygen",
        formula="O2",
        molar_mass_kg_mol=0.0319988,
        t_triple_k=54.361,
        p_triple_pa=146.3,
        t_nbp_k=90.188,
        t_crit_k=154.581,
        p_crit_pa=5043000.0,
        rho_crit_kg_m3=436.1,
        acentric_factor=0.0222,
        rho_l_nbp=1141.2,
        rho_v_nbp=4.467,
        h_fg_nbp_j_kg=213050.0,
        cp_l_nbp_j_kgk=1699.0,
        cp_v_nbp_j_kgk=970.0,
        mu_l_nbp_pas=1.95e-4,
        mu_v_nbp_pas=6.95e-6,
        k_l_nbp_w_mk=0.1499,
        k_v_nbp_w_mk=0.0082,
        sigma_0_n_m=0.03843,
        sigma_n=1.222,
        sound_speed_l_nbp=905.0,
        wagner_a1=-6.05960,
        wagner_a2=1.28105,
        wagner_a3=-1.11606,
        wagner_a4=-1.85990,
        jt_coeff_k_mpa=1.88,
        ortho_para_heat_j_kg=0.0,
    ),
    CryoFluidType.LCH4: CryoFluidProperties(
        name="Liquid Methane",
        formula="CH4",
        molar_mass_kg_mol=0.0160425,
        t_triple_k=90.694,
        p_triple_pa=11696.0,
        t_nbp_k=111.667,
        t_crit_k=190.564,
        p_crit_pa=4599200.0,
        rho_crit_kg_m3=162.66,
        acentric_factor=0.0114,
        rho_l_nbp=422.36,
        rho_v_nbp=1.816,
        h_fg_nbp_j_kg=510830.0,
        cp_l_nbp_j_kgk=3481.0,
        cp_v_nbp_j_kgk=2218.0,
        mu_l_nbp_pas=1.17e-4,
        mu_v_nbp_pas=4.48e-6,
        k_l_nbp_w_mk=0.1839,
        k_v_nbp_w_mk=0.0117,
        sigma_0_n_m=0.03857,
        sigma_n=1.215,
        sound_speed_l_nbp=1335.0,
        wagner_a1=-6.02242,
        wagner_a2=1.26652,
        wagner_a3=-0.92069,
        wagner_a4=-1.52360,
        jt_coeff_k_mpa=2.45,
        ortho_para_heat_j_kg=0.0,
    ),
    CryoFluidType.LH2: CryoFluidProperties(
        name="Liquid Hydrogen (Para-H2)",
        formula="H2",
        molar_mass_kg_mol=0.00201588,
        t_triple_k=13.803,
        p_triple_pa=7042.0,
        t_nbp_k=20.271,
        t_crit_k=32.938,
        p_crit_pa=1285800.0,
        rho_crit_kg_m3=31.26,
        acentric_factor=-0.219,
        rho_l_nbp=70.85,
        rho_v_nbp=1.338,
        h_fg_nbp_j_kg=445600.0,
        cp_l_nbp_j_kgk=9668.0,
        cp_v_nbp_j_kgk=12150.0,
        mu_l_nbp_pas=1.33e-5,
        mu_v_nbp_pas=1.08e-6,
        k_l_nbp_w_mk=0.1038,
        k_v_nbp_w_mk=0.0169,
        sigma_0_n_m=0.00547,
        sigma_n=1.090,
        sound_speed_l_nbp=1093.0,
        wagner_a1=-4.89789,
        wagner_a2=0.98856,
        wagner_a3=0.69864,
        wagner_a4=-0.52340,
        jt_coeff_k_mpa=3.80,
        ortho_para_heat_j_kg=702900.0,  # 702.9 kJ per kg of ortho-H2 converted to para-H2 (527 kJ/kg for 75% normal H2)
    ),
}


class CryoFluidEOS:
    """
    Real-fluid equation of state and saturation property evaluator for cryogenic fluids.
    """

    def __init__(self, fluid_type: CryoFluidType | str = CryoFluidType.LN2) -> None:
        if isinstance(fluid_type, str):
            fluid_type = CryoFluidType(fluid_type.upper())
        self.fluid_type: CryoFluidType = fluid_type
        self.props: CryoFluidProperties = FLUID_DATABASE[fluid_type]
        self.r_specific: float = R_UNIVERSAL / self.props.molar_mass_kg_mol

        # Compute exact anchor correction at T_nbp so P_sat(T_nbp) == 101325.0 Pa exactly
        self._wagner_offset: float = 0.0
        p_raw_nbp = self._raw_wagner_p_sat(self.props.t_nbp_k)
        self._wagner_scale: float = P_ATM / p_raw_nbp

    def _raw_wagner_p_sat(self, t_k: float) -> float:
        p = self.props
        t_clamped = float(np.clip(t_k, p.t_triple_k * 0.9, p.t_crit_k * 0.9999))
        tau = 1.0 - t_clamped / p.t_crit_k
        ln_pr = (p.t_crit_k / t_clamped) * (
            p.wagner_a1 * tau
            + p.wagner_a2 * (tau ** 1.5)
            + p.wagner_a3 * (tau ** 2.5)
            + p.wagner_a4 * (tau ** 5.0)
        )
        return p.p_crit_pa * math.exp(ln_pr)

    def saturation_pressure(self, t_k: float) -> float:
        """
        Compute saturation vapor pressure P_sat(T) in Pascals using the NIST-anchored
        4-term Wagner equation.
        """
        return self._raw_wagner_p_sat(t_k) * self._wagner_scale

    def saturation_temperature(self, p_pa: float) -> float:
        """
        Invert P_sat(T) = p_pa via damped Newton-Raphson with bisection safeguard.
        Returns saturation temperature T_sat in Kelvin.
        """
        p = self.props
        p_clamped = float(np.clip(p_pa, p.p_triple_pa * 0.95, p.p_crit_pa * 0.995))

        # Initial guess from Clausius-Clapeyron around T_nbp
        inv_t = (1.0 / p.t_nbp_k) - (self.r_specific / p.h_fg_nbp_j_kg) * math.log(p_clamped / P_ATM)
        t_est = float(np.clip(1.0 / max(inv_t, 1e-4), p.t_triple_k, p.t_crit_k * 0.998))

        t_low = p.t_triple_k * 0.92
        t_high = p.t_crit_k * 0.999
        for _ in range(25):
            f_val = self.saturation_pressure(t_est) - p_clamped
            if abs(f_val) / p_clamped < 1e-9:
                break
            if f_val > 0.0:
                t_high = t_est
            else:
                t_low = t_est
            dt = 1e-4 * t_est
            df_dt = (self.saturation_pressure(t_est + dt) - self.saturation_pressure(t_est - dt)) / (2.0 * dt)
            if abs(df_dt) > 1e-6:
                t_next = t_est - f_val / df_dt
                if t_low < t_next < t_high:
                    t_est = t_next
                else:
                    t_est = 0.5 * (t_low + t_high)
            else:
                t_est = 0.5 * (t_low + t_high)
        return t_est

    def latent_heat(self, t_k: float) -> float:
        """
        Watson correlation for latent heat of vaporization h_fg(T) [J/kg]:
            h_fg(T) = h_fg(T_nbp) * ((1 - T/T_c) / (1 - T_nbp/T_c))^0.38
        """
        p = self.props
        if t_k >= p.t_crit_k:
            return 0.0
        t_clamped = max(t_k, p.t_triple_k * 0.9)
        tr_num = max(1.0 - t_clamped / p.t_crit_k, 1e-6)
        tr_den = 1.0 - p.t_nbp_k / p.t_crit_k
        return p.h_fg_nbp_j_kg * ((tr_num / tr_den) ** 0.38)

    def liquid_density(self, t_k: float) -> float:
        """
        Saturated liquid density rho_l(T) [kg/m^3] using Rackett-Guggenheim scaling
        anchored at (T_nbp, rho_l_nbp) and (T_c, rho_crit).
        """
        p = self.props
        t_clamped = float(np.clip(t_k, p.t_triple_k * 0.9, p.t_crit_k * 0.999))
        tau = max(1.0 - t_clamped / p.t_crit_k, 1e-6)
        tau_nbp = 1.0 - p.t_nbp_k / p.t_crit_k
        delta_rho_nbp = p.rho_l_nbp - p.rho_crit_kg_m3
        return p.rho_crit_kg_m3 + delta_rho_nbp * ((tau / tau_nbp) ** (1.0 / 3.0))

    def compressibility_factor(self, t_k: float, p_pa: float) -> float:
        """
        Pitzer-type second-virial compressibility factor Z(T, P) for cryogenic vapor.
        """
        p = self.props
        tr = max(t_k / p.t_crit_k, 0.45)
        pr = min(p_pa / p.p_crit_pa, 0.95)
        b0 = 0.083 - 0.422 / (tr ** 1.6)
        b1 = 0.139 - 0.172 / (tr ** 4.2)
        z = 1.0 + (b0 + p.acentric_factor * b1) * (pr / tr)
        return float(np.clip(z, 0.35, 1.05))

    def vapor_density(self, t_k: float, p_pa: float | None = None) -> float:
        """
        Real-gas vapor density rho_v(T, P) [kg/m^3].
        If p_pa is None, evaluates along the saturation curve P = P_sat(T) anchored at T_nbp.
        """
        p = self.props
        if p_pa is None:
            p_eval = self.saturation_pressure(t_k)
        else:
            p_eval = max(p_pa, 100.0)
        z = self.compressibility_factor(t_k, p_eval)
        z_nbp = self.compressibility_factor(p.t_nbp_k, P_ATM)
        rho_ideal_nbp = P_ATM / (z_nbp * self.r_specific * p.t_nbp_k)
        anchor_ratio = p.rho_v_nbp / rho_ideal_nbp
        return anchor_ratio * p_eval / (z * self.r_specific * max(t_k, p.t_triple_k * 0.8))

    def surface_tension(self, t_k: float) -> float:
        """
        Van der Waals - Guggenheim surface tension sigma(T) [N/m]:
            sigma(T) = sigma_0 * (1 - T / T_c)^n
        """
        p = self.props
        if t_k >= p.t_crit_k:
            return 0.0
        t_clamped = max(t_k, p.t_triple_k * 0.9)
        return p.sigma_0_n_m * ((1.0 - t_clamped / p.t_crit_k) ** p.sigma_n)

    def liquid_viscosity(self, t_k: float) -> float:
        """Andrade temperature-dependent liquid dynamic viscosity mu_l(T) [Pa*s]."""
        p = self.props
        t_clamped = float(np.clip(t_k, p.t_triple_k * 0.9, p.t_crit_k * 0.99))
        b_visc = 1.85 * p.t_nbp_k
        return p.mu_l_nbp_pas * math.exp(b_visc * (1.0 / t_clamped - 1.0 / p.t_nbp_k))

    def vapor_viscosity(self, t_k: float) -> float:
        """Sutherland power-law cryogenic vapor dynamic viscosity mu_v(T) [Pa*s]."""
        p = self.props
        t_clamped = max(t_k, p.t_triple_k * 0.8)
        return p.mu_v_nbp_pas * ((t_clamped / p.t_nbp_k) ** 0.78)

    def liquid_conductivity(self, t_k: float) -> float:
        """Saturated liquid thermal conductivity k_l(T) [W/(m*K)]."""
        p = self.props
        t_clamped = float(np.clip(t_k, p.t_triple_k * 0.9, p.t_crit_k * 0.99))
        tau = (1.0 - t_clamped / p.t_crit_k) / (1.0 - p.t_nbp_k / p.t_crit_k)
        return p.k_l_nbp_w_mk * (max(tau, 0.05) ** 0.45)

    def vapor_conductivity(self, t_k: float) -> float:
        """Cryogenic vapor thermal conductivity k_v(T) [W/(m*K)]."""
        p = self.props
        t_clamped = max(t_k, p.t_triple_k * 0.8)
        return p.k_v_nbp_w_mk * ((t_clamped / p.t_nbp_k) ** 0.85)

    def liquid_cp(self, t_k: float) -> float:
        """Saturated liquid specific heat capacity c_{p,l}(T) [J/(kg*K)]."""
        p = self.props
        t_clamped = float(np.clip(t_k, p.t_triple_k * 0.9, p.t_crit_k * 0.96))
        tau = 1.0 - t_clamped / p.t_crit_k
        tau_nbp = 1.0 - p.t_nbp_k / p.t_crit_k
        return p.cp_l_nbp_j_kgk * ((tau_nbp / tau) ** 0.22)

    def vapor_cp(self, t_k: float) -> float:
        """Cryogenic vapor specific heat capacity c_{p,v}(T) [J/(kg*K)]."""
        return self.props.cp_v_nbp_j_kgk * (1.0 + 0.0005 * (t_k - self.props.t_nbp_k))

    def prandtl_liquid(self, t_k: float) -> float:
        """Liquid Prandtl number Pr_l = c_{p,l} * mu_l / k_l."""
        return self.liquid_cp(t_k) * self.liquid_viscosity(t_k) / self.liquid_conductivity(t_k)

    def prandtl_vapor(self, t_k: float) -> float:
        """Vapor Prandtl number Pr_v = c_{p,v} * mu_v / k_v."""
        return self.vapor_cp(t_k) * self.vapor_viscosity(t_k) / self.vapor_conductivity(t_k)

    def two_phase_sound_speed(self, alpha: float, t_k: float, p_pa: float | None = None) -> float:
        """
        Wood's homogeneous equilibrium / Jakob two-phase acoustic speed of sound a_tp(alpha) [m/s]:
            1 / (rho_m * a_tp^2) = alpha / (rho_v * a_v^2) + (1 - alpha) / (rho_l * a_l^2)
        Captures the dramatic acoustic speed drop in two-phase bubbly/slug mixtures and
        its sharp recovery to a_l (~850-1300 m/s) upon vapor collapse (cryo-hammer mechanism).
        """
        p = self.props
        alpha_c = float(np.clip(alpha, 0.0, 1.0))
        rho_l = self.liquid_density(t_k)
        rho_v = self.vapor_density(t_k, p_pa)
        a_l = p.sound_speed_l_nbp * math.sqrt(max(rho_l / p.rho_l_nbp, 0.2))
        gamma_v = 1.40
        a_v = math.sqrt(gamma_v * self.r_specific * max(t_k, p.t_triple_k))

        rho_m = alpha_c * rho_v + (1.0 - alpha_c) * rho_l
        comp_tp = (alpha_c / (rho_v * a_v * a_v)) + ((1.0 - alpha_c) / (rho_l * a_l * a_l))
        a_wood = 1.0 / math.sqrt(max(rho_m * comp_tp, 1e-12))
        return float(np.clip(a_wood, 15.0, a_l))

    def joule_thomson_cooling(
        self, p_in_pa: float, p_out_pa: float, t_in_k: float, m_dot_jt_kg_s: float
    ) -> tuple[float, float]:
        """
        Evaluate Joule-Thomson isenthalpic throttling across the TVS expansion valve.
        Returns (t_out_k, q_cooling_watts) where q_cooling_watts is the total refrigeration
        capacity available when the throttled two-phase stream is vaporized in the TVS heat exchanger.
        """
        p = self.props
        delta_p_mpa = max((p_in_pa - p_out_pa) * 1e-6, 0.0)
        t_sat_out = self.saturation_temperature(p_out_pa)
        t_jt_drop = p.jt_coeff_k_mpa * delta_p_mpa
        t_out_k = min(t_sat_out, t_in_k - t_jt_drop)

        # Refrigeration capacity from latent heat at lower saturation pressure + sensible superheat
        h_fg_out = self.latent_heat(t_out_k)
        cp_l = self.liquid_cp(t_in_k)
        subcool_loss = max(0.0, cp_l * (t_in_k - t_out_k) * 0.25)
        specific_cooling_j_kg = max(0.0, h_fg_out - subcool_loss)
        return t_out_k, m_dot_jt_kg_s * specific_cooling_j_kg

    def equilibrium_para_h2_fraction(self, t_k: float) -> float:
        """
        Equilibrium para-hydrogen mole fraction x_p,eq(T) for LH2 spin isomers.
        At T = 20.27 K (NBP), x_p,eq = 0.9979 (99.8% para). At T >= 300 K, x_p,eq -> 0.25 (normal H2).
        """
        if self.fluid_type != CryoFluidType.LH2:
            return 1.0
        t_c = max(t_k, 10.0)
        theta_rot = 85.4  # Characteristic rotational temperature of H2 [K]
        # Sum first 6 rotational states (even J = para, odd J = ortho with 3x nuclear spin degeneracy)
        z_para = sum((2 * j + 1) * math.exp(-j * (j + 1) * theta_rot / t_c) for j in (0, 2, 4))
        z_ortho = sum(3 * (2 * j + 1) * math.exp(-j * (j + 1) * theta_rot / t_c) for j in (1, 3, 5))
        return z_para / (z_para + z_ortho)

    def ortho_para_conversion_heat_watts(
        self, t_k: float, current_para_fraction: float, liquid_mass_kg: float, catalyzed: bool = False
    ) -> tuple[float, float]:
        """
        Compute exothermic ortho-to-para H2 conversion heat load [W] and rate of para fraction increase [1/s].
        Uncatalyzed autothermal conversion is a second-order process in ortho fraction (k_2 ~ 0.0114 h^-1 = 3.17e-6 s^-1).
        """
        if self.fluid_type != CryoFluidType.LH2:
            return 0.0, 0.0
        x_p_eq = self.equilibrium_para_h2_fraction(t_k)
        x_ortho = max(0.0, 1.0 - current_para_fraction)
        x_ortho_eq = max(0.0, 1.0 - x_p_eq)
        if x_ortho <= x_ortho_eq:
            return 0.0, 0.0
        k_rate = 1.5e-4 if catalyzed else 3.17e-6  # 1/s
        dx_para_dt = k_rate * (x_ortho ** 2 - x_ortho_eq ** 2)
        q_watts = liquid_mass_kg * dx_para_dt * self.props.ortho_para_heat_j_kg
        return q_watts, dx_para_dt


def get_cryo_fluid(fluid: CryoFluidType | str = CryoFluidType.LN2) -> CryoFluidEOS:
    """Factory helper returning a configured CryoFluidEOS instance."""
    return CryoFluidEOS(fluid)
