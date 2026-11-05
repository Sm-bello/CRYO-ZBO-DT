"""
Microgravity Orbital Propellant Depot Tank Thermodynamics, Capillary Liquid Acquisition
Device (LAD) Bubble-Point Margin & 3D Equivalent Pendulum Slosh Dynamics.

Models:
  - Stratified 3-zone thermal network (Bulk Liquid, Stratified Interface Layer, Ullage Gas)
  - Lockheed Multi-Layer Insulation (MLI) radiative/conductive heat leak + strut conduction
  - Ortho-to-para hydrogen exothermic spin conversion heat load (for LH2)
  - Low-Bond-number (Bo = rho * a_eff * R^2 / sigma) regime classification
  - Dutch-Twill 325x2300 Capillary LAD screen bubble-point pressure & vapor-ingestion safety margin
  - 3D lateral (x, y) equivalent mechanical pendulum slosh coupled to spacecraft attitude settling thrusts
  - Active Thermodynamic Vent System (J-T expansion + spray-bar destratification + ZBO cryocooler)
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from cryo_zbo_dt.cryo_fluids import CryoFluidEOS, CryoFluidType, get_cryo_fluid


@dataclass
class DepotTankConfig:
    """Configuration for the orbital cryogenic propellant depot tank simulator."""

    fluid: CryoFluidType = CryoFluidType.LOX
    tank_radius_m: float = 1.65              # 3.3 m diameter orbital depot tank
    tank_cyl_height_m: float = 4.20          # Cylindrical section length
    initial_fill_fraction: float = 0.78      # 78% liquid volume fill
    target_pressure_pa: float = 180000.0     # 180.0 kPa regulated depot setpoint
    pressure_band_limit_pa: float = 1500.0   # +/- 1.5 kPa strict ZBO regulation band
    passive_relief_band_pa: float = 4200.0   # Passive relief valve cracks at P_set + 4.2 kPa
    mli_layers: int = 45                     # 45-layer perforated Double-Aluminized Mylar MLI
    mli_layer_density_per_cm: float = 12.0
    external_hot_temp_k: float = 285.0       # Orbital solar/albedo effective outer shell temp
    strut_parasitic_watts: float = 14.5      # Low-conductivity G10/Ti support strut heat leak
    avionics_internal_watts: float = 6.5     # Internal capacitance/temperature probe + pump idle heat
    microgravity_accel_m_s2: float = 2.5e-4  # ~25 micro-g orbital drag / settling acceleration
    lad_pore_diameter_m: float = 14.8e-6     # 325x2300 Dutch Twill stainless mesh micron rating
    lad_contact_angle_deg: float = 8.0       # Wetting contact angle for cryogenic liquids on SS304L
    lad_outflow_demand_kg_s: float = 0.45    # Depot transfer outflow demand through LAD screen
    slosh_baffle_damping_ratio: float = 0.105 # Anti-slosh ring baffle + screen viscous damping
    max_jt_mass_flow_kg_s: float = 0.0018    # Maximum J-T emergency open-loop throttle flow [kg/s]
    jt_exhaust_pressure_pa: float = 95000.0  # Backpressure in J-T expansion heat exchanger
    max_spray_flow_kg_s: float = 0.35        # Spray-bar recirculation pump max flow [kg/s]
    max_cryocooler_lift_watts: float = 65.0  # Reverse-Brayton / Pulse-Tube cold-head lift at cryo temp
    initial_para_h2_fraction: float = 0.88   # Only active if fluid == LH2


@dataclass
class TankStepState:
    """Instantaneous telemetry state of the orbital propellant depot tank."""

    time_s: float
    ullage_pressure_pa: float
    pressure_error_pa: float
    t_bulk_k: float
    t_interface_k: float
    t_ullage_k: float
    liquid_mass_kg: float
    ullage_mass_kg: float
    cumulative_boiloff_vented_kg: float
    cumulative_jt_consumed_kg: float
    total_propellant_lost_kg: float
    q_parasitic_total_watts: float
    q_tvs_cooling_watts: float
    bond_number: float
    lad_bubble_point_pa: float
    lad_total_pressure_drop_pa: float
    lad_bubble_margin: float
    slosh_disp_x_m: float
    slosh_disp_y_m: float
    slosh_wave_amplitude_m: float
    slosh_torque_nm: float
    para_h2_fraction: float


class MicrogravityDepotTankSimulator:
    """
    Orbital Cryogenic Propellant Depot Tank Simulator coupling stratified self-pressurization,
    low-Bond-number capillary LAD bubble-point margin, and 3D pendulum slosh dynamics.
    """

    def __init__(self, config: DepotTankConfig | None = None) -> None:
        self.cfg = config or DepotTankConfig()
        self.eos: CryoFluidEOS = get_cryo_fluid(self.cfg.fluid)

        # Total tank volume (cylinder + 2 hemispherical domes)
        r = self.cfg.tank_radius_m
        self.tank_volume_m3 = math.pi * (r ** 2) * self.cfg.tank_cyl_height_m + (4.0 / 3.0) * math.pi * (r ** 3)
        self.tank_surface_area_m2 = 2.0 * math.pi * r * self.cfg.tank_cyl_height_m + 4.0 * math.pi * (r ** 2)
        self.interface_area_m2 = math.pi * (r ** 2)

        self.t_sat_set = self.eos.saturation_temperature(self.cfg.target_pressure_pa)
        self.reset()

    def reset(self) -> None:
        """Reset depot tank to initial orbital equilibrium state."""
        self.time_s: float = 0.0
        self.t_bulk_k: float = self.t_sat_set - 0.05
        self.t_interface_k: float = self.t_sat_set + 0.01
        self.t_ullage_k: float = self.t_sat_set + 1.20
        self.ullage_pressure_pa: float = self.cfg.target_pressure_pa

        rho_l = self.eos.liquid_density(self.t_bulk_k)
        rho_v = self.eos.vapor_density(self.t_ullage_k, self.ullage_pressure_pa)
        v_liq = self.cfg.initial_fill_fraction * self.tank_volume_m3
        v_ull = (1.0 - self.cfg.initial_fill_fraction) * self.tank_volume_m3

        self.liquid_mass_kg: float = v_liq * rho_l
        self.ullage_mass_kg: float = v_ull * rho_v
        self.initial_total_mass_kg: float = self.liquid_mass_kg + self.ullage_mass_kg

        self.cumulative_boiloff_vented_kg: float = 0.0
        self.cumulative_jt_consumed_kg: float = 0.0
        self.para_h2_fraction: float = (
            self.cfg.initial_para_h2_fraction if self.cfg.fluid == CryoFluidType.LH2 else 1.0
        )

        # 3D lateral pendulum slosh state: [x_s, vx_s, y_s, vy_s]
        # Start with an initial 38 mm orbital settling slosh displacement and consistent initial velocity
        self.slosh_x_m: float = 0.038
        self.slosh_vx_m_s: float = 0.00025
        self.slosh_y_m: float = -0.018
        self.slosh_vy_m_s: float = -0.00012

    def compute_mli_heat_leak_watts(self) -> float:
        """
        Lockheed Equation for Multi-Layer Insulation (MLI) radiative + solid-conduction
        parasitic heat flux across N layers in high vacuum.
        """
        n_l = max(self.cfg.mli_layers, 5)
        n_bar = self.cfg.mli_layer_density_per_cm
        t_h = self.cfg.external_hot_temp_k
        t_c = self.t_bulk_k
        t_m = 0.5 * (t_h + t_c)

        c_s = 8.95e-8
        c_r = 5.39e-10
        eps_eff = 0.031

        q_cond_flux = (c_s * (n_bar ** 2.56) * t_m * (t_h - t_c)) / n_l
        q_rad_flux = (c_r * eps_eff * ((t_h ** 4.67) - (t_c ** 4.67))) / n_l
        q_mli_w = (q_cond_flux + q_rad_flux) * self.tank_surface_area_m2
        return q_mli_w

    def compute_bond_number(self, axial_accel_m_s2: float) -> float:
        """
        Dimensionless Bond number Bo = rho_l * a_eff * R_tank^2 / sigma(T_bulk).
        Characterizes the ratio of gravitational/settling body forces to surface tension forces.
        """
        rho_l = self.eos.liquid_density(self.t_bulk_k)
        sigma = max(self.eos.surface_tension(self.t_bulk_k), 1e-5)
        a_eff = max(abs(axial_accel_m_s2), 1e-6)
        return (rho_l * a_eff * (self.cfg.tank_radius_m ** 2)) / sigma

    def compute_lad_bubble_point_margin(
        self, slosh_amp_m: float, spray_subcooling_boost_k: float, axial_accel_m_s2: float
    ) -> tuple[float, float, float]:
        """
        Evaluate the capillary Liquid Acquisition Device (LAD) screen bubble-point pressure
        Delta_P_bubble [Pa], total operating screen pressure drop Delta_P_total [Pa],
        and normalized Vapor-Ingestion Safety Margin M_bubble = (Delta_P_bubble - Delta_P_total) / Delta_P_bubble.
        """
        # LAD channels are submerged in bulk liquid, with upper screen exposed to stratification when unmixed
        t_lad = 0.80 * self.t_bulk_k + 0.20 * self.t_interface_k - 0.85 * spray_subcooling_boost_k
        sigma = self.eos.surface_tension(t_lad)
        cos_theta = math.cos(math.radians(self.cfg.lad_contact_angle_deg))
        dp_bubble = (4.0 * sigma * cos_theta) / self.cfg.lad_pore_diameter_m

        rho_l = self.eos.liquid_density(t_lad)
        mu_l = self.eos.liquid_viscosity(t_lad)
        a_eff = max(abs(axial_accel_m_s2), self.cfg.microgravity_accel_m_s2)

        # 1. Hydrostatic head across exposed screen length
        dp_hydro = rho_l * a_eff * self.cfg.tank_cyl_height_m
        # 2. Flow-through-screen (FTS) Armour-Cannon viscous + inertial loss
        u_fts = self.cfg.lad_outflow_demand_kg_s / (rho_l * 0.18)  # 0.18 m^2 effective LAD screen window area
        dp_fts = 1850.0 * mu_l * u_fts + 14.5 * rho_l * (u_fts ** 2)
        # 3. Dynamic slosh pressure perturbation on submerged LAD screen
        omega_slosh_est = math.sqrt(max(1e-4, (1.8412 / self.cfg.tank_radius_m) * a_eff))
        dp_slosh = 0.5 * rho_l * ((omega_slosh_est * slosh_amp_m) ** 2) + rho_l * a_eff * slosh_amp_m
        # 4. Thermal stratification incipient cavitation penalty (if local P_sat exceeds ullage pressure)
        dp_strat_cav = max(0.0, self.eos.saturation_pressure(t_lad) - self.ullage_pressure_pa)

        dp_total = dp_hydro + dp_fts + dp_slosh + dp_strat_cav
        margin = (dp_bubble - dp_total) / max(dp_bubble, 1.0)
        return dp_bubble, dp_total, float(np.clip(margin, -0.5, 0.99))

    def step(
        self,
        dt_s: float,
        u_jt: float = 0.0,
        u_spray: float = 0.0,
        u_cryocooler: float = 0.0,
        lateral_thrust_accel_x_m_s2: float = 0.0,
        lateral_thrust_accel_y_m_s2: float = 0.0,
        axial_settling_accel_m_s2: float | None = None,
        passive_vent_enabled: bool = True,
    ) -> TankStepState:
        """
        Advance the orbital depot tank thermal, capillary LAD, and 3D slosh state by dt_s.
        """
        u_jt_c = float(np.clip(u_jt, 0.0, 1.0))
        u_spray_c = float(np.clip(u_spray, 0.0, 1.0))
        u_cryo_c = float(np.clip(u_cryocooler, 0.0, 1.0))
        a_axial = (
            self.cfg.microgravity_accel_m_s2
            if axial_settling_accel_m_s2 is None
            else axial_settling_accel_m_s2
        )

        # 1. Compute total parasitic heat loads [W]
        q_mli = self.compute_mli_heat_leak_watts()
        q_op, dx_para_dt = self.eos.ortho_para_conversion_heat_watts(
            self.t_bulk_k, self.para_h2_fraction, self.liquid_mass_kg
        )
        self.para_h2_fraction = float(np.clip(self.para_h2_fraction + dx_para_dt * dt_s, 0.0, 1.0))
        q_pump_dissipation = 3.2 * (u_spray_c ** 2)
        q_parasitic_total = q_mli + self.cfg.strut_parasitic_watts + self.cfg.avionics_internal_watts + q_op + q_pump_dissipation

        # 2. Active TVS Cooling (J-T throttling + ZBO Cryocooler lift)
        m_dot_jt = u_jt_c * self.cfg.max_jt_mass_flow_kg_s
        _, q_jt_watts = self.eos.joule_thomson_cooling(
            self.ullage_pressure_pa, self.cfg.jt_exhaust_pressure_pa, self.t_bulk_k, m_dot_jt
        )
        q_cryo_watts = u_cryo_c * self.cfg.max_cryocooler_lift_watts
        q_tvs_total = q_jt_watts + q_cryo_watts

        # 3. 3D Equivalent Mechanical Pendulum Slosh Dynamics
        rho_l = self.eos.liquid_density(self.t_bulk_k)
        sigma = self.eos.surface_tension(self.t_bulk_k)
        r = self.cfg.tank_radius_m
        k_11 = 1.8412 / r  # First antisymmetric Bessel mode wave number
        h_liq = max(0.25, self.cfg.initial_fill_fraction * self.cfg.tank_cyl_height_m)
        omega_sq = k_11 * math.tanh(k_11 * h_liq) * (a_axial + (k_11 ** 2) * sigma / rho_l)
        omega_slosh = math.sqrt(max(omega_sq, 1e-4))
        # Active spray circulation through ring manifold also augments interfacial wave damping
        zeta_eff = self.cfg.slosh_baffle_damping_ratio + 0.095 * u_spray_c

        # Sub-step pendulum ODE for unconditional stability
        n_sub = max(1, int(math.ceil(dt_s / 0.2)))
        dt_sub = dt_s / n_sub
        for _ in range(n_sub):
            ax_s = -2.0 * zeta_eff * omega_slosh * self.slosh_vx_m_s - (omega_slosh ** 2) * self.slosh_x_m - lateral_thrust_accel_x_m_s2
            ay_s = -2.0 * zeta_eff * omega_slosh * self.slosh_vy_m_s - (omega_slosh ** 2) * self.slosh_y_m - lateral_thrust_accel_y_m_s2
            self.slosh_vx_m_s += ax_s * dt_sub
            self.slosh_x_m += self.slosh_vx_m_s * dt_sub
            self.slosh_vy_m_s += ay_s * dt_sub
            self.slosh_y_m += self.slosh_vy_m_s * dt_sub

        slosh_amp = math.hypot(self.slosh_x_m, self.slosh_y_m)
        m_slosh = 0.28 * self.liquid_mass_kg
        slosh_torque = m_slosh * (omega_slosh ** 2) * slosh_amp * (0.45 * self.cfg.tank_cyl_height_m)

        # 4. Stratified Thermal Network (Bulk Liquid vs Interface Boundary Layer vs Ullage)
        # In microgravity, ~58% of parasitic heat channels into the wall meniscus / stratified interface layer
        cp_l = self.eos.liquid_cp(self.t_bulk_k)
        m_interface_layer = max(0.032 * self.liquid_mass_kg, 5.0)
        m_bulk_core = max(self.liquid_mass_kg - m_interface_layer, 50.0)

        q_to_interface = 0.58 * q_parasitic_total
        q_to_bulk = 0.42 * q_parasitic_total - q_tvs_total

        # Spray-bar destratification mixing rate [W/K] between interface layer, ullage, and subcooled bulk
        m_dot_spray = u_spray_c * self.cfg.max_spray_flow_kg_s
        q_mix_spray = m_dot_spray * cp_l * (self.t_interface_k - self.t_bulk_k)
        q_cond_conduction = 4.5 * self.interface_area_m2 * (self.t_interface_k - self.t_bulk_k)

        dt_interface = ((q_to_interface - q_mix_spray - q_cond_conduction) * dt_s) / (m_interface_layer * cp_l)
        dt_bulk = ((q_to_bulk + q_mix_spray + q_cond_conduction) * dt_s) / (m_bulk_core * cp_l)

        self.t_bulk_k += dt_bulk
        self.t_interface_k = max(self.t_bulk_k - 0.1, self.t_interface_k + dt_interface)
        # Ullage gas tracks interface temperature with slight wall superheat reduced by spray droplets
        target_t_ull = self.t_interface_k + (1.25 - 0.95 * u_spray_c)
        self.t_ullage_k += 0.35 * (target_t_ull - self.t_ullage_k)

        # 5. Ullage Pressure Evolution from Interface Saturation Pressure + Ullage Compression
        p_sat_interface = self.eos.saturation_pressure(self.t_interface_k)
        p_sat_bulk = self.eos.saturation_pressure(self.t_bulk_k)
        spray_condensation_factor = float(np.clip(0.82 * u_spray_c, 0.0, 0.85))
        p_eq = (1.0 - spray_condensation_factor) * p_sat_interface + spray_condensation_factor * p_sat_bulk

        # Relaxation toward phase-equilibrium vapor pressure at interface/spray temperature
        tau_p = 4.0 / (1.0 + 4.5 * u_spray_c)
        alpha_relax = min(1.0, dt_s / max(tau_p, 0.5))
        self.ullage_pressure_pa += alpha_relax * (p_eq - self.ullage_pressure_pa)

        # Account for J-T mass consumption
        dm_jt = m_dot_jt * dt_s
        self.liquid_mass_kg = max(0.0, self.liquid_mass_kg - dm_jt)
        self.cumulative_jt_consumed_kg += dm_jt

        # 6. Passive Relief Valve Venting if Ullage Pressure Exceeds Relief Threshold
        p_relief = self.cfg.target_pressure_pa + self.cfg.passive_relief_band_pa
        if passive_vent_enabled and self.ullage_pressure_pa > p_relief:
            # Vent vapor overboard to drop pressure back toward setpoint + 1.2 kPa reseat
            dp_excess = self.ullage_pressure_pa - (self.cfg.target_pressure_pa + 1200.0)
            v_ull = max((1.0 - self.cfg.initial_fill_fraction) * self.tank_volume_m3, 0.5)
            dm_vent = max(0.0, (dp_excess * v_ull) / (self.eos.r_specific * self.t_ullage_k))
            # Also account for flash evaporation cooling of interface during blowdown
            h_fg = self.eos.latent_heat(self.t_interface_k)
            dm_flash = max(0.0, (0.55 * q_parasitic_total * dt_s * 6.0) / h_fg)
            total_vent_step = dm_vent + dm_flash
            self.cumulative_boiloff_vented_kg += total_vent_step
            self.liquid_mass_kg = max(0.0, self.liquid_mass_kg - total_vent_step)
            self.t_interface_k -= (total_vent_step * h_fg) / (m_interface_layer * cp_l)
            self.ullage_pressure_pa = self.eos.saturation_pressure(self.t_interface_k)

        # 7. Evaluate Bond Number and Capillary LAD Bubble-Point Margin
        bo_num = self.compute_bond_number(a_axial)
        spray_subcool_boost = u_spray_c * max(self.t_interface_k - self.t_bulk_k, 0.35)
        dp_bubble, dp_total, lad_margin = self.compute_lad_bubble_point_margin(
            slosh_amp, spray_subcool_boost, a_axial
        )

        self.time_s += dt_s
        p_err = self.ullage_pressure_pa - self.cfg.target_pressure_pa

        return TankStepState(
            time_s=self.time_s,
            ullage_pressure_pa=self.ullage_pressure_pa,
            pressure_error_pa=p_err,
            t_bulk_k=self.t_bulk_k,
            t_interface_k=self.t_interface_k,
            t_ullage_k=self.t_ullage_k,
            liquid_mass_kg=self.liquid_mass_kg,
            ullage_mass_kg=self.ullage_mass_kg,
            cumulative_boiloff_vented_kg=self.cumulative_boiloff_vented_kg,
            cumulative_jt_consumed_kg=self.cumulative_jt_consumed_kg,
            total_propellant_lost_kg=self.cumulative_boiloff_vented_kg + self.cumulative_jt_consumed_kg,
            q_parasitic_total_watts=q_parasitic_total,
            q_tvs_cooling_watts=q_tvs_total,
            bond_number=bo_num,
            lad_bubble_point_pa=dp_bubble,
            lad_total_pressure_drop_pa=dp_total,
            lad_bubble_margin=lad_margin,
            slosh_disp_x_m=self.slosh_x_m,
            slosh_disp_y_m=self.slosh_y_m,
            slosh_wave_amplitude_m=slosh_amp,
            slosh_torque_nm=slosh_torque,
            para_h2_fraction=self.para_h2_fraction,
        )
