"""
End-to-End TRL 4 Benchmark Campaign Runner for CRYO-ZBO-DT.

Executes:
  1. Multi-Fluid Cryogenic Property Table Verification (LN2, LOX, LCH4, LH2)
  2. LN2 Benchtop Transfer Line Chilldown Benchmark:
     - Baseline A: Uncontrolled Continuous Flood Chilldown
     - Active   B: CRYO-ZBO-DT Intermittent Pulse-Chilldown MPC over Binary CCSDS HIL Bridge
  3. Orbital LOX Propellant Depot ZBO & Microgravity Capillary LAD / Slosh Benchmark:
     - Baseline A: Passive Relief-Valve Venting without Destratification
     - Active   B: CRYO-ZBO-DT Active J-T TVS + Spray-Bar + ZBO Cryocooler MPC
  4. Twin-Run Cryptographic SHA-256 Reproducibility Audit & 7-Pillar TRL 4 Certification
  5. Generation of HTML5 Control Room Scorecard, Markdown Dossier, and JSON Audit Manifest
"""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import time
from typing import Any, Dict
import yaml

import numpy as np

from cryo_zbo_dt.chilldown_twophase import ChilldownLineConfig, TwoPhaseChilldownSolver
from cryo_zbo_dt.cryo_fluids import CryoFluidType, get_cryo_fluid
from cryo_zbo_dt.depot_tank_slosh import DepotTankConfig, MicrogravityDepotTankSimulator
from cryo_zbo_dt.hil_bridge import CryoAvionicsHILBridge, FaultInjectionConfig, HILTransportMode
from cryo_zbo_dt.reporting import generate_all_reports
from cryo_zbo_dt.tvs_mpc_controller import (
    ActiveTVSMPCController,
    PulseChilldownMPC,
    PulseChilldownMPCConfig,
    TVSMPCConfig,
)
from cryo_zbo_dt.verification import compute_trajectory_sha256, evaluate_trl4_campaign


def _run_deterministic_pass(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one complete deterministic simulation pass of all baseline and active benchmarks."""
    c_cfg = cfg["chilldown_benchtop"]
    t_cfg = cfg["orbital_depot_tank"]
    h_cfg = cfg["hil_bridge"]

    line_cfg = ChilldownLineConfig(
        fluid=CryoFluidType(c_cfg["fluid"]),
        n_nodes=int(c_cfg["n_nodes"]),
        line_length_m=float(c_cfg["line_length_m"]),
        inner_diameter_m=float(c_cfg["inner_diameter_m"]),
        wall_thickness_m=float(c_cfg["wall_thickness_m"]),
        inlet_pressure_pa=float(c_cfg["inlet_pressure_pa"]),
        back_pressure_pa=float(c_cfg["back_pressure_pa"]),
        initial_wall_temp_k=float(c_cfg["initial_wall_temp_k"]),
        max_mass_flow_kg_s=float(c_cfg["max_mass_flow_kg_s"]),
    )
    dt_chill = float(c_cfg["dt_s"])
    n_steps_chill = int(round(float(c_cfg["duration_s"]) / dt_chill))

    # --- 1A. Baseline Continuous Flood Chilldown ---
    solver_flood = TwoPhaseChilldownSolver(line_cfg)
    flood_times = []
    flood_tw_down = []
    flood_mass = []
    flood_peaks = []
    mass_to_chill_flood = None

    for _ in range(n_steps_chill):
        # Continuous flood opens valve abruptly to 1.0 until chilled, then holds at 0.35
        u_flood = 0.35 if (mass_to_chill_flood is not None) else 1.0
        res = solver_flood.step(dt_chill, valve_command=u_flood)
        flood_times.append(res.time_s)
        flood_tw_down.append(float(res.wall_temps_k[-1]))
        flood_mass.append(res.total_mass_injected_kg)
        flood_peaks.append(res.line_pressure_pa)
        if res.is_chilled_down and mass_to_chill_flood is None:
            mass_to_chill_flood = res.total_mass_injected_kg

    if mass_to_chill_flood is None:
        mass_to_chill_flood = solver_flood.total_mass_injected_kg

    # --- 1B. Active CRYO-ZBO-DT Intermittent Pulse-Chilldown MPC over Binary CCSDS HIL ---
    solver_pulse = TwoPhaseChilldownSolver(line_cfg)
    pulse_mpc = PulseChilldownMPC(PulseChilldownMPCConfig(dt_control_s=dt_chill))
    fault_cfg = FaultInjectionConfig(
        enable_thermocouple_dropout=bool(h_cfg["enable_thermocouple_dropout"]),
        dropout_start_s=float(h_cfg["dropout_start_s"]),
        dropout_end_s=float(h_cfg["dropout_end_s"]),
        enable_sensor_drift=bool(h_cfg["enable_sensor_drift"]),
        enable_valve_stiction=bool(h_cfg["enable_valve_stiction"]),
        corrupt_every_nth_frame=int(h_cfg["corrupt_every_nth_frame"]),
    )
    hil = CryoAvionicsHILBridge(mode=HILTransportMode(h_cfg["transport_mode"]), fault_config=fault_cfg)

    pulse_times = []
    pulse_tw_down = []
    pulse_mass = []
    pulse_peaks = []
    max_mass_err = 0.0
    max_energy_err = 0.0
    mass_to_chill_pulse = None
    step_latencies_ms = []

    for step_idx in range(n_steps_chill):
        t_step_start = time.perf_counter()
        tlm = hil.transmit_telemetry(
            timestamp_s=solver_pulse.time_s,
            wall_temps_k=solver_pulse.wall_temps_k,
            ullage_pressure_pa=180000.0,
            t_bulk_k=solver_pulse.t_sat_ref,
            t_interface_k=solver_pulse.t_sat_ref,
            line_pressure_pa=solver_pulse.cfg.inlet_pressure_pa,
            void_fraction_out=float(solver_pulse.void_fractions[-1]),
            lad_bubble_margin=0.65,
        )
        decision = pulse_mpc.compute_action(
            time_s=tlm.timestamp_s,
            wall_temps_k=np.array(tlm.wall_temps_k, dtype=float),
            t_sat_k=solver_pulse.t_sat_ref,
            t_mfb_k=solver_pulse.t_mfb_k,
            line_pressure_pa=tlm.line_pressure_pa,
            void_fraction_out=tlm.void_fraction_out,
        )
        cmd = hil.transmit_command(
            u_chilldown_valve=decision.valve_command,
            u_jt_valve=0.0,
            u_spray_pump=0.2,
            u_cryocooler=0.35,
            corrupt_wire=(step_idx == 75),
        )
        step_latencies_ms.append((time.perf_counter() - t_step_start) * 1000.0)

        res_p = solver_pulse.step(dt_chill, valve_command=cmd.u_chilldown_valve)
        pulse_times.append(res_p.time_s)
        pulse_tw_down.append(float(res_p.wall_temps_k[-1]))
        pulse_mass.append(res_p.total_mass_injected_kg)
        pulse_peaks.append(res_p.line_pressure_pa)
        max_mass_err = max(max_mass_err, res_p.mass_balance_rel_error)
        max_energy_err = max(max_energy_err, res_p.energy_balance_rel_error)
        if res_p.is_chilled_down and mass_to_chill_pulse is None:
            mass_to_chill_pulse = res_p.total_mass_injected_kg

    if mass_to_chill_pulse is None:
        mass_to_chill_pulse = solver_pulse.total_mass_injected_kg

    mass_savings_pct = (1.0 - (mass_to_chill_pulse / max(mass_to_chill_flood, 1e-6))) * 100.0
    flood_peak_surge_kpa = (max(flood_peaks) - line_cfg.back_pressure_pa) * 1e-3
    pulse_peak_surge_kpa = (max(pulse_peaks) - line_cfg.back_pressure_pa) * 1e-3
    surge_reduction_pct = (1.0 - (pulse_peak_surge_kpa / max(flood_peak_surge_kpa, 1e-6))) * 100.0

    # --- 2. Orbital Depot Tank Passive Venting vs Active J-T TVS + Spray + Cryocooler ---
    depot_cfg = DepotTankConfig(
        fluid=CryoFluidType(t_cfg["fluid"]),
        tank_radius_m=float(t_cfg["tank_radius_m"]),
        tank_cyl_height_m=float(t_cfg["tank_cyl_height_m"]),
        initial_fill_fraction=float(t_cfg["initial_fill_fraction"]),
        target_pressure_pa=float(t_cfg["target_pressure_pa"]),
        pressure_band_limit_pa=float(t_cfg["pressure_band_limit_pa"]),
        passive_relief_band_pa=float(t_cfg["passive_relief_band_pa"]),
        mli_layers=int(t_cfg["mli_layers"]),
        microgravity_accel_m_s2=float(t_cfg["microgravity_accel_m_s2"]),
        lad_pore_diameter_m=float(t_cfg["lad_pore_diameter_m"]),
    )
    dt_tank = float(t_cfg["dt_s"])
    n_steps_tank = int(round(float(t_cfg["duration_s"]) / dt_tank))

    sim_passive = MicrogravityDepotTankSimulator(depot_cfg)
    sim_active = MicrogravityDepotTankSimulator(depot_cfg)
    tvs_mpc = ActiveTVSMPCController(
        TVSMPCConfig(
            dt_control_s=dt_tank,
            target_pressure_pa=depot_cfg.target_pressure_pa,
            pressure_band_pa=depot_cfg.pressure_band_limit_pa,
        )
    )

    tank_times = []
    pass_p_ull = []
    act_p_ull = []
    pass_p_err = []
    act_p_err = []
    pass_lad_margin = []
    act_lad_margin = []
    act_slosh_amp = []
    act_bonds = []

    init_slosh_amp = math.hypot(sim_active.slosh_x_m, sim_active.slosh_y_m)

    for k in range(n_steps_tank):
        # Attitude thruster settling pulse disturbance around t = 120s
        a_lat_x = 1.8e-4 if (55 <= k <= 65) else 0.0
        a_lat_y = -1.2e-4 if (55 <= k <= 65) else 0.0

        # Passive step
        st_pass = sim_passive.step(
            dt_tank,
            u_jt=0.0,
            u_spray=0.0,
            u_cryocooler=0.0,
            lateral_thrust_accel_x_m_s2=a_lat_x,
            lateral_thrust_accel_y_m_s2=a_lat_y,
            passive_vent_enabled=True,
        )

        # Active step via CCSDS HIL bridge
        t0_act = time.perf_counter()
        _, _, cur_lad_margin = sim_active.compute_lad_bubble_point_margin(
            math.hypot(sim_active.slosh_x_m, sim_active.slosh_y_m),
            0.25,
            depot_cfg.microgravity_accel_m_s2,
        )
        tlm_tank = hil.transmit_telemetry(
            timestamp_s=sim_active.time_s,
            wall_temps_k=solver_pulse.wall_temps_k,
            ullage_pressure_pa=sim_active.ullage_pressure_pa,
            t_bulk_k=sim_active.t_bulk_k,
            t_interface_k=sim_active.t_interface_k,
            line_pressure_pa=line_cfg.back_pressure_pa,
            void_fraction_out=0.0,
            lad_bubble_margin=cur_lad_margin,
        )
        tvs_dec = tvs_mpc.compute_action(
            ullage_pressure_pa=tlm_tank.ullage_pressure_pa,
            t_bulk_k=tlm_tank.t_bulk_k,
            t_interface_k=tlm_tank.t_interface_k,
            t_sat_set_k=sim_active.t_sat_set,
            lad_bubble_margin=tlm_tank.lad_bubble_margin,
            q_parasitic_est_watts=38.5,
            max_cryocooler_lift_watts=depot_cfg.max_cryocooler_lift_watts,
        )
        cmd_tank = hil.transmit_command(
            u_chilldown_valve=0.35,
            u_jt_valve=tvs_dec.u_jt,
            u_spray_pump=tvs_dec.u_spray,
            u_cryocooler=tvs_dec.u_cryocooler,
        )
        step_latencies_ms.append((time.perf_counter() - t0_act) * 1000.0)

        st_act = sim_active.step(
            dt_tank,
            u_jt=cmd_tank.u_jt_valve,
            u_spray=cmd_tank.u_spray_pump,
            u_cryocooler=cmd_tank.u_cryocooler,
            lateral_thrust_accel_x_m_s2=a_lat_x,
            lateral_thrust_accel_y_m_s2=a_lat_y,
            passive_vent_enabled=True,
        )

        tank_times.append(st_act.time_s)
        pass_p_ull.append(st_pass.ullage_pressure_pa)
        act_p_ull.append(st_act.ullage_pressure_pa)
        pass_p_err.append(abs(st_pass.pressure_error_pa))
        act_p_err.append(abs(st_act.pressure_error_pa))
        pass_lad_margin.append(st_pass.lad_bubble_margin)
        act_lad_margin.append(st_act.lad_bubble_margin)
        act_slosh_amp.append(st_act.slosh_wave_amplitude_m)
        act_bonds.append(st_act.bond_number)

    hil.close()

    boiloff_savings_pct = (
        1.0 - (sim_active.cumulative_boiloff_vented_kg + sim_active.cumulative_jt_consumed_kg)
        / max(sim_passive.cumulative_boiloff_vented_kg, 1e-6)
    ) * 100.0
    peak_slosh_amp = max(max(act_slosh_amp), 1e-6)
    slosh_damping_pct = (1.0 - (act_slosh_amp[-1] / peak_slosh_amp)) * 100.0

    sha_digest = compute_trajectory_sha256(
        [
            np.array(pulse_tw_down, dtype=np.float64),
            np.array(pulse_mass, dtype=np.float64),
            np.array(act_p_ull, dtype=np.float64),
            np.array(act_lad_margin, dtype=np.float64),
        ]
    )

    # Multi-fluid property verification table
    fluid_table = []
    for ftype in (CryoFluidType.LN2, CryoFluidType.LOX, CryoFluidType.LCH4, CryoFluidType.LH2):
        eos = get_cryo_fluid(ftype)
        p = eos.props
        fluid_table.append(
            {
                "fluid": ftype.value,
                "formula": p.formula,
                "t_nbp_k": p.t_nbp_k,
                "t_crit_k": p.t_crit_k,
                "p_crit_mpa": p.p_crit_pa * 1e-6,
                "rho_l_nbp": eos.liquid_density(p.t_nbp_k),
                "h_fg_nbp_kj_kg": eos.latent_heat(p.t_nbp_k) * 1e-3,
                "sigma_nbp_mn_m": eos.surface_tension(p.t_nbp_k) * 1e3,
                "sound_speed_l_m_s": p.sound_speed_l_nbp,
            }
        )

    return {
        "chilldown_flood": {
            "time_s": flood_times,
            "downstream_wall_temp_k": flood_tw_down,
            "cumulative_mass_kg": flood_mass,
            "mass_to_chill_kg": mass_to_chill_flood,
            "peak_surge_kpa": flood_peak_surge_kpa,
        },
        "chilldown_pulse": {
            "time_s": pulse_times,
            "downstream_wall_temp_k": pulse_tw_down,
            "cumulative_mass_kg": pulse_mass,
            "mass_to_chill_kg": mass_to_chill_pulse,
            "mass_savings_vs_flood_pct": mass_savings_pct,
            "peak_surge_kpa": pulse_peak_surge_kpa,
            "surge_reduction_vs_flood_pct": surge_reduction_pct,
            "leidenfrost_temp_k": solver_pulse.t_mfb_k,
            "t_sat_k": solver_pulse.t_sat_ref,
            "final_max_wall_temp_k": float(np.max(solver_pulse.wall_temps_k)),
            "final_exit_void_fraction": float(solver_pulse.void_fractions[-1]),
            "max_mass_balance_rel_error": max_mass_err,
            "max_energy_balance_rel_error": max_energy_err,
            "wall_energy_removed_kj": solver_pulse.cumulative_wall_energy_removed_j * 1e-3,
        },
        "nukiyama_curve": solver_pulse.compute_nukiyama_curve(),
        "tank_passive": {
            "time_s": tank_times,
            "ullage_pressure_pa": pass_p_ull,
            "max_abs_pressure_error_pa": float(max(pass_p_err)),
            "total_propellant_lost_kg": sim_passive.cumulative_boiloff_vented_kg,
            "lad_bubble_margin": pass_lad_margin,
            "min_lad_bubble_margin": float(min(pass_lad_margin)),
        },
        "tank_active": {
            "time_s": tank_times,
            "ullage_pressure_pa": act_p_ull,
            "max_abs_pressure_error_pa": float(max(act_p_err)),
            "total_propellant_lost_kg": sim_active.cumulative_boiloff_vented_kg + sim_active.cumulative_jt_consumed_kg,
            "boiloff_savings_pct": boiloff_savings_pct,
            "lad_bubble_margin": act_lad_margin,
            "min_lad_bubble_margin": float(min(act_lad_margin)),
            "slosh_amplitude_reduction_pct": slosh_damping_pct,
            "mean_bond_number": float(np.mean(act_bonds)),
        },
        "hil_stats": asdict(hil.stats),
        "fluid_table": fluid_table,
        "step_latencies_ms": step_latencies_ms,
        "sha256": sha_digest,
    }


def run_campaign(config_path: str | Path | None = None, output_dir: str | Path | None = None) -> Dict[str, Any]:
    """Run the complete twin-pass TRL 4 verification campaign and generate all reports."""
    root_dir = Path(__file__).resolve().parent.parent
    cfg_file = Path(config_path) if config_path else (root_dir / "configs" / "cryo_campaign.yaml")
    out_dir = Path(output_dir) if output_dir else (root_dir / "reports")

    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))

    # Execute Run 1 and Run 2 independently to verify Pillar 6 (SHA-256 Bitwise Reproducibility)
    run1 = _run_deterministic_pass(cfg)
    run2 = _run_deterministic_pass(cfg)

    campaign_data = dict(run1)
    campaign_data["sha256_run1"] = run1["sha256"]
    campaign_data["sha256_run2"] = run2["sha256"]

    report = evaluate_trl4_campaign(campaign_data)
    artifact_paths = generate_all_reports(report, campaign_data, out_dir)

    print("=" * 86)
    print(" CRYO-ZBO-DT: TRL 4 ZERO-BOIL-OFF CRYOGENIC TWO-PHASE & TVS DIGITAL TWIN SCORECARD")
    print("=" * 86)
    for p in report.pillars:
        status = "PASS" if p.passed else "FAIL"
        print(f" [Pillar {p.pillar_id}] {status:4s} | {p.name:48s} | {p.measured_value:.4f} {p.units}")
    print("-" * 86)
    print(f" Overall TRL 4 Status : {'PASS' if report.overall_passed else 'FAIL'} ({report.pillars_passed_count}/{report.total_pillars} Pillars)")
    print(f" Trajectory SHA-256   : {report.sha256_trajectory_digest}")
    print(f" HTML5 Scorecard      : {artifact_paths['html_scorecard']}")
    print(f" Markdown Dossier     : {artifact_paths['markdown_dossier']}")
    print(f" JSON Audit Manifest  : {artifact_paths['json_audit']}")
    print("=" * 86)

    return {
        "report": report,
        "campaign_data": campaign_data,
        "artifact_paths": artifact_paths,
    }


if __name__ == "__main__":
    import math
    run_campaign()
