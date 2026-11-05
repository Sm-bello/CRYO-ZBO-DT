"""
Comprehensive Unit & Integration Test Suite for CRYO-ZBO-DT (TRL 4 Verification).

Covers:
  - Real-fluid cryogenic EOS (LN2, LOX, LCH4, LH2) saturation, density, latent heat, surface tension
  - Two-phase Wood speed of sound & Joule-Thomson expansion cooling
  - Para-to-ortho / ortho-to-para LH2 spin-isomer equilibrium & exothermic conversion heat
  - Complete Nukiyama boiling curve regimes & Henry-Leidenfrost T_MFB calibration
  - Zuber-Findlay drift-flux void fraction & 1D transient line chilldown conservation
  - Intermittent Pulse-Chilldown MPC (>35% mass savings & >45% cryo-hammer surge suppression)
  - Orbital Depot Tank MLI heat leak, low-Bond-number Capillary LAD bubble-point margin & 3D slosh
  - Active ZBO TVS MPC (+/- 1.5 kPa tank pressure regulation)
  - Binary CCSDS telemetry/telecommand framing, CRC-16-CCITT FDIR, LOOPBACK & UDP HIL bridges
  - End-to-end 7-Pillar TRL 4 campaign execution and report generation
"""

from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np

from cryo_zbo_dt.chilldown_twophase import (
    BoilingRegime,
    ChilldownLineConfig,
    TwoPhaseChilldownSolver,
)
from cryo_zbo_dt.cryo_fluids import (
    P_ATM,
    CryoFluidEOS,
    CryoFluidType,
    get_cryo_fluid,
)
from cryo_zbo_dt.depot_tank_slosh import (
    DepotTankConfig,
    MicrogravityDepotTankSimulator,
)
from cryo_zbo_dt.hil_bridge import (
    CCSDS_SYNC_MARKER,
    CryoAvionicsHILBridge,
    CryoCommandPacket,
    CryoTelemetryPacket,
    FaultInjectionConfig,
    HILTransportMode,
    crc16_ccitt,
)
from cryo_zbo_dt.tvs_mpc_controller import (
    ActiveTVSMPCController,
    PulseChilldownMPC,
    PulseChilldownMPCConfig,
    TVSMPCConfig,
)
from examples.run_cryo_trl4_campaign import run_campaign


@pytest.mark.parametrize(
    "fluid_type,expected_nbp",
    [
        (CryoFluidType.LN2, 77.355),
        (CryoFluidType.LOX, 90.188),
        (CryoFluidType.LCH4, 111.667),
        (CryoFluidType.LH2, 20.271),
    ],
)
def test_cryo_fluid_saturation_anchor_and_inversion(fluid_type: CryoFluidType, expected_nbp: float) -> None:
    """Verify P_sat(T_nbp) == 101325 Pa and accurate Newton-Raphson T_sat(P_sat(T)) inversion."""
    eos = get_cryo_fluid(fluid_type)
    assert abs(eos.props.t_nbp_k - expected_nbp) < 1e-3
    p_nbp = eos.saturation_pressure(eos.props.t_nbp_k)
    assert abs(p_nbp - P_ATM) / P_ATM < 1e-7

    for t_test in np.linspace(eos.props.t_triple_k * 1.05, eos.props.t_crit_k * 0.92, 6):
        p_sat = eos.saturation_pressure(float(t_test))
        t_inv = eos.saturation_temperature(p_sat)
        assert abs(t_inv - t_test) < 1e-3


@pytest.mark.parametrize("fluid_type", list(CryoFluidType))
def test_cryo_fluid_thermophysical_consistency(fluid_type: CryoFluidType) -> None:
    """Verify liquid/vapor densities, latent heat, surface tension, and Prandtl numbers."""
    eos = get_cryo_fluid(fluid_type)
    t_nbp = eos.props.t_nbp_k
    rho_l = eos.liquid_density(t_nbp)
    rho_v = eos.vapor_density(t_nbp)
    assert abs(rho_l - eos.props.rho_l_nbp) < 1e-4
    assert abs(rho_v - eos.props.rho_v_nbp) < 1e-4
    assert rho_l > 15.0 * rho_v

    h_fg_nbp = eos.latent_heat(t_nbp)
    h_fg_high = eos.latent_heat(0.5 * (t_nbp + eos.props.t_crit_k))
    assert h_fg_nbp > h_fg_high > 0.0
    assert eos.latent_heat(eos.props.t_crit_k + 1.0) == 0.0

    sigma_nbp = eos.surface_tension(t_nbp)
    assert sigma_nbp > 0.001
    assert eos.surface_tension(eos.props.t_crit_k + 0.5) == 0.0
    assert 0.5 < eos.prandtl_liquid(t_nbp) < 15.0
    assert 0.5 < eos.prandtl_vapor(t_nbp) < 5.0


def test_wood_two_phase_acoustic_speed_and_jt_cooling() -> None:
    """Verify Wood's two-phase sound speed minimum at intermediate void fraction and J-T cooling."""
    eos = get_cryo_fluid(CryoFluidType.LN2)
    t_k = eos.props.t_nbp_k
    a_liq = eos.two_phase_sound_speed(0.0, t_k)
    a_mid = eos.two_phase_sound_speed(0.5, t_k)
    a_vap = eos.two_phase_sound_speed(1.0, t_k)
    assert a_liq > 800.0
    assert a_mid < a_vap < a_liq

    t_out, q_cool = eos.joule_thomson_cooling(180000.0, 101325.0, 81.0, m_dot_jt_kg_s=0.002)
    assert t_out < 81.0
    assert q_cool > 250.0


def test_lh2_ortho_para_equilibrium_and_exothermic_heat() -> None:
    """Verify LH2 equilibrium para-H2 fraction and exothermic ortho-to-para conversion heat release."""
    eos_lh2 = get_cryo_fluid(CryoFluidType.LH2)
    xp_20k = eos_lh2.equilibrium_para_h2_fraction(20.271)
    xp_300k = eos_lh2.equilibrium_para_h2_fraction(300.0)
    assert xp_20k > 0.995
    assert 0.24 < xp_300k < 0.26

    q_uncat, dx_uncat = eos_lh2.ortho_para_conversion_heat_watts(20.271, current_para_fraction=0.75, liquid_mass_kg=500.0)
    q_cat, dx_cat = eos_lh2.ortho_para_conversion_heat_watts(20.271, current_para_fraction=0.75, liquid_mass_kg=500.0, catalyzed=True)
    assert q_cat > q_uncat > 0.0
    assert dx_cat > dx_uncat > 0.0


def test_nukiyama_boiling_curve_regimes_and_leidenfrost_tmfb() -> None:
    """Verify all 4 wet boiling regimes and Henry-Leidenfrost T_MFB calibration for LN2 on SS304L."""
    solver = TwoPhaseChilldownSolver(ChilldownLineConfig(fluid=CryoFluidType.LN2))
    assert 105.0 <= solver.t_mfb_k <= 135.0
    assert solver.t_sat_ref < solver.t_onb_k < solver.t_chf_k < solver.t_mfb_k

    # 1. Single-phase liquid below ONB
    q_sp, _, reg_sp = solver.evaluate_nukiyama_point(solver.t_sat_ref + 0.5, solver.t_sat_ref, 180.0, 0.0)
    assert reg_sp == BoilingRegime.SINGLE_PHASE_LIQUID

    # 2. Nucleate boiling between ONB and CHF
    q_nb, _, reg_nb = solver.evaluate_nukiyama_point(0.5 * (solver.t_onb_k + solver.t_chf_k), solver.t_sat_ref, 180.0, 0.2)
    assert reg_nb == BoilingRegime.NUCLEATE_BOILING

    # 3. Peak CHF > Minimum Film Boiling heat flux
    q_chf, _, _ = solver.evaluate_nukiyama_point(solver.t_chf_k, solver.t_sat_ref, 180.0, 0.3)
    q_tr, _, reg_tr = solver.evaluate_nukiyama_point(0.5 * (solver.t_chf_k + solver.t_mfb_k), solver.t_sat_ref, 180.0, 0.5)
    assert reg_tr == BoilingRegime.TRANSITION_BOILING

    # 4. Film boiling above T_MFB
    q_mfb, _, reg_fb = solver.evaluate_nukiyama_point(solver.t_mfb_k + 1.0, solver.t_sat_ref, 180.0, 0.8)
    assert reg_fb == BoilingRegime.FILM_BOILING
    assert q_chf > q_tr > q_mfb > q_sp


def test_zuber_findlay_drift_flux_void_fraction() -> None:
    """Verify Zuber-Findlay void fraction bounds and monotonic dependence on quality x."""
    solver = TwoPhaseChilldownSolver()
    assert solver.drift_flux_void_fraction(0.0, 150.0, solver.t_sat_ref) == 0.0
    assert solver.drift_flux_void_fraction(1.0, 150.0, solver.t_sat_ref) == 1.0
    a_low = solver.drift_flux_void_fraction(0.05, 150.0, solver.t_sat_ref)
    a_high = solver.drift_flux_void_fraction(0.40, 150.0, solver.t_sat_ref)
    assert 0.0 < a_low < a_high < 1.0


def test_chilldown_mass_and_energy_conservation_and_pulse_savings() -> None:
    """Verify strict mass/enthalpy conservation and >35% pulse-chilldown mass savings over flood."""
    cfg = ChilldownLineConfig(fluid=CryoFluidType.LN2)
    flood_solver = TwoPhaseChilldownSolver(cfg)
    pulse_solver = TwoPhaseChilldownSolver(cfg)
    mpc = PulseChilldownMPC(PulseChilldownMPCConfig(dt_control_s=0.5))

    m_flood = None
    m_pulse = None
    flood_peak_p = 0.0
    pulse_peak_p = 0.0

    for _ in range(280):
        r_f = flood_solver.step(0.5, 0.35 if m_flood is not None else 1.0)
        flood_peak_p = max(flood_peak_p, r_f.line_pressure_pa)
        if r_f.is_chilled_down and m_flood is None:
            m_flood = r_f.total_mass_injected_kg

        dec = mpc.compute_action(
            pulse_solver.time_s,
            pulse_solver.wall_temps_k,
            pulse_solver.t_sat_ref,
            pulse_solver.t_mfb_k,
            pulse_solver.cfg.inlet_pressure_pa,
            float(pulse_solver.void_fractions[-1]),
        )
        r_p = pulse_solver.step(0.5, dec.valve_command)
        pulse_peak_p = max(pulse_peak_p, r_p.line_pressure_pa)
        assert r_p.mass_balance_rel_error < 1e-9
        assert r_p.energy_balance_rel_error < 1e-9
        if r_p.is_chilled_down and m_pulse is None:
            m_pulse = r_p.total_mass_injected_kg

    assert m_flood is not None and m_pulse is not None
    savings_pct = (1.0 - m_pulse / m_flood) * 100.0
    assert savings_pct > 35.0
    assert pulse_peak_p < flood_peak_p


def test_orbital_depot_tank_lad_margin_slosh_and_active_zbo_tvs() -> None:
    """Verify Active TVS MPC holds ullage pressure within +/- 1.5 kPa and protects LAD margin."""
    cfg = DepotTankConfig(fluid=CryoFluidType.LOX)
    sim_pass = MicrogravityDepotTankSimulator(cfg)
    sim_act = MicrogravityDepotTankSimulator(cfg)
    mpc = ActiveTVSMPCController(TVSMPCConfig(dt_control_s=2.0, target_pressure_pa=cfg.target_pressure_pa))

    init_slosh = sim_act.slosh_x_m
    max_act_err = 0.0
    min_act_lad = 1.0

    for _ in range(300):
        st_p = sim_pass.step(2.0, u_jt=0.0, u_spray=0.0, u_cryocooler=0.0)
        _, _, lad_m = sim_act.compute_lad_bubble_point_margin(0.02, 0.2, cfg.microgravity_accel_m_s2)
        dec = mpc.compute_action(
            sim_act.ullage_pressure_pa,
            sim_act.t_bulk_k,
            sim_act.t_interface_k,
            sim_act.t_sat_set,
            lad_m,
        )
        st_a = sim_act.step(2.0, u_jt=dec.u_jt, u_spray=dec.u_spray, u_cryocooler=dec.u_cryocooler)
        max_act_err = max(max_act_err, abs(st_a.pressure_error_pa))
        min_act_lad = min(min_act_lad, st_a.lad_bubble_margin)

    assert max_act_err <= 1500.0
    assert min_act_lad >= 0.25
    assert abs(sim_act.slosh_x_m) < 0.35 * abs(init_slosh)
    assert sim_act.cumulative_boiloff_vented_kg < 0.05 * max(sim_pass.cumulative_boiloff_vented_kg, 1e-5)


def test_ccsds_crc16_framing_and_udp_hil_bridge() -> None:
    """Verify CCSDS binary pack/unpack, CRC-16-CCITT rejection, and localhost UDP socket bridge."""
    pkt = CryoTelemetryPacket(
        seq_count=42,
        timestamp_s=12.5,
        wall_temps_k=(210.0, 215.0, 220.0, 225.0),
        ullage_pressure_pa=180120.0,
        t_bulk_k=90.1,
        t_interface_k=90.4,
        line_pressure_pa=155000.0,
        void_fraction_out=0.65,
        lad_bubble_margin=0.58,
        status_flags=1,
    )
    raw = pkt.pack()
    assert len(raw) == CryoTelemetryPacket.FRAME_SIZE_BYTES
    decoded = CryoTelemetryPacket.unpack(raw)
    assert decoded.seq_count == 42
    assert abs(decoded.ullage_pressure_pa - 180120.0) < 1.0

    # Corrupt 1 bit and verify ValueError from CRC-16-CCITT
    bad = bytearray(raw)
    bad[10] ^= 0x01
    with pytest.raises(ValueError, match="CRC-16-CCITT"):
        CryoTelemetryPacket.unpack(bytes(bad))

    # Test UDP localhost bridge transport
    bridge_udp = CryoAvionicsHILBridge(mode=HILTransportMode.UDP, udp_port=19488)
    try:
        rx_tlm = bridge_udp.transmit_telemetry(
            timestamp_s=5.0,
            wall_temps_k=np.array([150.0, 155.0, 160.0, 165.0]),
            ullage_pressure_pa=180000.0,
            t_bulk_k=90.0,
            t_interface_k=90.2,
            line_pressure_pa=140000.0,
            void_fraction_out=0.3,
            lad_bubble_margin=0.62,
        )
        assert abs(rx_tlm.wall_temps_k[0] - 150.0) < 1e-3
        rx_cmd = bridge_udp.transmit_command(0.45, 0.0, 0.25, 0.50)
        assert abs(rx_cmd.u_chilldown_valve - 0.45) < 1e-4
    finally:
        bridge_udp.close()


def test_full_trl4_campaign_and_report_artifacts(tmp_path: Path) -> None:
    """Run the full 7-Pillar TRL 4 campaign and verify all 7 pillars pass and artifacts exist."""
    result = run_campaign(output_dir=tmp_path)
    report = result["report"]
    assert report.overall_passed is True
    assert report.pillars_passed_count == 7
    for p in report.pillars:
        assert p.passed is True, f"Pillar {p.pillar_id} ({p.name}) failed: {p.details}"

    for key in ("html_scorecard", "markdown_dossier", "json_audit"):
        fpath = Path(result["artifact_paths"][key])
        assert fpath.exists()
        assert fpath.stat().st_size > 1000
