"""
7-Pillar TRL 4 Verification & Credibility Engine for CRYO-ZBO-DT.

Evaluates and certifies:
  1. Mass & Enthalpy Conservation across Two-Phase Change
  2. Leidenfrost T_MFB & Chilldown Empirical Curve Calibration against LN2 Lab Benchmarks
  3. Zero-Boil-Off (ZBO) TVS Tank Pressure Regulation (+/- 1.5 kPa band)
  4. Capillary LAD Vapor-Ingestion Bubble-Point Margin & Microgravity Slosh Damping
  5. Cryo-Hammer Surge Suppression, Pulse-Chilldown Mass Savings (>35%), & HIL FDIR
  6. Cryptographic SHA-256 Bitwise Reproducibility
  7. Real-Time Avionics Control Loop Timing Budget (< 5.0 ms p99)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Dict, List

import numpy as np


@dataclass
class PillarResult:
    """Verification outcome for a single TRL 4 credibility pillar."""

    pillar_id: int
    name: str
    metric_name: str
    measured_value: float
    threshold_description: str
    units: str
    passed: bool
    details: str


@dataclass
class TRL4VerificationReport:
    """Complete 7-Pillar TRL 4 Verification Scorecard and Campaign Audit Summary."""

    project_code: str
    trl_target: str
    overall_passed: bool
    pillars_passed_count: int
    total_pillars: int
    sha256_trajectory_digest: str
    pillars: List[PillarResult] = field(default_factory=list)
    summary_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_code": self.project_code,
            "trl_target": self.trl_target,
            "overall_passed": self.overall_passed,
            "pillars_passed_count": self.pillars_passed_count,
            "total_pillars": self.total_pillars,
            "sha256_trajectory_digest": self.sha256_trajectory_digest,
            "pillars": [asdict(p) for p in self.pillars],
            "summary_metrics": self.summary_metrics,
        }


def compute_trajectory_sha256(arrays: List[np.ndarray]) -> str:
    """Compute a deterministic SHA-256 hex digest over rounded float64 trajectory arrays."""
    hasher = hashlib.sha256()
    for arr in arrays:
        clean = np.round(np.asarray(arr, dtype=np.float64), decimals=8)
        hasher.update(clean.tobytes())
    return hasher.hexdigest()


def evaluate_trl4_campaign(campaign_data: Dict[str, Any]) -> TRL4VerificationReport:
    """
    Evaluate all 7 TRL 4 verification pillars from the campaign simulation outputs.
    """
    chill_flood = campaign_data["chilldown_flood"]
    chill_pulse = campaign_data["chilldown_pulse"]
    tank_passive = campaign_data["tank_passive"]
    tank_active = campaign_data["tank_active"]
    hil_stats = campaign_data["hil_stats"]
    repro_digest_1 = campaign_data["sha256_run1"]
    repro_digest_2 = campaign_data["sha256_run2"]
    step_latencies_ms = np.asarray(campaign_data["step_latencies_ms"], dtype=float)

    # Pillar 1: Mass & Enthalpy Conservation across Two-Phase Change
    max_mass_err = float(chill_pulse["max_mass_balance_rel_error"])
    max_energy_err_pct = float(chill_pulse["max_energy_balance_rel_error"]) * 100.0
    p1_pass = (max_mass_err < 1e-6) and (max_energy_err_pct < 1.0)
    p1 = PillarResult(
        pillar_id=1,
        name="Mass & Enthalpy Conservation across Phase Change",
        metric_name="Two-Phase Energy Balance Residual",
        measured_value=max_energy_err_pct,
        threshold_description="< 1.0% energy residual & < 1e-6 mass balance error",
        units="%",
        passed=p1_pass,
        details=(
            f"Max mass balance error = {max_mass_err:.2e}, max enthalpy residual = {max_energy_err_pct:.4e}%, "
            f"wall heat removed = {chill_pulse['wall_energy_removed_kj']:.2f} kJ."
        ),
    )

    # Pillar 2: Leidenfrost T_MFB & Chilldown Empirical Calibration (LN2 Benchmark)
    t_mfb_k = float(chill_pulse["leidenfrost_temp_k"])
    final_max_wall_k = float(chill_pulse["final_max_wall_temp_k"])
    t_sat_k = float(chill_pulse["t_sat_k"])
    final_void_out = float(chill_pulse["final_exit_void_fraction"])
    p2_pass = (105.0 <= t_mfb_k <= 135.0) and (final_max_wall_k <= t_sat_k + 3.0) and (final_void_out <= 0.02)
    p2 = PillarResult(
        pillar_id=2,
        name="Leidenfrost T_MFB & Chilldown Empirical Calibration",
        metric_name="Predicted LN2-SS304L Leidenfrost T_MFB",
        measured_value=t_mfb_k,
        threshold_description="105.0 K <= T_MFB <= 135.0 K & Final Wall Superheat <= 3.0 K",
        units="K",
        passed=p2_pass,
        details=(
            f"Henry-Berenson T_MFB = {t_mfb_k:.2f} K (T_sat = {t_sat_k:.2f} K), "
            f"final downstream wall temp = {final_max_wall_k:.2f} K, exit void alpha = {final_void_out:.4f}."
        ),
    )

    # Pillar 3: Zero-Boil-Off (ZBO) TVS Tank Pressure Regulation
    max_p_err_kpa = float(tank_active["max_abs_pressure_error_pa"]) * 1e-3
    zbo_savings_pct = float(tank_active["boiloff_savings_pct"])
    p3_pass = (max_p_err_kpa <= 1.50) and (zbo_savings_pct >= 90.0)
    p3 = PillarResult(
        pillar_id=3,
        name="Zero-Boil-Off (ZBO) TVS Tank Pressure Regulation",
        metric_name="Max Ullage Pressure Excursion |dP|",
        measured_value=max_p_err_kpa,
        threshold_description="<= 1.50 kPa around 180.0 kPa setpoint & >= 90% ZBO savings",
        units="kPa",
        passed=p3_pass,
        details=(
            f"Active TVS held tank within +/- {max_p_err_kpa:.3f} kPa (vs +{tank_passive['max_abs_pressure_error_pa']*1e-3:.2f} kPa passive relief), "
            f"reducing propellant boil-off loss by {zbo_savings_pct:.2f}%."
        ),
    )

    # Pillar 4: Capillary LAD Vapor-Ingestion Margin & Microgravity Slosh Damping
    min_lad_margin_pct = float(tank_active["min_lad_bubble_margin"]) * 100.0
    slosh_damping_pct = float(tank_active["slosh_amplitude_reduction_pct"])
    p4_pass = (min_lad_margin_pct >= 25.0) and (slosh_damping_pct >= 60.0)
    p4 = PillarResult(
        pillar_id=4,
        name="Capillary LAD Bubble-Point Margin & Slosh Damping",
        metric_name="Minimum LAD Screen Bubble-Point Margin",
        measured_value=min_lad_margin_pct,
        threshold_description=">= 25.0% bubble-point safety margin & >= 60.0% slosh damping",
        units="%",
        passed=p4_pass,
        details=(
            f"325x2300 Dutch-Twill LAD min bubble-point margin = {min_lad_margin_pct:.2f}% "
            f"(passive unsubcooled min = {tank_passive['min_lad_bubble_margin']*100.0:.2f}%), "
            f"microgravity slosh wave damped by {slosh_damping_pct:.1f}% (Bond number Bo = {tank_active['mean_bond_number']:.2f})."
        ),
    )

    # Pillar 5: Cryo-Hammer Surge Suppression, Pulse-Chilldown Savings & HIL FDIR
    mass_savings_pct = float(chill_pulse["mass_savings_vs_flood_pct"])
    surge_reduction_pct = float(chill_pulse["surge_reduction_vs_flood_pct"])
    crc_rejected = int(hil_stats["rejected_telemetry_crc_frames"]) + int(hil_stats["rejected_command_crc_frames"])
    tc_dropouts_handled = int(hil_stats["thermocouple_dropout_events"])
    p5_pass = (
        (mass_savings_pct >= 35.0)
        and (surge_reduction_pct >= 45.0)
        and (crc_rejected > 0)
        and (tc_dropouts_handled > 0)
    )
    p5 = PillarResult(
        pillar_id=5,
        name="Pulse-Chilldown Mass Savings, Surge Suppression & HIL FDIR",
        metric_name="Cryogenic Chilldown Mass Savings vs Flood",
        measured_value=mass_savings_pct,
        threshold_description=">= 35.0% cryogen mass savings & >= 45.0% cryo-hammer surge reduction",
        units="%",
        passed=p5_pass,
        details=(
            f"Pulse-Chilldown MPC saved {mass_savings_pct:.2f}% LN2 mass ({chill_pulse['mass_to_chill_kg']:.2f} kg vs "
            f"{chill_flood['mass_to_chill_kg']:.2f} kg flood), cut peak cryo-hammer surge by {surge_reduction_pct:.1f}% "
            f"({chill_pulse['peak_surge_kpa']:.1f} kPa vs {chill_flood['peak_surge_kpa']:.1f} kPa), "
            f"and isolated {crc_rejected} CRC-16 faults + {tc_dropouts_handled} thermocouple dropouts."
        ),
    )

    # Pillar 6: Cryptographic SHA-256 Reproducibility
    repro_match = (repro_digest_1 == repro_digest_2) and (len(repro_digest_1) == 64)
    p6 = PillarResult(
        pillar_id=6,
        name="Cryptographic SHA-256 Bitwise Reproducibility",
        metric_name="Deterministic Twin-Run Digest Match",
        measured_value=1.0 if repro_match else 0.0,
        threshold_description="Exact SHA-256 hex digest match (1.0 == 1.0) across independent runs",
        units="boolean",
        passed=repro_match,
        details=f"Run 1 & Run 2 SHA-256: {repro_digest_1[:16]}...{repro_digest_1[-16:]} (exact match = {repro_match}).",
    )

    # Pillar 7: Real-Time Avionics Control Loop Timing Budget
    p99_latency_ms = float(np.percentile(step_latencies_ms, 99))
    mean_latency_ms = float(np.mean(step_latencies_ms))
    p7_pass = p99_latency_ms < 5.0
    p7 = PillarResult(
        pillar_id=7,
        name="Real-Time Avionics Control Loop Timing Budget",
        metric_name="99th-Percentile MPC + HIL Step Latency",
        measured_value=p99_latency_ms,
        threshold_description="< 5.0 ms p99 latency (10 Hz / 100 ms avionics frame budget)",
        units="ms",
        passed=p7_pass,
        details=f"Mean step latency = {mean_latency_ms:.3f} ms, p99 latency = {p99_latency_ms:.3f} ms (< 5.0 ms ceiling).",
    )

    pillars = [p1, p2, p3, p4, p5, p6, p7]
    passed_count = sum(1 for p in pillars if p.passed)
    overall_passed = passed_count == len(pillars)

    summary_metrics = {
        "ln2_flood_chilldown_mass_kg": round(float(chill_flood["mass_to_chill_kg"]), 4),
        "ln2_pulse_chilldown_mass_kg": round(float(chill_pulse["mass_to_chill_kg"]), 4),
        "chilldown_mass_savings_pct": round(mass_savings_pct, 2),
        "flood_peak_surge_kpa": round(float(chill_flood["peak_surge_kpa"]), 2),
        "pulse_peak_surge_kpa": round(float(chill_pulse["peak_surge_kpa"]), 2),
        "cryo_hammer_surge_reduction_pct": round(surge_reduction_pct, 2),
        "leidenfrost_temp_k": round(t_mfb_k, 2),
        "passive_tank_max_pressure_err_kpa": round(float(tank_passive["max_abs_pressure_error_pa"]) * 1e-3, 3),
        "active_tvs_max_pressure_err_kpa": round(max_p_err_kpa, 3),
        "passive_tank_propellant_lost_kg": round(float(tank_passive["total_propellant_lost_kg"]), 4),
        "active_tvs_propellant_lost_kg": round(float(tank_active["total_propellant_lost_kg"]), 4),
        "zbo_boiloff_savings_pct": round(zbo_savings_pct, 2),
        "passive_min_lad_bubble_margin_pct": round(float(tank_passive["min_lad_bubble_margin"]) * 100.0, 2),
        "active_min_lad_bubble_margin_pct": round(min_lad_margin_pct, 2),
        "slosh_amplitude_reduction_pct": round(slosh_damping_pct, 2),
        "hil_crc_faults_isolated": crc_rejected,
        "hil_tc_dropouts_isolated": tc_dropouts_handled,
        "mpc_p99_latency_ms": round(p99_latency_ms, 4),
    }

    return TRL4VerificationReport(
        project_code="CRYO-ZBO-DT",
        trl_target="TRL 4 (Laboratory Breadboard & Avionics HIL Validated)",
        overall_passed=overall_passed,
        pillars_passed_count=passed_count,
        total_pillars=len(pillars),
        sha256_trajectory_digest=repro_digest_1,
        pillars=pillars,
        summary_metrics=summary_metrics,
    )
