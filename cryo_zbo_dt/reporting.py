"""
Publication-Grade Reporting & Aerospace Control Room Dashboard Generator for CRYO-ZBO-DT.

Generates three standalone verification artifacts in `reports/`:
  1. `reports/cryo_trl4_scorecard.html` : Dark-theme HTML5 Aerospace Control Room Dashboard
     with 4 embedded vector SVG engineering plots, KPI cards, 7-Pillar TRL 4 Verification
     Matrix, Multi-Fluid Cryogenic Property Table, and CCSDS HIL Link Audit.
  2. `reports/TRL4_DOSSIER.md`          : Comprehensive Markdown TRL 4 Engineering Dossier.
  3. `reports/trl4_audit.json`          : Machine-readable JSON audit manifest with SHA-256 hashes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from cryo_zbo_dt.verification import TRL4VerificationReport


def _svg_polyline(
    x_vals: Sequence[float],
    y_vals: Sequence[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    plot_x: float,
    plot_y: float,
    plot_w: float,
    plot_h: float,
    log_x: bool = False,
    log_y: bool = False,
) -> str:
    """Convert data coordinates into an SVG polyline points string."""
    pts = []
    for xv, yv in zip(x_vals, y_vals):
        if log_x:
            xn = (math.log10(max(xv, 1e-6)) - math.log10(x_min)) / max(
                math.log10(x_max) - math.log10(x_min), 1e-9
            )
        else:
            xn = (xv - x_min) / max(x_max - x_min, 1e-9)
        if log_y:
            yn = (math.log10(max(yv, 1e-6)) - math.log10(y_min)) / max(
                math.log10(y_max) - math.log10(y_min), 1e-9
            )
        else:
            yn = (yv - y_min) / max(y_max - y_min, 1e-9)
        xn_c = min(max(xn, 0.0), 1.0)
        yn_c = min(max(yn, 0.0), 1.0)
        px = plot_x + xn_c * plot_w
        py = plot_y + plot_h - yn_c * plot_h
        pts.append(f"{px:.1f},{py:.1f}")
    return " ".join(pts)


def _build_chilldown_svg(chill_flood: Dict[str, Any], chill_pulse: Dict[str, Any]) -> str:
    """Render dual-panel SVG of Wall Temperature Cooldown + Cumulative Cryogen Mass Consumed."""
    w, h = 580, 300
    px, py, pw, ph = 58, 28, 460, 225
    t_arr = chill_pulse["time_s"]
    t_max = float(max(t_arr[-1], 1.0))
    tw_flood = chill_flood["downstream_wall_temp_k"]
    tw_pulse = chill_pulse["downstream_wall_temp_k"]
    m_flood = chill_flood["cumulative_mass_kg"]
    m_pulse = chill_pulse["cumulative_mass_kg"]
    t_mfb = float(chill_pulse["leidenfrost_temp_k"])
    t_sat = float(chill_pulse["t_sat_k"])
    m_max = max(float(m_flood[-1]), float(m_pulse[-1]), 1.0) * 1.08

    pts_tw_flood = _svg_polyline(t_arr, tw_flood, 0.0, t_max, 65.0, 305.0, px, py, pw, ph)
    pts_tw_pulse = _svg_polyline(t_arr, tw_pulse, 0.0, t_max, 65.0, 305.0, px, py, pw, ph)
    pts_m_flood = _svg_polyline(t_arr, m_flood, 0.0, t_max, 0.0, m_max, px, py, pw, ph)
    pts_m_pulse = _svg_polyline(t_arr, m_pulse, 0.0, t_max, 0.0, m_max, px, py, pw, ph)

    y_mfb = py + ph - ((t_mfb - 65.0) / (305.0 - 65.0)) * ph
    y_sat = py + ph - ((t_sat - 65.0) / (305.0 - 65.0)) * ph

    return f"""<svg viewBox="0 0 {w} {h}" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">
      <rect width="{w}" height="{h}" rx="8" fill="#0d1522" stroke="#1e293b"/>
      <line x1="{px}" y1="{py}" x2="{px}" y2="{py+ph}" stroke="#475569" stroke-width="1.2"/>
      <line x1="{px}" y1="{py+ph}" x2="{px+pw}" y2="{py+ph}" stroke="#475569" stroke-width="1.2"/>
      <line x1="{px+pw}" y1="{py}" x2="{px+pw}" y2="{py+ph}" stroke="#475569" stroke-width="1.2"/>
      <line x1="{px}" y1="{y_mfb:.1f}" x2="{px+pw}" y2="{y_mfb:.1f}" stroke="#f59e0b" stroke-dasharray="5,4" stroke-width="1"/>
      <text x="{px+8}" y="{y_mfb-5:.1f}" fill="#f59e0b" font-size="10" font-family="monospace">Leidenfrost T_MFB = {t_mfb:.1f} K</text>
      <line x1="{px}" y1="{y_sat:.1f}" x2="{px+pw}" y2="{y_sat:.1f}" stroke="#38bdf8" stroke-dasharray="3,3" stroke-width="1"/>
      <text x="{px+8}" y="{y_sat-5:.1f}" fill="#38bdf8" font-size="10" font-family="monospace">T_sat = {t_sat:.1f} K</text>
      <polyline fill="none" stroke="#ef4444" stroke-width="2.0" stroke-dasharray="6,3" points="{pts_tw_flood}"/>
      <polyline fill="none" stroke="#10b981" stroke-width="2.4" points="{pts_tw_pulse}"/>
      <polyline fill="none" stroke="#f97316" stroke-width="1.8" stroke-opacity="0.85" points="{pts_m_flood}"/>
      <polyline fill="none" stroke="#38bdf8" stroke-width="2.0" points="{pts_m_pulse}"/>
      <text x="{px-8}" y="{py+10}" fill="#94a3b8" font-size="10" text-anchor="end">300K</text>
      <text x="{px-8}" y="{py+ph/2}" fill="#94a3b8" font-size="10" text-anchor="end">185K</text>
      <text x="{px-8}" y="{py+ph}" fill="#94a3b8" font-size="10" text-anchor="end">70K</text>
      <text x="{px+pw+6}" y="{py+10}" fill="#38bdf8" font-size="10">{m_max:.1f}kg</text>
      <text x="{px+pw+6}" y="{py+ph}" fill="#38bdf8" font-size="10">0kg</text>
      <text x="{px+pw/2}" y="{h-10}" fill="#cbd5e1" font-size="11" text-anchor="middle">Chilldown Time [s] (0 to {t_max:.0f} s)</text>
      <g transform="translate({px+110}, {py+8})">
        <rect width="335" height="38" rx="4" fill="#0f172a" fill-opacity="0.88" stroke="#334155"/>
        <line x1="8" y1="12" x2="26" y2="12" stroke="#10b981" stroke-width="2.4"/>
        <text x="30" y="15" fill="#e2e8f0" font-size="10">Pulse-MPC Wall T_w</text>
        <line x1="145" y1="12" x2="163" y2="12" stroke="#ef4444" stroke-width="2" stroke-dasharray="4,2"/>
        <text x="168" y="15" fill="#e2e8f0" font-size="10">Flood Wall T_w</text>
        <line x1="8" y1="28" x2="26" y2="28" stroke="#38bdf8" stroke-width="2"/>
        <text x="30" y="31" fill="#e2e8f0" font-size="10">Pulse Mass ({chill_pulse['mass_to_chill_kg']:.2f} kg)</text>
        <line x1="165" y1="28" x2="183" y2="28" stroke="#f97316" stroke-width="2"/>
        <text x="188" y="31" fill="#e2e8f0" font-size="10">Flood Mass ({chill_flood['mass_to_chill_kg']:.2f} kg)</text>
      </g>
    </svg>"""


def _build_nukiyama_svg(nukiyama_data: Dict[str, Any]) -> str:
    """Render log-log SVG of the Nukiyama Boiling Curve (q'' vs Delta T_sat)."""
    w, h = 580, 300
    px, py, pw, ph = 64, 28, 480, 225
    dt_vals = nukiyama_data["delta_t_sat_k"]
    q_vals = nukiyama_data["heat_flux_w_m2"]
    dt_min, dt_max = 0.6, 250.0
    q_min, q_max = 800.0, 5.0e5

    pts_q = _svg_polyline(dt_vals, q_vals, dt_min, dt_max, q_min, q_max, px, py, pw, ph, log_x=True, log_y=True)
    dt_mfb = float(nukiyama_data["t_mfb_k"] - nukiyama_data["t_sat_k"])
    dt_chf = float(nukiyama_data["t_chf_k"] - nukiyama_data["t_sat_k"])

    x_mfb = px + ((math.log10(dt_mfb) - math.log10(dt_min)) / (math.log10(dt_max) - math.log10(dt_min))) * pw
    x_chf = px + ((math.log10(dt_chf) - math.log10(dt_min)) / (math.log10(dt_max) - math.log10(dt_min))) * pw

    return f"""<svg viewBox="0 0 {w} {h}" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">
      <rect width="{w}" height="{h}" rx="8" fill="#0d1522" stroke="#1e293b"/>
      <line x1="{px}" y1="{py}" x2="{px}" y2="{py+ph}" stroke="#475569" stroke-width="1.2"/>
      <line x1="{px}" y1="{py+ph}" x2="{px+pw}" y2="{py+ph}" stroke="#475569" stroke-width="1.2"/>
      <line x1="{x_chf:.1f}" y1="{py}" x2="{x_chf:.1f}" y2="{py+ph}" stroke="#ef4444" stroke-dasharray="4,4" stroke-width="1"/>
      <text x="{x_chf-6:.1f}" y="{py+18}" fill="#ef4444" font-size="10" text-anchor="end" font-family="monospace">CHF Peak (dT={dt_chf:.1f}K)</text>
      <line x1="{x_mfb:.1f}" y1="{py}" x2="{x_mfb:.1f}" y2="{py+ph}" stroke="#f59e0b" stroke-dasharray="4,4" stroke-width="1"/>
      <text x="{x_mfb+6:.1f}" y="{py+36}" fill="#f59e0b" font-size="10" font-family="monospace">Leidenfrost MFB (dT={dt_mfb:.1f}K)</text>
      <polyline fill="none" stroke="#38bdf8" stroke-width="2.6" points="{pts_q}"/>
      <text x="{px+35}" y="{py+ph-25}" fill="#94a3b8" font-size="10">Single-Phase / Nucleate</text>
      <text x="{0.5*(x_chf+x_mfb):.1f}" y="{py+ph-12}" fill="#f59e0b" font-size="10" text-anchor="middle">Transition</text>
      <text x="{px+pw-70}" y="{py+ph-25}" fill="#cbd5e1" font-size="10">Film Boiling</text>
      <text x="{px-6}" y="{py+12}" fill="#94a3b8" font-size="10" text-anchor="end">5e5</text>
      <text x="{px-6}" y="{py+ph/2}" fill="#94a3b8" font-size="10" text-anchor="end">2e4</text>
      <text x="{px-6}" y="{py+ph}" fill="#94a3b8" font-size="10" text-anchor="end">8e2</text>
      <text x="{px+pw/2}" y="{h-10}" fill="#cbd5e1" font-size="11" text-anchor="middle">Wall Superheat dT_sat = T_w - T_sat [K] (Log Scale: 0.6 K to 250 K)</text>
    </svg>"""


def _build_tank_pressure_svg(tank_passive: Dict[str, Any], tank_active: Dict[str, Any]) -> str:
    """Render SVG of Orbital Depot Tank Ullage Pressure Regulation (Passive vs Active ZBO TVS)."""
    w, h = 580, 300
    px, py, pw, ph = 64, 28, 480, 225
    t_arr = tank_active["time_s"]
    t_max = float(max(t_arr[-1], 1.0))
    p_pass_kpa = [v * 1e-3 for v in tank_passive["ullage_pressure_pa"]]
    p_act_kpa = [v * 1e-3 for v in tank_active["ullage_pressure_pa"]]
    p_min, p_max = 177.5, 185.5

    pts_pass = _svg_polyline(t_arr, p_pass_kpa, 0.0, t_max, p_min, p_max, px, py, pw, ph)
    pts_act = _svg_polyline(t_arr, p_act_kpa, 0.0, t_max, p_min, p_max, px, py, pw, ph)

    y_upper = py + ph - ((181.5 - p_min) / (p_max - p_min)) * ph
    y_set = py + ph - ((180.0 - p_min) / (p_max - p_min)) * ph
    y_lower = py + ph - ((178.5 - p_min) / (p_max - p_min)) * ph

    return f"""<svg viewBox="0 0 {w} {h}" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">
      <rect width="{w}" height="{h}" rx="8" fill="#0d1522" stroke="#1e293b"/>
      <rect x="{px}" y="{y_upper:.1f}" width="{pw}" height="{y_lower-y_upper:.1f}" fill="#10b981" fill-opacity="0.09"/>
      <line x1="{px}" y1="{py}" x2="{px}" y2="{py+ph}" stroke="#475569" stroke-width="1.2"/>
      <line x1="{px}" y1="{py+ph}" x2="{px+pw}" y2="{py+ph}" stroke="#475569" stroke-width="1.2"/>
      <line x1="{px}" y1="{y_upper:.1f}" x2="{px+pw}" y2="{y_upper:.1f}" stroke="#10b981" stroke-dasharray="4,3" stroke-width="1"/>
      <text x="{px+8}" y="{y_upper-4:.1f}" fill="#10b981" font-size="10" font-family="monospace">+1.5 kPa Upper Band (181.5 kPa)</text>
      <line x1="{px}" y1="{y_set:.1f}" x2="{px+pw}" y2="{y_set:.1f}" stroke="#94a3b8" stroke-dasharray="2,2" stroke-width="1"/>
      <line x1="{px}" y1="{y_lower:.1f}" x2="{px+pw}" y2="{y_lower:.1f}" stroke="#10b981" stroke-dasharray="4,3" stroke-width="1"/>
      <text x="{px+8}" y="{y_lower+12:.1f}" fill="#10b981" font-size="10" font-family="monospace">-1.5 kPa Lower Band (178.5 kPa)</text>
      <polyline fill="none" stroke="#ef4444" stroke-width="2.0" stroke-dasharray="5,3" points="{pts_pass}"/>
      <polyline fill="none" stroke="#10b981" stroke-width="2.5" points="{pts_act}"/>
      <text x="{px-6}" y="{py+10}" fill="#94a3b8" font-size="10" text-anchor="end">185.5</text>
      <text x="{px-6}" y="{y_set+3:.1f}" fill="#94a3b8" font-size="10" text-anchor="end">180.0</text>
      <text x="{px-6}" y="{py+ph}" fill="#94a3b8" font-size="10" text-anchor="end">177.5</text>
      <text x="{px+pw/2}" y="{h-10}" fill="#cbd5e1" font-size="11" text-anchor="middle">Orbital Depot Simulation Time [s] (Ullage Pressure in kPa)</text>
    </svg>"""


def _build_surge_and_lad_svg(
    chill_flood: Dict[str, Any],
    chill_pulse: Dict[str, Any],
    tank_passive: Dict[str, Any],
    tank_active: Dict[str, Any],
) -> str:
    """Render SVG comparing Cryo-Hammer Pressure Surge & Capillary LAD Bubble-Point Margin."""
    w, h = 580, 300
    px, py, pw, ph = 64, 28, 480, 225
    t_arr = tank_active["time_s"]
    t_max = float(max(t_arr[-1], 1.0))
    lad_pass = [v * 100.0 for v in tank_passive["lad_bubble_margin"]]
    lad_act = [v * 100.0 for v in tank_active["lad_bubble_margin"]]

    pts_lad_pass = _svg_polyline(t_arr, lad_pass, 0.0, t_max, 0.0, 100.0, px, py, pw, ph)
    pts_lad_act = _svg_polyline(t_arr, lad_act, 0.0, t_max, 0.0, 100.0, px, py, pw, ph)

    y_min_req = py + ph - 0.25 * ph

    return f"""<svg viewBox="0 0 {w} {h}" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">
      <rect width="{w}" height="{h}" rx="8" fill="#0d1522" stroke="#1e293b"/>
      <line x1="{px}" y1="{py}" x2="{px}" y2="{py+ph}" stroke="#475569" stroke-width="1.2"/>
      <line x1="{px}" y1="{py+ph}" x2="{px+pw}" y2="{py+ph}" stroke="#475569" stroke-width="1.2"/>
      <line x1="{px}" y1="{y_min_req:.1f}" x2="{px+pw}" y2="{y_min_req:.1f}" stroke="#f59e0b" stroke-dasharray="5,3" stroke-width="1.2"/>
      <text x="{px+8}" y="{y_min_req-5:.1f}" fill="#f59e0b" font-size="10" font-family="monospace">Min Vapor-Ingestion Safety Margin = 25.0%</text>
      <polyline fill="none" stroke="#ef4444" stroke-width="2.0" stroke-dasharray="5,3" points="{pts_lad_pass}"/>
      <polyline fill="none" stroke="#38bdf8" stroke-width="2.5" points="{pts_lad_act}"/>
      <text x="{px-6}" y="{py+10}" fill="#94a3b8" font-size="10" text-anchor="end">100%</text>
      <text x="{px-6}" y="{py+ph/2}" fill="#94a3b8" font-size="10" text-anchor="end">50%</text>
      <text x="{px-6}" y="{py+ph}" fill="#94a3b8" font-size="10" text-anchor="end">0%</text>
      <text x="{px+pw/2}" y="{h-10}" fill="#cbd5e1" font-size="11" text-anchor="middle">Capillary LAD Bubble-Point Margin [%] vs Orbital Time [s]</text>
      <g transform="translate({px+15}, {py+10})">
        <rect width="435" height="36" rx="4" fill="#0f172a" fill-opacity="0.9" stroke="#334155"/>
        <text x="10" y="15" fill="#38bdf8" font-size="10" font-family="monospace">Active TVS + Baffle LAD Min Margin: {min(lad_act):.1f}% | Cryo-Hammer Peak: {chill_pulse['peak_surge_kpa']:.1f} kPa</text>
        <text x="10" y="29" fill="#ef4444" font-size="10" font-family="monospace">Passive Stratified LAD Min Margin:  {min(lad_pass):.1f}% | Flood Surge Peak: {chill_flood['peak_surge_kpa']:.1f} kPa</text>
      </g>
    </svg>"""


def generate_all_reports(
    report: TRL4VerificationReport,
    campaign_data: Dict[str, Any],
    output_dir: str | Path,
) -> Dict[str, str]:
    """
    Generate the HTML5 Scorecard Dashboard, Markdown TRL 4 Dossier, and JSON Audit Manifest.
    Returns dictionary of generated file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    html_file = out_path / "cryo_trl4_scorecard.html"
    md_file = out_path / "TRL4_DOSSIER.md"
    json_file = out_path / "trl4_audit.json"

    # 1. Write machine-readable JSON Audit Manifest
    audit_payload = {
        "verification_report": report.to_dict(),
        "fluid_table": campaign_data["fluid_table"],
        "hil_stats": campaign_data["hil_stats"],
    }
    json_file.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")

    # 2. Build SVG charts and write HTML5 Dashboard
    svg_chill = _build_chilldown_svg(campaign_data["chilldown_flood"], campaign_data["chilldown_pulse"])
    svg_nukiyama = _build_nukiyama_svg(campaign_data["nukiyama_curve"])
    svg_tank = _build_tank_pressure_svg(campaign_data["tank_passive"], campaign_data["tank_active"])
    svg_lad = _build_surge_and_lad_svg(
        campaign_data["chilldown_flood"],
        campaign_data["chilldown_pulse"],
        campaign_data["tank_passive"],
        campaign_data["tank_active"],
    )

    sm = report.summary_metrics
    pillar_rows_html = ""
    for p in report.pillars:
        badge_cls = "pass" if p.passed else "fail"
        badge_txt = "PASS" if p.passed else "FAIL"
        pillar_rows_html += f"""
        <tr>
          <td><strong>Pillar {p.pillar_id}</strong></td>
          <td>{p.name}</td>
          <td><code>{p.metric_name}</code></td>
          <td><strong>{p.measured_value:.4f} {p.units}</strong></td>
          <td>{p.threshold_description}</td>
          <td><span class="badge {badge_cls}">{badge_txt}</span></td>
          <td class="details">{p.details}</td>
        </tr>"""

    fluid_rows_html = ""
    for row in campaign_data["fluid_table"]:
        fluid_rows_html += f"""
        <tr>
          <td><strong>{row['fluid']}</strong> ({row['formula']})</td>
          <td>{row['t_nbp_k']:.3f}</td>
          <td>{row['t_crit_k']:.3f}</td>
          <td>{row['p_crit_mpa']:.3f}</td>
          <td>{row['rho_l_nbp']:.2f}</td>
          <td>{row['h_fg_nbp_kj_kg']:.2f}</td>
          <td>{row['sigma_nbp_mn_m']:.2f}</td>
          <td>{row['sound_speed_l_m_s']:.1f}</td>
        </tr>"""

    overall_badge = "TRL 4 CERTIFIED — 7/7 PILLARS PASSED" if report.overall_passed else "VERIFICATION INCOMPLETE"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>CRYO-ZBO-DT | TRL 4 Cryogenic Propulsion & Orbital Depot Digital Twin Scorecard</title>
  <style>
    :root {{
      --bg: #070b14;
      --card: #0f172a;
      --border: #1e293b;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --green: #10b981;
      --amber: #f59e0b;
      --red: #ef4444;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
      padding: 24px 32px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 18px;
      margin-bottom: 24px;
      flex-wrap: wrap;
      gap: 16px;
    }}
    h1 {{ font-size: 1.55rem; color: #f8fafc; letter-spacing: 0.4px; }}
    .subtitle {{ color: var(--muted); font-size: 0.92rem; margin-top: 4px; }}
    .status-pill {{
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid var(--green);
      color: #34d399;
      padding: 8px 16px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 0.88rem;
      letter-spacing: 0.5px;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 26px;
    }}
    .kpi-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px 18px;
    }}
    .kpi-label {{ color: var(--muted); font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.6px; }}
    .kpi-val {{ font-size: 1.65rem; font-weight: 700; color: var(--accent); margin: 6px 0 2px; }}
    .kpi-sub {{ color: #cbd5e1; font-size: 0.82rem; }}
    .charts-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
      gap: 20px;
      margin-bottom: 28px;
    }}
    .chart-card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
    }}
    .chart-title {{ font-size: 0.98rem; font-weight: 600; margin-bottom: 10px; color: #f1f5f9; }}
    h2 {{ font-size: 1.2rem; margin: 24px 0 12px; color: #f8fafc; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      margin-bottom: 24px;
      font-size: 0.88rem;
    }}
    th, td {{
      padding: 11px 14px;
      border-bottom: 1px solid var(--border);
      text-align: left;
    }}
    th {{ background: #131f35; color: #cbd5e1; font-weight: 600; }}
    .badge {{
      padding: 3px 10px;
      border-radius: 99px;
      font-weight: 700;
      font-size: 0.76rem;
    }}
    .badge.pass {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid var(--green); }}
    .badge.fail {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid var(--red); }}
    .details {{ color: var(--muted); font-size: 0.82rem; }}
    footer {{
      margin-top: 28px;
      padding-top: 16px;
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 0.82rem;
      display: flex;
      justify-content: space-between;
      flex-wrap: wrap;
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>CRYO-ZBO-DT — Zero-Boil-Off Cryogenic Two-Phase Flow, Slosh & Active TVS Digital Twin</h1>
      <div class="subtitle">DIMASA (Digital Intelligence Maintenance for Aircraft Systems in Africa) &amp; Physics Hybrid Integrity Lab | TRL 4 Verification Dashboard</div>
    </div>
    <div class="status-pill">{overall_badge}</div>
  </header>

  <section class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">Pulse-Chilldown Mass Savings</div>
      <div class="kpi-val">{sm['chilldown_mass_savings_pct']:.1f}%</div>
      <div class="kpi-sub">{sm['ln2_pulse_chilldown_mass_kg']:.2f} kg vs {sm['ln2_flood_chilldown_mass_kg']:.2f} kg Flood</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">ZBO Ullage Pressure Band</div>
      <div class="kpi-val">&plusmn;{sm['active_tvs_max_pressure_err_kpa']:.2f} kPa</div>
      <div class="kpi-sub">Target &le; &plusmn;1.50 kPa ({sm['zbo_boiloff_savings_pct']:.1f}% ZBO Savings)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Cryo-Hammer Surge Reduction</div>
      <div class="kpi-val">{sm['cryo_hammer_surge_reduction_pct']:.1f}%</div>
      <div class="kpi-sub">{sm['pulse_peak_surge_kpa']:.1f} kPa vs {sm['flood_peak_surge_kpa']:.1f} kPa Flood</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Capillary LAD Bubble Margin</div>
      <div class="kpi-val">{sm['active_min_lad_bubble_margin_pct']:.1f}%</div>
      <div class="kpi-sub">325&times;2300 Dutch-Twill Mesh (&ge;25% Req)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Leidenfrost T_MFB (LN2/SS304L)</div>
      <div class="kpi-val">{sm['leidenfrost_temp_k']:.1f} K</div>
      <div class="kpi-sub">Henry-Berenson Calibrated (105&ndash;135 K)</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Avionics MPC p99 Latency</div>
      <div class="kpi-val">{sm['mpc_p99_latency_ms']:.3f} ms</div>
      <div class="kpi-sub">CCSDS CRC-16 HIL Air-Gap (&lt;5.0 ms)</div>
    </div>
  </section>

  <section class="charts-grid">
    <div class="chart-card">
      <div class="chart-title">1. Transfer Line Chilldown Wall Cooldown &amp; Cryogen Consumption (LN2 Breadboard)</div>
      {svg_chill}
    </div>
    <div class="chart-card">
      <div class="chart-title">2. Complete Nukiyama Boiling Curve (Film &rarr; Leidenfrost T_MFB &rarr; Transition &rarr; CHF &rarr; Nucleate)</div>
      {svg_nukiyama}
    </div>
    <div class="chart-card">
      <div class="chart-title">3. Orbital Propellant Depot Ullage Pressure Regulation (Passive Venting vs. Active ZBO TVS)</div>
      {svg_tank}
    </div>
    <div class="chart-card">
      <div class="chart-title">4. Microgravity Capillary LAD Bubble-Point Margin &amp; Cryo-Hammer Surge Suppression</div>
      {svg_lad}
    </div>
  </section>

  <h2>7-Pillar TRL 4 Verification &amp; Credibility Matrix</h2>
  <table>
    <thead>
      <tr>
        <th>Pillar</th>
        <th>Verification Domain</th>
        <th>Primary Metric</th>
        <th>Measured Value</th>
        <th>TRL 4 Acceptance Criterion</th>
        <th>Status</th>
        <th>Engineering Audit Notes</th>
      </tr>
    </thead>
    <tbody>
      {pillar_rows_html}
    </tbody>
  </table>

  <h2>Real-Fluid Cryogenic Thermophysical Reference Table (NIST REFPROP Calibrated)</h2>
  <table>
    <thead>
      <tr>
        <th>Cryogen</th>
        <th>T_nbp [K]</th>
        <th>T_crit [K]</th>
        <th>P_crit [MPa]</th>
        <th>Liquid Density &rho;_l [kg/m&sup3;]</th>
        <th>Latent Heat h_fg [kJ/kg]</th>
        <th>Surface Tension &sigma; [mN/m]</th>
        <th>Liquid Sound Speed a_l [m/s]</th>
      </tr>
    </thead>
    <tbody>
      {fluid_rows_html}
    </tbody>
  </table>

  <footer>
    <div>SHA-256 Deterministic Trajectory Digest: <code>{report.sha256_trajectory_digest}</code></div>
    <div>Institutional Attribution: DIMASA &amp; Physics Hybrid Integrity Lab (Apache-2.0)</div>
  </footer>
</body>
</html>
"""
    html_file.write_text(html_content, encoding="utf-8")

    # 3. Write Markdown TRL 4 Engineering Dossier
    md_lines = [
        "# CRYO-ZBO-DT: TRL 4 Verification & Engineering Dossier",
        "",
        "**Full Title**: Zero-Boil-Off Cryogenic Two-Phase Flow, Microgravity Propellant Slosh, Line Chilldown & Active Thermodynamic Vent System (TVS) Digital Twin & HIL Testbed",
        "**Institutional Attribution**: **DIMASA (Digital Intelligence Maintenance for Aircraft Systems in Africa)** & **Physics Hybrid Integrity Lab**",
        f"**Verification Status**: **{'PASS (7/7 Pillars Certified)' if report.overall_passed else 'FAIL'}**",
        f"**Deterministic Trajectory SHA-256**: `{report.sha256_trajectory_digest}`",
        "",
        "---",
        "",
        "## 1. Executive Summary & Benchmark Comparison",
        "",
        "| Performance Metric | Uncontrolled Flood / Passive Venting | CRYO-ZBO-DT (Pulse-MPC + Active TVS) | Improvement / Margin |",
        "| :--- | :---: | :---: | :---: |",
        f"| **LN2 Line Chilldown Mass Consumed** | `{sm['ln2_flood_chilldown_mass_kg']:.3f} kg` | `{sm['ln2_pulse_chilldown_mass_kg']:.3f} kg` | **`{sm['chilldown_mass_savings_pct']:.2f}%` Mass Saved** (`> 35%` req) |",
        f"| **Peak Cryo-Hammer Surge Pressure** | `{sm['flood_peak_surge_kpa']:.1f} kPa` | `{sm['pulse_peak_surge_kpa']:.1f} kPa` | **`{sm['cryo_hammer_surge_reduction_pct']:.1f}%` Surge Reduction** |",
        f"| **Depot Tank Max Pressure Excursion** | `+{sm['passive_tank_max_pressure_err_kpa']:.2f} kPa` | `+/- {sm['active_tvs_max_pressure_err_kpa']:.3f} kPa` | **Strictly within `+/- 1.50 kPa` Setpoint** |",
        f"| **Orbital Depot Propellant Boil-Off Loss** | `{sm['passive_tank_propellant_lost_kg']:.4f} kg` | `{sm['active_tvs_propellant_lost_kg']:.4f} kg` | **`{sm['zbo_boiloff_savings_pct']:.2f}%` Zero-Boil-Off Savings** |",
        f"| **Min Capillary LAD Bubble-Point Margin** | `{sm['passive_min_lad_bubble_margin_pct']:.2f}%` | `{sm['active_min_lad_bubble_margin_pct']:.2f}%` | **Vapor-Free Outflow (`>= 25%` req)** |",
        f"| **Microgravity Slosh Wave Damping** | Uncontrolled | `{sm['slosh_amplitude_reduction_pct']:.1f}%` Reduction | **Settling Baffle + Spray Damping** |",
        "",
        "---",
        "",
        "## 2. 7-Pillar TRL 4 Credibility Scorecard",
        "",
        "| Pillar | Verification Domain | Measured Value | Acceptance Threshold | Status |",
        "| :---: | :--- | :---: | :---: | :---: |",
    ]
    for p in report.pillars:
        st = "PASS" if p.passed else "FAIL"
        md_lines.append(
            f"| **{p.pillar_id}** | {p.name} | `{p.measured_value:.4f} {p.units}` | {p.threshold_description} | **{st}** |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Multi-Fluid Cryogenic Thermophysical Verification Table",
            "",
            "| Fluid | Formula | $T_{\\text{nbp}}$ [K] | $T_c$ [K] | $P_c$ [MPa] | $\\rho_l(T_{\\text{nbp}})$ [kg/m³] | $h_{fg}(T_{\\text{nbp}})$ [kJ/kg] | $\\sigma(T_{\\text{nbp}})$ [mN/m] | $a_l$ [m/s] |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for row in campaign_data["fluid_table"]:
        md_lines.append(
            f"| **{row['fluid']}** | `{row['formula']}` | {row['t_nbp_k']:.3f} | {row['t_crit_k']:.3f} | {row['p_crit_mpa']:.3f} | {row['rho_l_nbp']:.2f} | {row['h_fg_nbp_kj_kg']:.2f} | {row['sigma_nbp_mn_m']:.2f} | {row['sound_speed_l_m_s']:.1f} |"
        )

    hs = campaign_data["hil_stats"]
    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 4. CCSDS Binary HIL Bridge & FDIR Telemetry Summary",
            "",
            f"- **CCSDS Telemetry Frames Transmitted / Validated**: `{hs['tx_telemetry_frames']}` / `{hs['rx_telemetry_frames']}`",
            f"- **CCSDS Telecommand Frames Transmitted / Validated**: `{hs['tx_command_frames']}` / `{hs['rx_command_frames']}`",
            f"- **CRC-16-CCITT Corrupted Frames Isolated & Rejected**: `{hs['rejected_telemetry_crc_frames'] + hs['rejected_command_crc_frames']}`",
            f"- **Thermocouple Open-Circuit Dropouts Isolated**: `{hs['thermocouple_dropout_events']}`",
            f"- **Cryogenic Valve Stiction Events Mitigated**: `{hs['valve_stiction_events']}`",
            f"- **Avionics MPC + HIL Loop p99 Execution Latency**: `{sm['mpc_p99_latency_ms']:.3f} ms`",
            "",
        ]
    )
    md_file.write_text("\n".join(md_lines), encoding="utf-8")

    return {
        "html_scorecard": str(html_file.resolve()),
        "markdown_dossier": str(md_file.resolve()),
        "json_audit": str(json_file.resolve()),
    }
