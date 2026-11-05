"""
CRYO-ZBO-DT: Zero-Boil-Off Cryogenic Two-Phase Flow, Microgravity Propellant Slosh,
Line Chilldown & Active Thermodynamic Vent System (TVS) Digital Twin & HIL Testbed.

Developed under the Digital Intelligence Maintenance for Aircraft Systems in Africa (DIMASA)
and Physics Hybrid Integrity Lab (PHIL) engineering standard (TRL 4 Validated).
"""

__version__ = "1.0.0"
__author__ = "DIMASA & Physics Hybrid Integrity Lab"

from cryo_zbo_dt.cryo_fluids import (
    CryoFluidType,
    CryoFluidProperties,
    CryoFluidEOS,
    get_cryo_fluid,
)
from cryo_zbo_dt.chilldown_twophase import (
    BoilingRegime,
    ChilldownLineConfig,
    ChilldownStepResult,
    TwoPhaseChilldownSolver,
)
from cryo_zbo_dt.depot_tank_slosh import (
    DepotTankConfig,
    TankStepState,
    MicrogravityDepotTankSimulator,
)
from cryo_zbo_dt.tvs_mpc_controller import (
    PulseChilldownMPCConfig,
    PulseChilldownMPC,
    TVSMPCConfig,
    ActiveTVSMPCController,
)
from cryo_zbo_dt.hil_bridge import (
    HILTransportMode,
    CryoTelemetryPacket,
    CryoCommandPacket,
    FaultInjectionConfig,
    CryoAvionicsHILBridge,
    crc16_ccitt,
)
from cryo_zbo_dt.verification import (
    PillarResult,
    TRL4VerificationReport,
    evaluate_trl4_campaign,
)
from cryo_zbo_dt.reporting import (
    generate_all_reports,
)

__all__ = [
    "__version__",
    "CryoFluidType",
    "CryoFluidProperties",
    "CryoFluidEOS",
    "get_cryo_fluid",
    "BoilingRegime",
    "ChilldownLineConfig",
    "ChilldownStepResult",
    "TwoPhaseChilldownSolver",
    "DepotTankConfig",
    "TankStepState",
    "MicrogravityDepotTankSimulator",
    "PulseChilldownMPCConfig",
    "PulseChilldownMPC",
    "TVSMPCConfig",
    "ActiveTVSMPCController",
    "HILTransportMode",
    "CryoTelemetryPacket",
    "CryoCommandPacket",
    "FaultInjectionConfig",
    "CryoAvionicsHILBridge",
    "crc16_ccitt",
    "PillarResult",
    "TRL4VerificationReport",
    "evaluate_trl4_campaign",
    "generate_all_reports",
]
