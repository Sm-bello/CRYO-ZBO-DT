"""
TRL 4 SIL/PIL/HIL Binary CCSDS Cryogenic Avionics Valve & DAQ Telemetry Bridge.

Implements an air-gapped binary communication protocol between the Cryogenic Plant
Simulator (Transfer Line Chilldown + Orbital Depot Tank) and the Active TVS / Pulse-Chilldown
MPC Flight Software:
  - Standard CCSDS Space Packet framing with 0x1ACFFC1D Attached Sync Marker (ASM)
  - CRC-16-CCITT (polynomial 0x1021, initial value 0xFFFF) frame integrity verification
  - Three selectable transport backends:
      1. LOOPBACK : Deterministic in-memory binary byte-stream air-gap (zero shared object references)
      2. UDP      : Localhost/network UDP datagram socket bridge for PLC / Flight Computer PIL
      3. SERIAL   : Hardware UART serial bridge via pyserial for physical LN2 benchtop HIL rigs
  - Fault Injection & FDIR Validation:
      * Thermocouple open-circuit dropout (NaN / rail spike injection)
      * Silicon-diode temperature sensor calibration drift
      * Cryogenic proportional valve stem stiction / deadband freeze
      * Bit-flip wire corruption for verifying CRC-16-CCITT rejection & safe-hold command retention
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import socket
import struct
from typing import Tuple

import numpy as np

try:
    import serial  # type: ignore
except ImportError:  # pragma: no cover
    serial = None


CCSDS_SYNC_MARKER = 0x1ACFFC1D
APID_CRYO_TELEMETRY = 0x0241
APID_CRYO_COMMAND = 0x0242


class HILTransportMode(str, Enum):
    """Supported physical/logical transport layers for the HIL air-gap bridge."""

    LOOPBACK = "LOOPBACK"
    UDP = "UDP"
    SERIAL = "SERIAL"


def crc16_ccitt(data: bytes, init_crc: int = 0xFFFF) -> int:
    """
    Compute CRC-16-CCITT (polynomial 0x1021, init 0xFFFF, non-reflected) over a byte buffer.
    Standardized by CCSDS 131.0-B and ECSS-E-ST-70-41C for spacecraft telemetry/telecommand.
    """
    crc = init_crc & 0xFFFF
    for byte in data:
        crc ^= (byte & 0xFF) << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


@dataclass
class CryoTelemetryPacket:
    """
    Binary CCSDS Telemetry Frame from Cryogenic DAQ to Avionics Controller.
    Payload layout (big-endian):
      - seq_count          : uint16
      - timestamp_s        : float32
      - wall_temps_k[4]    : 4 x float32 (axial thermocouple stations T_w1..T_w4)
      - ullage_pressure_pa : float32
      - t_bulk_k           : float32
      - t_interface_k      : float32
      - line_pressure_pa   : float32
      - void_fraction_out  : float32
      - lad_bubble_margin  : float32
      - status_flags       : uint16
    Total payload = 44 bytes; with 4-byte ASM + 6-byte CCSDS header + 2-byte CRC16 = 56 bytes.
    """

    seq_count: int
    timestamp_s: float
    wall_temps_k: Tuple[float, float, float, float]
    ullage_pressure_pa: float
    t_bulk_k: float
    t_interface_k: float
    line_pressure_pa: float
    void_fraction_out: float
    lad_bubble_margin: float
    status_flags: int = 0

    _STRUCT_FMT = ">IHf4fffffffHH"  # ASM(4) + APID_Ver(2) + Seq(2) + Len(2) -> custom compact CCSDS header
    # Explicit CCSDS header + payload + CRC16 format:
    # > I (ASM:4) H (Ver/APID:2) H (SeqFlags/Count:2) H (DataLen:2) f (time:4) 4f (Tw:16) 6f (sensors:24) H (flags:2) H (crc:2) = 58 bytes
    _FRAME_NO_CRC_FMT = ">IHHHf4f6fH"
    _FULL_FRAME_FMT = ">IHHHf4f6fHH"
    FRAME_SIZE_BYTES = struct.calcsize(_FULL_FRAME_FMT)

    def pack(self) -> bytes:
        """Serialize telemetry frame into CCSDS binary packet with CRC-16-CCITT."""
        ver_apid = (0 << 13) | (0 << 12) | (1 << 11) | (APID_CRYO_TELEMETRY & 0x07FF)
        seq_word = (3 << 14) | (self.seq_count & 0x3FFF)
        data_len = 46 - 1  # CCSDS Data Length = (octets in data field) - 1
        tw = self.wall_temps_k
        header_and_payload = struct.pack(
            self._FRAME_NO_CRC_FMT,
            CCSDS_SYNC_MARKER,
            ver_apid,
            seq_word,
            data_len,
            float(self.timestamp_s),
            float(tw[0]),
            float(tw[1]),
            float(tw[2]),
            float(tw[3]),
            float(self.ullage_pressure_pa),
            float(self.t_bulk_k),
            float(self.t_interface_k),
            float(self.line_pressure_pa),
            float(self.void_fraction_out),
            float(self.lad_bubble_margin),
            int(self.status_flags) & 0xFFFF,
        )
        # Compute CRC-16-CCITT over CCSDS header + payload (excluding 4-byte ASM sync marker)
        crc = crc16_ccitt(header_and_payload[4:])
        return header_and_payload + struct.pack(">H", crc)

    @classmethod
    def unpack(cls, raw_bytes: bytes) -> "CryoTelemetryPacket":
        """Deserialize and validate a binary CCSDS telemetry frame."""
        if len(raw_bytes) != cls.FRAME_SIZE_BYTES:
            raise ValueError(f"Invalid telemetry frame length: {len(raw_bytes)} != {cls.FRAME_SIZE_BYTES}")
        unpacked = struct.unpack(cls._FULL_FRAME_FMT, raw_bytes)
        asm = unpacked[0]
        if asm != CCSDS_SYNC_MARKER:
            raise ValueError(f"Invalid CCSDS Sync Marker: 0x{asm:08X}")
        rx_crc = unpacked[-1]
        calc_crc = crc16_ccitt(raw_bytes[4:-2])
        if rx_crc != calc_crc:
            raise ValueError(f"CRC-16-CCITT mismatch on telemetry frame: rx=0x{rx_crc:04X} != calc=0x{calc_crc:04X}")

        seq_count = unpacked[2] & 0x3FFF
        timestamp_s = float(unpacked[4])
        tw = (float(unpacked[5]), float(unpacked[6]), float(unpacked[7]), float(unpacked[8]))
        return cls(
            seq_count=seq_count,
            timestamp_s=timestamp_s,
            wall_temps_k=tw,
            ullage_pressure_pa=float(unpacked[9]),
            t_bulk_k=float(unpacked[10]),
            t_interface_k=float(unpacked[11]),
            line_pressure_pa=float(unpacked[12]),
            void_fraction_out=float(unpacked[13]),
            lad_bubble_margin=float(unpacked[14]),
            status_flags=int(unpacked[15]),
        )


@dataclass
class CryoCommandPacket:
    """
    Binary CCSDS Telecommand Frame from Avionics MPC Controller to Cryogenic Valve Actuators.
    Fields:
      - seq_count         : uint16
      - u_chilldown_valve : float32 [0..1]
      - u_jt_valve        : float32 [0..1]
      - u_spray_pump      : float32 [0..1]
      - u_cryocooler      : float32 [0..1]
      - command_flags     : uint16
    """

    seq_count: int
    u_chilldown_valve: float
    u_jt_valve: float
    u_spray_pump: float
    u_cryocooler: float
    command_flags: int = 0

    _FRAME_NO_CRC_FMT = ">IHHH4fH"
    _FULL_FRAME_FMT = ">IHHH4fHH"
    FRAME_SIZE_BYTES = struct.calcsize(_FULL_FRAME_FMT)

    def pack(self) -> bytes:
        """Serialize telecommand into CCSDS binary packet with CRC-16-CCITT."""
        ver_apid = (0 << 13) | (1 << 12) | (1 << 11) | (APID_CRYO_COMMAND & 0x07FF)
        seq_word = (3 << 14) | (self.seq_count & 0x3FFF)
        data_len = 18 - 1
        header_and_payload = struct.pack(
            self._FRAME_NO_CRC_FMT,
            CCSDS_SYNC_MARKER,
            ver_apid,
            seq_word,
            data_len,
            float(self.u_chilldown_valve),
            float(self.u_jt_valve),
            float(self.u_spray_pump),
            float(self.u_cryocooler),
            int(self.command_flags) & 0xFFFF,
        )
        crc = crc16_ccitt(header_and_payload[4:])
        return header_and_payload + struct.pack(">H", crc)

    @classmethod
    def unpack(cls, raw_bytes: bytes) -> "CryoCommandPacket":
        """Deserialize and verify CRC-16-CCITT of a binary CCSDS telecommand frame."""
        if len(raw_bytes) != cls.FRAME_SIZE_BYTES:
            raise ValueError(f"Invalid command frame length: {len(raw_bytes)} != {cls.FRAME_SIZE_BYTES}")
        unpacked = struct.unpack(cls._FULL_FRAME_FMT, raw_bytes)
        asm = unpacked[0]
        if asm != CCSDS_SYNC_MARKER:
            raise ValueError(f"Invalid CCSDS Sync Marker: 0x{asm:08X}")
        rx_crc = unpacked[-1]
        calc_crc = crc16_ccitt(raw_bytes[4:-2])
        if rx_crc != calc_crc:
            raise ValueError(f"CRC-16-CCITT mismatch on command frame: rx=0x{rx_crc:04X} != calc=0x{calc_crc:04X}")

        return cls(
            seq_count=unpacked[2] & 0x3FFF,
            u_chilldown_valve=float(unpacked[4]),
            u_jt_valve=float(unpacked[5]),
            u_spray_pump=float(unpacked[6]),
            u_cryocooler=float(unpacked[7]),
            command_flags=int(unpacked[8]),
        )


@dataclass
class FaultInjectionConfig:
    """Configuration for deterministic sensor, actuator, and link fault injection."""

    enable_thermocouple_dropout: bool = False
    dropout_start_s: float = 35.0
    dropout_end_s: float = 48.0
    dropout_channel_idx: int = 1

    enable_sensor_drift: bool = False
    pressure_drift_pa_per_s: float = 1.2

    enable_valve_stiction: bool = False
    valve_stiction_deadband: float = 0.035

    corrupt_every_nth_frame: int = 0  # If > 0, flips a payload bit every Nth frame to test CRC-16 FDIR


@dataclass
class HILBridgeStats:
    """Telemetry and command link health counters."""

    tx_telemetry_frames: int = 0
    rx_telemetry_frames: int = 0
    rejected_telemetry_crc_frames: int = 0
    tx_command_frames: int = 0
    rx_command_frames: int = 0
    rejected_command_crc_frames: int = 0
    thermocouple_dropout_events: int = 0
    valve_stiction_events: int = 0
    safe_hold_fallbacks: int = 0


class CryoAvionicsHILBridge:
    """
    Air-gapped CCSDS Binary Hardware-in-the-Loop (HIL) / Processor-in-the-Loop (PIL)
    Bridge with CRC-16-CCITT verification and deterministic fault injection.
    """

    def __init__(
        self,
        mode: HILTransportMode | str = HILTransportMode.LOOPBACK,
        fault_config: FaultInjectionConfig | None = None,
        udp_host: str = "127.0.0.1",
        udp_port: int = 19440,
        serial_port: str = "COM3",
        serial_baudrate: int = 115200,
    ) -> None:
        if isinstance(mode, str):
            mode = HILTransportMode(mode.upper())
        self.mode = mode
        self.fault_cfg = fault_config or FaultInjectionConfig()
        self.udp_host = udp_host
        self.udp_port = udp_port
        self.serial_port = serial_port
        self.serial_baudrate = serial_baudrate

        self.stats = HILBridgeStats()
        self._seq_tlm: int = 0
        self._seq_cmd: int = 0
        self._last_valid_tlm = CryoTelemetryPacket(
            seq_count=0,
            timestamp_s=0.0,
            wall_temps_k=(293.15, 293.15, 293.15, 293.15),
            ullage_pressure_pa=180000.0,
            t_bulk_k=90.0,
            t_interface_k=90.0,
            line_pressure_pa=105000.0,
            void_fraction_out=1.0,
            lad_bubble_margin=0.65,
            status_flags=0,
        )
        self._last_valid_cmd = CryoCommandPacket(
            seq_count=0,
            u_chilldown_valve=0.0,
            u_jt_valve=0.0,
            u_spray_pump=0.20,
            u_cryocooler=0.35,
            command_flags=0,
        )
        self._prev_physical_chill_valve: float = 0.0

        self._udp_tx_sock: socket.socket | None = None
        self._udp_rx_sock: socket.socket | None = None
        self._serial_conn = None

        if self.mode == HILTransportMode.UDP:
            self._udp_rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp_rx_sock.bind((self.udp_host, self.udp_port))
            self._udp_rx_sock.settimeout(0.05)
            self._udp_tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        elif self.mode == HILTransportMode.SERIAL:
            if serial is None:
                raise RuntimeError("pyserial is required for SERIAL HIL transport mode.")
            self._serial_conn = serial.Serial(
                port=self.serial_port, baudrate=self.serial_baudrate, timeout=0.05
            )

    def close(self) -> None:
        """Close any active UDP or UART hardware handles."""
        if self._udp_tx_sock is not None:
            self._udp_tx_sock.close()
            self._udp_tx_sock = None
        if self._udp_rx_sock is not None:
            self._udp_rx_sock.close()
            self._udp_rx_sock = None
        if self._serial_conn is not None:
            self._serial_conn.close()
            self._serial_conn = None

    def _transport_bytes(self, wire_bytes: bytes) -> bytes:
        """Transmit raw frame bytes across the configured physical/logical medium."""
        if self.mode == HILTransportMode.LOOPBACK:
            return bytes(wire_bytes)
        if self.mode == HILTransportMode.UDP:
            assert self._udp_tx_sock is not None and self._udp_rx_sock is not None
            self._udp_tx_sock.sendto(wire_bytes, (self.udp_host, self.udp_port))
            rx_data, _ = self._udp_rx_sock.recvfrom(256)
            return rx_data
        if self.mode == HILTransportMode.SERIAL:
            assert self._serial_conn is not None
            self._serial_conn.write(wire_bytes)
            return self._serial_conn.read(len(wire_bytes))
        return bytes(wire_bytes)

    def transmit_telemetry(
        self,
        timestamp_s: float,
        wall_temps_k: np.ndarray,
        ullage_pressure_pa: float,
        t_bulk_k: float,
        t_interface_k: float,
        line_pressure_pa: float,
        void_fraction_out: float,
        lad_bubble_margin: float,
    ) -> CryoTelemetryPacket:
        """
        Package plant state into a 4-station thermocouple + pressure CCSDS binary packet,
        apply configured sensor/wire fault injection, transmit across the air-gap,
        verify CRC-16-CCITT, and return the decoded telemetry packet (or safe-hold last good packet).
        """
        self._seq_tlm = (self._seq_tlm + 1) & 0x3FFF
        self.stats.tx_telemetry_frames += 1

        # Sample 4 evenly spaced axial wall thermocouple stations
        idx = np.linspace(0, len(wall_temps_k) - 1, 4).astype(int)
        tw_list = [float(wall_temps_k[i]) for i in idx]
        flags = 0

        # Fault 1: Thermocouple open-circuit dropout (injects NaN on selected channel)
        if (
            self.fault_cfg.enable_thermocouple_dropout
            and self.fault_cfg.dropout_start_s <= timestamp_s <= self.fault_cfg.dropout_end_s
        ):
            ch = int(np.clip(self.fault_cfg.dropout_channel_idx, 0, 3))
            tw_list[ch] = float("nan")
            flags |= 0x0001
            self.stats.thermocouple_dropout_events += 1

        # Fault 2: Pressure transducer slow calibration drift
        p_ull_tx = ullage_pressure_pa
        if self.fault_cfg.enable_sensor_drift:
            p_ull_tx += self.fault_cfg.pressure_drift_pa_per_s * math.sin(0.05 * timestamp_s)
            flags |= 0x0002

        tx_pkt = CryoTelemetryPacket(
            seq_count=self._seq_tlm,
            timestamp_s=timestamp_s,
            wall_temps_k=(tw_list[0], tw_list[1], tw_list[2], tw_list[3]),
            ullage_pressure_pa=p_ull_tx,
            t_bulk_k=t_bulk_k,
            t_interface_k=t_interface_k,
            line_pressure_pa=line_pressure_pa,
            void_fraction_out=void_fraction_out,
            lad_bubble_margin=lad_bubble_margin,
            status_flags=flags,
        )
        wire_bytes = bytearray(tx_pkt.pack())

        # Fault 3: Periodic single-bit wire corruption to verify CRC-16-CCITT detection
        if (
            self.fault_cfg.corrupt_every_nth_frame > 0
            and (self.stats.tx_telemetry_frames % self.fault_cfg.corrupt_every_nth_frame) == 0
        ):
            wire_bytes[14] ^= 0x08  # Flip bit in payload

        rx_bytes = self._transport_bytes(bytes(wire_bytes))

        try:
            rx_pkt = CryoTelemetryPacket.unpack(rx_bytes)
            self.stats.rx_telemetry_frames += 1
            # Sanitize NaN thermocouple channel inside receiver FDIR if flagged
            tw_rx = list(rx_pkt.wall_temps_k)
            for i in range(4):
                if not math.isfinite(tw_rx[i]):
                    tw_rx[i] = self._last_valid_tlm.wall_temps_k[i]
            sanitized_pkt = CryoTelemetryPacket(
                seq_count=rx_pkt.seq_count,
                timestamp_s=rx_pkt.timestamp_s,
                wall_temps_k=(tw_rx[0], tw_rx[1], tw_rx[2], tw_rx[3]),
                ullage_pressure_pa=rx_pkt.ullage_pressure_pa,
                t_bulk_k=rx_pkt.t_bulk_k,
                t_interface_k=rx_pkt.t_interface_k,
                line_pressure_pa=rx_pkt.line_pressure_pa,
                void_fraction_out=rx_pkt.void_fraction_out,
                lad_bubble_margin=rx_pkt.lad_bubble_margin,
                status_flags=rx_pkt.status_flags,
            )
            self._last_valid_tlm = sanitized_pkt
            return sanitized_pkt
        except ValueError:
            # CRC-16-CCITT rejected corrupted packet -> retain last valid telemetry frame
            self.stats.rejected_telemetry_crc_frames += 1
            self.stats.safe_hold_fallbacks += 1
            return self._last_valid_tlm

    def transmit_command(
        self,
        u_chilldown_valve: float,
        u_jt_valve: float,
        u_spray_pump: float,
        u_cryocooler: float,
        corrupt_wire: bool = False,
    ) -> CryoCommandPacket:
        """
        Serialize controller actuator commands into a binary CCSDS telecommand frame,
        transmit across the air-gap, validate CRC-16-CCITT, and apply cryogenic valve
        stiction physics if configured.
        """
        self._seq_cmd = (self._seq_cmd + 1) & 0x3FFF
        self.stats.tx_command_frames += 1

        tx_cmd = CryoCommandPacket(
            seq_count=self._seq_cmd,
            u_chilldown_valve=u_chilldown_valve,
            u_jt_valve=u_jt_valve,
            u_spray_pump=u_spray_pump,
            u_cryocooler=u_cryocooler,
            command_flags=0,
        )
        wire_bytes = bytearray(tx_cmd.pack())
        if corrupt_wire:
            wire_bytes[11] ^= 0x04

        rx_bytes = self._transport_bytes(bytes(wire_bytes))
        try:
            rx_cmd = CryoCommandPacket.unpack(rx_bytes)
            self.stats.rx_command_frames += 1

            # Apply cryogenic solenoid/proportional valve stiction deadband if enabled
            u_chill_phys = rx_cmd.u_chilldown_valve
            if self.fault_cfg.enable_valve_stiction:
                if abs(u_chill_phys - self._prev_physical_chill_valve) < self.fault_cfg.valve_stiction_deadband:
                    u_chill_phys = self._prev_physical_chill_valve
                    self.stats.valve_stiction_events += 1
            self._prev_physical_chill_valve = u_chill_phys

            applied_cmd = CryoCommandPacket(
                seq_count=rx_cmd.seq_count,
                u_chilldown_valve=u_chill_phys,
                u_jt_valve=rx_cmd.u_jt_valve,
                u_spray_pump=rx_cmd.u_spray_pump,
                u_cryocooler=rx_cmd.u_cryocooler,
                command_flags=rx_cmd.command_flags,
            )
            self._last_valid_cmd = applied_cmd
            return applied_cmd
        except ValueError:
            self.stats.rejected_command_crc_frames += 1
            self.stats.safe_hold_fallbacks += 1
            return self._last_valid_cmd
