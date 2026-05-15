"""Tactile sensing + gripper-safety wrapper for xArm policy deployment.

Reimplemented locally (no dependency on the tactile-data-collection repo) to
preserve openpi as a self-contained checkout. The behavior matches
`/data/edward/tactile-data-collection/environment/tactile.py` so that the
deployment-time safety threshold reads identically to the collection-time one:

  - One TactileSensor thread per A31301 ESP32 board.
  - Each thread owns its pyserial.Serial, parses `S,ts,idx,addr,conn,x,y,z`
    rows, accumulates a 9-taxel frame keyed on the shared device ts_ms, and
    publishes the latest complete frame under a lock.
  - Handles ESP32 reboots: parser goes silent on the panic / boot lines and
    re-syncs on the next BEGIN_STREAM.
  - TactileSensors bundles N readers and exposes .safety() -> (metric, is_safe)
    used by XArm.execute_delta to clamp closing commands.
  - Stale data (no frame within stale_after_sec) and zero-connected-taxel
    snapshots are treated as UNSAFE-to-close (fail-safe).
  - Optional per-cell baseline subtracted from xyz BEFORE the metric reduces
    over cells. When baseline is set, safety_threshold is interpreted in
    delta-from-idle units — which is the operating mode in
    tactile_config.py:SAFETY_THRESHOLD = 1500.0 (sum_abs_z delta).
"""
from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# pyserial is imported lazily inside TactileSensor.run() so the pure-Python
# helpers (compute_safety_metric / evaluate_safety / TactileConfig) can be
# imported and unit-tested in environments where pyserial isn't installed.

logger = logging.getLogger("xarm-infer.tactile")


# Lines emitted by the ESP32 firmware on a reboot/panic. While any of these is
# being seen on the wire, the parser ignores everything until a BEGIN_STREAM.
_REBOOT_PATTERNS = (
    "rst:0x", "boot:0x", "configsip:", "ets ",
    "Guru Meditation", "Backtrace:", "assert failed:", "abort()",
    "panic_handler", "LoadProhibited", "StoreProhibited",
    "IllegalInstruction", "IntegerDivideByZero",
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TactileConfig:
    ports: List[str] = dataclasses.field(
        default_factory=lambda: ["/dev/ttyACM0", "/dev/ttyACM1"]
    )
    baud: int = 115200
    n_taxels: int = 9

    # Reduction over (n_sensors, n_taxels, 3) -> scalar:
    #   "sum_abs_z"  sum(|Bz|) across connected taxels  — empirically the only one
    #                that works reliably for this magnet mounting
    #   "max_abs_z"  max(|Bz|) across connected taxels  — pins to the resting cell
    #   "max_norm"   max(||xyz||) across connected taxels
    safety_metric: str = "sum_abs_z"
    safety_threshold: float = 1500.0   # delta-from-idle when baseline is set
    stale_after_sec: float = 0.2

    # Optional per-cell baseline (shape: (n_sensors, n_taxels, 3)) subtracted
    # from raw xyz before the metric reduces over cells. None = raw-value mode
    # (threshold then has to be huge — e.g. ~30000 for sum_abs_z).
    baseline: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _is_reboot(line: str) -> bool:
    return any(p in line for p in _REBOOT_PATTERNS)


def _parse_sample(line: str):
    """Parse one `S,ts,idx,addr,conn,x,y,z` row. Returns tuple or None."""
    if not line.startswith("S,"):
        return None
    parts = line.split(",")
    if len(parts) != 8:
        return None
    try:
        return (
            int(parts[1]),     # ts_ms
            int(parts[2]),     # idx
            int(parts[4]),     # connected
            float(parts[5]),   # x
            float(parts[6]),   # y
            float(parts[7]),   # z
        )
    except Exception:
        return None


def compute_safety_metric(
    states: Sequence[Dict],
    metric: str,
    stale_after_sec: float,
    baseline: Optional[np.ndarray] = None,
) -> Tuple[float, bool]:
    """Reduce per-sensor snapshots to a scalar.

    Returns (metric_value, all_fresh). all_fresh=False when ANY sensor either
    has zero connected taxels, no published frame yet, or a frame older than
    `stale_after_sec`. The caller must treat all_fresh=False as unsafe.

    If `baseline` is provided, per-cell idle xyz is subtracted before reducing
    over cells. This is the operating mode for the data-collection threshold
    of 1500.0 sum_abs_z — raw idle sums are ~30000, so without baseline
    subtraction the threshold would have to be >30100.
    """
    now = time.time()
    per_sensor: List[float] = []
    all_fresh = True
    for i, st in enumerate(states):
        host_ts = float(st.get("host_timestamp", 0.0))
        conn = np.asarray(st.get("connected", np.zeros(0)), dtype=np.int32)
        xyz = np.asarray(st.get("xyz", np.zeros((0, 3))), dtype=np.float32)
        if (
            conn.size == 0
            or not np.any(conn)
            or host_ts <= 0.0
            or (now - host_ts) > stale_after_sec
        ):
            all_fresh = False
            continue
        mask = conn > 0
        if baseline is not None and i < baseline.shape[0]:
            xyz_use = xyz - np.asarray(baseline[i], dtype=np.float32)
        else:
            xyz_use = xyz
        if metric == "max_abs_z":
            per_sensor.append(float(np.max(np.abs(xyz_use[mask, 2]))))
        elif metric == "max_norm":
            per_sensor.append(float(np.max(np.linalg.norm(xyz_use[mask], axis=1))))
        elif metric == "sum_abs_z":
            per_sensor.append(float(np.sum(np.abs(xyz_use[mask, 2]))))
        else:
            raise ValueError(f"Unknown safety_metric: {metric!r}")
    if not per_sensor:
        return 0.0, False
    if metric == "sum_abs_z":
        return float(sum(per_sensor)), all_fresh
    return float(max(per_sensor)), all_fresh


def evaluate_safety(states: Sequence[Dict], cfg: TactileConfig) -> Tuple[float, bool]:
    """(metric_value, is_safe_to_close).

    is_safe=False => caller MUST NOT increase grasp closure. Stale/missing
    data forces is_safe=False (fail-safe). Opening is always allowed by the
    caller.
    """
    metric_val, fresh = compute_safety_metric(
        states, cfg.safety_metric, cfg.stale_after_sec, baseline=cfg.baseline,
    )
    if not fresh:
        return metric_val, False
    return metric_val, metric_val <= cfg.safety_threshold


# ---------------------------------------------------------------------------
# Per-board reader thread
# ---------------------------------------------------------------------------


class TactileSensor(threading.Thread):
    """Owns one serial port. Publishes the latest 9-taxel frame under a lock.

    open_failed is set True if the port could not be opened — callers can check
    it after start() to fail loudly (the openpi deployment path does this).
    """

    def __init__(self, port: str, config: TactileConfig, name: Optional[str] = None):
        super().__init__(daemon=True, name=name or f"TactileSensor-{port}")
        self.port = port
        self.config = config
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.open_failed = False

        n = config.n_taxels
        self._lock = threading.Lock()
        self._latest = {
            "xyz": np.zeros((n, 3), dtype=np.float32),
            "connected": np.zeros(n, dtype=np.int32),
            "device_ts_ms": np.int64(0),
            "host_timestamp": 0.0,   # 0 -> always stale until first publish
        }

    def get_state(self) -> Dict:
        with self._lock:
            return {
                "xyz": self._latest["xyz"].copy(),
                "connected": self._latest["connected"].copy(),
                "device_ts_ms": int(self._latest["device_ts_ms"]),
                "host_timestamp": float(self._latest["host_timestamp"]),
            }

    def stop(self):
        self.stop_event.set()

    def run(self):
        import serial  # pyserial — only required when actually streaming data
        cfg = self.config
        try:
            ser = serial.Serial(
                self.port, cfg.baud, timeout=1, rtscts=False, dsrdtr=False
            )
            # Hold DTR/RTS so opening doesn't reset the ESP32.
            try:
                ser.setDTR(True)
                ser.setRTS(True)
            except Exception:
                pass
        except Exception as e:
            logger.error("[%s] open %s failed: %s", self.name, self.port, e)
            self.open_failed = True
            self.ready_event.set()
            return

        # Nudge the device into streaming.
        try:
            ser.write(b"CMD,GET,STATE\n")
            ser.flush()
        except Exception as e:
            logger.warning("[%s] failed to send CMD,GET,STATE: %s", self.name, e)

        self.ready_event.set()

        n = cfg.n_taxels
        frame_xyz = np.zeros((n, 3), dtype=np.float32)
        frame_conn = np.zeros(n, dtype=np.int32)
        seen = np.zeros(n, dtype=bool)
        current_ts: Optional[int] = None
        reboot_detected = False

        try:
            while not self.stop_event.is_set():
                try:
                    raw = ser.readline().decode(errors="ignore")
                except Exception as e:
                    logger.error("[%s] read error: %s", self.name, e)
                    break
                if not raw:
                    continue
                line = raw.strip()
                if not line:
                    continue

                if _is_reboot(line):
                    if not reboot_detected:
                        logger.warning("[%s] device reboot/crash: %s", self.name, line)
                        reboot_detected = True
                    current_ts = None
                    seen[:] = False
                    continue
                if reboot_detected:
                    if line.startswith("BEGIN_STREAM"):
                        logger.info("[%s] recovered: %s", self.name, line)
                        reboot_detected = False
                        current_ts = None
                        seen[:] = False
                    continue

                if line.startswith("BEGIN_STREAM") or line.startswith("END_STREAM"):
                    continue

                parsed = _parse_sample(line)
                if parsed is None:
                    continue
                ts_ms, idx, conn, x, y, z = parsed
                if not (0 <= idx < n):
                    continue

                if current_ts is None:
                    current_ts = ts_ms

                # New ts_ms -> finalize the (possibly partial) previous frame.
                if ts_ms != current_ts:
                    self._publish(frame_xyz, frame_conn, current_ts)
                    frame_xyz[:] = 0
                    frame_conn[:] = 0
                    seen[:] = False
                    current_ts = ts_ms

                frame_xyz[idx] = (x, y, z)
                frame_conn[idx] = conn
                seen[idx] = True

                if seen.all():
                    self._publish(frame_xyz, frame_conn, current_ts)
                    frame_xyz[:] = 0
                    frame_conn[:] = 0
                    seen[:] = False
                    current_ts = None
        finally:
            try:
                ser.setDTR(True)
                ser.setRTS(True)
                ser.close()
            except Exception:
                pass

    def _publish(self, xyz: np.ndarray, conn: np.ndarray, ts_ms: int):
        with self._lock:
            self._latest["xyz"] = xyz.copy()
            self._latest["connected"] = conn.copy()
            self._latest["device_ts_ms"] = np.int64(ts_ms)
            self._latest["host_timestamp"] = time.time()


# ---------------------------------------------------------------------------
# Multi-board bundle
# ---------------------------------------------------------------------------


class TactileSensors:
    """N TactileSensor threads, one per port. Use as a context manager.

    Use `wait_until_fresh()` after entering to require live data before
    proceeding — the deployment script hard-fails if that times out so we
    never start moving the robot with a dead sensor.
    """

    def __init__(self, config: TactileConfig, names: Optional[List[str]] = None):
        self.config = config
        labels = names or [f"sensor{i}" for i in range(len(config.ports))]
        if len(labels) != len(config.ports):
            raise ValueError("names length must match config.ports length")
        self.sensors: List[TactileSensor] = [
            TactileSensor(port, config, name=lbl)
            for port, lbl in zip(config.ports, labels)
        ]

    def __enter__(self):
        for s in self.sensors:
            s.start()
        for s in self.sensors:
            s.ready_event.wait(timeout=1.0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for s in self.sensors:
            s.stop()
        for s in self.sensors:
            s.join(timeout=2.0)

    @property
    def all_open(self) -> bool:
        return all(not s.open_failed for s in self.sensors)

    @property
    def any_open(self) -> bool:
        return any(not s.open_failed for s in self.sensors)

    @property
    def failed_ports(self) -> List[str]:
        return [s.port for s in self.sensors if s.open_failed]

    def get_latest(self) -> List[Dict]:
        return [s.get_state() for s in self.sensors]

    def wait_until_fresh(self, timeout_sec: float = 5.0) -> bool:
        """Block until every sensor has published a non-stale frame, or timeout.

        Returns True on success, False on timeout. The deployment script hard-
        fails on False so we never start a rollout with a board that opened
        the port but isn't actually streaming.
        """
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            states = self.get_latest()
            now = time.time()
            ok = True
            for st in states:
                host_ts = float(st.get("host_timestamp", 0.0))
                conn = np.asarray(st.get("connected", np.zeros(0)), dtype=np.int32)
                if (
                    host_ts <= 0.0
                    or (now - host_ts) > self.config.stale_after_sec
                    or not np.any(conn)
                ):
                    ok = False
                    break
            if ok:
                return True
            time.sleep(0.05)
        return False

    def safety(self) -> Tuple[float, bool]:
        return evaluate_safety(self.get_latest(), self.config)
