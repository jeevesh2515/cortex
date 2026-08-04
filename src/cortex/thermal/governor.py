"""Thermal-aware indexing governor.

The MacBook Air has no fan. Sustained inference on an M-series Air follows a
well-documented curve: roughly 35 tok/s for the first two minutes, then a slide
to about 13 tok/s once the chassis saturates -- a ~60% collapse that no amount
of software cleverness reverses. The MacBook Pro, with fans, holds its peak
indefinitely.

Most local-RAG projects ignore this and index at full tilt, which on an Air
means the machine is hot, slow and loud-in-the-lap for the entire backfill, and
the user's *interactive* queries land in the throttled regime.

Cortex treats thermal headroom as a schedulable resource. The rule that governs
everything here:

    **Background indexing yields. Interactive queries never do.**

Backfill is deferred, serialised or paused based on thermal and power state. A
query the user is waiting on always runs at full concurrency, because a second
brain that is slow to answer is a second brain nobody opens.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "GovernorConfig",
    "MacOSProbe",
    "PowerSource",
    "Reading",
    "StaticProbe",
    "SystemProbe",
    "ThermalGovernor",
    "ThermalState",
]


class ThermalState(StrEnum):
    BOOST = "boost"
    """Cool, on mains, machine idle. Backfill aggressively."""

    NOMINAL = "nominal"
    """Normal operating range. Standard concurrency."""

    THROTTLED = "throttled"
    """Hot or on battery. Serialise, defer backfill."""

    CRITICAL = "critical"
    """Heavily throttled. Suspend background work entirely."""


class PowerSource(StrEnum):
    AC = "ac"
    BATTERY = "battery"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Reading:
    """A point-in-time snapshot of machine state."""

    power: PowerSource = PowerSource.UNKNOWN
    cpu_speed_limit: int = 100
    """Percent of nominal clock the OS is currently permitting. Below 100 means
    the thermal management system is actively throttling."""

    load_average: float = 0.0
    battery_percent: int | None = None
    cpu_count: int = 1

    @property
    def load_ratio(self) -> float:
        return self.load_average / max(1, self.cpu_count)

    @property
    def is_throttling(self) -> bool:
        return self.cpu_speed_limit < 100


@runtime_checkable
class SystemProbe(Protocol):
    """Reads machine state. Abstracted so the policy is testable off-Mac."""

    def read(self) -> Reading: ...


class StaticProbe:
    """Fixed reading. Used by tests and by the Linux/CI path."""

    def __init__(self, reading: Reading | None = None) -> None:
        self.reading = reading or Reading(power=PowerSource.AC, cpu_count=os.cpu_count() or 1)

    def read(self) -> Reading:
        return self.reading


class MacOSProbe:
    """Reads thermal and power state on macOS.

    Uses ``pmset``, which reports the thermal speed limit **without requiring
    sudo** -- unlike ``powermetrics``, which does. That matters because a
    background LaunchAgent prompting for an admin password is unshippable.
    """

    _SPEED_RE = re.compile(r"CPU_Speed_Limit\s*=\s*(\d+)")
    _BATTERY_RE = re.compile(r"(\d+)%")

    def __init__(self, *, timeout: float = 2.0) -> None:
        self.timeout = timeout
        self._pmset = shutil.which("pmset")
        self._cpu_count = os.cpu_count() or 1

    @property
    def available(self) -> bool:
        return platform.system() == "Darwin" and self._pmset is not None

    def _run(self, *args: str) -> str:
        if not self._pmset:
            return ""
        try:
            proc = subprocess.run(
                [self._pmset, *args],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            return proc.stdout
        except (subprocess.SubprocessError, OSError) as exc:
            logger.debug("pmset %s failed: %s", args, exc)
            return ""

    def read(self) -> Reading:
        therm = self._run("-g", "therm")
        speed_match = self._SPEED_RE.search(therm)
        speed_limit = int(speed_match.group(1)) if speed_match else 100

        batt = self._run("-g", "batt")
        if "AC Power" in batt:
            power = PowerSource.AC
        elif "Battery Power" in batt:
            power = PowerSource.BATTERY
        else:
            power = PowerSource.UNKNOWN

        pct_match = self._BATTERY_RE.search(batt)
        battery = int(pct_match.group(1)) if pct_match else None

        try:
            load = os.getloadavg()[0]
        except (OSError, AttributeError):
            load = 0.0

        return Reading(
            power=power,
            cpu_speed_limit=speed_limit,
            load_average=load,
            battery_percent=battery,
            cpu_count=self._cpu_count,
        )


@dataclass(frozen=True, slots=True)
class GovernorConfig:
    """Policy thresholds.

    Defaults are tuned for a fanless 16 GB Air: conservative, because the cost
    of being wrong is a hot laptop and a slow interactive query.
    """

    boost_workers: int = 4
    nominal_workers: int = 2
    throttled_workers: int = 1
    critical_workers: int = 0

    throttle_speed_limit: int = 95
    """CPU_Speed_Limit at or below which we consider the machine throttled."""

    critical_speed_limit: int = 60
    boost_load_ratio: float = 0.35
    """Machine must be quieter than this (load / cores) to earn BOOST."""

    nominal_load_ratio: float = 1.5
    min_battery_for_backfill: int = 30
    """Below this charge, background indexing is suspended regardless of heat."""

    sample_interval: float = 10.0
    hysteresis_samples: int = 2
    """Consecutive agreeing samples required before easing restrictions.

    Loosening instantly on one cool reading causes oscillation: the machine
    heats, throttles, cools for one sample, resumes at full tilt and
    immediately re-throttles. Tightening is deliberately allowed to happen at
    once -- reacting slowly to heat is the failure we care about.
    """


@dataclass(slots=True)
class ThermalGovernor:
    """Decides how much background work the machine can absorb right now."""

    probe: SystemProbe = field(default_factory=StaticProbe)
    config: GovernorConfig = field(default_factory=GovernorConfig)
    _state: ThermalState = ThermalState.NOMINAL
    _pending: ThermalState | None = None
    _pending_count: int = 0
    _last_sample: float = 0.0
    _last_reading: Reading | None = None
    _initialised: bool = False
    """False until the first reading lands. Hysteresis guards against flapping
    between states, which presupposes a prior state -- on cold start there is
    nothing to flap from, so the first observation is adopted directly rather
    than spending two intervals climbing out of the default."""

    @property
    def state(self) -> ThermalState:
        return self._state

    @property
    def last_reading(self) -> Reading | None:
        return self._last_reading

    def _classify(self, reading: Reading) -> ThermalState:
        cfg = self.config

        if reading.cpu_speed_limit <= cfg.critical_speed_limit:
            return ThermalState.CRITICAL
        if (
            reading.power is PowerSource.BATTERY
            and reading.battery_percent is not None
            and reading.battery_percent < cfg.min_battery_for_backfill
        ):
            return ThermalState.CRITICAL
        if reading.cpu_speed_limit <= cfg.throttle_speed_limit:
            return ThermalState.THROTTLED
        # On battery we never boost: the user is mobile and the thermal budget
        # is shared with whatever they are actually doing.
        if reading.power is PowerSource.BATTERY:
            return ThermalState.THROTTLED
        if reading.load_ratio <= cfg.boost_load_ratio:
            return ThermalState.BOOST
        if reading.load_ratio <= cfg.nominal_load_ratio:
            return ThermalState.NOMINAL
        return ThermalState.THROTTLED

    _SEVERITY: ClassVar[dict[ThermalState, int]] = {
        ThermalState.BOOST: 0,
        ThermalState.NOMINAL: 1,
        ThermalState.THROTTLED: 2,
        ThermalState.CRITICAL: 3,
    }

    def sample(self, *, now: float | None = None, force: bool = False) -> ThermalState:
        """Take a reading and update state, honouring hysteresis."""
        clock = time.monotonic() if now is None else now
        if not force and clock - self._last_sample < self.config.sample_interval:
            return self._state
        self._last_sample = clock

        reading = self.probe.read()
        self._last_reading = reading
        observed = self._classify(reading)

        if not self._initialised:
            self._initialised = True
            self._state = observed
            self._pending = None
            self._pending_count = 0
            return self._state

        if observed is self._state:
            self._pending = None
            self._pending_count = 0
            return self._state

        tightening = self._SEVERITY[observed] > self._SEVERITY[self._state]
        if tightening:
            # React to heat immediately.
            logger.info("thermal state %s -> %s", self._state.value, observed.value)
            self._state = observed
            self._pending = None
            self._pending_count = 0
            return self._state

        # Loosening requires sustained agreement.
        if self._pending is observed:
            self._pending_count += 1
        else:
            self._pending = observed
            self._pending_count = 1

        if self._pending_count >= self.config.hysteresis_samples:
            logger.info("thermal state %s -> %s (settled)", self._state.value, observed.value)
            self._state = observed
            self._pending = None
            self._pending_count = 0
        return self._state

    def workers(self, *, interactive: bool = False) -> int:
        """Concurrency budget for the next unit of work.

        Interactive work is never throttled -- that is the core policy. A user
        waiting on an answer gets the whole machine.
        """
        if interactive:
            return max(1, self.config.boost_workers)
        return {
            ThermalState.BOOST: self.config.boost_workers,
            ThermalState.NOMINAL: self.config.nominal_workers,
            ThermalState.THROTTLED: self.config.throttled_workers,
            ThermalState.CRITICAL: self.config.critical_workers,
        }[self._state]

    def may_backfill(self) -> bool:
        """Whether a bulk re-index may proceed right now."""
        return self._state in (ThermalState.BOOST, ThermalState.NOMINAL)

    def cooldown_hint(self) -> float:
        """Seconds to pause between background batches."""
        return {
            ThermalState.BOOST: 0.0,
            ThermalState.NOMINAL: 0.05,
            ThermalState.THROTTLED: 0.5,
            ThermalState.CRITICAL: 5.0,
        }[self._state]

    def describe(self) -> dict[str, object]:
        """Human-readable status. Surfaced by the CLI and MCP so the governor
        is observable rather than a black box making the machine feel slow."""
        reading = self._last_reading
        return {
            "state": self._state.value,
            "workers": self.workers(),
            "may_backfill": self.may_backfill(),
            "power": reading.power.value if reading else "unknown",
            "cpu_speed_limit": reading.cpu_speed_limit if reading else None,
            "battery_percent": reading.battery_percent if reading else None,
            "load_ratio": round(reading.load_ratio, 2) if reading else None,
        }


def default_probe() -> SystemProbe:
    """Pick the right probe for this machine."""
    mac = MacOSProbe()
    if mac.available:
        return mac
    return StaticProbe(Reading(power=PowerSource.AC, cpu_count=os.cpu_count() or 1))
