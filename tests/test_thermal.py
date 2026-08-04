"""Thermal governor tests.

The governor is pure policy over an injected probe, so all of this runs on any
platform. The behaviours that matter: heat tightens immediately, cooling is
hysteretic, and interactive work is never throttled.
"""

from __future__ import annotations

from cortex.thermal.governor import (
    GovernorConfig,
    MacOSProbe,
    PowerSource,
    Reading,
    StaticProbe,
    ThermalGovernor,
    ThermalState,
    default_probe,
)

FAST_CONFIG = GovernorConfig(sample_interval=0.0, hysteresis_samples=2)


def gov(reading: Reading, config: GovernorConfig | None = None) -> ThermalGovernor:
    return ThermalGovernor(probe=StaticProbe(reading), config=config or FAST_CONFIG)


def cool_ac(**kw: object) -> Reading:
    base = {
        "power": PowerSource.AC,
        "cpu_speed_limit": 100,
        "load_average": 0.5,
        "cpu_count": 8,
    }
    base.update(kw)
    return Reading(**base)  # type: ignore[arg-type]


class TestClassification:
    def test_cool_idle_on_mains_boosts(self) -> None:
        g = gov(cool_ac())
        assert g.sample(force=True) is ThermalState.BOOST

    def test_busy_on_mains_is_nominal(self) -> None:
        g = gov(cool_ac(load_average=6.0))
        assert g.sample(force=True) is ThermalState.NOMINAL

    def test_very_busy_throttles(self) -> None:
        g = gov(cool_ac(load_average=20.0))
        assert g.sample(force=True) is ThermalState.THROTTLED

    def test_battery_never_boosts(self) -> None:
        # Mobile means the thermal budget is shared with the user's real work.
        g = gov(cool_ac(power=PowerSource.BATTERY, battery_percent=90))
        assert g.sample(force=True) is ThermalState.THROTTLED

    def test_speed_limit_triggers_throttle(self) -> None:
        g = gov(cool_ac(cpu_speed_limit=80))
        assert g.sample(force=True) is ThermalState.THROTTLED

    def test_severe_speed_limit_is_critical(self) -> None:
        g = gov(cool_ac(cpu_speed_limit=45))
        assert g.sample(force=True) is ThermalState.CRITICAL

    def test_low_battery_is_critical(self) -> None:
        g = gov(cool_ac(power=PowerSource.BATTERY, battery_percent=15))
        assert g.sample(force=True) is ThermalState.CRITICAL

    def test_low_battery_on_ac_is_fine(self) -> None:
        # Charging at 15% is not a reason to stop working.
        g = gov(cool_ac(power=PowerSource.AC, battery_percent=15))
        assert g.sample(force=True) is ThermalState.BOOST


class TestHysteresis:
    def test_tightening_is_immediate(self) -> None:
        g = ThermalGovernor(probe=StaticProbe(cool_ac()), config=FAST_CONFIG)
        assert g.sample(force=True) is ThermalState.BOOST
        # Machine heats up.
        g.probe = StaticProbe(cool_ac(cpu_speed_limit=50))  # type: ignore[attr-defined]
        assert g.sample(force=True) is ThermalState.CRITICAL, "heat must apply at once"

    def test_loosening_requires_sustained_agreement(self) -> None:
        g = ThermalGovernor(probe=StaticProbe(cool_ac(cpu_speed_limit=50)), config=FAST_CONFIG)
        assert g.sample(force=True) is ThermalState.CRITICAL
        g.probe = StaticProbe(cool_ac())  # type: ignore[attr-defined]
        # One cool reading is not enough -- that is what causes oscillation.
        assert g.sample(force=True) is ThermalState.CRITICAL
        assert g.sample(force=True) is ThermalState.BOOST

    def test_flapping_reading_does_not_loosen(self) -> None:
        g = ThermalGovernor(probe=StaticProbe(cool_ac(cpu_speed_limit=50)), config=FAST_CONFIG)
        g.sample(force=True)
        g.probe = StaticProbe(cool_ac())  # type: ignore[attr-defined]
        g.sample(force=True)
        # Heat returns before the loosening settles.
        g.probe = StaticProbe(cool_ac(cpu_speed_limit=50))  # type: ignore[attr-defined]
        g.sample(force=True)
        g.probe = StaticProbe(cool_ac())  # type: ignore[attr-defined]
        assert g.sample(force=True) is ThermalState.CRITICAL

    def test_sample_interval_respected(self) -> None:
        config = GovernorConfig(sample_interval=100.0)
        g = ThermalGovernor(probe=StaticProbe(cool_ac()), config=config)
        g.sample(now=0.0)
        g.probe = StaticProbe(cool_ac(cpu_speed_limit=10))  # type: ignore[attr-defined]
        # Too soon: the cached state stands.
        assert g.sample(now=1.0) is not ThermalState.CRITICAL


class TestWorkerBudget:
    def test_scales_with_state(self) -> None:
        assert gov(cool_ac()).sample(force=True) is ThermalState.BOOST
        boost = gov(cool_ac())
        boost.sample(force=True)
        assert boost.workers() == FAST_CONFIG.boost_workers

        hot = gov(cool_ac(cpu_speed_limit=80))
        hot.sample(force=True)
        assert hot.workers() == FAST_CONFIG.throttled_workers

    def test_critical_halts_background_work(self) -> None:
        g = gov(cool_ac(cpu_speed_limit=40))
        g.sample(force=True)
        assert g.workers() == 0
        assert not g.may_backfill()

    def test_interactive_is_never_throttled(self) -> None:
        # The core policy: a query the user is waiting on gets the machine.
        g = gov(cool_ac(cpu_speed_limit=40))
        g.sample(force=True)
        assert g.state is ThermalState.CRITICAL
        assert g.workers() == 0
        assert g.workers(interactive=True) >= 1

    def test_may_backfill_only_when_cool(self) -> None:
        cool = gov(cool_ac())
        cool.sample(force=True)
        assert cool.may_backfill()

        hot = gov(cool_ac(cpu_speed_limit=70))
        hot.sample(force=True)
        assert not hot.may_backfill()

    def test_cooldown_grows_with_heat(self) -> None:
        cool = gov(cool_ac())
        cool.sample(force=True)
        hot = gov(cool_ac(cpu_speed_limit=40))
        hot.sample(force=True)
        assert hot.cooldown_hint() > cool.cooldown_hint()


class TestReading:
    def test_load_ratio(self) -> None:
        assert Reading(load_average=4.0, cpu_count=8).load_ratio == 0.5

    def test_load_ratio_guards_zero_cores(self) -> None:
        assert Reading(load_average=4.0, cpu_count=0).load_ratio == 4.0

    def test_is_throttling(self) -> None:
        assert Reading(cpu_speed_limit=80).is_throttling
        assert not Reading(cpu_speed_limit=100).is_throttling


class TestObservability:
    def test_describe_exposes_state(self) -> None:
        g = gov(cool_ac(battery_percent=88))
        g.sample(force=True)
        described = g.describe()
        assert described["state"] == "boost"
        assert described["power"] == "ac"
        assert described["battery_percent"] == 88
        assert described["may_backfill"] is True


class TestMacOSProbe:
    def test_speed_limit_regex(self) -> None:
        output = "CPU_Scheduler_Limit = 100\nCPU_Speed_Limit \t= 74\n"
        assert MacOSProbe._SPEED_RE.search(output).group(1) == "74"  # type: ignore[union-attr]

    def test_battery_regex(self) -> None:
        output = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1234)\t87%; discharging"
        assert MacOSProbe._BATTERY_RE.search(output).group(1) == "87"  # type: ignore[union-attr]

    def test_unavailable_off_darwin(self) -> None:
        import platform

        probe = MacOSProbe()
        if platform.system() != "Darwin":
            assert not probe.available

    def test_default_probe_returns_something_usable(self) -> None:
        assert default_probe().read() is not None
