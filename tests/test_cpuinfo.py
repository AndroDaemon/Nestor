"""Tests for the CPU topology helpers (``nestor.cpuinfo``).

The interesting behaviour is the *decision* logic: how "auto" resolves to a
thread count, and that the affinity/priority tricks stay off on a non-hybrid
CPU (so i5-8250U-class machines behave exactly as before).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nestor import cpuinfo  # noqa: E402


def test_resolve_threads_explicit_value_wins():
    assert cpuinfo.resolve_threads(6) == 6
    assert cpuinfo.resolve_threads(1) == 1


def test_resolve_threads_auto_uses_physical_cores_capped(monkeypatch):
    monkeypatch.setattr(cpuinfo, "is_hybrid", lambda: False)
    monkeypatch.setattr(cpuinfo, "physical_cores", lambda: 16)
    assert cpuinfo.resolve_threads(0) == cpuinfo.AUTO_THREAD_CAP
    monkeypatch.setattr(cpuinfo, "physical_cores", lambda: 4)
    assert cpuinfo.resolve_threads(0) == 4
    monkeypatch.setattr(cpuinfo, "physical_cores", lambda: 1)
    assert cpuinfo.resolve_threads(0) == 2  # never below two threads


def test_resolve_threads_auto_uses_all_physical_cores_on_hybrid(monkeypatch):
    # i5-1235U: 2 P + 8 E = 10 physical cores. "Auto" must use all of them
    # (capped at 8), not just the 4 logical P-core CPUs: measured 17:50 with 8
    # threads versus 19:22 with the 4 threads pinned to the P-cores.
    monkeypatch.setattr(cpuinfo, "is_hybrid", lambda: True)
    monkeypatch.setattr(cpuinfo, "physical_cores", lambda: 10)
    monkeypatch.setattr(cpuinfo, "performance_logical_count", lambda: 4)
    assert cpuinfo.resolve_threads(0) == cpuinfo.AUTO_THREAD_CAP
    # The global cap still applies on a wide chip, and an explicit user value
    # always wins, hybrid or not.
    monkeypatch.setattr(cpuinfo, "physical_cores", lambda: 14)
    monkeypatch.setattr(cpuinfo, "performance_logical_count", lambda: 12)
    assert cpuinfo.resolve_threads(0) == cpuinfo.AUTO_THREAD_CAP
    assert cpuinfo.resolve_threads(4) == 4
    assert cpuinfo.resolve_threads(8) == 8


def test_resolve_diar_threads_explicit_value_wins():
    assert cpuinfo.resolve_diar_threads(6) == 6
    assert cpuinfo.resolve_diar_threads(8) == 8


def test_resolve_diar_threads_auto_is_capped_low(monkeypatch):
    # Diarization slows down past a handful of threads (quadratic clustering):
    # on an hour of audio 8 threads took 72 s against 44.5 s for 4. "Auto"
    # must therefore stay well below the ASR thread count.
    monkeypatch.setattr(cpuinfo, "physical_cores", lambda: 10)
    assert cpuinfo.resolve_diar_threads(0) == cpuinfo.DIAR_THREAD_CAP
    monkeypatch.setattr(cpuinfo, "physical_cores", lambda: 2)
    assert cpuinfo.resolve_diar_threads(0) == 2
    monkeypatch.setattr(cpuinfo, "physical_cores", lambda: 1)
    assert cpuinfo.resolve_diar_threads(0) == 2  # never below two threads


def test_layout_picks_the_higher_efficiency_class_as_performance(monkeypatch):
    """Win32: a *higher* EfficiencyClass is the faster core.

    Regression guard for the i5-1235U bug: the 2 P-cores report a higher
    class than the 8 E-cores, so min() used to pick the E-core cluster.
    """
    # (efficiency_class, affinity_mask) per physical core.
    # Modelled on the i5-1235U: 8 E-cores (class 0, one logical each) and
    # 2 P-cores with HT (class 1, two logical each) = 10 physical / 12 logical.
    fake = [(0, 0b0000000000010000), (0, 0b0000000000100000),
            (0, 0b0000000001000000), (0, 0b0000000010000000),
            (0, 0b0000000100000000), (0, 0b0000001000000000),
            (0, 0b0000010000000000), (0, 0b0000100000000000),
            (1, 0b0000000000000011), (1, 0b0000000000001100)]
    monkeypatch.setattr(cpuinfo, "_processor_core_entries", lambda: fake)
    monkeypatch.setattr(cpuinfo, "_layout_cache", None)
    try:
        assert cpuinfo.is_hybrid() is True
        # the two P-cores with HT -> 4 performance logical processors
        assert cpuinfo.performance_logical_count() == 4
        assert cpuinfo.physical_cores() == 10
        # "auto" uses all physical cores (capped), not just the P-cores
        assert cpuinfo.resolve_threads(0) == cpuinfo.AUTO_THREAD_CAP
    finally:
        monkeypatch.setattr(cpuinfo, "_layout_cache", None)


def test_resolve_threads_bad_input_falls_back_to_auto():
    assert cpuinfo.resolve_threads(None) >= 2
    assert cpuinfo.resolve_threads(-5) >= 2
    assert cpuinfo.resolve_threads("auto") >= 2


def test_physical_cores_is_positive():
    assert cpuinfo.physical_cores() >= 1


def test_layout_reports_no_hybrid(monkeypatch):
    monkeypatch.setattr(
        cpuinfo,
        "_layout",
        lambda: {"physical": 4, "hybrid": False, "perf_mask": 0, "perf_logical": 0},
    )
    assert cpuinfo.is_hybrid() is False
    assert cpuinfo.performance_mask() == 0
    assert cpuinfo.performance_logical_count() == 0


def test_tuned_is_noop_on_non_hybrid(monkeypatch):
    monkeypatch.setattr(cpuinfo, "_layout", lambda: {"hybrid": False})
    calls: list = []
    monkeypatch.setattr(cpuinfo, "_set_affinity", lambda mask: calls.append(mask) or True)
    with cpuinfo.tuned(4, priority=False, pin_performance=True):
        pass
    assert calls == []  # nothing is pinned when there are no P-cores


def test_begin_pins_to_performance_cores_when_hybrid(monkeypatch):
    monkeypatch.setattr(cpuinfo, "is_hybrid", lambda: True)
    monkeypatch.setattr(cpuinfo, "performance_mask", lambda: 0b1111)
    monkeypatch.setattr(cpuinfo, "performance_logical_count", lambda: 4)
    monkeypatch.setattr(cpuinfo, "_get_affinity", lambda: 0b111111)
    applied: list = []
    monkeypatch.setattr(
        cpuinfo, "_set_affinity", lambda mask: (applied.append(mask), True)[1]
    )
    state = cpuinfo._begin(4, priority=False, pin_performance=True)
    assert applied == [0b1111]  # intersection of P-cores and the old mask
    assert state["affinity"] == 0b111111
    cpuinfo._end(state)
    assert applied[-1] == 0b111111  # original mask restored


def test_begin_skips_pin_when_more_threads_than_perf_cores(monkeypatch):
    monkeypatch.setattr(cpuinfo, "is_hybrid", lambda: True)
    monkeypatch.setattr(cpuinfo, "performance_mask", lambda: 0b1111)
    monkeypatch.setattr(cpuinfo, "performance_logical_count", lambda: 4)
    monkeypatch.setattr(cpuinfo, "_get_affinity", lambda: 0b111111)
    applied: list = []
    monkeypatch.setattr(cpuinfo, "_set_affinity", lambda mask: applied.append(mask) or True)
    cpuinfo._begin(8, priority=False, pin_performance=True)
    assert applied == []  # 8 threads do not fit the 4 P-cores: leave it to Windows


def test_priority_is_raised_and_restored(monkeypatch):
    previous = 0x20  # NORMAL_PRIORITY_CLASS
    monkeypatch.setattr(cpuinfo, "_set_above_normal", lambda: previous)
    monkeypatch.setattr(cpuinfo, "is_hybrid", lambda: False)
    restored: list = []
    monkeypatch.setattr(cpuinfo, "_restore_priority", lambda value: restored.append(value))
    state = cpuinfo._begin(4, priority=True, pin_performance=True)
    assert state["priority"] == previous
    cpuinfo._end(state)
    assert restored == [previous]


def test_describe_pinning_reports_disabled(monkeypatch):
    monkeypatch.setattr(cpuinfo, "is_hybrid", lambda: True)
    info = cpuinfo.describe_pinning(4, pin_performance=False)
    assert info["requested"] is False
    assert info["applied"] is False
    assert info["reason"] == "disabled"
    assert info["message"]


def test_describe_pinning_reports_non_hybrid(monkeypatch):
    monkeypatch.setattr(cpuinfo, "is_hybrid", lambda: False)
    info = cpuinfo.describe_pinning(4, pin_performance=True)
    assert info["applied"] is False
    assert info["reason"] == "not_hybrid"


def test_describe_pinning_reports_too_many_threads(monkeypatch):
    # the exact case from the i5-1235U benchmark: 8 threads, 4 logical P-cores
    monkeypatch.setattr(cpuinfo, "is_hybrid", lambda: True)
    monkeypatch.setattr(cpuinfo, "performance_logical_count", lambda: 4)
    monkeypatch.setattr(cpuinfo, "performance_mask", lambda: 0b1111)
    info = cpuinfo.describe_pinning(8, pin_performance=True)
    assert info["applied"] is False
    assert info["reason"] == "too_many_threads"
    assert "8" in info["message"] and "4" in info["message"]


def test_describe_pinning_reports_applied(monkeypatch):
    monkeypatch.setattr(cpuinfo, "is_hybrid", lambda: True)
    monkeypatch.setattr(cpuinfo, "performance_logical_count", lambda: 4)
    monkeypatch.setattr(cpuinfo, "performance_mask", lambda: 0b1111)
    monkeypatch.setattr(cpuinfo, "physical_cores", lambda: 10)
    # an explicit 4 threads still fit the P-cores, so the pin can apply
    info = cpuinfo.describe_pinning(4, pin_performance=True)
    assert info["applied"] is True
    assert info["reason"] == "applied"
    assert info["threads"] == 4


def test_describe_pinning_reports_missing_p_cores(monkeypatch):
    monkeypatch.setattr(cpuinfo, "is_hybrid", lambda: True)
    monkeypatch.setattr(cpuinfo, "performance_logical_count", lambda: 0)
    monkeypatch.setattr(cpuinfo, "performance_mask", lambda: 0)
    info = cpuinfo.describe_pinning(4, pin_performance=True)
    assert info["applied"] is False
    assert info["reason"] == "no_perf_cores"


def test_describe_pinning_does_not_touch_the_process(monkeypatch):
    """Pure decision helper: it must not change affinity or priority."""
    monkeypatch.setattr(cpuinfo, "is_hybrid", lambda: True)
    monkeypatch.setattr(cpuinfo, "performance_logical_count", lambda: 4)
    monkeypatch.setattr(cpuinfo, "performance_mask", lambda: 0b1111)
    touched: list = []
    monkeypatch.setattr(cpuinfo, "_set_affinity", lambda mask: touched.append(mask) or True)
    monkeypatch.setattr(cpuinfo, "_set_above_normal", lambda: touched.append("priority") or 0)
    cpuinfo.describe_pinning(0, pin_performance=True)
    assert touched == []


def test_default_settings_do_not_pin_to_performance_cores():
    """Pinning measured slower than the default spread (19:22 vs 17:50)."""
    from nestor import settings

    assert settings.DEFAULTS["pin_performance"] is False
    assert settings.DEFAULTS["threads"] == 0


# ------------------------------------------------- machine description (report)
def test_cpu_name_is_cached(monkeypatch):
    """The report asks for the CPU name on every run; the registry read is once."""
    monkeypatch.setattr(cpuinfo, "IS_WINDOWS", True)
    monkeypatch.setattr(cpuinfo, "_cpu_name_cache", None)
    calls = {"n": 0}

    def fake_registry():
        calls["n"] += 1
        # _registry_cpu_name normalises whitespace itself; the cache sits above it
        return "Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz"

    monkeypatch.setattr(cpuinfo, "_registry_cpu_name", fake_registry)

    assert cpuinfo.cpu_name() == "Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz"
    assert cpuinfo.cpu_name() == "Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz"
    assert calls["n"] == 1


def test_total_ram_bytes_is_plausible_and_cached(monkeypatch):
    monkeypatch.setattr(cpuinfo, "_ram_total_cache", None)
    total = cpuinfo.total_ram_bytes()
    # 0 means "unknown" on an exotic platform; anything else must be real RAM
    assert total == 0 or total >= 256 * 1024 * 1024

    monkeypatch.setattr(cpuinfo, "_ram_total_cache", 123456)
    assert cpuinfo.total_ram_bytes() == 123456
