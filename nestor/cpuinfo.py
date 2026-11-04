"""CPU topology helpers (Windows, stdlib only - no psutil).

Two jobs, both deliberately conservative:

* pick a sensible worker-thread count when the user left the setting on
  "auto" (physical cores, capped - the ONNX runtime gains little past a
  handful of cores and small laptop parts just throttle);
* on a *hybrid* CPU (Intel P+E cores) keep the heavy worker on the
  performance cores and optionally raise the process priority.

Every entry point degrades to a safe no-op when the information is
unavailable or the platform is not Windows, so non-hybrid machines
(i5-8250U and friends) behave exactly as before.
"""

from __future__ import annotations

import ctypes
import os
import sys
from contextlib import contextmanager
from typing import Dict, Optional

IS_WINDOWS = sys.platform == "win32"

# "Auto" stops here: more threads barely help the ONNX runtime and a 15 W
# laptop throttles sooner instead of finishing faster.
AUTO_THREAD_CAP = 8

# Diarization is a different animal: its agglomerative clustering is
# quadratic in the number of speech segments, so throwing more threads at it
# slows the whole stage down. Measured on an hour of audio: 4 threads beat 8
# on both test laptops (28.4 s vs 33.1 s on a 300 s slice, and 44.5 s vs
# 72.0 s on a loaded machine). "Auto" therefore keeps diarization at a small
# fixed budget instead of following the ASR thread count.
DIAR_THREAD_CAP = 4

_RELATION_PROCESSOR_CORE = 0

_ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000

_layout_cache: Optional[Dict[str, object]] = None


# --------------------------------------------------------------------- winapi
def _kernel32():
    """kernel32 with explicit signatures (pseudo-handles are pointers)."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetLogicalProcessorInformationEx.argtypes = [
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    kernel32.GetLogicalProcessorInformationEx.restype = ctypes.c_int
    kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel32.SetPriorityClass.restype = ctypes.c_int
    kernel32.GetPriorityClass.argtypes = [ctypes.c_void_p]
    kernel32.GetPriorityClass.restype = ctypes.c_uint
    kernel32.GetProcessAffinityMask.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.GetProcessAffinityMask.restype = ctypes.c_int
    kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel32.SetProcessAffinityMask.restype = ctypes.c_int
    return kernel32


def _processor_core_entries() -> Optional[list]:
    """``[(efficiency_class, affinity_mask), ...]`` per physical core, or None.

    Parses ``GetLogicalProcessorInformationEx(RelationProcessorCore)`` by hand
    because the structure ends in a variable-length array.
    """
    if not IS_WINDOWS:
        return None
    try:
        kernel32 = _kernel32()
        size = ctypes.c_ulong(0)
        kernel32.GetLogicalProcessorInformationEx(
            _RELATION_PROCESSOR_CORE, None, ctypes.byref(size)
        )
        if size.value == 0:
            return None
        buf = ctypes.create_string_buffer(size.value)
        if not kernel32.GetLogicalProcessorInformationEx(
            _RELATION_PROCESSOR_CORE, buf, ctypes.byref(size)
        ):
            return None
        raw = buf.raw[: size.value]
        ptr_size = ctypes.sizeof(ctypes.c_void_p)
        entries = []
        offset = 0
        while offset + 8 <= len(raw):
            entry_size = int.from_bytes(raw[offset + 4: offset + 8], "little")
            if entry_size <= 0 or offset + entry_size > len(raw):
                break
            body = raw[offset + 8: offset + entry_size]
            # PROCESSOR_RELATIONSHIP:
            #   Flags(1), EfficiencyClass(1), Reserved(20), GroupCount(2),
            #   GroupMask[] = GROUP_AFFINITY { Mask(ptr), Group(2), Reserved }
            if len(body) >= 24:
                efficiency_class = body[1]
                group_count = int.from_bytes(body[22:24], "little")
                mask = 0
                stride = ptr_size + 8
                for group in range(max(1, group_count)):
                    base = 24 + group * stride
                    if base + ptr_size <= len(body):
                        mask |= int.from_bytes(
                            body[base: base + ptr_size], "little"
                        )
                entries.append((efficiency_class, mask))
            offset += entry_size
        return entries or None
    except Exception:  # noqa: BLE001 - topology is a nicety, never a blocker
        return None


# -------------------------------------------------------------------- topology
def _layout() -> Dict[str, object]:
    """Cached topology: physical cores, classes and the performance-core mask."""
    global _layout_cache
    if _layout_cache is not None:
        return _layout_cache

    entries = _processor_core_entries()
    if not entries:
        _layout_cache = {
            "physical": None,
            "hybrid": False,
            "perf_mask": 0,
            "perf_logical": 0,
        }
        return _layout_cache

    classes: Dict[int, list] = {}
    for efficiency_class, mask in entries:
        bucket = classes.setdefault(efficiency_class, [0, 0])
        bucket[0] += 1  # physical cores in this class
        bucket[1] |= mask  # logical processors in this class

    hybrid = len(classes) > 1
    # Higher EfficiencyClass == greater performance (Win32 docs on
    # PROCESSOR_RELATIONSHIP): on a Core i5-1235U the 2 P-cores report a
    # higher class than the 8 E-cores. Picking min() here selected the E-core
    # cluster - "auto" then asked for 8 threads and pinned the process to the
    # slow cores, which is why the nominally stronger laptop ran slower.
    # Only meaningful - and only used - on a hybrid CPU.
    perf_class = max(classes)
    perf_mask = classes[perf_class][1] if hybrid else 0
    _layout_cache = {
        "physical": len(entries),
        "hybrid": hybrid,
        "perf_mask": perf_mask,
        "perf_logical": bin(perf_mask).count("1") if perf_mask else 0,
    }
    return _layout_cache


def physical_cores() -> int:
    """Physical core count (falls back to logical ``os.cpu_count``)."""
    layout = _layout()
    count = layout.get("physical")
    if isinstance(count, int) and count > 0:
        return count
    return max(1, os.cpu_count() or 2)


def is_hybrid() -> bool:
    """True when the CPU mixes performance and efficiency cores."""
    return bool(_layout().get("hybrid"))


def performance_mask() -> int:
    """Affinity mask of the performance (P) cores, 0 when not hybrid."""
    return int(_layout().get("perf_mask") or 0)


def performance_logical_count() -> int:
    """Number of logical processors on the performance cores."""
    return int(_layout().get("perf_logical") or 0)


def resolve_threads(threads: int) -> int:
    """Threads to actually use: the explicit value, or physical cores (capped).

    "Auto" uses *all* physical cores, E-cores included.  An earlier version
    capped auto at the P-core count on a hybrid CPU, on the theory that the
    slow small cores become stragglers at every ASR chunk barrier.  That
    theory came from the first i5-1235U run, which was taken while the P/E
    clusters were swapped; on the corrected runs the opposite is true: 8
    threads finished an hour of audio in 17:50 against 19:22 for 4 threads
    locked to the P-cores.  Transcription is the stage that scales, and the
    E-cores help it.  The global cap stays: on a 15 W laptop more threads
    throttle instead of finishing sooner.  Diarization does *not* follow this
    value - see :func:`resolve_diar_threads`.
    """
    try:
        value = int(threads)
    except (TypeError, ValueError):
        value = 0
    if value > 0:
        return value
    return max(2, min(physical_cores(), AUTO_THREAD_CAP))


def resolve_diar_threads(threads: int) -> int:
    """Threads for the diarizer: the explicit value, or a small fixed budget.

    Unlike transcription, diarization gets slower with too many threads (see
    ``DIAR_THREAD_CAP``): its clustering cost grows with the number of speech
    segments while the segmentation network itself is small, so extra threads
    only add contention.  An explicit user value still wins, so the setting
    stays honest; "auto" is capped at a handful of cores.
    """
    try:
        value = int(threads)
    except (TypeError, ValueError):
        value = 0
    if value > 0:
        return value
    return max(2, min(physical_cores(), DIAR_THREAD_CAP))


# --------------------------------------------------------- priority & affinity
def _set_above_normal() -> Optional[int]:
    """Raise this process to ABOVE_NORMAL; return the previous class or None."""
    if not IS_WINDOWS:
        return None
    try:
        kernel32 = _kernel32()
        handle = kernel32.GetCurrentProcess()
        previous = kernel32.GetPriorityClass(handle)
        if not kernel32.SetPriorityClass(handle, _ABOVE_NORMAL_PRIORITY_CLASS):
            return None
        return int(previous)
    except Exception:  # noqa: BLE001
        return None


def _restore_priority(priority_class: int) -> None:
    if not IS_WINDOWS:
        return
    try:
        kernel32 = _kernel32()
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), priority_class)
    except Exception:  # noqa: BLE001
        pass


def _get_affinity() -> Optional[int]:
    if not IS_WINDOWS:
        return None
    try:
        kernel32 = _kernel32()
        process_mask = ctypes.c_size_t(0)
        system_mask = ctypes.c_size_t(0)
        if not kernel32.GetProcessAffinityMask(
            kernel32.GetCurrentProcess(),
            ctypes.byref(process_mask),
            ctypes.byref(system_mask),
        ):
            return None
        return int(process_mask.value)
    except Exception:  # noqa: BLE001
        return None


def _set_affinity(mask: int) -> bool:
    if not IS_WINDOWS or not mask:
        return False
    try:
        kernel32 = _kernel32()
        return bool(
            kernel32.SetProcessAffinityMask(
                kernel32.GetCurrentProcess(), ctypes.c_size_t(mask)
            )
        )
    except Exception:  # noqa: BLE001
        return False


def _begin(
    threads: int, priority: bool, pin_performance: bool
) -> Dict[str, object]:
    state: Dict[str, object] = {"priority": None, "affinity": None}

    if priority:
        previous = _set_above_normal()
        if previous is not None:
            state["priority"] = previous

    if pin_performance and is_hybrid():
        mask = performance_mask()
        perf_logical = performance_logical_count()
        # Only pin when the whole workload still fits on the P-cores; with more
        # threads than that the E-cores should help, so leave it to Windows.
        if mask and perf_logical and resolve_threads(threads) <= perf_logical:
            previous_mask = _get_affinity()
            chosen = mask & previous_mask if previous_mask else mask
            if chosen and _set_affinity(chosen):
                state["affinity"] = previous_mask

    return state


def _end(state: Dict[str, object]) -> None:
    previous_mask = state.get("affinity")
    if isinstance(previous_mask, int) and previous_mask:
        _set_affinity(previous_mask)
    previous_priority = state.get("priority")
    if isinstance(previous_priority, int):
        _restore_priority(previous_priority)


@contextmanager
def tuned(threads: int, priority: bool = False, pin_performance: bool = True):
    """Scope that (optionally) raises priority / pins the process to P-cores."""
    state = _begin(threads, priority, pin_performance)
    try:
        yield
    finally:
        _end(state)


def describe_pinning(threads: int, pin_performance: bool) -> Dict[str, object]:
    """Why the process was, or was not, pinned to the performance cores.

    Pure bookkeeping for the diagnostics report: it answers the question the
    benchmark kept raising ("the setting is on, but did it actually do
    anything?") without touching affinity. ``reason`` is a stable machine
    code; ``message`` is the human sentence shown in the UI.
    """
    resolved = resolve_threads(threads)
    info: Dict[str, object] = {
        "requested": pin_performance,
        "applied": False,
        "threads": resolved,
        "hybrid": is_hybrid(),
        "perf_cores": performance_logical_count(),
        "reason": "",
        "message": "",
    }
    if not pin_performance:
        reason, message = "disabled", "выключено в настройках"
    elif not is_hybrid():
        reason, message = "not_hybrid", "процессор не гибридный — привязка не нужна"
    elif not performance_logical_count():
        reason, message = "no_perf_cores", "не удалось определить быстрые ядра"
    elif resolved > performance_logical_count():
        reason = "too_many_threads"
        message = (
            f"потоков {resolved} больше, чем быстрых ядер "
            f"{performance_logical_count()} — оставлено на усмотрение Windows"
        )
    else:
        reason, message = "applied", "обработка закреплена за быстрыми ядрами"
        info["applied"] = True
    info["reason"] = reason
    info["message"] = message
    return info
