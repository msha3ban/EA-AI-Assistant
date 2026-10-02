from __future__ import annotations

import os
import re
import resource
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self


@dataclass(frozen=True)
class ResourceReading:
    vram_mib: float | None
    available_ram_mib: float | None
    process_ram_mib: float


class ResourceSampler(Protocol):
    def sample(self) -> ResourceReading: ...


class SystemResourceSampler:
    def sample(self) -> ResourceReading:
        vram = None
        if shutil.which("nvidia-smi"):
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                values = [
                    float(line.strip())
                    for line in result.stdout.splitlines()
                    if line.strip()
                ]
                vram = max(values) if values else None
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
        available = None
        try:
            content = Path("/proc/meminfo").read_text(encoding="ascii")
            match = re.search(r"^MemAvailable:\s+(\d+) kB", content, re.MULTILINE)
            if match:
                available = int(match.group(1)) / 1024
        except OSError:
            pass
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        process = usage / 1024 if os.name != "darwin" else usage / (1024 * 1024)
        return ResourceReading(vram, available, process)


class SamplingWindow:
    def __init__(self, sampler: ResourceSampler, interval: float = 0.5) -> None:
        self.sampler, self.interval = sampler, interval
        self.before = sampler.sample()
        self.readings = [self.before]
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.wait(self.interval):
            self.readings.append(self.sampler.sample())

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join()
        self.readings.append(self.sampler.sample())

    def result(self) -> dict[str, float | None]:
        vram_values = [r.vram_mib for r in self.readings if r.vram_mib is not None]
        ram_values = [
            r.available_ram_mib
            for r in self.readings
            if r.available_ram_mib is not None
        ]
        peak_vram = max(vram_values) if vram_values else None
        vram_delta = (
            max((v - self.before.vram_mib for v in vram_values), default=None)
            if self.before.vram_mib is not None
            else None
        )
        ram_delta = (
            max((self.before.available_ram_mib - r for r in ram_values), default=None)
            if self.before.available_ram_mib is not None
            else None
        )
        process_peak = max(
            [self.before.process_ram_mib, *(r.process_ram_mib for r in self.readings)]
        )
        ram_peaks = [value for value in (process_peak, ram_delta) if value is not None]
        return {
            "peak_vram_mib": peak_vram,
            "peak_vram_delta_mib": vram_delta,
            "peak_ram_mib": max(ram_peaks) if ram_peaks else None,
            "peak_process_ram_mib": process_peak,
            "system_ram_increase_mib": ram_delta,
        }
