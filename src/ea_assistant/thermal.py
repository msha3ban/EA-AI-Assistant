from __future__ import annotations

from .adapters import Sensors
from .config import ThermalConfig


class ThermalStop(RuntimeError):
    """Processing must stop because current machine conditions are unsafe."""


class ThermalGuard:
    def __init__(self, sensors: Sensors, config: ThermalConfig) -> None:
        self.sensors, self.config = sensors, config

    def check(self) -> None:
        temp = self.sensors.gpu_temperature()
        ac = self.sensors.on_ac()
        if temp is None:
            raise ThermalStop(
                "Thermal guard stopped processing: GPU temperature unavailable"
            )
        if temp >= self.config.hard_stop_c:
            raise ThermalStop(
                f"Thermal guard stopped processing: GPU temperature {temp} C reached hard stop"
            )
        if self.config.require_ac and ac is not True:
            raise ThermalStop(
                "Thermal guard stopped processing: AC power is off or unknown"
            )
