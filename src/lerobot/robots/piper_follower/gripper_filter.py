# Copyright 2026 jokeru8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Piper gripper post-processing for real-robot policy deployment."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass
class PiperGripperFilterConfig:
    """Schmitt trigger, vote window and time-based ramp in Piper gripper units.

    Piper datasets use gripper stroke in meters: ``0`` is closed and a larger
    value (normally around ``0.08``) is open. Filtering is disabled by default
    so existing checkpoints keep their current continuous-gripper behaviour.
    """

    enabled: bool = False
    close_threshold: float = 0.024
    open_threshold: float = 0.040
    window_n: int = 4
    ratio: float = 0.75
    ramp_s: float = 0.3
    closed_position: float = 0.0
    open_position: float = 0.08

    def __post_init__(self) -> None:
        if not self.closed_position <= self.close_threshold < self.open_threshold <= self.open_position:
            raise ValueError(
                "gripper positions must satisfy closed <= close_threshold < open_threshold <= open"
            )
        if self.window_n < 1:
            raise ValueError("window_n must be >= 1")
        if not 0.0 < self.ratio <= 1.0:
            raise ValueError("ratio must be in (0, 1]")
        if self.ramp_s <= 0.0:
            raise ValueError("ramp_s must be > 0")


class GripperHysteresisWindow:
    """Convert a noisy continuous command into a stable open/closed command.

    This is the Marvin/UMI Schmitt-window-ramp design adapted to Piper's
    physical convention (small stroke means closed, large stroke means open).
    """

    def __init__(self, config: PiperGripperFilterConfig, *, initial: float | None = None) -> None:
        self.config = config
        initial = config.open_position if initial is None else float(initial)
        self.target = self._nearest_endpoint(initial)
        self.cmd = float(initial)
        self._ramp_from = self.cmd
        self._ramp_to = self.target
        self._ramp_t0 = 0.0
        # -1 closes, +1 opens.
        self._samples: deque[int] = deque(maxlen=config.window_n)

    def _nearest_endpoint(self, value: float) -> float:
        cfg = self.config
        midpoint = 0.5 * (cfg.closed_position + cfg.open_position)
        return cfg.open_position if value >= midpoint else cfg.closed_position

    @property
    def state(self) -> float:
        return self.target

    def reset(self, initial: float | None = None) -> None:
        cfg = self.config
        initial = cfg.open_position if initial is None else float(initial)
        self.target = self._nearest_endpoint(initial)
        self.cmd = initial
        self._ramp_from = initial
        self._ramp_to = self.target
        self._ramp_t0 = 0.0
        self._samples.clear()

    def _vote(self, raw: float) -> int:
        cfg = self.config
        if self.target == cfg.closed_position:
            # Once closed, only a clear high command votes to open.
            return 1 if raw > cfg.open_threshold else -1
        # Once open, only a clear low command votes to close.
        return -1 if raw < cfg.close_threshold else 1

    def _update_latch(self, now: float) -> None:
        cfg = self.config
        need = cfg.ratio * cfg.window_n
        close_n = sum(sample == -1 for sample in self._samples)
        open_n = sum(sample == 1 for sample in self._samples)
        new_target = self.target
        if close_n >= need and open_n < need:
            new_target = cfg.closed_position
        elif open_n >= need and close_n < need:
            new_target = cfg.open_position
        if new_target != self.target:
            self._ramp_from = self.cmd
            self._ramp_to = new_target
            self._ramp_t0 = now
            self.target = new_target
            self._samples.clear()

    def _interpolate(self, now: float) -> float:
        cfg = self.config
        if self._ramp_from == self._ramp_to:
            self.cmd = self._ramp_to
            return self.cmd
        elapsed = now - self._ramp_t0
        if elapsed <= 0.0:
            self.cmd = self._ramp_from
        elif elapsed >= cfg.ramp_s:
            self.cmd = self._ramp_to
            self._ramp_from = self._ramp_to
        else:
            alpha = elapsed / cfg.ramp_s
            self.cmd = self._ramp_from + alpha * (self._ramp_to - self._ramp_from)
        return self.cmd

    def update(self, raw: float, *, now: float | None = None) -> float:
        if not self.config.enabled:
            self.cmd = float(raw)
            return self.cmd
        timestamp = time.monotonic() if now is None else float(now)
        self._samples.append(self._vote(float(raw)))
        self._update_latch(timestamp)
        return self._interpolate(timestamp)
