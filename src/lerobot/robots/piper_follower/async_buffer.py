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

"""Marvin-style thread-safe playback buffer for Piper action chunks."""

from __future__ import annotations

import logging
import threading
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class StreamActionBuffer:
    """Align late chunks and linearly cross-fade them into remaining actions."""

    def __init__(self, delay: int, exec_horizon: int, min_smooth_steps: int = 8) -> None:
        if delay < 0:
            raise ValueError("delay must be >= 0")
        if exec_horizon < 1:
            raise ValueError("exec_horizon must be >= 1")
        if delay >= exec_horizon:
            raise ValueError("delay must be smaller than exec_horizon")
        if min_smooth_steps < 1:
            raise ValueError("min_smooth_steps must be >= 1")
        self.delay = int(delay)
        self.exec_horizon = int(exec_horizon)
        self.min_smooth_steps = int(min_smooth_steps)
        self._lock = threading.Lock()
        self._queue: deque[np.ndarray] = deque()
        self._step = 0
        self._epoch = 0
        self._last_launch_epoch = -1
        self._last_launch_step = 0
        self._last_action: np.ndarray | None = None
        self._play_horizon = self.exec_horizon

    def reset(self) -> None:
        with self._lock:
            self._queue.clear()
            self._step = 0
            self._epoch = 0
            self._last_launch_epoch = -1
            self._last_launch_step = 0
            self._last_action = None
            self._play_horizon = self.exec_horizon

    def mark_launch_if_ready(self) -> bool:
        """Return true once per chunk when it is time to request the next chunk."""

        with self._lock:
            if self._last_launch_epoch == self._epoch:
                return False
            launch_step = max(0, self._play_horizon - self.delay - 1)
            if self._step < launch_step:
                return False
            self._last_launch_epoch = self._epoch
            self._last_launch_step = self._step
            return True

    def integrate_first_chunk(self, actions: np.ndarray) -> None:
        values = self._validate_chunk(actions)
        with self._lock:
            self._queue = deque(action.copy() for action in values)
            self._step = 0
            self._epoch = 0
            self._last_launch_epoch = -1
            self._last_launch_step = 0
            self._play_horizon = min(self.exec_horizon, len(values))

    def integrate_new_chunk(self, actions: np.ndarray) -> bool:
        values = self._validate_chunk(actions)
        with self._lock:
            drop_n = max(0, self._step - self._last_launch_step)
            if drop_n >= len(values):
                logger.warning("Discarding late Piper chunk: drop=%s length=%s", drop_n, len(values))
                # The gRPC client cannot synchronously bootstrap like Marvin's
                # local broker, so permit a fresh observation/request at once.
                self._last_launch_epoch = -1
                return False
            new = [action.copy() for action in values[drop_n:]]
            if self._queue:
                old = list(self._queue)
            elif self._last_action is not None:
                old = [self._last_action.copy() for _ in range(self.min_smooth_steps)]
            else:
                self._queue = deque(new)
                self._step = 0
                self._epoch += 1
                self._play_horizon = min(self.exec_horizon, len(new))
                return True

            if len(old) < self.min_smooth_steps:
                old.extend(old[-1].copy() for _ in range(self.min_smooth_steps - len(old)))
            overlap = min(len(old), len(new))
            weights = np.array([1.0]) if overlap == 1 else np.linspace(1.0, 0.0, overlap)
            blended = [
                weights[index] * old[index] + (1.0 - weights[index]) * new[index] for index in range(overlap)
            ]
            combined = blended + new[overlap:]
            self._queue = deque(action.copy() for action in combined)
            self._step = 0
            self._epoch += 1
            self._play_horizon = min(self.exec_horizon, len(combined))
            return True

    def get_next_action(self) -> np.ndarray | None:
        with self._lock:
            if not self._queue:
                return None
            action = self._queue.popleft()
            self._last_action = action.copy()
            self._step += 1
            return action

    @staticmethod
    def _validate_chunk(actions: np.ndarray) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float64)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError(f"action chunk must have shape (T, A) with T > 0, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("action chunk contains NaN or infinity")
        return values

    @property
    def remaining(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch
