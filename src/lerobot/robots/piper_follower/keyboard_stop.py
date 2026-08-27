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

"""Interactive hold/home hotkeys for Piper deployment."""

from __future__ import annotations

import logging
import select
import sys
import threading
from collections.abc import Callable

from .deployment import StopAt

logger = logging.getLogger(__name__)


class KeyboardStopWatcher:
    def __init__(self, on_stop: Callable[[StopAt], None]) -> None:
        self._on_stop = on_stop
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        if not sys.stdin.isatty():
            logger.info("stdin is not a TTY; Piper s/h stop hotkeys are disabled")
            return False
        self._thread = threading.Thread(target=self._loop, name="piper-keyboard-stop", daemon=True)
        self._thread.start()
        logger.info("Piper hotkeys: s=hold and stop, h=smooth-home and stop")
        return True

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            except (OSError, ValueError):
                break
            if not ready:
                continue
            try:
                key = sys.stdin.read(1).lower()
            except OSError:
                break
            if key == "s":
                self._on_stop("hold")
                break
            if key == "h":
                self._on_stop("home")
                break
