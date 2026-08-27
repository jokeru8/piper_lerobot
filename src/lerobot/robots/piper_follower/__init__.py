from typing import TYPE_CHECKING, Any

from .async_buffer import StreamActionBuffer
from .config_piper_follower import PIPERFollowerConfig
from .deployment import PiperAsyncBufferConfig, PiperDeploymentConfig, PiperDeploymentController
from .ik_adapter import CallablePiperIKAdapter, PiperIKAdapter

if TYPE_CHECKING:
    from .piper_follower import PIPERFollower


def __getattr__(name: str) -> Any:
    """Keep pure deployment utilities importable without the hardware SDK."""

    if name == "PIPERFollower":
        from .piper_follower import PIPERFollower

        return PIPERFollower
    raise AttributeError(name)
