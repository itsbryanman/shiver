"""Phase-based visual vibration recovery."""

from .recover import RecoveryConfig, RecoveryResult, recover_signal
from .rolling_shutter import (
    RollingRecoveryResult,
    RollingShutterTiming,
    recover_rolling_shutter,
)
from .synthetic import SyntheticClip, make_synthetic_clip

__all__ = [
    "RecoveryConfig",
    "RecoveryResult",
    "RollingRecoveryResult",
    "RollingShutterTiming",
    "SyntheticClip",
    "make_synthetic_clip",
    "recover_signal",
    "recover_rolling_shutter",
]

__version__ = "0.1.0"
