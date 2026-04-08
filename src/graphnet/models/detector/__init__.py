"""Detector-specific modules, for data ingestion and standardisation."""

from .icecube import IceCube86, IceCubeDeepCore, IceCubeUpgrade
from .detector import Detector
from .liquido import LiquidO_v0, LiquidO_v1, LiquidO_v2
from .prometheus import ORCA150
from .eos import Eos_v0
