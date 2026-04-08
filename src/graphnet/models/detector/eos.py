"""Eos-specific `Detector` class(es)."""

from typing import Dict, Callable
import torch
import os
import numpy as np

from graphnet.models.detector.detector import Detector
from graphnet.constants import EOS_GEOMETRY_TABLE_DIR

class Eos_v0(Detector):
    """`Detector` class for Eos.  Includes PMT angles."""

    geometry_table_path = os.path.join(
        EOS_GEOMETRY_TABLE_DIR, "eos.parquet"
    )
    xyz = ["pmt_x", "pmt_y", "pmt_z"]
    string_id_column = "pmt_type"
    sensor_id_column = "pmt_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "pmt_x": self._pmt_xyz,
            "pmt_y": self._pmt_xyz,
            "pmt_z": self._pmt_xyz,
            "t": self._t,
            "pmt_zenith": self._angle,
            "pmt_azimuth": self._angle,
            "charge": self._charge,
        }
        return feature_map

    def _angle(self, x:torch.tensor) -> torch.tensor:
        return x / 2 # standardize by 2. (angles range between 0-2pi)

    def _pmt_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 1000 # Divide spatial dimensions (in mm) by 1000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return np.arcsinh(x) # Handles large times while being safe for negative times

    def _charge(self, x: torch.tensor) -> torch.tensor:
        return np.arcsinh(x) # Handles large charges while being safe for negative charges
