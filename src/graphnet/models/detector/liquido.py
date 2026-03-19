"""LiquidO-specific `Detector` class(es)."""

from typing import Dict, Callable
import torch
import os
import numpy as np

from graphnet.models.detector.detector import Detector
from graphnet.constants import LIQUIDO_GEOMETRY_TABLE_DIR

class LiquidO_v0(Detector):
    """`Detector` class for LiquidO prototype."""

    geometry_table_path = os.path.join(
        LIQUIDO_GEOMETRY_TABLE_DIR, "liquido_v1.parquet"
    )
    xyz = ["sipm_x", "sipm_y", "sipm_z"]
    string_id_column = "fiber_id"
    sensor_id_column = "sipm_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sipm_x": self._sipm_xyz,
            "sipm_y": self._sipm_xyz,
            "sipm_z": self._sipm_xyz,
            "t": self._t,
        }
        return feature_map
    
    def _sipm_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 1000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 500


class LiquidO_v1(Detector):
    """`Detector` class for LiquidO prototype.  Includes SiPM angles."""

    geometry_table_path = os.path.join(
        LIQUIDO_GEOMETRY_TABLE_DIR, "liquido_v1.parquet"
    )
    xyz = ["sipm_x", "sipm_y", "sipm_z"]
    string_id_column = "fiber_id"
    sensor_id_column = "sipm_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sipm_x": self._sipm_xyz,
            "sipm_y": self._sipm_xyz,
            "sipm_z": self._sipm_xyz,
            "t": self._t,
            "sipm_zenith": self._angle, # added to handle stereo layers
            "sipm_azimuth": self._angle, # added to handle stereo layers
            "charge": self._charge,
        }
        return feature_map

    # Add standardization for the sipm zenith and azimuth angles
    def _angle(self, x:torch.tensor) -> torch.tensor:
        return x / 2 # standardize by 2. (angles range between 0-pi)
    
    def _sipm_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 1000

    def _t(self, x: torch.tensor) -> torch.tensor:
        #return x / 500
        #return np.log(0.01 + np.abs(x)) # time ranges from about 0.5 to 1000., but is strongly peaked at small values
        return np.arcsinh(x)

    def _charge(self, x: torch.tensor) -> torch.tensor:
        return np.arcsinh(x)

class LiquidO_v2(Detector):
    """`Detector` class for LiquidO prototype.  Includes SiPM angles."""

    geometry_table_path = os.path.join(
        LIQUIDO_GEOMETRY_TABLE_DIR, "liquido_v1.parquet" # DFC: Does this need to be updated?
    )
    xyz = ["sipm_x", "sipm_y", "sipm_z"]
    string_id_column = "fiber_id"
    sensor_id_column = "sipm_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sipm_x": self._sipm_xyz,
            "sipm_y": self._sipm_xyz,
            "sipm_z": self._sipm_xyz,
            "time": self._time, # time derived by DAQ
            "charge": self._charge, # charge derived by DAQ
        }
        return feature_map

    # Add standardization for the sipm zenith and azimuth angles
    def _charge(self, x:torch.tensor) -> torch.tensor:
        # per-channel charge (pC) varies from ~3 to ~500, skewed to low values, hence using logarithm
        # Add 0.01 for insurance in case charge is zero, although it should always be positive
        # High charge values are relevant and we don't want to diminish their impact.
        return np.log(0.01 + x)
    
    def _sipm_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 1000

    def _time(self, x: torch.tensor) -> torch.tensor:
        # return x / 500 # less than ideal.

        # DAQData times range between 0-500, peaked at low values.  
        # Using arcsinh(x)/3 maps -10<->1000 to -1<->2.3.  Handles possible negative times from SiPM dark noise etc.
        # High time values are for highly scattered photons that carry less information about the event,
        # and in the future will likely mostly come from  SiPM dark noise.  Dividing by 3 helps to diminish their impact.
        return np.arcsinh(x)/3. 
