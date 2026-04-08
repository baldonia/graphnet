"""ratpac-two data extractor for Ntuple Files."""
from typing import Dict, Any, Optional
import numpy as np

from graphnet.data.extractors import Extractor

class NtupleHitExtractor(Extractor):
    """
    Ntuple Extractor for `HitData`. 
    Handles digit and lognormal charge types.
    Applies time smearing strictly for training data.
    """

    def __init__(
        self, 
        charge_type: str, 
        is_training: bool,
        output_keys: Optional[Dict[str, str]] = None
    ) -> None:
        super().__init__(extractor_name="HitData")
        self.charge_type = charge_type.lower()
        self.is_training = is_training
        
        if self.charge_type not in ['digit', 'lognormal']:
            raise ValueError(f"Unsupported charge type: '{self.charge_type}'. Must be 'digit' or 'lognormal'.")
        
        # Default output keys
        self._output_keys = {
            'id': 'Photosensor_id',
            'x': 'photosensor_x',
            'y': 'photosensor_y',
            'z': 'photosensor_z',
            'ze': 'photosensor_zenith',
            'az': 'photosensor_azimuth',
            't': 'photosensor_time',
            'charge': 'charge',
        }
        if output_keys is not None:
            self._output_keys.update(output_keys)

    def __call__(self, event_data: Dict[str, Any], maps: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Select input keys based on charge_type
        if self.charge_type == 'lognormal':
            idx = event_data['fit_pmtid_Lognormal']
            base_time = event_data['fit_time_Lognormal']
            charge = event_data['fit_charge_Lognormal']
        elif self.charge_type == 'digit':
            idx = event_data['digitPMTID']
            base_time = event_data['digitTime']
            charge = event_data['digitCharge']

        # 2. Geometry Mapping
        pmtu = maps['pmtU'][0][idx].astype(np.float32)
        pmtv = maps['pmtV'][0][idx].astype(np.float32)
        pmtw = maps['pmtW'][0][idx].astype(np.float32)

        # 3. Time calculation
        # Training events removes trigger offset and adds normal smearing
        # Reco events (both sim and data) use the raw hit time
        if self.is_training:
            time = base_time + event_data['triggerTime'] + np.random.normal(0, 20)
        else:
            time = base_time

        # 4. Build output dictionary
        return {
            self._output_keys['id']: idx,
            self._output_keys['x']: maps['pmtX'][0][idx].astype(np.float32),
            self._output_keys['y']: maps['pmtY'][0][idx].astype(np.float32),
            self._output_keys['z']: maps['pmtZ'][0][idx].astype(np.float32),
            self._output_keys['ze']: np.arccos(pmtw).astype(np.float32),
            self._output_keys['az']: np.mod(np.arctan2(pmtv, pmtu), 2 * np.pi).astype(np.float32),
            self._output_keys['t']: time.astype(np.float32),
            self._output_keys['charge']: charge.astype(np.float32),
        }


class NtupleTruthExtractor(Extractor):
    """
    Ntuple Extractor for `TruthData`. 
    Handles real Monte Carlo truth extraction or zero-spoofing for blind data.
    """

    def __init__(self, is_data: bool = False) -> None:
        super().__init__(extractor_name="TruthData")
        self.is_data = is_data

    def __call__(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.is_data:
            # Blind Data: Return dummy values to satisfy the GraphNet schema
            return {
                "vertex_x": 0.0, "vertex_y": 0.0, "vertex_z": 0.0,
                "zenith": 0.0, "azimuth": 0.0, "interaction_time": 0.0,
                "energy": 0.0, "pid": 0,
            }
        else:
            # Simulation: Extract real MC truth
            mcu, mcv, mcw = event_data['mcu'], event_data['mcv'], event_data['mcw']
            
            mcaz = np.mod(np.arctan2(mcv, mcu), 2 * np.pi).astype(np.float32)
            mcze = np.arccos(mcw).astype(np.float32)

            return {
                "vertex_x": event_data['mcx'].astype(np.float32),
                "vertex_y": event_data['mcy'].astype(np.float32),
                "vertex_z": event_data['mcz'].astype(np.float32),
                "zenith": mcze,
                "azimuth": mcaz,
                "interaction_time": event_data['mct'],
                "energy": event_data['mcke'].astype(np.float32),
                "pid": event_data['mcpdg'],
            }
