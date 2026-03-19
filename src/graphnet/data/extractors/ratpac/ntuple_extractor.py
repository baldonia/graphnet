"""ratpac-two data extractor for Ntuple Files."""
from typing import Dict, Any, List
import numpy as np

from graphnet.data.extractors import Extractor


class NtupleExtractor(Extractor):

    def __init__(self, extractor_name: str, column_names: List[str]):
        # Member variable(s)
        self._table = extractor_name
        self._column_names = column_names
        super().__init__(extractor_name=extractor_name)

class MCHitExtractorTrain_lognormal(Extractor):
    """Extractor for `HitData` in ratpac-two ntuple ROOT files. Uses lognormal pulse fitting."""

    def __init__(self, output_keys: Dict[str, str] = None) -> None:
        """
        Parameters:
            output_keys (Dict[str, str]): A mapping from internal names to desired output keys.
                The default mapping is:
                    {
                        'id': 'Photosensor_id',
                        'x': 'photosensor_x',
                        'y': 'photosensor_y',
                        'z': 'photosensor_z',
                        'ze': 'photosensor_zenith',
                        'az': 'photosensor_azimuth',
                        't': 'photosensor_time',
                        'charge': 'charge',
                    }
        """
        super().__init__(extractor_name="HitData")
        # Default keys
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
        # Update with provided keys if any
        if output_keys is not None:
            self._output_keys.update(output_keys)

    def __call__(self, event_data: Dict[str, Any], maps: Dict[str, Any]) -> Dict[str, Any]:

        idx = event_data['fit_pmtid_Lognormal']
        #I need to consider the implications of returning none before adding this cut
        #if len(idx) > 3:
        pmtu = maps['pmtU'][0][idx].astype(np.float32)
        pmtv = maps['pmtV'][0][idx].astype(np.float32)
        pmtw = maps['pmtW'][0][idx].astype(np.float32)
        # Center events around 0 and add random time spread
        time = event_data['fit_time_Lognormal'] + event_data['triggerTime'] + np.random.normal(0, 20)

        data = {
            self._output_keys['id']: idx,
            self._output_keys['x']: maps['pmtX'][0][idx].astype(np.float32),
            self._output_keys['y']: maps['pmtY'][0][idx].astype(np.float32),
            self._output_keys['z']: maps['pmtZ'][0][idx].astype(np.float32),
            self._output_keys['ze']: np.arccos(pmtw).astype(np.float32),
            self._output_keys['az']: np.mod(np.arctan2(pmtv, pmtu), 2 * np.pi).astype(np.float32),
            self._output_keys['t']: time.astype(np.float32),
            self._output_keys['charge']: event_data['fit_charge_Lognormal'].astype(np.float32),
        }

        return data

        #else:

        #    return None


class MCHitExtractorReco_lognormal(Extractor):
    """Extractor for `HitData` in ratpac-two ntuple ROOT files. Uses lognormal pulse fitting."""

    def __init__(self, output_keys: Dict[str, str] = None) -> None:
        """
        Parameters:
            output_keys (Dict[str, str]): A mapping from internal names to desired output keys.
                The default mapping is:
                    {
                        'id': 'Photosensor_id',
                        'x': 'photosensor_x',
                        'y': 'photosensor_y',
                        'z': 'photosensor_z',
                        'ze': 'photosensor_zenith',
                        'az': 'photosensor_azimuth',
                        't': 'photosensor_time',
                        'charge': 'charge',
                    }
        """
        super().__init__(extractor_name="HitData")
        # Default keys
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
        # Update with provided keys if any
        if output_keys is not None:
            self._output_keys.update(output_keys)

    def __call__(self, event_data: Dict[str, Any], maps: Dict[str, Any]) -> Dict[str, Any]:

        idx = event_data['fit_pmtid_Lognormal']
        #I need to consider the implications of returning none before adding this cut
        #if len(idx) > 3:
        pmtu = maps['pmtU'][0][idx].astype(np.float32)
        pmtv = maps['pmtV'][0][idx].astype(np.float32)
        pmtw = maps['pmtW'][0][idx].astype(np.float32)

        data = {
            self._output_keys['id']: idx,
            self._output_keys['x']: maps['pmtX'][0][idx].astype(np.float32),
            self._output_keys['y']: maps['pmtY'][0][idx].astype(np.float32),
            self._output_keys['z']: maps['pmtZ'][0][idx].astype(np.float32),
            self._output_keys['ze']: np.arccos(pmtw).astype(np.float32),
            self._output_keys['az']: np.mod(np.arctan2(pmtv, pmtu), 2 * np.pi).astype(np.float32),
            self._output_keys['t']: event_data['fit_time_Lognormal'].astype(np.float32),
            self._output_keys['charge']: event_data['fit_charge_Lognormal'].astype(np.float32),
        }

        return data

        #else:

        #    return None

class MCHitExtractorTrain_digit(Extractor):
    """Extractor for `HitData` in ratpac-two ntuple ROOT files. Uses integrated wfm charge."""

    def __init__(self, output_keys: Dict[str, str] = None) -> None:
        """
        Parameters:
            output_keys (Dict[str, str]): A mapping from internal names to desired output keys.
                The default mapping is:
                    {
                        'id': 'Photosensor_id',
                        'x': 'photosensor_x',
                        'y': 'photosensor_y',
                        'z': 'photosensor_z',
                        'ze': 'photosensor_zenith',
                        'az': 'photosensor_azimuth',
                        't': 'photosensor_time',
                        'charge': 'charge',
                    }
        """
        super().__init__(extractor_name="HitData")
        # Default keys
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
        # Update with provided keys if any
        if output_keys is not None:
            self._output_keys.update(output_keys)

    def __call__(self, event_data: Dict[str, Any], maps: Dict[str, Any]) -> Dict[str, Any]:

        idx = event_data['digitPMTID']
        #I need to consider the implications of returning none before adding this cut
        #if len(idx) > 3:
        pmtu = maps['pmtU'][0][idx].astype(np.float32)
        pmtv = maps['pmtV'][0][idx].astype(np.float32)
        pmtw = maps['pmtW'][0][idx].astype(np.float32)
        # Center events around 0 and add random time spread
        time = event_data['digitTime'] + event_data['triggerTime'] + np.random.normal(0, 20)

        data = {
            self._output_keys['id']: idx,
            self._output_keys['x']: maps['pmtX'][0][idx].astype(np.float32),
            self._output_keys['y']: maps['pmtY'][0][idx].astype(np.float32),
            self._output_keys['z']: maps['pmtZ'][0][idx].astype(np.float32),
            self._output_keys['ze']: np.arccos(pmtw).astype(np.float32),
            self._output_keys['az']: np.mod(np.arctan2(pmtv, pmtu), 2 * np.pi).astype(np.float32),
            self._output_keys['t']: time.astype(np.float32),
            self._output_keys['charge']: event_data['digitCharge'].astype(np.float32),
        }

        return data

        #else:

        #    return None

class MCHitExtractorReco_digit(Extractor):
    """Extractor for `HitData` in ratpac-two ntuple ROOT files. Uses integrated wfm charge."""

    def __init__(self, output_keys: Dict[str, str] = None) -> None:
        """
        Parameters:
            output_keys (Dict[str, str]): A mapping from internal names to desired output keys.
                The default mapping is:
                    {
                        'id': 'Photosensor_id',
                        'x': 'photosensor_x',
                        'y': 'photosensor_y',
                        'z': 'photosensor_z',
                        'ze': 'photosensor_zenith',
                        'az': 'photosensor_azimuth',
                        't': 'photosensor_time',
                        'charge': 'charge',
                    }
        """
        super().__init__(extractor_name="HitData")
        # Default keys
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
        # Update with provided keys if any
        if output_keys is not None:
            self._output_keys.update(output_keys)

    def __call__(self, event_data: Dict[str, Any], maps: Dict[str, Any]) -> Dict[str, Any]:

        idx = event_data['digitPMTID']
        #I need to consider the implications of returning none before adding this cut
        #if len(idx) > 3:
        pmtu = maps['pmtU'][0][idx].astype(np.float32)
        pmtv = maps['pmtV'][0][idx].astype(np.float32)
        pmtw = maps['pmtW'][0][idx].astype(np.float32)

        data = {
            self._output_keys['id']: idx,
            self._output_keys['x']: maps['pmtX'][0][idx].astype(np.float32),
            self._output_keys['y']: maps['pmtY'][0][idx].astype(np.float32),
            self._output_keys['z']: maps['pmtZ'][0][idx].astype(np.float32),
            self._output_keys['ze']: np.arccos(pmtw).astype(np.float32),
            self._output_keys['az']: np.mod(np.arctan2(pmtv, pmtu), 2 * np.pi).astype(np.float32),
            self._output_keys['t']: event_data['digitTime'].astype(np.float32),
            self._output_keys['charge']: event_data['digitCharge'].astype(np.float32),
        }

        return data

        #else:

        #    return None

class MCTruthExtractor(Extractor):
    """Extractor for `TruthData` in Eos ntuple files for simulation."""

    def __init__(self) -> None:
        super().__init__(extractor_name="TruthData")

    def __call__(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        # Get particle gun vertex information (unit: mm)
        mcx = event_data['mcx']
        mcy = event_data['mcy']
        mcz = event_data['mcz']
        # Get initial direction information (u v w share the same coordinate system as x y z)
        mcu = event_data['mcu']
        mcv = event_data['mcv']
        mcw = event_data['mcw']
        # get particle gun time (unit: ns)
        mct = event_data['mct']
        # initial kinetic energy of particle (unit: MeV)
        mcke = event_data['mcke']
        # pdg identifier for initial particle
        mcpdg = event_data['mcpdg']

        #convert direction from cartesian to spherical coordinates
        mcaz = np.mod(np.arctan2(mcv, mcu), 2 * np.pi).astype(np.float32)
        mcze = np.arccos(mcw).astype(np.float32)

        data = {
            "vertex_x": mcx.astype(np.float32),
            "vertex_y": mcy.astype(np.float32),
            "vertex_z": mcz.astype(np.float32),
            "zenith": mcze,
            "azimuth": mcaz,
            "interaction_time": mct,
            "energy": mcke.astype(np.float32),
            "pid": mcpdg,
        }
        return data

class MCTruthExtractor_data(Extractor):
    """Extractor for `TruthData` in Eos ntuple files for data. Data events do not know truth information."""

    def __init__(self) -> None:
        super().__init__(extractor_name="TruthData")

    def __call__(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        # Get particle gun vertex information (unit: mm)
        mcx = 0
        mcy = 0
        mcz = 0
        # Get initial direction information (u v w share the same coordinate system as x y z)
        mcu = 0
        mcv = 0
        mcw = 0
        # get particle gun time (unit: ns)
        mct = 0
        # initial kinetic energy of particle (unit: MeV)
        mcke = 0
        # pdg identifier for initial particle
        mcpdg = 0

        #convert direction from cartesian to spherical coordinates
        mcaz = 0
        mcze = 0

        data = {
            "vertex_x": mcx,
            "vertex_y": mcy,
            "vertex_z": mcz,
            "zenith": mcze,
            "azimuth": mcaz,
            "interaction_time": mct,
            "energy": mcke,
            "pid": mcpdg,
        }
        return data
