"""Modules for reading data files from Eos."""

import os
from glob import glob
from typing import List, Union, Dict, Any

import numpy as np
import uproot

from graphnet.data.extractors.ratpac import MCHitExtractorTrain_lognormal, MCHitExtractorReco_lognormal, MCHitExtractorTrain_digit, MCHitExtractorReco_digit, MCTruthExtractor, MCTruthExtractor_data
from .graphnet_file_reader import GraphNeTFileReader


class NtupleReader(GraphNeTFileReader):
    """A class for reading ntuple ROOT files from ratpac-two."""

    _accepted_file_extensions = [".root"]
    _accepted_extractors = [MCHitExtractorTrain_lognormal, MCHitExtractorReco_lognormal, MCHitExtractorTrain_digit, MCHitExtractorReco_digit, MCTruthExtractor]

    def __call__(self, file_path: str) -> List[Dict[str, Dict[str, Any]]]:
        outputs = []

        with uproot.open(file_path) as file:
            out_key = self.get_valid_out_key(file)
            obsdata = file[out_key].arrays(
                filter_name=[
                    'fit_charge_Lognormal', 'fit_pmtid_Lognormal', 'fit_time_Lognormal', 'triggerTime',
                    'digitPMTID', 'digitTime', 'digitCharge',
                    'mcx', 'mcy', 'mcz', 'mcu', 'mcv', 'mcw', 'mct', 'mcke', 'mcpdg',
                    'evid', 'subev'
                ],
                library='np'
            )
            maps = file["meta;1"].arrays(
                filter_name=['pmtX', 'pmtY', 'pmtZ', 'pmtU', 'pmtV', 'pmtW'],
                library='np'
            )

            n_events = len(obsdata['mcx'])  # Number of events

            for i in range(n_events):
                if (obsdata['subev'][i]!=0): # get rid of sub events
                    continue
                event_data = {key: obsdata[key][i] for key in obsdata.keys()}
                event_outputs = {}

                for extractor in self._extractors:
                    if isinstance(extractor, (MCHitExtractorTrain_lognormal, MCHitExtractorReco_lognormal, MCHitExtractorTrain_digit, MCHitExtractorReco_digit)):
                        data = extractor(event_data, maps)
                    else:
                        data = extractor(event_data)
                    if data is not None:
                        event_outputs[extractor._extractor_name] = data

                if event_outputs:
                    outputs.append(event_outputs)

        return outputs

    def get_valid_out_key(self, file) -> str:
        """Determine the valid output key by selecting the one with the highest numeric suffix."""
        out_keys = [key for key in file.keys() if key.startswith('output')]
        if not out_keys:
            raise ValueError(f"No valid output keys found in file.")
        out_num = np.array([int(key[-1]) for key in out_keys])
        return out_keys[np.argmax(out_num)]

    def find_files(self, path: Union[str, List[str]]) -> List[str]:
        """Search folder(s) for ROOT files."""
        files = []
        if isinstance(path, str):
            path = [path]
        for p in path:
            files.extend(glob(os.path.join(p, "*.root")))
        return files

class NtupleReader_digitdata(GraphNeTFileReader):
    """A class for reading ntuple ROOT files from ratpac-two. Function used for data files using integrated charge."""

    _accepted_file_extensions = [".root"]
    _accepted_extractors = [MCHitExtractorTrain_lognormal, MCHitExtractorReco_lognormal, MCHitExtractorTrain_digit, MCHitExtractorReco_digit, MCTruthExtractor, MCTruthExtractor_data]

    def __call__(self, file_path: str) -> List[Dict[str, Dict[str, Any]]]:
        outputs = []

        with uproot.open(file_path) as file:
            out_key = self.get_valid_out_key(file)
            obsdata = file[out_key].arrays(
                filter_name=[
                    'digitPMTID', 'digitTime', 'digitCharge',
                    'evid', 'subev'
                ],
                library='np'
            )
            maps = file["meta;1"].arrays(
                filter_name=['pmtX', 'pmtY', 'pmtZ', 'pmtU', 'pmtV', 'pmtW'],
                library='np'
            )

            n_events = len(obsdata['evid'])  # Number of events

            for i in range(n_events):
                if (obsdata['subev'][i]!=0): # get rid of sub events
                    continue
                event_data = {key: obsdata[key][i] for key in obsdata.keys()}
                event_outputs = {}

                for extractor in self._extractors:
                    if isinstance(extractor, (MCHitExtractorTrain_lognormal, MCHitExtractorReco_lognormal, MCHitExtractorTrain_digit, MCHitExtractorReco_digit)):
                        data = extractor(event_data, maps)
                    else:
                        data = extractor(event_data)
                    if data is not None:
                        event_outputs[extractor._extractor_name] = data

                if event_outputs:
                    outputs.append(event_outputs)

        return outputs

    def get_valid_out_key(self, file) -> str:
        """Determine the valid output key by selecting the one with the highest numeric suffix."""
        out_keys = [key for key in file.keys() if key.startswith('output')]
        if not out_keys:
            raise ValueError(f"No valid output keys found in file.")
        out_num = np.array([int(key[-1]) for key in out_keys])
        return out_keys[np.argmax(out_num)]

    def find_files(self, path: Union[str, List[str]]) -> List[str]:
        """Search folder(s) for ROOT files."""
        files = []
        if isinstance(path, str):
            path = [path]
        for p in path:
            files.extend(glob(os.path.join(p, "*.root")))
        return files

class NtupleReader_lognormaldata(GraphNeTFileReader):
    """A class for reading ntuple ROOT files from ratpac-two. Function used for data files using lognormal fitted charge."""

    _accepted_file_extensions = [".root"]
    _accepted_extractors = [MCHitExtractorTrain_lognormal, MCHitExtractorReco_lognormal, MCHitExtractorTrain_digit, MCHitExtractorReco_digit, MCTruthExtractor, MCTruthExtractor_data]

    def __call__(self, file_path: str) -> List[Dict[str, Dict[str, Any]]]:
        outputs = []

        with uproot.open(file_path) as file:
            out_key = self.get_valid_out_key(file)
            obsdata = file[out_key].arrays(
                filter_name=[
                    'fit_charge_Lognormal', 'fit_pmtid_Lognormal', 'fit_time_Lognormal',
                    'evid', 'subev'
                ],
                library='np'
            )
            maps = file["meta;1"].arrays(
                filter_name=['pmtX', 'pmtY', 'pmtZ', 'pmtU', 'pmtV', 'pmtW'],
                library='np'
            )

            n_events = len(obsdata['evid'])  # Number of events

            for i in range(n_events):
                if (obsdata['subev'][i]!=0): # get rid of sub events
                    continue
                event_data = {key: obsdata[key][i] for key in obsdata.keys()}
                event_outputs = {}

                for extractor in self._extractors:
                    if isinstance(extractor, (MCHitExtractorTrain_lognormal, MCHitExtractorReco_lognormal, MCHitExtractorTrain_digit, MCHitExtractorReco_digit)):
                        data = extractor(event_data, maps)
                    else:
                        data = extractor(event_data)
                    if data is not None:
                        event_outputs[extractor._extractor_name] = data

                if event_outputs:
                    outputs.append(event_outputs)

        return outputs

    def get_valid_out_key(self, file) -> str:
        """Determine the valid output key by selecting the one with the highest numeric suffix."""
        out_keys = [key for key in file.keys() if key.startswith('output')]
        if not out_keys:
            raise ValueError(f"No valid output keys found in file.")
        out_num = np.array([int(key[-1]) for key in out_keys])
        return out_keys[np.argmax(out_num)]

    def find_files(self, path: Union[str, List[str]]) -> List[str]:
        """Search folder(s) for ROOT files."""
        files = []
        if isinstance(path, str):
            path = [path]
        for p in path:
            files.extend(glob(os.path.join(p, "*.root")))
        return files
