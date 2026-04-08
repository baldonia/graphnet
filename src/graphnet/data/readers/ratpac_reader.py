"""Modules for reading data files from Eos."""

import os
from glob import glob
from typing import List, Union, Dict, Any

import numpy as np
import uproot

from graphnet.data.extractors.ratpac import NtupleHitExtractor, NtupleTruthExtractor
from .graphnet_file_reader import GraphNeTFileReader


class NtupleReader(GraphNeTFileReader):
    """A class for reading ntuple ROOT files from ratpac-two."""

    _accepted_file_extensions = [".root"]
    _accepted_extractors = [NtupleHitExtractor, NtupleTruthExtractor]

    def __init__(self, charge_type: str, is_data: bool):
        """
        Args:
            charge_type: 'digit' or 'lognormal'.
            is_data: True if reading real experimental data (skips MC truth columns).
        """
        super().__init__()
        self.charge_type = charge_type.lower()
        self.is_data = is_data

        # 1. Dynamically build the list of columns to read from the ROOT tree
        self.keys_to_read = ['evid', 'subev'] # Base keys needed for all files

        # Add charge-specific keys
        if self.charge_type == 'lognormal':
            self.keys_to_read.extend(['fit_charge_Lognormal', 'fit_pmtid_Lognormal', 'fit_time_Lognormal'])
        elif self.charge_type == 'digit':
            self.keys_to_read.extend(['digitPMTID', 'digitTime', 'digitCharge'])
        else:
            raise ValueError(f"Unsupported charge type: {self.charge_type}")

        # Add MC truth keys ONLY if it is a simulation
        if not self.is_data:
            self.keys_to_read.extend([
                'triggerTime', 'mcx', 'mcy', 'mcz', 
                'mcu', 'mcv', 'mcw', 'mct', 'mcke', 'mcpdg'
            ])

    def __call__(self, file_path: str) -> List[Dict[str, Dict[str, Any]]]:
        outputs = []

        with uproot.open(file_path) as file:
            out_key = self.get_valid_out_key(file)
            
            # Use our dynamically generated list to load only what we need
            obsdata = file[out_key].arrays(filter_name=self.keys_to_read, library='np')
            
            maps = file["meta;1"].arrays(
                filter_name=['pmtX', 'pmtY', 'pmtZ', 'pmtU', 'pmtV', 'pmtW'],
                library='np'
            )

            # Use 'evid' to count events
            n_events = len(obsdata['evid'])  

            for i in range(n_events):
                if obsdata['subev'][i] != 0: # get rid of sub events
                    continue
                
                event_data = {key: obsdata[key][i] for key in obsdata.keys()}
                event_outputs = {}

                for extractor in self._extractors:
                    if isinstance(extractor, NtupleHitExtractor):
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
            raise ValueError("No valid output keys found in file.")
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
