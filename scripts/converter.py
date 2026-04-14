#!/usr/bin/env python3
"""
Master Data Conversion Script
Converts ntuple sets (real reco data, sim reco data, and training sim) to SQLite for GraphNet.
"""

import argparse
import sys

from graphnet.data.dataconverter import DataConverter
from graphnet.data.writers import SQLiteWriter

# Import our unified Reader and Extractors
# (Ensure these match the actual file paths where you saved the unified classes)
from graphnet.data.readers import NtupleReader
from graphnet.data.extractors.ratpac import (
    NtupleHitExtractor,
    NtupleTruthExtractor
)

def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="Run data conversion from ntuple to SQLite.")
    
    # Standard IO arguments
    parser.add_argument('--input_dir', required=True, help="Directory containing input ROOT files.")
    parser.add_argument('--output_dir', required=True, help="Directory to save SQLite output.")
    parser.add_argument('--num_workers', type=int, default=60, help="Number of workers for multiprocessing.")
    
    # Configuration arguments
    parser.add_argument('--dataset_type', type=str, choices=['reco_data', 'reco_sim', 'train'], required=True,
                        help="Specify the pipeline: 'reco_data' (blind), 'reco_sim' (keeps truth), or 'train'.")
    parser.add_argument('--charge_type', type=str, choices=['digit', 'lognormal'], required=True,
                        help="Specify the charge type: 'digit' or 'lognormal'.")

    args = parser.parse_args()

    # Define custom output keys for the hit extractor
    custom_keys = {
        'id': 'pmtID',
        'x': 'pmt_x',
        'y': 'pmt_y',
        'z': 'pmt_z',
        'ze': 'pmt_zenith',
        'az': 'pmt_azimuth',
        't': 't',
        'charge': 'charge',
    }

    # 2. Configure Dynamic Flags
    is_data = (args.dataset_type == 'reco_data')
    is_training = (args.dataset_type == 'train')

    print(f"--- Conversion Configuration ---")
    print(f"Dataset Type: {args.dataset_type.upper()}")
    print(f"Charge Type:  {args.charge_type.upper()}")
    print(f"Is Training:  {is_training} (Smears hit times)")
    print(f"Spoof Truth:  {is_data} (Zeroes out MC truth)")
    print(f"--------------------------------\n")

    # 3. Instantiate the Ntuple Classes
    reader = NtupleReader(
        charge_type=args.charge_type, 
        is_data=is_data
    )
    
    hit_extractor = NtupleHitExtractor(
        charge_type=args.charge_type, 
        is_training=is_training,
        output_keys=custom_keys
    )
    
    truth_extractor = NtupleTruthExtractor(
        is_data=is_data
    )

    # 4. Run Converter
    converter = DataConverter(
        file_reader=reader,
        save_method=SQLiteWriter(),
        extractors=[hit_extractor, truth_extractor],
        outdir=args.output_dir,
        num_workers=args.num_workers,
    )

    print(f"Starting conversion from {args.input_dir} -> {args.output_dir}")
    converter(input_dir=args.input_dir)
    print("Conversion complete!")

if __name__ == "__main__":
    main()
