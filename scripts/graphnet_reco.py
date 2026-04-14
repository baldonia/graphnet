#!/usr/bin/env python3
"""
GraphNet Master Inference Script
Handles both simulated events (extracts truth) and real data (blind inference).
"""

import os
import glob
import sqlite3
import argparse
from typing import List

# GraphNet core imports
from graphnet.models import Model
from graphnet.utilities.config import ModelConfig
from graphnet.data.constants import TRUTH
from graphnet.data.datamodule import GraphNeTDataModule
from graphnet.data.dataset import SQLiteDataset
from graphnet.models.detector import Eos_v0
from graphnet.models.graphs import KNNGraph


def get_event_numbers(data_paths: List[str], fraction: float) -> List[List[int]]:
    """
    Extracts a specific fraction of event numbers from the TruthData table.
    """
    selections = []
    fraction = max(0.0, min(1.0, fraction))
    
    for db_file in data_paths:
        print(f"Fetching event IDs from {db_file}...")
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        if fraction == 1.0:
            cursor.execute("SELECT event_no FROM TruthData ORDER BY event_no ASC")
        else:
            cursor.execute("SELECT COUNT(*) FROM TruthData")
            total_events = cursor.fetchone()[0]
            num_to_keep = int(total_events * fraction)
            
            cursor.execute(f"SELECT event_no FROM TruthData ORDER BY event_no ASC LIMIT {num_to_keep}")
            
        events = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if events:
            print(f"  -> Selected {len(events)} events (Min: {min(events)}, Max: {max(events)})")
        selections.append(events)
        
    return selections


def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description='GraphNet Master Inference')
    parser.add_argument('-dp', '--DataPath', type=str, required=True, 
                        help="Wildcard filename for input databases (e.g., 'data/*.db')")
    parser.add_argument('-m', '--ModelPath', type=str, required=True, 
                        help="Path to directory containing model_config.yml and state_dict.pth")
    parser.add_argument('-t', '--DataType', type=str, choices=['sim', 'data'], required=True,
                        help="Specify 'sim' to extract truth labels, or 'data' for blind inference.")
    parser.add_argument('-od', '--OutDir', type=str, default="inference_output", 
                        help="Directory to save the results")
    parser.add_argument('-b', '--BatchSize', type=int, default=128)
    parser.add_argument('-nw', '--NumWorkers', type=int, default=4)
    parser.add_argument('-g', '--GpuIndex', type=int, default=0,
                        help="GPU index to use (default: 0). Use -1 for CPU.")
    parser.add_argument('-f', '--Fraction', type=float, default=1.0, 
                        help="Fraction of events to reconstruct (0.0 to 1.0). Default is 1.0.")
    
    args = parser.parse_args()

    # 2. Resolve paths and inputs
    data_paths = sorted(glob.glob(args.DataPath))
    if not data_paths:
        raise ValueError(f"No files found matching DataPath: {args.DataPath}")
    
    print(f"Running {args.DataType.upper()} inference on {len(data_paths)} database files.")
    os.makedirs(args.OutDir, exist_ok=True)

    # 3. Configure Detector Features
    features = ['pmt_x', 'pmt_y', 'pmt_z', 't', 'pmt_zenith', 'pmt_azimuth', 'charge']
    pulsemap = 'HitData'
    graph_definition = KNNGraph(detector=Eos_v0())

    # 4. Set up the GraphNeTDataModule
    selections = get_event_numbers(data_paths, args.Fraction)
    
    data_module = GraphNeTDataModule(
        dataset_reference=SQLiteDataset,
        dataset_args={
            "truth_table": 'TruthData',
            "pulsemaps": pulsemap,
            "truth": TRUTH.EOS,
            "features": features,
            "path": data_paths,
            "graph_definition": graph_definition,
        },
        test_dataloader_kwargs={
            "batch_size": args.BatchSize,
            "num_workers": args.NumWorkers,
            "shuffle": False,
            "prefetch_factor": 2,
            "persistent_workers": True if args.NumWorkers > 0 else False
        },
        test_selection=selections,
    )
    
    test_dataloader = data_module.test_dataloader

    # 5. Load the Model
    config_file = os.path.join(args.ModelPath, "model_config.yml")
    state_file = os.path.join(args.ModelPath, "state_dict.pth")
    
    print(f"Loading architecture from: {config_file}")
    model_config = ModelConfig.load(config_file)
    model = Model.from_config(model_config, trust=True)
    
    print(f"Loading weights from: {state_file}")
    model.load_state_dict(state_file)

    # 6. Run Inference & Define Output Columns
    if args.DataType == 'sim':
        additional_attributes = ['energy', 'pid', 'vertex_x', 'vertex_y', 'vertex_z', 'zenith', 'azimuth', 'event_no']
    else:
        additional_attributes = ['event_no']
        
    print("Starting predictions...")
    gpus = [args.GpuIndex] if args.GpuIndex >= 0 else None
    
    results = model.predict_as_dataframe(
        test_dataloader,
        additional_attributes=additional_attributes,
        gpus=gpus,
    )

    # 7. Save Output
    csv_out = os.path.join(args.OutDir, "results.csv")
    parquet_out = os.path.join(args.OutDir, "results.parquet")
    
    results.to_csv(csv_out, index=False)
    results.to_parquet(parquet_out, index=False)
    
    print(f"Inference complete! Saved {len(results)} rows to {args.OutDir}")


if __name__ == "__main__":
    main()
