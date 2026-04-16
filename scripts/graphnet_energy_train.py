"""Example of training Model."""
# Train to predict energy.

import os
from typing import Any, Dict, List, Optional

import torch
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


from graphnet.data.constants import TRUTH
from graphnet.models import StandardModel
from graphnet.models.detector import Eos_v0
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.training.loss_functions import LogCoshLoss
from graphnet.data.datamodule import GraphNeTDataModule
from graphnet.data.dataset import SQLiteDataset
from torch_geometric.data import Data
from torch import Tensor

import pandas as pd
import sqlite3
import glob

from pytorch_lightning.loggers import WandbLogger # Use WandB to track loss, acc, etc.

# Added as part of effort to remove ProgressBar
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def main(
    path: str,
    pulsemap: str,
    target: str,
    truth_table: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
    outdir: str,
    features: List[str],
    truth: List[str],
    selections, # Holds the event numbers for the Test dataset, one group of numbers per file
    trainval_selections, # Holds the non-test event numbers for train/val, one group per file
) -> None:
    """Run example."""

    # Configuration
    config: Dict[str, Any] = {
        "path": path,
        "pulsemap": pulsemap,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "target": target,
        "early_stopping_patience": early_stopping_patience,
        "fit": {
            "gpus": gpus,
            "max_epochs": max_epochs,
        },
    }

    # Define graph representation
    graph_definition = KNNGraph(detector=Eos_v0())

    # Get DataLoaders
    data_module = GraphNeTDataModule(dataset_reference=SQLiteDataset,
                                     dataset_args = {
                                    "truth_table": truth_table,
                                    "pulsemaps": pulsemap,
                                    "truth": truth,
                                    "features": features,
                                    "path": data_path,
                                    "graph_definition": graph_definition,
                                    },
                                    train_dataloader_kwargs={"batch_size": batch_size,
                                                             "num_workers": num_workers,
                                                             "shuffle": False,
                                                             "prefetch_factor": 8,
                                                             "persistent_workers": True
                                                             },
                                    test_selection = selections,
                                    selection = trainval_selections,
                                )
    training_dataloader = data_module.train_dataloader
    validation_dataloader = data_module.val_dataloader
    test_dataloader = data_module.test_dataloader


    # Building model
    backbone = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )
    task = EnergyReconstruction(
        hidden_size=backbone.nb_outputs,
        target_labels=config["target"],
        loss_function=LogCoshLoss(),
        transform_prediction_and_target=lambda x: torch.log10(x),
        transform_inference=lambda x: torch.pow(10, x),
    )

    model = StandardModel(
        graph_definition=graph_definition,
        backbone=backbone,
        tasks=[task],
        optimizer_class=Adam,
        # Don't use learning rate scheduler:
        ###optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        # Use learning rate scheduler:
        optimizer_kwargs={"lr": 1e-04},
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={
            "patience": 3,
        },
        scheduler_config={
            "frequency": 1,
            "monitor": "val_loss",
        },
    )

    # Create the exact same callbacks that would be created by default, minus ProgressBar
    # Use callbacks=callbacks in model.fit below
    callbacks = [] # here, standard_model.py uses callbacks = [ProgressBar()]
    if validation_dataloader is not None:
        # Add Early Stopping (exactly as in default)
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=config["early_stopping_patience"],
            )
        )
        # Add Model Check Point (exactly as in default)
        callbacks.append(
            ModelCheckpoint(
                save_top_k=1,
                monitor="val_loss",
                mode="min",
                filename=f"{model.backbone.__class__.__name__}-" + 
                        "{epoch}-{val_loss:.2f}-{train_loss:.2f}",
            )
        )


    # Training model
    model.fit(
        training_dataloader,
        validation_dataloader,
        early_stopping_patience=config["early_stopping_patience"],
        callbacks=callbacks,
        logger = wandb_logger,
        enable_progress_bar=True,
        **config["fit"],
    )

    # Get predictions
    additional_attributes = ['energy', 'pid', 'vertex_x', 'vertex_y', 'vertex_z', 'zenith', 'azimuth']
    assert isinstance(additional_attributes, list)

    results = model.predict_as_dataframe(
        test_dataloader,
        additional_attributes=additional_attributes + ["event_no"],
        gpus=[0], # Just use a single GPU for inference.  N GPUs overwrite the .csv file N times,
                  # leaving 1/N of the events after finishing training and inference.
    )

    # Save predictions and model to file
    os.makedirs(outdir, exist_ok=True)

    # Save results as .csv and/or .parquet
    results.to_csv(f"{outdir}/results.csv")
    #results.to_parquet(f"{outdir}/results.parquet")

    # Save full model (including weights) to .pth file - not version safe
    # Note: Models saved as .pth files in one version of graphnet
    #       may not be compatible with a different version of graphnet.
    model.save(f"{outdir}/model.pth")

    # Save model config and state dict - Version safe save method.
    # This method of saving models is the safest way.
    model.save_state_dict(f"{outdir}/state_dict.pth")
    model.save_config(f"{outdir}/model_config.yml")    
    
    # Put this at the end in case it crashes the job...
    results.to_parquet(f"{outdir}/results.parquet")

def GetTestEventNumberLists(data_path, percent_to_keep):
    """
    Extracts the last 10% of event numbers from the "TruthData" table in each SQLite database file.

    Args:
    - data_path (list): List of file paths to SQLite database files.
    - percent_to_keep (float): Percentage of event numbers to keep from each database.

    Returns:
    - test_selections (list): List containing test event numbers for each database file.
    - trainval_selections (list): List containing non-test event numbers for each database file.
    """
    # Lists to store selections for each file
    test_selections = []
    trainval_selections = []

    # Iterate over each file path
    for db_file in data_path:
        print(f"Processing {db_file}")
        
        # Connect to the SQLite database
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Get the total number of events
        cursor.execute("SELECT COUNT(*) FROM TruthData")
        total_events = cursor.fetchone()[0]
        
        # Calculate the number of events to keep (last 10%)
        num_to_keep = int(total_events * percent_to_keep)
        
        # Get ALL event numbers, ordered descending
        cursor.execute("SELECT event_no FROM TruthData ORDER BY event_no DESC")
        all_events = [row[0] for row in cursor.fetchall()]
        
        # Close the connection to the database
        conn.close()
        
        # Split: first num_to_keep are the test events (highest event numbers),
        # remainder are for train/val
        last_events = all_events[:num_to_keep]
        remaining_events = all_events[num_to_keep:]

        print(f'  Test: {len(last_events)} events, Train/Val: {len(remaining_events)} events')

        # Append to selection lists
        test_selections.append(last_events)
        trainval_selections.append(remaining_events)

    return test_selections, trainval_selections

if __name__ == "__main__":
    # Constants
    features = ['pmt_x', 'pmt_y', 'pmt_z', 't', 'pmt_zenith', 'pmt_azimuth', 'charge'] # detector Eos_v0
    pulsemap = 'HitData'
    truth = TRUTH.EOS
    truth_table = 'TruthData'
    # data_path is defined via command line arg and glob, see below
    target = 'energy'
    gpus = [0,1,2,3] # multiple GPUs only used for training (overridden in code for inference step)
    max_epochs = 100
    early_stopping_patience = 7
    batch_size = 128
    num_workers = 8 # (Should this match --cpus-per-task= in SLURM script?)

    # Build the output directory path.
    # Add 'target', len(gpus), batch_size: energy_g{len(gpus)}_b{batch_size}
    # Add optional label using argparse

    import os
    import argparse

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='GraphNet training args')
    parser.add_argument('-l','--OutdirLabel', 
                        type=str, 
                        default='', 
                        help='Optional label for outdir name.')
    parser.add_argument("-dp","--DataPath",
                    type=str,
                    default = '',
                    help="wildcard filename for creating data_path list")

    args = parser.parse_args()
    outdir_label = args.OutdirLabel
    DataPath = args.DataPath

    data_path = glob.glob(DataPath)
    data_path.sort() # sort the data_path filenames

    if DataPath != '':
        print(f'Running with db files from DataPath: {DataPath}')
    print(f'Running with {len(data_path)} files.')

    base_path = os.path.dirname(data_path[0]) # use the full base pathname
        
    # Build the output directory path
    outdir = os.path.join(base_path, f"{target}-g{len(gpus)}-bs{batch_size}-me{max_epochs}-{outdir_label}".strip("-"))

    print(f'Output directory path: {outdir}')

    # Create the directory if it does not exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        print(f"Created output directory:\n   {outdir}")
    else:
        print(f"Output directory already exists:\n   {outdir}")

    # Create WandB Log directory
    wandblogdir = os.path.join(outdir, 'WandBLogs')
    if not os.path.exists(wandblogdir):
        os.makedirs(wandblogdir)

    # Set up WandB logger
    wandb_logger = WandbLogger(
                project="GraphNet-Energy",
                name = outdir_label,
                save_dir = wandblogdir,
                log_model=True,
            )

    # Get lists of events, one per file, of the last 10% in each file.
    # Use for Test data
    percent_to_keep = 0.10
    selections, trainval_selections = GetTestEventNumberLists(data_path, percent_to_keep)

    for i, selection in enumerate(selections):
        min_event_no = min(selection)
        max_event_no = max(selection)
        print(f"For file {i}:")
        print(f"  Minimum event_no: {min_event_no}")
        print(f"  Maximum event_no: {max_event_no}")

    main(
        path = data_path,
        pulsemap = pulsemap,
        target = target,
        truth_table = truth_table,
        gpus = gpus,
        max_epochs = max_epochs,
        early_stopping_patience = early_stopping_patience,
        batch_size = batch_size,
        num_workers = num_workers,
        outdir = outdir,
        features = features,
        truth = truth,
        selections = selections,
        trainval_selections = trainval_selections,
    )
