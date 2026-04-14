import os
import argparse
import torch
from graphnet.models import StandardModel
from graphnet.models.gnn import DynEdge
from graphnet.models.task.classification import BinaryClassificationTask
from graphnet.training.loss_functions import BinaryCrossEntropyLoss
from graphnet.models.graphs import KNNGraph
from graphnet.models.detector import Eos_v0

def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description="Extract and save GraphNet model components from a checkpoint.")
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True, 
        help="Path to the .ckpt file"
    )
    parser.add_argument(
        "--outdir", 
        type=str, 
        required=True, 
        help="Directory where the extracted files will be saved"
    )

    args = parser.parse_args()

    # 2. Re-create the architecture components
    graph_definition = KNNGraph(detector=Eos_v0())

    backbone = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )

    task = BinaryClassificationTask(
        hidden_size=backbone.nb_outputs,
        target_labels="binary_label",
        prediction_labels=['positron_pred'],
        loss_function=BinaryCrossEntropyLoss(),
    )

    # 3. Load the model from the checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    model = StandardModel.load_from_checkpoint(
        args.checkpoint,
        graph_definition=graph_definition,
        backbone=backbone,
        tasks=[task]
    )

    # 4. Create outdir if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)

    # 5. Generate the version-safe files
    model.save(os.path.join(args.outdir, "model.pth"))
    model.save_state_dict(os.path.join(args.outdir, "state_dict.pth"))
    model.save_config(os.path.join(args.outdir, "model_config.yml"))

    print(f"Extraction complete. Files saved in: {args.outdir}")

if __name__ == "__main__":
    main()
