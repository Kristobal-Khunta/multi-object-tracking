import argparse
import os.path as osp
from pathlib import Path

import motmetrics as mm
import numpy as np
import torch
from torch.utils.data import DataLoader

from mot.data.data_gnn import LongTrackTrainingDataset
from mot.data.data_track import MOT16Sequences
from mot.eval import run_tracker
from mot.models.gnn import SimilarityNet
from mot.tracker.advanced import MPNTracker
from mot.trainer import train_one_epoch

mm.lap.default_solver = "lap"


def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return None


def setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model_type", type=str, default="softmax", choices=["softmax", "triplet"]
    )

    parser.add_argument(
        "--metric_fn",
        type=str,
        default="cosine",
        choices=["cosine", "euclidian"],
        help="metric_fn",
    )
    parser.add_argument("--max_epoch", type=int, default=30)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    set_all_seeds(12347)
    parser = setup_parser()
    args = parser.parse_args()

    root_dir = Path(__file__).parent.parent
    root_dir = str(root_dir)

    train_db = torch.load(
        osp.join(root_dir, "data/preprocessed_data/preprocessed_data_train_2.pth")
    )
    MAX_PATIENCE = 20
    # MAX_EPOCHS = 15
    # EVAL_FREQ = 1

    device = torch.device("cpu")

    # Define our model, and init
    similarity_net = SimilarityNet(
        reid_network=None,  # Not needed since we work with precomputed features
        node_dim=32,
        edge_dim=64,
        reid_dim=512,
        edges_in_dim=6,
        num_steps=10,
    ).to(device)

    # We only keep two sequences for validation. You can
    dataset = LongTrackTrainingDataset(
        dataset="MOT16-train_wo_val2",
        db=train_db,
        root_dir=osp.join(root_dir, "data/MOT16"),
        max_past_frames=MAX_PATIENCE,
        vis_threshold=0.25,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=lambda x: x,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    optimizer = torch.optim.Adam(similarity_net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)
    tracker = None

    best_idf1 = 0.0
    for epoch in range(1, args.max_epoch + 1):
        print(f"-------- EPOCH {epoch:2d} --------")
        train_one_epoch(
            model=similarity_net,
            data_loader=data_loader,
            optimizer=optimizer,
            print_freq=args.print_freq,
        )
        scheduler.step()

        if epoch % args.eval_freq == 0:
            tracker = MPNTracker(
                obj_detect=None,
                reid_model=None,
                similarity_net=similarity_net.eval(),
                patience=MAX_PATIENCE,
            )
            val_sequences = MOT16Sequences(
                "MOT16-val2", osp.join(root_dir, "data/MOT16"), vis_threshold=0.0
            )
            res = run_tracker(
                val_sequences, db=train_db, tracker=tracker, output_dir=None
            )
            idf1 = res.loc["OVERALL"]["idf1"]
            if idf1 > best_idf1:
                best_idf1 = idf1
                torch.save(
                    similarity_net.state_dict(),
                    osp.join(root_dir, "models", "best_ckpt.pth"),
                )


if __name__ == "__main__":
    main()
