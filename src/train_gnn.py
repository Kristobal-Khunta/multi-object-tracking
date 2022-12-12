import argparse
import os.path as osp
from pathlib import Path

import motmetrics as mm
import torch
from torch.utils.data import DataLoader

from mot.data.data_gnn import LongTrackTrainingDataset
from mot.data.data_track import MOT16Sequences
from mot.eval import run_tracker
from mot.models.gnn import SimilarityNet
from mot.tracker.advanced import MPNTracker
from mot.trainer import train_one_epoch
from mot.utils import set_all_seeds

mm.lap.default_solver = "lap"


def parse_args():
    """Set up Python's ArgumentParser with tracker settings and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max_epoch", type=int, default=30)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--max_patient", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--help", "-h", action="help")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    set_all_seeds(12347)

    print("parse args1845")
    root_dir = Path(__file__).parent.parent
    root_dir = str(root_dir)
    train_db = torch.load(
        osp.join(root_dir, "data/preprocessed_data/preprocessed_data_train_2.pth")
    )
    # Define our model
    similarity_net = SimilarityNet(
        node_dim=32,
        edge_dim=64,
        reid_dim=512,
        edges_in_dim=6,
        num_steps=10,
    )
    similarity_net = similarity_net.to(args.device)
    # We only keep two sequences for validation. You can
    dataset = LongTrackTrainingDataset(
        dataset="MOT16-train_wo_val2",
        db=train_db,
        root_dir=osp.join(root_dir, "data/MOT16"),
        max_past_frames=args.max_patient,
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
            device=args.device,
            optimizer=optimizer,
            print_freq=args.print_freq,
        )
        scheduler.step()

        if epoch % args.eval_freq == 0:
            print("EVAL LOOP")
            tracker = MPNTracker(
                obj_detect=None,
                reid_model=None,
                similarity_net=similarity_net.eval(),
                patience=args.max_patient,
            )
            val_sequences = MOT16Sequences(
                "MOT16-val2", osp.join(root_dir, "data/MOT16"), vis_threshold=0.0
            )
            res, _ = run_tracker(
                val_sequences, tracker=tracker, db=train_db, output_dir=None
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
