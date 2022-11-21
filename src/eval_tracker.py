import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm.autonotebook import tqdm
import torch
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment as linear_assignment
import os.path as osp
import motmetrics as mm
mm.lap.default_solver = 'lap'


import argparse
import os.path as osp
from pathlib import Path

import motmetrics as mm
import numpy as np
import torch
from torch.utils.data import DataLoader


from mot.eval import run_tracker
from mot.models.gnn import SimilarityNet
from mot.tracker.advanced import MPNTracker
from mot.trainer import train_one_epoch

mm.lap.default_solver = "lap"
from mot.data.data_track import MOT16Sequences
from mot.data.data_obj_detect import MOT16ObjDetect
from mot.models.object_detector import FRCNN_FPN
from mot.tracker.base import Tracker
from mot.visualize import plot_sequence
from mot.transforms import obj_detect_transforms
from mot.eval import evaluate_mot_accums, get_mot_accum, evaluate_obj_detect
from mot.models.gnn import BipartiteNeuralMessagePassingLayer, SimilarityNet
from mot.tracker.advanced import MPNTracker
from market.models import build_model


def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return None


def setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max_epoch", type=int, default=30)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--max_patient", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--help", "-h", action="help")
    return parser


def main(args):
    set_all_seeds(12347)
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    obj_detect_nms_thresh = 0.3
    print("parse args")
    root_dir = Path(__file__).parent.parent
    root_dir = str(root_dir)
    #### model paths ####
    reid_model_file = os.path.join(root_dir, "models/resnet50_reid.pth")
    obj_detect_model_file = os.path.join(root_dir, "models/faster_rcnn_fpn.model")

    #### Load reid model ####
    reid_model = build_model("resnet34", 751, loss="softmax", pretrained=False)
    reid_ckpt = torch.load(reid_model_file, map_location=lambda storage, loc: storage)
    reid_model.load_state_dict(reid_ckpt)
    reid_model = reid_model.to(device)

    #### load object detector ####
    obj_detect = FRCNN_FPN(num_classes=2, nms_thresh=obj_detect_nms_thresh)
    obj_detect_state_dict = torch.load(
        obj_detect_model_file, map_location=lambda storage, loc: storage
    )
    obj_detect.load_state_dict(obj_detect_state_dict)
    obj_detect.eval()
    obj_detect = obj_detect.to(device)

    #### load similarity net
    similarity_net = SimilarityNet(
        node_dim=32,
        edge_dim=64,
        reid_dim=512,
        edges_in_dim=6,
        num_steps=10,
    )

    best_ckpt = torch.load(
        osp.join(root_dir, "models", "best_ckpt.pth"),
        map_location=lambda storage, loc: storage,
    )
    similarity_net.load_state_dict(best_ckpt)
    similarity_net = similarity_net.to(device)

    # tracker = ReIDHungarianTracker(obj_detect)
    # tracker = ReIDHungarianTracker2(obj_detect)

    tracker = MPNTracker(
        similarity_net=similarity_net.eval(),
        reid_model=reid_model.eval(),
        obj_detect=obj_detect.eval()
    )


    val_sequences = MOT16Sequences(
        "MOT16-val2", osp.join(root_dir, "data/MOT16"), vis_threshold=0.0
    )

    print("seqs", [str(s) for s in val_sequences if not s.no_gt])

    result_mot, results_seq = run_tracker(
                val_sequences, tracker=tracker, db=None, output_dir=None
            )
    #plot_sequence(
    #    results_seq["MOT16-02"],
    #    [s for s in val_sequences if str(s) == "MOT16-02"][0],
    #    first_n_frames=3,
    #)
    print(result_mot)

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
