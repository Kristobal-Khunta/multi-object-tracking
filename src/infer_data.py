import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from market.models import build_model
from mot.data.data_track import MOT16Sequences
from mot.models.object_detector import FRCNN_FPN
from mot.models.reid import compute_reid_features, get_crop_from_boxes


def setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--help", "-h", action="help")
    return parser


def prepare_obj_detect_model(weight_path, obj_detect_nms_thresh=0.3):
    obj_detect = FRCNN_FPN(num_classes=2, nms_thresh=obj_detect_nms_thresh)
    obj_detect_state_dict = torch.load(
        weight_path, map_location=lambda storage, loc: storage
    )
    obj_detect.load_state_dict(obj_detect_state_dict)
    return obj_detect


def prepare_reid_model(weight_path):
    #### Load reid model ####
    reid_model = build_model("resnet34", 751, loss="softmax", pretrained=False)
    reid_ckpt = torch.load(weight_path, map_location=lambda storage, loc: storage)
    reid_model.load_state_dict(reid_ckpt)
    reid_model = reid_model
    return reid_model


def main():
    parser = setup_parser()
    args = parser.parse_args()
    device = args.device
    root_dir = Path(__file__).parent.parent
    root_dir = str(root_dir)
    stored_data_filename = "preprocessed_data/preprocessed_data_{}.pth"

    #### model paths ####
    reid_model_file = os.path.join(root_dir, "models/resnet50_reid.pth")
    obj_detect_model_file = os.path.join(root_dir, "models/faster_rcnn_fpn.model")

    reid_model = prepare_reid_model(reid_model_file)
    reid_model = reid_model.eval().to(device)

    obj_detect = prepare_obj_detect_model(obj_detect_model_file)
    obj_detect = obj_detect.eval().to(device)
    #### load object detector ####

    for train_test in ("train", "test"):
        db = {}
        sequences = MOT16Sequences(
            f"MOT16-{train_test}",
            os.path.join(root_dir, "data/MOT16"),
            vis_threshold=0.0,
        )

        for seq in sequences:
            print(f"Processing sequence {seq}")
            data_loader = DataLoader(seq, batch_size=1, shuffle=False)
            db[str(seq)] = []
            with torch.no_grad():
                for frame_num, frame in tqdm(enumerate(data_loader)):

                    # img = frame["img"]
                    img_path = frame["img_path"][0]
                    assert frame_num + 1 == int(img_path.split("/")[-1].split(".")[0])

                    # Store detected boxes
                    # boxes, scores = obj_detect.detect(frame['img'])
                    boxes, scores = obj_detect.detect(frame["img"])
                    crops = get_crop_from_boxes(boxes, frame["img"])
                    det_reid = compute_reid_features(reid_model, crops)

                    # Store ReID embeddings

                    det = {
                        "boxes": boxes.cpu(),
                        "scores": scores.cpu(),
                        "reid": det_reid.cpu(),
                    }

                    # Store ground truth box info
                    if "gt" in frame and len(frame["gt"]) > 0:
                        gt_ids = torch.as_tensor(list(frame["gt"].keys()))
                        gt_boxes = []
                        gt_vis = []
                        for id_ in gt_ids:
                            bbox = frame["gt"][id_.item()]
                            vis = frame["vis"][id_.item()]
                            if bbox.min() < 0:
                                continue
                            gt_boxes.append(bbox)
                            gt_vis.append(vis)

                        gt_boxes = torch.cat(gt_boxes, dim=0)
                        gt_vis = torch.cat(gt_vis, dim=0)

                        gt_crops = get_crop_from_boxes(gt_boxes, frame["img"])
                        gt_reid = compute_reid_features(reid_model, gt_crops)

                        gt = {
                            "ids": gt_ids.cpu(),
                            "boxes": gt_boxes.cpu(),
                            "vis": gt_vis.cpu(),
                            "reid": gt_reid.cpu(),
                        }
                    else:
                        gt = None

                    db[str(seq)].append({"det": det, "gt": gt})
            assert len(db[str(seq)]) == len(data_loader)

        torch.save(
            db, os.path.join(root_dir, "data", stored_data_filename.format(train_test))
        )


if __name__ == "__main__":
    main()
