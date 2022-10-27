#########################################
# Still ugly file with helper functions #
#########################################

import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import motmetrics as mm
import numpy as np
import torch
from cycler import cycler as cy
from torchvision.transforms import functional as F
from tqdm.auto import tqdm
import time
import copy


def get_mot_accum(results, seq):
    mot_accum = mm.MOTAccumulator(auto_id=True)

    # for i, data in enumerate(seq):
    for i in range(len(seq)):
        # data = self.data[idx]
        gt = seq.data[i]["gt"]
        gt_ids = []
        if gt:
            gt_boxes = []
            for gt_id, box in gt.items():
                gt_ids.append(gt_id)
                gt_boxes.append(box)

            gt_boxes = np.stack(gt_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            gt_boxes = np.stack(
                (
                    gt_boxes[:, 0],
                    gt_boxes[:, 1],
                    gt_boxes[:, 2] - gt_boxes[:, 0],
                    gt_boxes[:, 3] - gt_boxes[:, 1],
                ),
                axis=1,
            )
        else:
            gt_boxes = np.array([])

        track_ids = []
        track_boxes = []
        for track_id, frames in results.items():
            if i in frames:
                track_ids.append(track_id)
                # frames = x1, y1, x2, y2, score
                track_boxes.append(frames[i][:4])

        if track_ids:
            track_boxes = np.stack(track_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            track_boxes = np.stack(
                (
                    track_boxes[:, 0],
                    track_boxes[:, 1],
                    track_boxes[:, 2] - track_boxes[:, 0],
                    track_boxes[:, 3] - track_boxes[:, 1],
                ),
                axis=1,
            )
        else:
            track_boxes = np.array([])

        distance = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)

        mot_accum.update(gt_ids, track_ids, distance)

    return mot_accum


def evaluate_mot_accums(accums, names, generate_overall=False):
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accums,
        metrics=mm.metrics.motchallenge_metrics,
        names=names,
        generate_overall=generate_overall,
    )

    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )
    print(str_summary)
    return summary


def evaluate_obj_detect(model, data_loader):
    model.eval()
    device = list(model.parameters())[0].device
    results = {}
    for imgs, targets in tqdm(data_loader):
        imgs = [img.to(device) for img in imgs]

        with torch.no_grad():
            preds = model(imgs)

        for pred, target in zip(preds, targets):
            results[target["image_id"].item()] = {
                "boxes": pred["boxes"].cpu(),
                "scores": pred["scores"].cpu(),
            }

    data_loader.dataset.print_eval(results)


def obj_detect_transforms(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


#####
def run_tracker(val_sequences, db, tracker, output_dir=None):
    time_total = 0
    mot_accums = []
    results_seq = {}
    for seq in val_sequences:
        # break
        tracker.reset()
        now = time.time()

        print(f"Tracking: {seq}")

        # data_loader = DataLoader(seq, batch_size=1, shuffle=False)
        with torch.no_grad():
            # for i, frame in enumerate(tqdm(data_loader)):
            for frame in db[str(seq)]:
                tracker.step(frame)

        results = tracker.get_results()
        results_seq[str(seq)] = results

        if seq.no_gt:
            print("No GT evaluation data available.")
        else:
            mot_accums.append(get_mot_accum(results, seq))

        time_total += time.time() - now

        print(f"Tracks found: {len(results)}")
        print(f"Runtime for {seq}: {time.time() - now:.1f} s.")

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            seq.write_results(results, os.path.join(output_dir))

    print(f"Runtime for all sequences: {time_total:.1f} s.")
    if mot_accums:
        return evaluate_mot_accums(
            mot_accums,
            [str(s) for s in val_sequences if not s.no_gt],
            generate_overall=True,
        )


def cosine_distance(input1, input2):
    """Computes cosine distance.
    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = torch.nn.functional.normalize(input1, p=2, dim=1)
    input2_normed = torch.nn.functional.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat


def ltrb_to_ltwh(ltrb_boxes):
    ltwh_boxes = copy.deepcopy(ltrb_boxes)
    ltwh_boxes[:, 2] = ltrb_boxes[:, 2] - ltrb_boxes[:, 0]
    ltwh_boxes[:, 3] = ltrb_boxes[:, 3] - ltrb_boxes[:, 1]

    return ltwh_boxes


def ltrb_to_xcycwh(ltrb_boxes):
    xcycwh = copy.deepcopy(ltrb_boxes)
    xcycwh[:, 0] = (ltrb_boxes[:, 2] + ltrb_boxes[:, 0]) / 2  # x_ceter = (rx+lx)/2
    xcycwh[:, 1] = (ltrb_boxes[:, 3] + ltrb_boxes[:, 1]) / 2  #
    xcycwh[:, 2] = ltrb_boxes[:, 2] - ltrb_boxes[:, 0]
    xcycwh[:, 3] = ltrb_boxes[:, 3] - ltrb_boxes[:, 1]
    return xcycwh

