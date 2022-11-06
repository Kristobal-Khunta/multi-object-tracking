import torch
import copy
import motmetrics as mm
import numpy as np
from torch.nn import functional as F
import torchvision.transforms.functional as TF
import market.metrics as metrics


def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.
    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    distmat = torch.cdist(input1, input2, p=2.0)
    return distmat**2


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


def compute_iou_reid_distance_matrix(
    track_features,
    pred_features,
    track_boxes,
    boxes,
    metric_fn,
    unmatched_cost=255.0,
    alpha=0.0,
):
    # UNMATCHED_COST = 255.0

    # Build cost matrix.
    distance = mm.distances.iou_matrix(
        ltrb_to_ltwh(track_boxes).numpy(), ltrb_to_ltwh(boxes).numpy(), max_iou=0.5
    )

    appearance_distance = metrics.compute_distance_matrix(
        track_features, pred_features, metric_fn=metric_fn
    )
    appearance_distance = appearance_distance.numpy() * 0.5
    # return appearance_distance

    assert np.alltrue(appearance_distance >= -0.1)
    assert np.alltrue(appearance_distance <= 1.1)

    combined_costs = alpha * distance + (1 - alpha) * appearance_distance

    # Set all unmatched costs to _UNMATCHED_COST.
    distance = np.where(np.isnan(distance), unmatched_cost, combined_costs)

    distance = np.where(appearance_distance > 0.1, unmatched_cost, distance)

    return distance


def get_crop_from_boxes(boxes, frame, height=256, width=128):
    """Crops all persons from a frame given the boxes.
    Args:
            boxes: The bounding boxes.
            frame: The current frame.
            height (int, optional): [description]. Defaults to 256.
            width (int, optional): [description]. Defaults to 128.
    """
    person_crops = []
    norm_mean = [0.485, 0.456, 0.406]  # imagenet mean
    norm_std = [0.229, 0.224, 0.225]  # imagenet std
    for box in boxes:
        box = box.to(torch.int32)
        res = frame[:, :, box[1] : box[3], box[0] : box[2]]
        res = F.interpolate(res, (height, width), mode="bilinear")
        res = TF.normalize(res[0, ...], norm_mean, norm_std)
        person_crops.append(res.unsqueeze(0))

    return person_crops


def compute_reid_features(model, crops):
    f_ = []
    model.eval()
    with torch.no_grad():
        for data in crops:
            img = data.cuda()
            features = model(img)
            features = features.cpu().clone()
            f_.append(features)
        f_ = torch.cat(f_, 0)
        return f_
