import torch
import motmetrics as mm
import numpy as np
from torch.nn import functional as F
import torchvision.transforms.functional as TF
import market.metrics as metrics
from mot.utils import ltrb_to_ltwh
from typing import Callable


def compute_iou_reid_distance_matrix(
    track_features: torch.Tensor,
    pred_features: torch.Tensor,
    track_boxes: torch.Tensor,
    boxes: torch.Tensor,
    metric_fn: Callable,
    unmatched_cost: float = 255.0,
    alpha: float = 0.0,
) -> np.array:
    """
    calculate distance between existing tracks and new bboxes
    """

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


def get_crop_from_boxes(
    boxes: torch.Tensor, frame: torch.Tensor, height: int = 256, width: int = 128
) -> list[torch.Tensor]:
    """Crops all persons from a frame given the boxes.
    Args:
        boxes: The bounding boxes.
        frame: The current frame.
        height (int, optional): [description]. Defaults to 256.
        width (int, optional): [description]. Defaults to 128.
    Returns:
        list with small images inside bbox and resized to (height, width) shape

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


def compute_reid_features(
    model: torch.nn.Module, crops: list[torch.Tensor]
) -> torch.Tensor:
    """
    compute reid feature for each pedestrian crop and stack in umion tensor
    Args:
        model: reidentification model
        crops: list with pedestrian crop resized to size, that model expects
    Returns:
        f_: reid feature per each crop pedestrian
    """
    f_ = []
    device = list(model.parameters())[0].device
    model.eval()
    with torch.no_grad():
        for data in crops:
            img = data.to(device)
            features = model(img)
            features = features.cpu().clone()
            f_.append(features)
        f_ = torch.cat(f_, 0)
    return f_
