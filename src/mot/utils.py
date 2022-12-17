import torch
import copy
import numpy as np
from typing import Union


def euclidean_squared_distance(
    input1: torch.Tensor, input2: torch.Tensor
) -> torch.Tensor:
    """Computes euclidean squared distance.
    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    distmat = torch.cdist(input1, input2, p=2.0)
    return distmat**2


def cosine_distance(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
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


def ltrb_to_ltwh(
    ltrb_boxes: Union[np.array, torch.Tensor]
) -> Union[np.array, torch.Tensor]:
    """
    convert boxes from left-top right-bottom format to  [x_left_top,y_left_top,width,height]
    Args:
        ltrb_boxes: torch.Tensor or np.array with shape (N,4)
    Return:
        torch.Tensor or np.array with shape (N,4): boxes 
    """
    ltwh_boxes = copy.deepcopy(ltrb_boxes)
    ltwh_boxes[:, 2] = ltrb_boxes[:, 2] - ltrb_boxes[:, 0]
    ltwh_boxes[:, 3] = ltrb_boxes[:, 3] - ltrb_boxes[:, 1]
    return ltwh_boxes


def ltrb_to_xcycwh(
    ltrb_boxes: Union[np.array, torch.Tensor]
) -> Union[np.array, torch.Tensor]:
    """
    convert boxes from left-top right-bottom format to  [x_center, y_center, width, height]
    Args:
        ltrb_boxes: torch.Tensor or np.array with shape (N,4)
    Return:
        torch.Tensor or np.array with shape (N,4): boxes in xcycwh format
    """
    xcycwh = copy.deepcopy(ltrb_boxes)
    xcycwh[:, 0] = (ltrb_boxes[:, 2] + ltrb_boxes[:, 0]) / 2  # x_ceter = (rx+lx)/2
    xcycwh[:, 1] = (ltrb_boxes[:, 3] + ltrb_boxes[:, 1]) / 2  #
    xcycwh[:, 2] = ltrb_boxes[:, 2] - ltrb_boxes[:, 0]
    xcycwh[:, 3] = ltrb_boxes[:, 3] - ltrb_boxes[:, 1]
    return xcycwh


def set_all_seeds(seed: int) -> None:
    """
    set all seeds
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return None
