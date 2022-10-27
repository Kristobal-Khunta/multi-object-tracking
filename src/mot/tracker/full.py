import collections

import motmetrics as mm
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import market.metrics as metrics


from .base import Tracker, Track


class ReIDTracker(Tracker):
    def add(self, new_boxes, new_scores, new_features):
        """Initializes new Track objects and saves them."""
        num_new = len(new_boxes)
        for i in range(num_new):
            self.tracks.append(
                Track(new_boxes[i], new_scores[i], self.track_num + i, new_features[i])
            )
        self.track_num += num_new

    def reset(self, hard=True):
        self.tracks = []
        # self.inactive_tracks = []

        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def data_association(self, boxes, scores, frame):
        self.tracks = []
        self.add(boxes, scores)

    def step(self, frame):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        # object detection
        boxes, scores = self.obj_detect.detect(frame["img"])

        self.data_association(boxes, scores, frame["img"])

        # results
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate(
                [t.box.cpu().numpy(), np.array([t.score])]
            )

        self.im_index += 1

    def get_crop_from_boxes(self, boxes, frame, height=256, width=128):
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

    def compute_reid_features(self, model, crops):
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

    def compute_distance_matrix(
        self, track_features, pred_features, track_boxes, boxes, metric_fn, alpha=0.0
    ):
        UNMATCHED_COST = 255.0

        # Build cost matrix.
        distance = mm.distances.iou_matrix(
            track_boxes.numpy(), boxes.numpy(), max_iou=0.5
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
        distance = np.where(np.isnan(distance), UNMATCHED_COST, combined_costs)
        return distance
