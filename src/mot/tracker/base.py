import collections

import motmetrics as mm
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import market.metrics as metrics
import abc
mm.lap.default_solver = "lap"


class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(self, box, score, track_id, feature=None, inactive=0):
        self.id = track_id
        self.box = box
        self.score = score
        self.feature = collections.deque([feature])
        self.inactive = inactive
        self.max_features_num = 10

    def add_feature(self, feature):
        """Adds new appearance features to the object."""
        self.feature.append(feature)
        if len(self.feature) > self.max_features_num:
            self.feature.popleft()

    def get_feature(self):
        if len(self.feature) > 1:
            feature = torch.stack(list(self.feature), dim=0)
        else:
            feature = self.feature[0].unsqueeze(0)
        # return feature.mean(0, keepdim=False)
        return feature[-1]

    def __repr__(self):
        return f"track_id = {self.id} score = {self.score:.2f} bbox = {self.box}"


class BaseTracker(abc.ABC):
    """The main tracking file, here is where magic happens."""

    def __init__(self, obj_detect):
        self.obj_detect = obj_detect
        self.tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}
        self.mot_accum = None

    def reset(self, hard=True):
        self.tracks = []

        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def get_pos(self):
        """Get the positions of all active tracks."""
        if len(self.tracks) == 1:
            box = self.tracks[0].box
        elif len(self.tracks) > 1:
            box = torch.stack([t.box for t in self.tracks], 0)
        else:
            box = torch.zeros(0).cuda()
        return box

    def update_results(self):
        # results
        for t in self.tracks:
            if t.id not in self.results:
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate(
                [t.box.cpu().numpy(), np.array([t.score])]
            )

        self.im_index += 1

    def get_results(self):
        return self.results

    @abc.abstractmethod
    def data_association(self):
        raise NotImplementedError

    @abc.abstractmethod
    def add(self):
        """Initializes new Track objects and saves them."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, frame):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        raise NotImplementedError


class Tracker(BaseTracker):
    """The main tracking file, here is where magic happens."""

    def add(self, new_boxes, new_scores,):
        """Initializes new Track objects and saves them."""
        num_new = len(new_boxes)
        for i in range(num_new):
            self.tracks.append(Track(new_boxes[i], new_scores[i], self.track_num + i))
        self.track_num += num_new

    def data_association(self, boxes, scores):
        self.tracks = []
        self.add(boxes, scores)

    def step(self, frame):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """

        # object detection
        boxes, scores = self.obj_detect.detect(frame["img"])
        # if predefined detections
        # boxes, scores = frame["det"]["boxes"], frame["det"]["scores"]
        self.data_association(boxes, scores)
        self.update_results()


############
class BaseReIDTracker(BaseTracker):
    def __init__(self, *args, **kwargs):
        self._UNMATCHED_COST = 255.0
        super().__init__(*args, **kwargs)

    def data_association(self, boxes, scores, frame):
        raise NotImplementedError

    def add(self, new_boxes, new_scores, new_features):
        """Initializes new Track objects and saves them."""
        num_new = len(new_boxes)
        for i in range(num_new):
            self.tracks.append(
                Track(new_boxes[i], new_scores[i], self.track_num + i, new_features[i])
            )
        self.track_num += num_new

    def step(self, frame):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        # object detection
        boxes, scores = self.obj_detect.detect(frame["img"])

        self.data_association(boxes, scores, frame["img"])
        # results
        self.update_results()

    @staticmethod
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

    @staticmethod
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

    def compute_distance_matrix(
        self, track_features, pred_features, track_boxes, boxes, metric_fn, alpha=0.0
    ):

        # Build cost matrix.
        distance = mm.distances.iou_matrix(
            track_boxes.numpy(), boxes.numpy(), max_iou=0.5
        )

        appearance_distance = metrics.compute_distance_matrix(
            track_features, pred_features, metric_fn=metric_fn
        )
        appearance_distance = appearance_distance.numpy() * 0.5
        # return appearance_distance

        if not np.alltrue(appearance_distance >= -0.1):
            raise AssertionError
        if not np.alltrue(appearance_distance <= 1.1):
            raise AssertionError

        combined_costs = alpha * distance + (1 - alpha) * appearance_distance

        # Set all unmatched costs to _UNMATCHED_COST.
        distance = np.where(np.isnan(distance), self._UNMATCHED_COST, combined_costs)
        return distance
