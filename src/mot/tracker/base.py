import collections

import motmetrics as mm
import numpy as np
import torch
import abc

mm.lap.default_solver = "lap"


class Track:
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


class Tracker(abc.ABC):
    """The main tracking file, here is where magic happens."""

    def __init__(self):
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

    def add(self, new_boxes, new_scores, new_features):
        """Initializes new Track objects and saves them."""
        num_new = len(new_boxes)
        for i in range(num_new):
            self.tracks.append(
                Track(new_boxes[i], new_scores[i], self.track_num + i, new_features[i])
            )
        self.track_num += num_new

    @abc.abstractmethod
    def data_association(self, boxes, scores, features=None):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, frame):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        raise NotImplementedError


# class MixinPredefinedDetections:
#     def step(self, frame):
#         """This function should be called every timestep to perform tracking with a blob
#         containing the image information.
#         """

#         # if predefined detections
#         boxes = frame["det"]["boxes"]
#         scores = frame["det"]["scores"]
#         self.data_association(boxes, scores)
#         self.update_results()
