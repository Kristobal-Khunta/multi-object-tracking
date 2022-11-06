import motmetrics as mm
import numpy as np

from mot.tracker.base import Tracker
from mot.utils import ltrb_to_ltwh
from scipy.optimize import linear_sum_assignment as linear_assignment

mm.lap.default_solver = "lap"


class BaselineTracker(Tracker):
    """The main tracking file, here is where magic happens."""

    def __init__(self, obj_detect):
        self.obj_detect = obj_detect
        self.tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}
        self.mot_accum = None

    def step(self, frame):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """

        if self.obj_detect:
            boxes, scores = self.obj_detect.detect(frame["img"])
        else:
            boxes = frame["det"]["boxes"]
            scores = frame["det"]["scores"]

        self.data_association(boxes, scores)
        self.update_results()

    def data_association(self, boxes, scores):
        self.tracks = []
        self.add(boxes, scores)


class IoUTracker(Tracker):
    def __init__(self, obj_detect):
        self.obj_detect = obj_detect
        self.tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}
        self.mot_accum = None

    def step(self, frame):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """

        if self.obj_detect:
            boxes, scores = self.obj_detect.detect(frame["img"])
        else:
            boxes = frame["det"]["boxes"]
            scores = frame["det"]["scores"]

        self.data_association(boxes, scores)
        self.update_results()

    def data_association(self, boxes, scores):
        # self.im_index - index of current proceeded image
        # num existing tracks = self.tracks = 0 at first step
        # new bboxes form new frame = boxes
        if self.tracks:
            # track_ids = [t.id for t in self.tracks] not needed in this tracker
            track_boxes = np.stack([t.box.numpy() for t in self.tracks], axis=0)

            iou_track_boxes = ltrb_to_ltwh(track_boxes)
            iou_boxes = ltrb_to_ltwh(boxes)
            distance = mm.distances.iou_matrix(
                iou_track_boxes, iou_boxes.numpy(), max_iou=0.5
            )
            # distance.shape = [num existing tracks,num new bboxes]
            # update existing tracks
            remove_track_ids = []
            for t, dist in zip(self.tracks, distance):
                # If there are no matching boxes for this track = all nans in row
                # we remove the track from tracking
                if np.isnan(dist).all():
                    remove_track_ids.append(t.id)
                else:
                    match_id = np.nanargmin(dist)
                    t.box = boxes[match_id]
            self.tracks = [t for t in self.tracks if t.id not in remove_track_ids]

            # add new tracks
            new_boxes = []
            new_scores = []
            for i, dist in enumerate(np.transpose(distance)):
                if np.isnan(dist).all():
                    new_boxes.append(boxes[i])
                    new_scores.append(scores[i])
            self.add(new_boxes, new_scores)

        else:
            # as self.tracks is empty add all bboxes as start of new tracks"
            self.add(boxes, scores)
            # after add all bboxes self.track is not empty "


class HungarianIoUTracker(IoUTracker):
    def __init__(self, obj_detect):
        super().__init__(obj_detect=obj_detect)
        self.tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}
        self.mot_accum = None
        self._UNMATCHED_COST = 255.0

    def data_association(self, boxes, scores):
        if self.tracks:
            # track_ids = [t.id for t in self.tracks] not needed in this tracker
            track_boxes = np.stack([t.box.numpy() for t in self.tracks], axis=0)

            # Build cost matrix.
            distance = mm.distances.iou_matrix(track_boxes, boxes.numpy(), max_iou=0.5)

            # Set all unmatched costs to _UNMATCHED_COST.
            distance = np.where(np.isnan(distance), self._UNMATCHED_COST, distance)
            # Perform Hungarian matching.
            # row_idx and col_idx are indices into track_boxes and boxes.
            # row_idx[i] and col_idx[i] define a match.
            # distance[row_idx[i], col_idx[i]] define the cost for that matching.
            row_idx, col_idx = linear_assignment(distance)
            self.update_tracks(row_idx, col_idx, distance, boxes, scores)
        else:
            # No tracks exist.
            self.add(boxes, scores)

    def update_tracks(self, row_idx, col_idx, distance, boxes, scores):

        min_dist = distance[row_idx, col_idx]

        # If the costs are equal to _UNMATCHED_COST, it's not a
        # match.

        condition = min_dist < self._UNMATCHED_COST
        bad_idx = np.argwhere(~condition)
        good_idx = np.argwhere(condition)

        # internal_idx_bad_track = row_idx[bad_idx].ravel()  not needed in this tracker
        internal_idx_bad_bbox = col_idx[bad_idx].ravel()

        internal_idx_good_tracks = row_idx[good_idx].ravel()
        internal_idx_good_boxes = col_idx[good_idx].ravel()
        tracks_to_update = []

        for int_track_idx, int_box_idx in zip(
            internal_idx_good_tracks, internal_idx_good_boxes
        ):
            t = self.tracks[int_track_idx]
            t.box = boxes[int_box_idx]
            tracks_to_update.append(t)
        self.tracks = tracks_to_update

        internal_idx_unseen_boxes = set(range(len(boxes))) - set(col_idx)
        internal_idx_new_boxes = internal_idx_unseen_boxes | set(internal_idx_bad_bbox)
        new_boxes = [boxes[i] for i in internal_idx_new_boxes]
        new_scores = [scores[i] for i in internal_idx_new_boxes]

        self.add(new_boxes, new_scores)
