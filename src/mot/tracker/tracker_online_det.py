import motmetrics as mm
import numpy as np
import torch

from mot.tracker.base import Tracker, ReIDTrackerBase
from mot.utils import ltrb_to_ltwh, cosine_distance
from scipy.optimize import linear_sum_assignment as linear_assignment


# tracker
class IoUTracker(Tracker):
    def data_association(self, boxes, scores):
        # self.im_index - index of current proceeded image
        # num existing tracks = self.tracks = 0 at first step
        # new bboxes form new frame = boxes
        if self.tracks:
            track_ids = [t.id for t in self.tracks]
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


class HungarianIoUTracker(Tracker):
    def __init__(self, *args, **kwargs):
        self._UNMATCHED_COST = 255.0
        super().__init__(*args, **kwargs)

    def data_association(self, boxes, scores):
        if self.tracks:
            track_ids = [t.id for t in self.tracks]
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
            min_dist = distance[row_idx, col_idx]

            # If the costs are equal to _UNMATCHED_COST, it's not a
            # match.

            condition = min_dist < self._UNMATCHED_COST
            bad_idx = np.argwhere(~condition)
            good_idx = np.argwhere(condition)

            internal_idx_bad_track = row_idx[bad_idx].ravel()
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
            internal_idx_new_boxes = internal_idx_unseen_boxes | set(
                internal_idx_bad_bbox
            )
            new_boxes = [boxes[i] for i in internal_idx_new_boxes]
            new_scores = [scores[i] for i in internal_idx_new_boxes]

            self.add(new_boxes, new_scores)
        else:
            # No tracks exist.
            self.add(boxes, scores)


class ReIDHungarianIoUTracker(ReIDTrackerBase):
    def __init__(self, reid_model, *args, **kwargs):
        self.reid_model = reid_model
        self._UNMATCHED_COST = 255.0
        super().__init__(*args, **kwargs)

    def data_association(self, boxes, scores, frame):
        crops = self.get_crop_from_boxes(boxes, frame)
        pred_features = self.compute_reid_features(self.reid_model, crops).cpu().clone()

        if self.tracks:
            track_ids = [t.id for t in self.tracks]
            track_boxes = torch.stack([t.box for t in self.tracks], axis=0)
            track_features = torch.stack([t.get_feature() for t in self.tracks], axis=0)

            # This will use your similarity measure. Please use cosine_distance!
            distance = self.compute_distance_matrix(
                track_features,
                pred_features,
                track_boxes,
                boxes,
                metric_fn=cosine_distance,
            )

            # Perform Hungarian matching.
            row_idx, col_idx = linear_assignment(distance)

            # row_idx and col_idx are indices into track_boxes and boxes.
            # row_idx[i] and col_idx[i] define a match.
            # distance[row_idx[i], col_idx[i]] define the cost for that matching.

            remove_track_ids = []
            seen_track_ids = []
            seen_box_idx = []
            # Update existing tracks and remove unmatched tracks.

            for track_idx, box_idx in zip(row_idx, col_idx):
                costs = distance[track_idx, box_idx]
                internal_track_id = track_ids[track_idx]
                seen_track_ids.append(internal_track_id)
                # If the costs are equal to _UNMATCHED_COST, it's not a match.
                if costs == self._UNMATCHED_COST:
                    remove_track_ids.append(internal_track_id)
                else:
                    self.tracks[track_idx].box = boxes[box_idx]
                    self.tracks[track_idx].add_feature(pred_features[box_idx])
                    seen_box_idx.append(box_idx)

            unseen_track_ids = set(track_ids) - set(seen_track_ids)
            remove_track_ids.extend(list(unseen_track_ids))
            self.tracks = [t for t in self.tracks if t.id not in remove_track_ids]

            # update the feature of a track by using add_feature:
            # self.tracks[my_track_id].add_feature(pred_features[my_feat_index])
            # use the mean feature from the last 10 frames for ReID.

            new_boxes_idx = set(range(len(boxes))) - set(seen_box_idx)
            new_boxes = [boxes[i] for i in new_boxes_idx]
            new_scores = [scores[i] for i in new_boxes_idx]
            new_features = [pred_features[i] for i in new_boxes_idx]
            self.add(new_boxes, new_scores, new_features)
        else:
            # No tracks exist.
            self.add(boxes, scores, pred_features)
