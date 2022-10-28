import market.metrics as metrics
import motmetrics as mm
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from mot.tracker.base import Tracker
from mot.utils import ltrb_to_ltwh
from scipy.optimize import linear_sum_assignment as linear_assignment


# tracker
class BaseTrackerIoU(Tracker):
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


class ReIDTracker(Tracker):
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
