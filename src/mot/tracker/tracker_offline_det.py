import motmetrics as mm
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment as linear_assignment

from mot.tracker.base import Track, Tracker
from mot.utils import cosine_distance, ltrb_to_ltwh
import market.metrics as metrics

mm.lap.default_solver = "lap"





class TrackerOfflineDet(Tracker):
    def step(self, frame):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """

        # if predefined detections
        boxes = frame["det"]["boxes"]
        scores = frame["det"]["scores"]
        self.data_association(boxes, scores)
        self.update_results()


class ReIDTrackerOfflineDet(TrackerOfflineDet):
    def __init__(self, unmatched_cost=255.0, *args, **kwargs):
        self._UNMATCHED_COST = unmatched_cost
        super().__init__(*args, **kwargs)

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

    def data_association(self, boxes, scores, features):
        raise NotImplementedError

    def step(self, frame):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.
        """
        boxes = frame["det"]["boxes"]
        scores = frame["det"]["scores"]
        reid_feats = frame["det"]["reid"].cpu()
        self.data_association(boxes, scores, reid_feats)

        # results
        self.update_results()

    def compute_distance_matrix(
        self, track_features, pred_features, track_boxes, boxes, metric_fn, alpha=0.0
    ):

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
        distance = np.where(np.isnan(distance), self._UNMATCHED_COST, combined_costs)

        distance = np.where(appearance_distance > 0.1, self._UNMATCHED_COST, distance)

        return distance


class ReIDHungarianTrackerOfflineDet(ReIDTrackerOfflineDet):
    def data_association(self, boxes, scores, pred_features):
        """Refactored from previous implementation to split it onto distance computation and track management"""
        if self.tracks:
            track_boxes = torch.stack([t.box for t in self.tracks], axis=0)
            track_features = torch.stack([t.get_feature() for t in self.tracks], axis=0)

            distance = self.compute_distance_matrix(
                track_features,
                pred_features,
                track_boxes,
                boxes,
                metric_fn=cosine_distance,
            )

            # Perform Hungarian matching.
            row_idx, col_idx = linear_assignment(distance)
            self.update_tracks(row_idx, col_idx, distance, boxes, scores, pred_features)

        else:
            # No tracks exist.
            self.add(boxes, scores, pred_features)

    def update_tracks(self, row_idx, col_idx, distance, boxes, scores, pred_features):
        """Updates existing tracks and removes unmatched tracks.
        Reminder: If the costs are equal to _UNMATCHED_COST, it's not a
        match.
        """
        track_ids = [t.id for t in self.tracks]

        unmatched_track_ids = []
        seen_track_ids = []
        seen_box_idx = []
        for track_idx, box_idx in zip(row_idx, col_idx):
            costs = distance[track_idx, box_idx]
            internal_track_id = track_ids[track_idx]
            seen_track_ids.append(internal_track_id)
            if costs == self._UNMATCHED_COST:
                unmatched_track_ids.append(internal_track_id)
            else:
                self.tracks[track_idx].box = boxes[box_idx]
                self.tracks[track_idx].add_feature(pred_features[box_idx])
                seen_box_idx.append(box_idx)

        unseen_track_ids = set(track_ids) - set(seen_track_ids)
        unmatched_track_ids.extend(list(unseen_track_ids))
        self.tracks = [t for t in self.tracks if t.id not in unmatched_track_ids]

        # Add new tracks.
        new_boxes_idx = set(range(len(boxes))) - set(seen_box_idx)
        new_boxes = [boxes[i] for i in new_boxes_idx]
        new_scores = [scores[i] for i in new_boxes_idx]
        new_features = [pred_features[i] for i in new_boxes_idx]
        self.add(new_boxes, new_scores, new_features)


class LongTermReIDHungarianTrackerOfflineDet(ReIDHungarianTrackerOfflineDet):
    def __init__(self, patience, *args, **kwargs):
        """Add a patience parameter"""
        self.patience = patience
        super().__init__(*args, **kwargs)

    def update_results(self):
        """Only store boxes for tracks that are active"""
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            if t.inactive == 0:
                self.results[t.id][self.im_index] = np.concatenate(
                    [t.box.cpu().numpy(), np.array([t.score])]
                )

        self.im_index += 1

    def update_tracks(self, row_idx, col_idx, distance, boxes, scores, pred_features):
        track_ids = [t.id for t in self.tracks]

        unmatched_track_ids = []
        seen_track_ids = []
        seen_box_idx = []
        for track_idx, box_idx in zip(row_idx, col_idx):
            costs = distance[track_idx, box_idx]
            internal_track_id = track_ids[track_idx]
            seen_track_ids.append(internal_track_id)
            if costs == self._UNMATCHED_COST:
                unmatched_track_ids.append(internal_track_id)

            else:
                self.tracks[track_idx].box = boxes[box_idx]
                self.tracks[track_idx].add_feature(pred_features[box_idx])

                # Note: the track is matched, therefore, inactive is set to 0
                self.tracks[track_idx].inactive = 0
                seen_box_idx.append(box_idx)

        unseen_track_ids = set(track_ids) - set(seen_track_ids)
        unmatched_track_ids.extend(list(unseen_track_ids))

        # Update the `inactive` attribute for those tracks that have been
        # not been matched. kill those for which the inactive parameter
        # is > self.patience

        active_tracks = []
        for t in self.tracks:
            if t.id not in unmatched_track_ids:
                active_tracks.append(t)
            elif t.inactive < self.patience:
                active_tracks.append(t)
                t.inactive += 1
            else:  #
                continue
        self.tracks = active_tracks

        new_boxes_idx = set(range(len(boxes))) - set(seen_box_idx)
        new_boxes = [boxes[i] for i in new_boxes_idx]
        new_scores = [scores[i] for i in new_boxes_idx]
        new_features = [pred_features[i] for i in new_boxes_idx]
        self.add(new_boxes, new_scores, new_features)


############


class MPNTrackerOfflineDet(LongTermReIDHungarianTrackerOfflineDet):
    def __init__(self, similarity_net, device='cuda', *args, **kwargs):
        self.similarity_net = similarity_net
        self.device = device
        super().__init__(*args, **kwargs)

    def data_association(self, boxes, scores, pred_features):
        if self.tracks:
            track_boxes = torch.stack([t.box for t in self.tracks], axis=0).to(self.device)
            track_features = torch.stack(
                [t.get_feature() for t in self.tracks], axis=0
            ).to(self.device)

            # Hacky way to recover the timestamps of boxes and tracks
            curr_t = self.im_index * torch.ones((pred_features.shape[0],)).to(self.device)
            track_t = torch.as_tensor(
                [self.im_index - t.inactive - 1 for t in self.tracks]
            ).to(self.device)

            # Do a forward pass through self.assign_net to obtain our costs.
            edges_raw_logits = self.similarity_net(
                track_features.to(self.device),
                pred_features.to(self.device),
                track_boxes.to(self.device),
                boxes.to(self.device),
                track_t,
                curr_t,
            )
            # Note: self.assign_net will return unnormalized probabilities.
            # apply the sigmoid function to them!
            pred_sim = torch.sigmoid(edges_raw_logits).detach().cpu().numpy()
            pred_sim = pred_sim[-1]  # Use predictions at last message passing step
            distance = 1 - pred_sim
            # bprint(pred_sim)
            # Do not allow mataches when sim < 0.5, to avoid low-confident associations
            distance = np.where(pred_sim < 0.5, self._UNMATCHED_COST, distance)

            # Perform Hungarian matching.
            row_idx, col_idx = linear_assignment(distance)
            self.update_tracks(row_idx, col_idx, distance, boxes, scores, pred_features)

        else:
            # No tracks exist.
            self.add(boxes, scores, pred_features)
