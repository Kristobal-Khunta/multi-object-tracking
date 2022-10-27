import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

from tracker.tracker.predef_tracker import LongTermReIDHungarianPredefTracker


_UNMATCHED_COST = 255


class MPNTracker(LongTermReIDHungarianPredefTracker):
    def __init__(self, assign_net, *args, **kwargs):
        self.assign_net = assign_net
        super().__init__(*args, **kwargs)

    def data_association(self, boxes, scores, pred_features):
        if self.tracks:
            track_boxes = torch.stack([t.box for t in self.tracks], axis=0).cuda()
            track_features = torch.stack(
                [t.get_feature() for t in self.tracks], axis=0
            ).cuda()

            # Hacky way to recover the timestamps of boxes and tracks
            curr_t = self.im_index * torch.ones((pred_features.shape[0],)).cuda()
            track_t = torch.as_tensor(
                [self.im_index - t.inactive - 1 for t in self.tracks]
            ).cuda()

            # Do a forward pass through self.assign_net to obtain our costs.
            edges_raw_logits = self.assign_net(
                track_features.cuda(),
                pred_features.cuda(),
                track_boxes.cuda(),
                boxes.cuda(),
                track_t,
                curr_t,
            )
            # Note: self.assign_net will return unnormalized probabilities.
            # apply the sigmoid function to them!
            pred_sim = torch.sigmoid(edges_raw_logits).detach().cpu().numpy()
            pred_sim = pred_sim[-1]  # Use predictions at last message passing step
            distance = 1 - pred_sim
            #print(pred_sim)
            # Do not allow mataches when sim < 0.5, to avoid low-confident associations
            distance = np.where(pred_sim < 0.5, _UNMATCHED_COST, distance)

            # Perform Hungarian matching.
            row_idx, col_idx = linear_assignment(distance)
            self.update_tracks(row_idx, col_idx, distance, boxes, scores, pred_features)

        else:
            # No tracks exist.
            self.add(boxes, scores, pred_features)
