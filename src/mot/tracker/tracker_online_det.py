# import motmetrics as mm
# import numpy as np
# import torch

# from mot.tracker.base import Tracker, BaseReIDTracker
# from mot.utils import ltrb_to_ltwh, cosine_distance
# from scipy.optimize import linear_sum_assignment as linear_assignment


# # tracker


# # class ReIDHungarianIoUTracker(BaseReIDTracker):
# #     def __init__(self, reid_model, unmatched_cost=255.0, *args, **kwargs):
# #         self.reid_model = reid_model
# #         self._UNMATCHED_COST = unmatched_cost
# #         super().__init__(*args, **kwargs)

# #     def data_association(self, boxes, scores, frame):
# #         crops = self.get_crop_from_boxes(boxes, frame)
# #         pred_features = self.compute_reid_features(self.reid_model, crops).cpu().clone()

# #         if self.tracks:
# #             track_ids = [t.id for t in self.tracks]
# #             track_boxes = torch.stack([t.box for t in self.tracks], axis=0)
# #             track_features = torch.stack([t.get_feature() for t in self.tracks], axis=0)

# #             # This will use your similarity measure. Please use cosine_distance!
# #             distance = self.compute_distance_matrix(
# #                 track_features,
# #                 pred_features,
# #                 track_boxes,
# #                 boxes,
# #                 metric_fn=cosine_distance,
# #             )

# #             # Perform Hungarian matching.
# #             row_idx, col_idx = linear_assignment(distance)

# #             # row_idx and col_idx are indices into track_boxes and boxes.
# #             # row_idx[i] and col_idx[i] define a match.
# #             # distance[row_idx[i], col_idx[i]] define the cost for that matching.

# #             remove_track_ids = []
# #             seen_track_ids = []
# #             seen_box_idx = []
# #             # Update existing tracks and remove unmatched tracks.

# #             for track_idx, box_idx in zip(row_idx, col_idx):
# #                 costs = distance[track_idx, box_idx]
# #                 internal_track_id = track_ids[track_idx]
# #                 seen_track_ids.append(internal_track_id)
# #                 # If the costs are equal to _UNMATCHED_COST, it's not a match.
# #                 if costs == self._UNMATCHED_COST:
# #                     remove_track_ids.append(internal_track_id)
# #                 else:
# #                     self.tracks[track_idx].box = boxes[box_idx]
# #                     self.tracks[track_idx].add_feature(pred_features[box_idx])
# #                     seen_box_idx.append(box_idx)

# #             unseen_track_ids = set(track_ids) - set(seen_track_ids)
# #             remove_track_ids.extend(list(unseen_track_ids))
# #             self.tracks = [t for t in self.tracks if t.id not in remove_track_ids]

# #             # update the feature of a track by using add_feature:
# #             # self.tracks[my_track_id].add_feature(pred_features[my_feat_index])
# #             # use the mean feature from the last 10 frames for ReID.

# #             new_boxes_idx = set(range(len(boxes))) - set(seen_box_idx)
# #             new_boxes = [boxes[i] for i in new_boxes_idx]
# #             new_scores = [scores[i] for i in new_boxes_idx]
# #             new_features = [pred_features[i] for i in new_boxes_idx]
# #             self.add(new_boxes, new_scores, new_features)
# #         else:
# #             # No tracks exist.
# #             self.add(boxes, scores, pred_features)


# class ReIDHungarianIoUTracker(BaseReIDTracker):
#     def __init__(self, obj_detect, reid_model, **kwargs):
#         super().__init__(obj_detect=obj_detect, **kwargs)
#         self.reid_model = reid_model
#         self.tracks = []
#         self.track_num = 0
#         self.im_index = 0
#         self.results = {}
#         self.mot_accum = None
#         self._UNMATCHED_COST = 255.0

#     def data_association(self, boxes, scores, frame):
#         crops = self.get_crop_from_boxes(boxes, frame)
#         pred_features = self.compute_reid_features(self.reid_model, crops).cpu().clone()

#         if self.tracks:
#             # track_ids = [t.id for t in self.tracks] # not needed
#             track_boxes = torch.stack([t.box for t in self.tracks], axis=0)
#             track_features = torch.stack([t.get_feature() for t in self.tracks], axis=0)

#             # This will use your similarity measure. Please use cosine_distance!
#             distance = self.compute_distance_matrix(
#                 track_features,
#                 pred_features,
#                 track_boxes,
#                 boxes,
#                 metric_fn=cosine_distance,
#             )

#             # Perform Hungarian matching.
#             row_idx, col_idx = linear_assignment(distance)
#             self.update_tracks(row_idx, col_idx, distance, boxes, scores, pred_features)

#         else:
#             # No tracks exist.
#             self.add(boxes, scores, pred_features)

#     def update_tracks(self, row_idx, col_idx, distance, boxes, scores, pred_features):
#         # row_idx and col_idx are indices into track_boxes and boxes.
#         # row_idx[i] and col_idx[i] define a match.
#         # distance[row_idx[i], col_idx[i]] define the cost for that matching.
#         track_ids = [t.id for t in self.tracks]

#         remove_track_ids = []
#         seen_track_ids = []
#         seen_box_idx = []
#         # Update existing tracks and remove unmatched tracks.

#         for track_idx, box_idx in zip(row_idx, col_idx):
#             costs = distance[track_idx, box_idx]
#             internal_track_id = track_ids[track_idx]
#             seen_track_ids.append(internal_track_id)
#             # If the costs are equal to _UNMATCHED_COST, it's not a match.
#             if costs == self._UNMATCHED_COST:
#                 remove_track_ids.append(internal_track_id)
#             else:
#                 self.tracks[track_idx].box = boxes[box_idx]
#                 self.tracks[track_idx].add_feature(pred_features[box_idx])
#                 seen_box_idx.append(box_idx)

#         unseen_track_ids = set(track_ids) - set(seen_track_ids)
#         remove_track_ids.extend(list(unseen_track_ids))
#         self.tracks = [t for t in self.tracks if t.id not in remove_track_ids]

#         # update the feature of a track by using add_feature:
#         # self.tracks[my_track_id].add_feature(pred_features[my_feat_index])
#         # use the mean feature from the last 10 frames for ReID.

#         new_boxes_idx = set(range(len(boxes))) - set(seen_box_idx)
#         new_boxes = [boxes[i] for i in new_boxes_idx]
#         new_scores = [scores[i] for i in new_boxes_idx]
#         new_features = [pred_features[i] for i in new_boxes_idx]
#         self.add(new_boxes, new_scores, new_features)


# class MPNTracker(ReIDHungarianIoUTracker):
#     def __init__(
#         self, obj_detect, reid_model, refine_gnn_net, *args, device="cuda", **kwargs
#     ):
#         self.obj_detect = obj_detect
#         self.reid_model = reid_model
#         self.refine_gnn_net = refine_gnn_net
#         self.device = device
#         self._UNMATCHED_COST = 255.0
#         self.tracks = []
#         self.track_num = 0
#         self.im_index = 0
#         self.results = {}
#         self.mot_accum = None
#         super().__init__(*args, **kwargs)

#     def data_association(self, boxes, scores, frame):  # pred_features
#         crops = self.get_crop_from_boxes(boxes, frame)
#         pred_features = self.compute_reid_features(self.reid_model, crops).cpu().clone()

#         if self.tracks:
#             track_boxes = torch.stack([t.box for t in self.tracks], axis=0).to(
#                 self.device
#             )
#             track_features = torch.stack(
#                 [t.get_feature() for t in self.tracks], axis=0
#             ).to(self.device)

#             # Hacky way to recover the timestamps of boxes and tracks
#             curr_t = self.im_index * torch.ones((pred_features.shape[0],)).to(
#                 self.device
#             )
#             track_t = torch.as_tensor(
#                 [self.im_index - t.inactive - 1 for t in self.tracks]
#             ).to(self.device)

#             # Do a forward pass through self.assign_net to obtain our costs.
#             edges_raw_logits = self.refine_gnn_net(
#                 track_features.to(self.device),
#                 pred_features.to(self.device),
#                 track_boxes.to(self.device),
#                 boxes.to(self.device),
#                 track_t,
#                 curr_t,
#             )
#             # Note: self.assign_net will return unnormalized probabilities.
#             # apply the sigmoid function to them!
#             pred_sim = torch.sigmoid(edges_raw_logits).detach().cpu().numpy()
#             pred_sim = pred_sim[-1]  # Use predictions at last message passing step
#             distance = 1 - pred_sim
#             # print(pred_sim)
#             # Do not allow mataches when sim < 0.5, to avoid low-confident associations
#             distance = np.where(pred_sim < 0.5, self._UNMATCHED_COST, distance)

#             # Perform Hungarian matching.
#             row_idx, col_idx = linear_assignment(distance)
#             self.update_tracks(row_idx, col_idx, distance, boxes, scores, pred_features)

#         else:
#             # No tracks exist.
#             self.add(boxes, scores, pred_features)
