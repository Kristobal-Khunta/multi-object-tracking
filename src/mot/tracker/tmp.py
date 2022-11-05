
# ############
# class BaseReIDTracker(BaseTracker):
#     def __init__(self, obj_detect, reid_model):
#         self.obj_detect = obj_detect
#         self.reid_model = reid_model
#         self.tracks = []
#         self.track_num = 0
#         self.im_index = 0
#         self.results = {}
#         self.mot_accum = None
#         self._UNMATCHED_COST = 255.0

#     def data_association(self, boxes, scores, frame):
#         raise NotImplementedError

#     def add(self, new_boxes, new_scores, new_features):
#         """Initializes new Track objects and saves them."""
#         num_new = len(new_boxes)
#         for i in range(num_new):
#             self.tracks.append(
#                 Track(new_boxes[i], new_scores[i], self.track_num + i, new_features[i])
#             )
#         self.track_num += num_new

#     def step(self, frame):
#         """This function should be called every timestep to perform tracking with a blob
#         containing the image information.
#         """
#         # object detection
#         boxes, scores = self.obj_detect.detect(frame["img"])

#         self.data_association(boxes, scores, frame["img"])
#         # results
#         self.update_results()

#     def compute_distance_matrix(
#         self, track_features, pred_features, track_boxes, boxes, metric_fn, alpha=0.0
#     ):

#         # Build cost matrix.
#         distance = mm.distances.iou_matrix(
#             track_boxes.numpy(), boxes.numpy(), max_iou=0.5
#         )

#         appearance_distance = metrics.compute_distance_matrix(
#             track_features, pred_features, metric_fn=metric_fn
#         )
#         appearance_distance = appearance_distance.numpy() * 0.5
#         # return appearance_distance

#         if not np.alltrue(appearance_distance >= -0.1):
#             raise AssertionError
#         if not np.alltrue(appearance_distance <= 1.1):
#             raise AssertionError

#         combined_costs = alpha * distance + (1 - alpha) * appearance_distance

#         # Set all unmatched costs to _UNMATCHED_COST.
#         distance = np.where(np.isnan(distance), self._UNMATCHED_COST, combined_costs)
#         return distance



# class ReIDHungarianTrackerOfflineDet(Tracker):
#     def __init__(self,):
#         super().__init__(obj_detect = None)
#         self.tracks = []
#         self.track_num = 0
#         self.im_index = 0
#         self.results = {}
#         self.mot_accum = None
#         self._UNMATCHED_COST = 255.0

    
#     def compute_distance_matrix(
#         self, track_features, pred_features, track_boxes, boxes, metric_fn, alpha=0.0
#     ):

#         # Build cost matrix.
#         distance = mm.distances.iou_matrix(
#             ltrb_to_ltwh(track_boxes).numpy(), ltrb_to_ltwh(boxes).numpy(), max_iou=0.5
#         )

#         appearance_distance = metrics.compute_distance_matrix(
#             track_features, pred_features, metric_fn=metric_fn
#         )
#         appearance_distance = appearance_distance.numpy() * 0.5
#         # return appearance_distance

#         if not np.alltrue(appearance_distance >= -0.1):
#             raise AssertionError
#         if not np.alltrue(appearance_distance <= 1.1):
#             raise AssertionError

#         combined_costs = alpha * distance + (1 - alpha) * appearance_distance

#         # Set all unmatched costs to _UNMATCHED_COST.
#         distance = np.where(np.isnan(distance), self._UNMATCHED_COST, combined_costs)

#         distance = np.where(appearance_distance > 0.1, self._UNMATCHED_COST, distance)

#         return distance

#     def data_association(self, boxes, scores, pred_features):
#         """Refactored from previous implementation to split it onto distance computation and track management"""
#         if self.tracks:
#             track_boxes = torch.stack([t.box for t in self.tracks], axis=0)
#             track_features = torch.stack([t.get_feature() for t in self.tracks], axis=0)

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
#         """Updates existing tracks and removes unmatched tracks.
#         Reminder: If the costs are equal to _UNMATCHED_COST, it's not a
#         match.
#         """
#         track_ids = [t.id for t in self.tracks]

#         unmatched_track_ids = []
#         seen_track_ids = []
#         seen_box_idx = []
#         for track_idx, box_idx in zip(row_idx, col_idx):
#             costs = distance[track_idx, box_idx]
#             internal_track_id = track_ids[track_idx]
#             seen_track_ids.append(internal_track_id)
#             if costs == self._UNMATCHED_COST:
#                 unmatched_track_ids.append(internal_track_id)
#             else:
#                 self.tracks[track_idx].box = boxes[box_idx]
#                 self.tracks[track_idx].add_feature(pred_features[box_idx])
#                 seen_box_idx.append(box_idx)

#         unseen_track_ids = set(track_ids) - set(seen_track_ids)
#         unmatched_track_ids.extend(list(unseen_track_ids))
#         self.tracks = [t for t in self.tracks if t.id not in unmatched_track_ids]

#         # Add new tracks.
#         new_boxes_idx = set(range(len(boxes))) - set(seen_box_idx)
#         new_boxes = [boxes[i] for i in new_boxes_idx]
#         new_scores = [scores[i] for i in new_boxes_idx]
#         new_features = [pred_features[i] for i in new_boxes_idx]
#         self.add(new_boxes, new_scores, new_features)


# class ReIDTrackerOfflineDet(TrackerOfflineDet):
#     def __init__(self, *args, **kwargs):
#         self.tracks = []
#         self.track_num = 0
#         self.im_index = 0
#         self.results = {}
#         self.mot_accum = None
#         self._UNMATCHED_COST = 255.0
#         super().__init__(*args, **kwargs)

#     def add(self, new_boxes, new_scores, new_features):
#         """Initializes new Track objects and saves them."""
#         num_new = len(new_boxes)
#         for i in range(num_new):
#             self.tracks.append(
#                 Track(new_boxes[i], new_scores[i], self.track_num + i, new_features[i])
#             )

#         self.track_num += num_new

#     def reset(self, hard=True):
#         self.tracks = []

#         if hard:
#             self.track_num = 0
#             self.results = {}
#             self.im_index = 0

#     def data_association(self, boxes, scores, features):
#         raise NotImplementedError

#     def step(self, frame):
#         """This function should be called every timestep to perform tracking with a blob
#         containing the image information.
#         """
#         boxes = frame["det"]["boxes"]
#         scores = frame["det"]["scores"]
#         reid_feats = frame["det"]["reid"].cpu()
#         self.data_association(boxes, scores, reid_feats)

#         # results
#         self.update_results()

#     def compute_distance_matrix(
#         self, track_features, pred_features, track_boxes, boxes, metric_fn, alpha=0.0
#     ):

#         # Build cost matrix.
#         distance = mm.distances.iou_matrix(
#             ltrb_to_ltwh(track_boxes).numpy(), ltrb_to_ltwh(boxes).numpy(), max_iou=0.5
#         )

#         appearance_distance = metrics.compute_distance_matrix(
#             track_features, pred_features, metric_fn=metric_fn
#         )
#         appearance_distance = appearance_distance.numpy() * 0.5
#         # return appearance_distance

#         if not np.alltrue(appearance_distance >= -0.1):
#             raise AssertionError
#         if not np.alltrue(appearance_distance <= 1.1):
#             raise AssertionError

#         combined_costs = alpha * distance + (1 - alpha) * appearance_distance

#         # Set all unmatched costs to _UNMATCHED_COST.
#         distance = np.where(np.isnan(distance), self._UNMATCHED_COST, combined_costs)

#         distance = np.where(appearance_distance > 0.1, self._UNMATCHED_COST, distance)

#         return distance