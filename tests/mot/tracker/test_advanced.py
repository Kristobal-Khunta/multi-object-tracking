import unittest
import torch
from src.mot.tracker.advanced import AdvancedTracker


class TestAdvancedTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = AdvancedTracker(device="cpu")

    def test_data_association_no_tracks(self):
        boxes = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]])
        scores = torch.tensor([0.9, 0.8])
        pred_features = torch.randn(2, 128)

        self.tracker.data_association(boxes, scores, pred_features)

        self.assertEqual(len(self.tracker.tracks), 2)

    def test_data_association_with_tracks(self):
        # Add initial tracks
        self.tracker.add(
            torch.tensor([[10, 10, 20, 20]]), torch.tensor([0.9]), torch.randn(1, 128)
        )

        # New detections
        boxes = torch.tensor([[11, 11, 21, 21], [30, 30, 40, 40]])
        scores = torch.tensor([0.95, 0.8])
        pred_features = torch.randn(2, 128)

        self.tracker.data_association(boxes, scores, pred_features)

        self.assertEqual(len(self.tracker.tracks), 2)

    def test_data_association_low_similarity(self):
        # Add initial track
        self.tracker.add(
            torch.tensor([[10, 10, 20, 20]]), torch.tensor([0.9]), torch.randn(1, 128)
        )

        # New detection with low similarity
        boxes = torch.tensor([[50, 50, 60, 60]])
        scores = torch.tensor([0.8])
        pred_features = torch.randn(1, 128)

        # Mock the similarity_net to return low similarity
        self.tracker.similarity_net = lambda *args: torch.tensor([[-10.0]])

        self.tracker.data_association(boxes, scores, pred_features)

        self.assertEqual(len(self.tracker.tracks), 2)

    def test_data_association_multiple_tracks(self):
        # Add initial tracks
        self.tracker.add(
            torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]]),
            torch.tensor([0.9, 0.8]),
            torch.randn(2, 128),
        )

        # New detections
        boxes = torch.tensor([[11, 11, 21, 21], [31, 31, 41, 41], [50, 50, 60, 60]])
        scores = torch.tensor([0.95, 0.85, 0.7])
        pred_features = torch.randn(3, 128)

        self.tracker.data_association(boxes, scores, pred_features)

        self.assertEqual(len(self.tracker.tracks), 3)

    def test_data_association_device_consistency(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tracker = AdvancedTracker(device=device)

        # Add initial track
        tracker.add(
            torch.tensor([[10, 10, 20, 20]], device=device),
            torch.tensor([0.9], device=device),
            torch.randn(1, 128, device=device),
        )

        # New detection
        boxes = torch.tensor([[11, 11, 21, 21]], device=device)
        scores = torch.tensor([0.95], device=device)
        pred_features = torch.randn(1, 128, device=device)

        tracker.data_association(boxes, scores, pred_features)

        self.assertEqual(tracker.tracks[0].box.device, device)
        self.assertEqual(tracker.tracks[0].score.device, device)
        self.assertEqual(tracker.tracks[0].features[-1].device, device)


if __name__ == "__main__":
    unittest.main()
