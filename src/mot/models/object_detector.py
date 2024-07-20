from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch


class FRCNN_FPN(FasterRCNN):
    def __init__(self, num_classes: int, nms_thresh: float = 0.5):
        backbone = resnet_fpn_backbone("resnet50", False)
        super(FRCNN_FPN, self).__init__(backbone, num_classes)

        self.roi_heads.nms_thresh = nms_thresh

    def detect(self, img: torch.Tensor):
        """
        Detect pedestrians on the image
        Args:
            img (torch.Tensor)
        Returns:
            tuple(
                boxes: torch.Tensor with shape (N,4)
                scores: torch.Tensor with shape (N,)
                )
        """
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self.forward(img)[0]

        return detections["boxes"].detach().cpu(), detections["scores"].detach().cpu()
