"""Metric monocular depth via Depth Anything V2."""

import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from ego_vision.config.settings import DEPTH_MODEL


class DepthEstimator:
    def __init__(self, model=DEPTH_MODEL, device=None):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model)
        self.model = AutoModelForDepthEstimation.from_pretrained(model).to(self.device).eval()

    def estimate_bgr(self, frame_bgr):
        """Returns float32 depth in meters at the input frame's resolution."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        depth = outputs.predicted_depth

        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        depth = torch.nn.functional.interpolate(
            depth, size=(h, w), mode="bicubic", align_corners=False
        ).squeeze()

        return depth.float().cpu().numpy()
