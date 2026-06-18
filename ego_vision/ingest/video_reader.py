"""Thin wrapper over cv2.VideoCapture."""

from dataclasses import dataclass

import cv2


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    frame_count: int


class VideoReader:
    def __init__(self, path):
        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.path}")

        self.meta = VideoMeta(
            fps=self._cap.get(cv2.CAP_PROP_FPS) or 30.0,
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def frames(self):
        idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                return
            yield idx, frame
            idx += 1

    def release(self):
        self._cap.release()
