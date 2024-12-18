from typing import Generator

import cv2
import numpy as np


class IOCapture:
    def __init__(self, iname: str, oname: str = ""):
        self.icap = cv2.VideoCapture(iname)
        self.ocap = (
            cv2.VideoWriter(
                oname,
                cv2.VideoWriter_fourcc(*"H264"),
                self.icap.get(cv2.CAP_PROP_FPS),
                (
                    int(self.icap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.icap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                ),
            )
            if oname
            else None
        )

    def read(self):
        return self.icap.read()

    def write(self, frame):
        if self.ocap is not None:
            self.ocap.write(frame)

    def release(self):
        self.icap.release()
        if self.ocap is not None:
            self.ocap.release()


def video_data(
    iname: str,
    oname: str = "",
    start: int = -1,
    final: int = -1,
) -> Generator[np.ndarray, None, None]:
    capture = IOCapture(iname, oname)
    capture.icap.set(cv2.CAP_PROP_POS_FRAMES, start)
    count = start
    try:
        while True:
            ret, frame = capture.read()
            count += 1
            if not ret:
                break
            if final > 0 and count >= final:
                break
            yield frame  # Yield the current frame
    finally:
        capture.release()
