import cv2
import numpy as np
import time
from contextlib import contextmanager


@contextmanager
def timer(name):
    start = time.time()
    try:
        yield
    finally:
        elapsed = (time.time() - start) * 1000
        print(f"{name}: {elapsed:.2f} ms")


def grabcut(image, roi):
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.grabCut(image, mask, roi, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
    print("mask", mask.sum(), mask.std())
    result = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    return (result * 255).astype(np.uint8)


def main():
    image = cv2.imread("sample.png")
    # roi = cv2.selectROI("ROI", image, fromCenter=False, showCrosshair=True)
    # cv2.destroyWindow("ROI")
    # print(roi)
    roi = 776, 786, 100, 100
    with timer("grabcut"):
        result = grabcut(image, roi)

    cv2.imshow("foreground", result)
    cv2.waitKey()


if __name__ == "__main__":
    main()
