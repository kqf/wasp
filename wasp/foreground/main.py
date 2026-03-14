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

def watershed(image, roi, pad=40, fg_shrink_ratio=0.35, border=5):
    x, y, w, h = roi

    # 1. Tighter crop = closer background wall = less room to expand
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image.shape[1], x + w + pad)
    y2 = min(image.shape[0], y + h + pad)
    crop = image[y1:y2, x1:x2].copy()
    ch, cw = crop.shape[:2]

    # 2. Pre-blur to soften weak gradients (prevents bleeding through noise)
    blurred = cv2.GaussianBlur(crop, (5, 5), 0)

    # 3. Boost gradients — watershed stops more aggressively at edges
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3,3), np.uint8))
    boosted = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

    markers = np.zeros((ch, cw), np.int32)

    markers[0:border, :] = 1
    markers[-border:, :] = 1
    markers[:, 0:border] = 1
    markers[:, -border:] = 1

    # 4. Larger fg_shrink = smaller seed = more conservative expansion
    fx1, fy1 = pad, pad
    fx2, fy2 = pad + w, pad + h
    fg_shrink = max(2, int(min(w, h) * fg_shrink_ratio))
    markers[
        fy1 + fg_shrink : fy2 - fg_shrink,
        fx1 + fg_shrink : fx2 - fg_shrink
    ] = 2

    # 5. Run on gradient image instead of raw — stops at edges not color
    cv2.watershed(boosted, markers)

    seg = np.where(markers == 2, 1, 0).astype(np.uint8)

    # 6. Morphological erosion — shrink result slightly
    kernel = np.ones((3, 3), np.uint8)
    seg = cv2.erode(seg, kernel, iterations=1)

    full_mask = np.zeros(image.shape[:2], np.uint8)
    full_mask[y1:y2, x1:x2] = seg

    return full_mask * 255

def threshold_segment(image, roi, pad=40, border=5):
    x, y, w, h = roi
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image.shape[1], x + w + pad)
    y2 = min(image.shape[0], y + h + pad)
    crop = image[y1:y2, x1:x2].copy()

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Otsu automatically picks the best threshold value
    _, seg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    full_mask = np.zeros(image.shape[:2], np.uint8)
    full_mask[y1:y2, x1:x2] = ~seg
    return full_mask

def floodfill_segment(image, roi, pad=40, tolerance=7):
    x, y, w, h = roi

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image.shape[1], x + w + pad)
    y2 = min(image.shape[0], y + h + pad)
    crop = image[y1:y2, x1:x2].copy()
    ch, cw = crop.shape[:2]

    # Seed from center of ROI
    seed = (pad + w // 2, pad + h // 2)  # (x, y) for floodFill

    # Mask must be 2px larger on each side for floodFill
    ff_mask = np.zeros((ch + 2, cw + 2), np.uint8)

    cv2.floodFill(
        crop, ff_mask, seed,
        newVal=(255, 255, 255),
        loDiff=(tolerance,) * 3,
        upDiff=(tolerance,) * 3,
        flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8)  # write 255 into mask
    )

    # Trim the 1px border padding floodFill requires
    seg = ff_mask[1:-1, 1:-1]

    full_mask = np.zeros(image.shape[:2], np.uint8)
    full_mask[y1:y2, x1:x2] = seg

    return full_mask


import cv2
import numpy as np


def waterflow_mbd(image, roi=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    costs = np.full(gray.shape, np.inf)
    costs[0, :], costs[-1, :], costs[:, 0], costs[:, -1] = 0, 0, 0, 0

    for _ in range(2):
        for y in range(1, gray.shape[0] - 1):
            for x in range(1, gray.shape[1] - 1):
                # Update cost based on neighbors + intensity barrier
                val = gray[y, x]
                costs[y, x] = min(
                    costs[y, x],
                    costs[y - 1, x] + abs(int(val) - int(gray[y - 1, x])),
                    costs[y, x - 1] + abs(int(val) - int(gray[y, x - 1])),
                )
        for y in range(gray.shape[0] - 2, 0, -1):
            for x in range(gray.shape[1] - 2, 0, -1):
                val = gray[y, x]
                costs[y, x] = min(
                    costs[y, x],
                    costs[y + 1, x] + abs(int(val) - int(gray[y + 1, x])),
                    costs[y, x + 1] + abs(int(val) - int(gray[y, x + 1])),
                )

    costs = cv2.normalize(costs, None, 0, 255, cv2.NORM_MINMAX)
    return costs.astype(np.uint8)


METHODS = {
    # "grabcut": grabcut,
    # "watershed": watershed,
    # "threshold": threshold_segment,
    "floodfill": floodfill_segment,
}

def evaluate(name, segment):
    image = cv2.imread("sample.png")
    roi = 776, 786, 100, 100

    with timer(f"Evaluating {name}"):
        result = segment(image, roi)

    cv2.imshow("foreground", result)
    cv2.waitKey()


def main():
    # roi = cv2.selectROI("ROI", image, fromCenter=False, showCrosshair=True)
    # cv2.destroyWindow("ROI")
    # print(roi)
    for name, method in METHODS.items():
        evaluate(name, method)

if __name__ == "__main__":
    main()
