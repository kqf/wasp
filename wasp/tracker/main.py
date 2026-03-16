import cv2
import numpy as np
import matplotlib.pyplot as plt
from toolz import compose

from wasp.timer import Timer
from wasp.tracker.capture import video_dataset
from wasp.tracker.color import GrayscaleTracker
from wasp.tracker.custom.cropped import CroppedTracker
from wasp.tracker.custom.plot import OverlayTracker, draw_bbox
from wasp.tracker.custom.resize import ResizedTracker
from wasp.tracker.filter import KalmanFilter
from wasp.tracker.segments import load_segments


class Metrics:
    def __init__(self):
        self.pred = []
        self.gt = []

    def update(self, pred_bbox, gt_bbox):
        px, py, pw, ph = pred_bbox
        gx, gy, gw, gh = gt_bbox

        pred_center = (px + pw / 2, py + ph / 2)
        gt_center = (gx + gw / 2, gy + gh / 2)

        self.pred.append(pred_center)
        self.gt.append(gt_center)

    def arrays(self):
        pred = np.array(self.pred)
        gt = np.array(self.gt)
        return pred, gt


def velocity(signal):
    return np.diff(signal)


def jitter(signal):
    # jitter = std of velocity
    return np.std(np.diff(signal))


def compute_lag(pred, gt):
    pred = pred - pred.mean()
    gt = gt - gt.mean()

    corr = np.correlate(pred, gt, mode="full")
    return corr.argmax() - (len(pred) - 1)


def plot_results(pred, gt):

    vx = velocity(pred[:, 0])
    vy = velocity(pred[:, 1])

    # Position plots
    plt.figure()
    plt.plot(pred[:, 0], label="pred_x")
    plt.plot(gt[:, 0], label="gt_x")
    plt.title("X Position")
    plt.legend()

    plt.figure()
    plt.plot(pred[:, 1], label="pred_y")
    plt.plot(gt[:, 1], label="gt_y")
    plt.title("Y Position")
    plt.legend()

    # Velocity plots
    plt.figure()
    plt.plot(vx)
    plt.title("Velocity X")

    plt.figure()
    plt.plot(vy)
    plt.title("Velocity Y")

    plt.show()


def main():
    segment = load_segments("wasp/tracker/segments.json")["sky"]
    frames = video_dataset(
        aname="test-annotations.json",
        iname="test.mov",
        start=segment.start_frame,
        final=segment.stop_frame,
    )
    bbox = segment.bbox
    tracker = None
    kfilter = None
    timer = Timer()

    metrics = Metrics()

    for i, (frame, label) in enumerate(frames):

        if tracker is None:

            kfilter = KalmanFilter(label.to_tuple())

            tracker = compose(
                OverlayTracker,
                GrayscaleTracker,
                CroppedTracker,
                ResizedTracker,
                cv2.TrackerMIL.create,
            )()

            tracker.init(frame, label.to_tuple())

        kfilter.correct(bbox)

        with timer():
            _, bbox = tracker.update(frame)

        gt_bbox = label.to_tuple()

        metrics.update(bbox, gt_bbox)

        draw_bbox(frame, bbox, (0, 255, 0))
        draw_bbox(frame, gt_bbox, (255, 0, 0))
        draw_bbox(frame, kfilter.predict(), (0, 0, 255))

        cv2.imshow("tracking", frame)

        if cv2.waitKey(1) == 27:
            break

    print(timer)

    cv2.destroyAllWindows()

    pred, gt = metrics.arrays()
    vx = velocity(pred[:, 0])
    vy = velocity(pred[:, 1])
    jitter_x = jitter(pred[:, 0])
    jitter_y = jitter(pred[:, 1])
    lag_x = compute_lag(pred[:, 0], gt[:, 0])
    lag_y = compute_lag(pred[:, 1], gt[:, 1])
    print("\n--- Tracking Metrics ---")
    print("Velocity std X:", np.std(vx))
    print("Velocity std Y:", np.std(vy))
    print("Jitter X:", jitter_x)
    print("Jitter Y:", jitter_y)
    print("Lag X (frames):", lag_x)
    print("Lag Y (frames):", lag_y)
    plot_results(pred, gt)


if __name__ == "__main__":
    main()
