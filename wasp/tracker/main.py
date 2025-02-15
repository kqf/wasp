import cv2
from toolz import compose

from wasp.timer import Timer
from wasp.tracker.capture import video_dataset
from wasp.tracker.color import GrayscaleTracker
from wasp.tracker.custom.cropped import CroppedTracker
from wasp.tracker.custom.plot import (  # PlotInternalTracker,
    OverlayTracker,
    draw_bbox,
)

# from wasp.tracker.custom.tm import TemplateMatchingTracker
from wasp.tracker.filter import KalmanFilter
from wasp.tracker.segments import load_segments


def build_tracker(
    featureSetNumFeatures=250,
    samplerSearchWinSize=25.0,
    samplerTrackInRadius=4.0,
    samplerTrackMaxNegNum=65,
    samplerTrackMaxPosNum=50_000,
) -> cv2.Tracker:
    params = cv2.TrackerMIL.Params()
    params.featureSetNumFeatures = featureSetNumFeatures
    params.samplerSearchWinSize = samplerSearchWinSize
    params.samplerTrackInRadius = samplerTrackInRadius
    params.samplerTrackMaxNegNum = samplerTrackMaxNegNum
    params.samplerTrackMaxPosNum = samplerTrackMaxPosNum
    return cv2.TrackerMIL.create(params)


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
    for i, (frame, label) in enumerate(frames):
        if tracker is None:
            kfilter = KalmanFilter(label.to_tuple())
            tracker = compose(
                OverlayTracker,
                GrayscaleTracker,
                CroppedTracker,
                cv2.TrackerCSRT.create,
            )()
            tracker.init(frame, label.to_tuple())

        kfilter.correct(bbox)
        with timer():
            _, bbox = tracker.update(frame)

        draw_bbox(frame, bbox, (0, 255, 0))
        draw_bbox(frame, label.to_tuple(), (255, 0, 0))
        draw_bbox(frame, kfilter.predict(), (255, 0, 0))
        cv2.imshow("tracking", frame)
        if cv2.waitKey() == 27:
            return
    print(timer)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
