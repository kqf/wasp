import cv2

from wasp.tracker.custom.of import OpticalFLowTracker
from wasp.tracker.custom.ofs import OpticalFLowSimplified
from wasp.tracker.custom.tm import TemplateMatchingTracker

TRACKERS = {
    "cv2.legacy.TrackerMOSSE_create": cv2.legacy.TrackerMOSSE_create,
    "cv2.legacy.TrackerBoosting_create": cv2.legacy.TrackerBoosting_create,
    "cv2.legacy.TrackerCSRT_create": cv2.legacy.TrackerCSRT_create,
    "cv2.legacy.TrackerMIL_create": cv2.legacy.TrackerMIL_create,
    "cv2.legacy.TrackerKCF_create": cv2.legacy.TrackerKCF_create,
    "cv2.legacy.TrackerMF_create": cv2.legacy.TrackerMF_create,
    "cv2.legacy.TrackerTLD_create": cv2.legacy.TrackerTLD_create,
    "OF": OpticalFLowTracker,
    "OFS": OpticalFLowSimplified,
    "TM": TemplateMatchingTracker,
}
