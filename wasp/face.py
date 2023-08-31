from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Face:
    bbox: np.ndarray
    kps: np.ndarray
    detection_score: float
    embedding: Optional[np.ndarray] = None
