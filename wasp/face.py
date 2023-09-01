from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.linalg import norm as l2norm


@dataclass
class Face:
    bbox: np.ndarray
    kps: np.ndarray
    detection_score: float
    embedding: Optional[np.ndarray] = None

    @property
    def embedding_norm(self):
        return None if self.embedding is None else l2norm(self.embedding)

    @property
    def normed_embedding(self):
        # sourcery skip: assign-if-exp, reintroduce-else
        if self.embedding is None:
            return None
        return self.embedding / self.embedding_norm
