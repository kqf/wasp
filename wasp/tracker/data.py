from dataclasses import dataclass, field
from typing import Optional

AbsoluteXYXY = tuple[float, float, float, float]
AbsoluteXYHW = tuple[float, float, float, float]


@dataclass_json
@dataclass
class Annotation:
    label: str
    segment: str
    xxyy: Optional[AbsoluteXYXY] = field(default=None)
    xyhw: Optional[AbsoluteXYHW] = field(default=None)

    def __post_init__(self):
        if self.xxyy and self.xyhw:
            raise ValueError("Provide only one of xxyy or xyhw, not both.")
        elif not self.xxyy and not self.xyhw:
            raise ValueError("You must provide one of xxyy or xyhw.")

        if self.xxyy:
            x1, y1, x2, y2 = self.xxyy
            self.xyhw = (x1, y1, x2 - x1, y2 - y1)
            return

        x, y, w, h = self.xyhw
        self.xxyy = (x, y, x + w, y + h)
