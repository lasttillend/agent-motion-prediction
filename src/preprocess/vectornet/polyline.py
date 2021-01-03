import copy
from typing import List, Optional

import numpy as np

class Vector:
    
    def __init__(self, start: np.ndarray, end: np.ndarray, attr: dict, idx: int):
        self._start = start.copy()
        self._end = end.copy()
        self._attr = copy.deepcopy(attr)
        self._idx = idx

    def __str__(self):
        return f"s: ({self._start[0]: .4f}, {self._start[1]: .4f}) -> e: ({self._end[0]: .4f}, {self._end[1]: .4f})"
        
    @property
    def start(self) -> np.ndarray:
        return self._start

    @property
    def end(self) -> np.ndarray:
        return self._end

    @property
    def attributes(self) -> dict:
        return self._attr
        
class Polyline:

    def __init__(self, lines: List[Vector], idx: int, element_type_name: str):
        self._lines = copy.deepcopy(lines)
        self._idx = idx
        self._element_type_name = element_type_name

    def __iter__(self):
        return iter(self._lines)
