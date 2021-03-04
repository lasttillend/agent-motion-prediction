import copy
from typing import List, Optional

import numpy as np
import torch

polyline_colors = dict(trajectory="#CCFFE5", lane="#000000")

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
    
    @property
    def ids(self) -> int:
        return self._idx

    @property
    def object_type(self) -> str:
        return self._attr["OBJECT_TYPE"]
        
class Polyline:

    def __init__(self, lines: List[Vector], idx: int, element_type_name: str):
        self._lines = copy.deepcopy(lines)
        self._idx = idx
        self._element_type_name = element_type_name
        self.is_padded = 0  # defualt is not padded

    def __iter__(self):
        return iter(self._lines)
    
    def __len__(self):
        return len(self._lines)

    @property
    def ids(self) -> int:
        return self._idx

    @property
    def object_type(self) -> str:
        return self._element_type_name

    def to_numpy(self) -> np.ndarray:
        vector_features = [] 
        for vector in self:
            start = vector.start
            end = vector.end
            attributes = list(vector.attributes.values())
            vector_feature = np.r_[start, end, attributes]
            vector_features.append(vector_feature)
        return np.array(vector_features, dtype=np.float)

    def to_tensor(self) -> torch.tensor:
        vector_features_np = self.to_numpy()
        return torch.from_numpy(vector_features_np)

    def draw(self, ax) -> None:
        vectors = self.to_numpy()[:, :4]  # [xmin, ymin, xmax, ymax] 
        for vector in vectors:
            dx = vector[2] - vector[0]
            dy = vector[3] - vector[1]
            ax.arrow(vector[0], vector[1], dx, dy, color=polyline_colors[self._element_type_name], length_includes_head=True, head_width=0.5, head_length=0.3)






