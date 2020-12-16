from typing import List, Optional
import numpy as np

class LaneSegment:
    
    def __init__(
        self,
        id: str,
        l_neighbor_id: Optional[str],
        r_neighbor_id: Optional[str],
        predecessors: Optional[List[str]],
        successors: Optional[List[str]],
        centerline: np.ndarray
    ) -> None:
        self.id = id
        self.l_neighbor_id = l_neighbor_id
        self.r_neighbor_id = r_neighbor_id
        self.predecessors = predecessors
        self.successors = successors
        self.centerline = centerline
        
