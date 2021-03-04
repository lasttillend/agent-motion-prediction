from typing import List

import numpy as np

from l5kit.data.proto.road_network_pb2 import MapElement

from preprocess.vectornet.custom_map import CustomMapAPI
from preprocess.vectornet.custom_map_element import CustomMapElement
from preprocess.vectornet.polyline import Polyline, Vector

class Lane(CustomMapElement):
        
    def __init__(self, element: MapElement):
        assert self.is_lane(element), "This is not a lane element!"
        self._element = element
        self.element_type = "lane"
        self.num_points = 20
        self.boundaries = None
    
    def vectorize(self, map_api: CustomMapAPI, polyline_id: int) -> List[Polyline]:
        lane_id = self.get_id()
        # Interpolate points on lane boundaries to have same number of vectors.
        new_left_bd, new_right_bd = map_api.interpolate_points_on_lane_boundaries(lane_id, self.num_points)
        
        # Vectorize left boundary.
        lines = []
        for i in range(len(new_left_bd) - 1):   # number of vectors in a polyline = number of boundary points - 1
            start, end = new_left_bd[i], new_left_bd[i + 1]
            attr = self.get_attributes()
            idx = polyline_id
            vector = Vector(start, end, attr, idx)
            lines.append(vector)
        left_polyline = Polyline(lines, polyline_id, self.element_type)
        polyline_id += 1

        # Vectorize right boundary.
        lines = []
        for i in range(len(new_right_bd) - 1):
            start, end = new_right_bd[i], new_right_bd[i + 1]
            attr = self.get_attributes()
            idx = polyline_id
            vector = Vector(start, end, attr, idx)
            lines.append(vector)
        right_polyline = Polyline(lines, polyline_id, self.element_type)

        return [left_polyline, right_polyline]
    
    def num_polylines(self) -> int:
        return 2  # number of polylines created after vectorization for a lane is 2
    
    def get_attributes(self) -> dict:
        turn_type = self.get_turn_type()
        attr = dict(OBJECT_TYPE=1, turn_type=turn_type)  # OBJECT_TYPE = 1 for lane
        return attr

    def get_boundaries(self, map_api: CustomMapAPI) -> List[np.ndarray]:
        if self.boundaries is not None:
            return self.boundaries
        lane_id = self.get_id()
        bd_coords = map_api.get_lane_coords(lane_id)
        left_bd = bd_coords["xyz_left"][:, :2]
        right_bd = bd_coords["xyz_right"][:, :2]

        self.boundaries = [left_bd, right_bd]

        return [left_bd, right_bd]
       
    def get_turn_type(self):
        """
        Lane MapElement has a field called "turn_type_in_parent_junction", which is of type _LANE_TURNTYPE.
        _LANE_TURNTYPE = 0 -> unknown
                         1 -> THROUGH
                         2 -> LEFT
                         3 -> SHARP_LEFT
                         4 -> RIGHT
                         5 -> SHARP_RIGHT
                         6 -> U_TURN
        """
        turn_type = self._element.element.lane.turn_type_in_parent_junction
        return turn_type
        
    @staticmethod
    def is_lane(element: MapElement) -> bool:
        return bool(element.element.HasField("lane")) 
        
        
