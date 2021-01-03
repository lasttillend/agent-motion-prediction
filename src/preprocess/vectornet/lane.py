from typing import List

from l5kit.data.proto.road_network_pb2 import MapElement

from preprocess.vectornet.custom_map import CustomMapAPI
from preprocess.vectornet.custom_map_element import CustomMapElement
from preprocess.vectornet.polyline import Polyline, Vector

class Lane(CustomMapElement):
        
    def __init__(self, element: MapElement):
        assert self.is_lane(element), "This is not a lane element!"
        self._element = element
        self.element_type = "lane"
    
    def vectorize(self, map_api: CustomMapAPI, polyline_id: int) -> List[Polyline]:
        lane_id = map_api.id_as_str(self._element.id)
        bd_coords = map_api.get_lane_coords(lane_id)
        left_bd = bd_coords["xyz_left"][:, :2]  # xy coords of left boundary
        right_bd = bd_coords["xyz_right"][:, :2]  # xy coords of right boundary

        # Vectorize left boundary.
        lines = []
        for i in range(len(left_bd) - 1):   # number of vectors in a polyline = number of boundary points - 1
            start, end = left_bd[i], left_bd[i + 1]
            attr = self.get_attributes()
            idx = polyline_id
            vector = Vector(start, end, attr, idx)
            lines.append(vector)
        left_polyline = Polyline(lines, polyline_id, self.element_type)
        polyline_id += 1

        # Vectorize right boundary.
        lines = []
        for i in range(len(right_bd) - 1):
            start, end = right_bd[i], right_bd[i + 1]
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
        attr = dict(OBJECT_TYPE="lane", turn_type=turn_type)
        return attr
    
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
        
        
