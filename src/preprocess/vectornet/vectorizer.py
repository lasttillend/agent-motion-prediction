import copy
from collections import defaultdict
from typing import List

import numpy as np

from l5kit.data.proto.road_network_pb2 import MapElement

from preprocess.vectornet.custom_map import CustomMapAPI
from preprocess.vectornet.custom_map_element import CustomMapElement
from preprocess.vectornet.polyline import Vector, Polyline


class Vectorizer:

    def __init__(self, map_api: CustomMapAPI):
        self._map_api = map_api
        self.polyline_ids = defaultdict(int)  # polyline counter for each map element type 

    def vectorize(self, map_elements: List[CustomMapElement]) -> List[Polyline]:
        vector_set = []
        element_type = map_elements[0].element_type
        for map_element in map_elements:
            polylines = map_element.vectorize(self._map_api, self.polyline_ids[element_type])  # a list of polylines
            vector_set.extend(polylines)
            self.polyline_ids[element_type] += map_element.num_polylines()

        return vector_set


