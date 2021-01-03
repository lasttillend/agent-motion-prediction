from abc import ABC, abstractmethod
from typing import List

from l5kit.data.proto.road_network_pb2 import MapElement

from preprocess.vectornet.custom_map import CustomMapAPI
from preprocess.vectornet.polyline import Polyline

class CustomMapElement(ABC):
    """
    Basic class for a custom map element. Every subclass is basically a wrapper of MapElement type.
    """

    def __init__(self, element: MapElement):
        self._element = element

    
    @abstractmethod
    def vectorize(self, map_api: CustomMapAPI, polyline_id: int) -> List[Polyline]:
        pass

    @abstractmethod
    def num_polylines(self) -> int:
        """
        Number of polylines created after vectorization.
        """
        pass 

    @abstractmethod
    def get_attributes(self) -> dict:
        """
        Attribute features of a map element.
        """
        pass 
