import copy
import os
import pickle
from typing import List, Tuple, Dict

import numpy as np

from l5kit.data.map_api import MapAPI
from l5kit.data.proto.road_network_pb2 import MapElement, TrafficControlElement, LatLngBox
from l5kit.geometry import geodetic_to_ecef, transform_point

MAP_ELEMENTS = ["lane"]  # currently considered
MAP_FILES_ROOT = "/home/han/study/projects/agent-motion-prediction/src/preprocess/vectornet/map_elements"

class CustomMapAPI(MapAPI):
    
    def __init__(self, protobuf_map_path: str, world_to_ecef: np.ndarray):
        super().__init__(protobuf_map_path, world_to_ecef)
        
    def get_elements_from_layer(self, layer_name: str) -> List[MapElement]:
        if layer_name in ["lane", "junction", "traffic_control_element"]:
            elements = [elem for elem in self.elements if self._element_of_type(elem, layer_name)]
        elif layer_name == "crosswalk":
            elements = self.get_crosswalks()
        elif layer_name == "parking_zone":
            elements = self.get_parking_zones()
        elif layer_name == "traffic_light":
            elements = self.get_traffic_lights()
        else:
            raise ValueError(f"Map element {layer_name} has not been defined!") 

        return elements

    def is_traffic_element(self, elem: MapElement) -> bool:
        return elem.element.HasField("traffic_control_element") 
    
    def is_a_point_traffic_element(self, elem: MapElement):
        """
        A "point traffic element" only has one coordinate pair, e.g., a stop sign.
        """
        assert self.is_traffic_element(elem), "This is not a traffic element!"
        return len(elem.element.traffic_control_element.points_x_deltas_cm) == 0
    
    def get_traffic_elements(self) -> List[MapElement]:
        return [elem for elem in self.elements if self.is_traffic_element(elem)]
    
    def get_traffic_element_coords(self, element_id: str) -> dict:
        """
        Get XYZ coordinates in world ref system for a traffic control element given its id.

        Args:
            element_id (str): traffic control element id
        Returns:
            dict: a dict with the polygon coordinates as an (Nx3) XYZ array
        """
        element = self[element_id]
        assert self.is_traffic_element(element), "This is not a traffic element!"
        traffic_element = element.element.traffic_control_element
  
        if self.is_a_point_traffic_element(element):
            dx, dy, dz = 0, 0, 0
        else:
            dx = traffic_element.points_x_deltas_cm
            dy = traffic_element.points_y_deltas_cm
            dz = traffic_element.points_z_deltas_cm
            
        xyz = self.unpack_deltas_cm(
            dx,
            dy,
            dz,
            traffic_element.geo_frame,        
        )
        
        return {"xyz": xyz}
   
    def get_crosswalks(self) -> List[TrafficControlElement]:
        all_traffic_elements = self.get_traffic_elements()
        crosswalks = []
        for traffic_element in all_traffic_elements:
            element_id = self.id_as_str(traffic_element.id)
            element = traffic_element.element.traffic_control_element
            if element.HasField("pedestrian_crosswalk") and element.points_x_deltas_cm:
                crosswalks.append(self[element_id])
                
        return crosswalks
    
    def get_parking_zones(self) -> List[TrafficControlElement]:
        all_traffic_elements = self.get_traffic_elements()
        parking_zones = []
        for traffic_element in all_traffic_elements:
            element_id = self.id_as_str(traffic_element.id)
            element = traffic_element.element.traffic_control_element
            if element.HasField("parking_zone") and element.points_x_deltas_cm:
                parking_zones.append(self[element_id])
                
        return parking_zones     
    
    def get_traffic_lights(self) -> List[TrafficControlElement]:
        all_traffic_elements = self.get_traffic_elements()
        traffic_lights = []
        for traffic_element in all_traffic_elements:
            element_id = self.id_as_str(traffic_element.id)
            element = traffic_element.element.traffic_control_element
            if element.HasField("traffic_light") and element.points_x_deltas_cm:
                traffic_lights.append(self[element_id])
        
        return traffic_lights

    def get_node_coords(self, element_id: str) -> dict:
        """
        Get XYZ coordinates in world ref system for a Road Network Node given its id.
        """
        element = self[element_id]
        node = element.element.node
        location = node.location
        lat, lon = self._undo_e7(location.lat_e7), self._undo_e7(location.lng_e7)
        altitude = location.altitude_cm / 100

        xyz_ecef = geodetic_to_ecef([lat, lon, altitude])
        xyz_world = transform_point(xyz_ecef, self.ecef_to_world)

        return {'xyz': xyz_world}
    
    def get_bbox(self, element_type: str, element_id: str) -> np.ndarray:
        if element_type == "lane":
            bbox = self._get_lane_bbox(element_id)
        elif element_type in ["crosswalk", "parking_zone", "traffic_light"]:
            bbox = self._get_traffic_element_bbox(element_id)
        elif element_type == "junction":
            bbox = self._get_junction_bbox(element_id)
        else:
            raise ValueError(f"Finding {element_type}'s bounding box has not been defined!")
        return bbox
    
    def get_sorted_lane_bbox(self) -> Tuple[Tuple, Tuple, Tuple, Tuple]:
        """
        Lane bboxes are sorted according to xmin/xmax, ymin/ymax.
        """
        all_lanes = self.get_elements_from_layer("lane")
        bboxes, ids = [], []
        for lane in all_lanes:
            lane_id = self.id_as_str(lane.id)
            lane_bbox = self.get_bbox("lane", lane_id)
            bboxes.append(lane_bbox)
            ids.append(lane_id)
        bboxes_np = np.array(bboxes)
        bboxes_xmin = bboxes_np[:, 0]
        bboxes_ymin = bboxes_np[:, 1]
        bboxes_xmax = bboxes_np[:, 2]
        bboxes_ymax = bboxes_np[:, 3]
    
        xmin_idx = np.argsort(bboxes_xmin)
        sorted_bboxes_xmin = list(bboxes_xmin[xmin_idx])
        sorted_xmin_element_ids = list(np.array(ids)[xmin_idx])

        xmax_idx = np.argsort(bboxes_xmax)
        sorted_bboxes_xmax = list(bboxes_xmax[xmax_idx])
        sorted_xmax_element_ids = list(np.array(ids)[xmax_idx])

        ymin_idx = np.argsort(bboxes_ymin)
        sorted_bboxes_ymin = list(bboxes_ymin[ymin_idx])
        sorted_ymin_element_ids = list(np.array(ids)[ymin_idx])
       
        ymax_idx = np.argsort(bboxes_ymax)
        sorted_bboxes_ymax = list(bboxes_ymax[ymax_idx])
        sorted_ymax_element_ids = list(np.array(ids)[ymax_idx])

        sorted_bbox_and_ids_pairs = (
                                        (sorted_bboxes_xmin, sorted_xmin_element_ids),
                                        (sorted_bboxes_xmax, sorted_xmax_element_ids),
                                        (sorted_bboxes_ymin, sorted_ymin_element_ids),
                                        (sorted_bboxes_ymax, sorted_ymax_element_ids),
                                    )

        return sorted_bbox_and_ids_pairs 
    
    def save_all_sorted_bbox(self) -> None:
        sorted_map_element_bboxes = dict()
        for element_type in MAP_ELEMENTS:
            if element_type == "lane":
                sorted_bbox_and_ids_pairs = self.get_sorted_lane_bbox()
            else:
                raise ValueError(f"Map element {element_type} has not been defined!")
            sorted_map_element_bboxes[element_type] = sorted_bbox_and_ids_pairs
        
        output_path = os.path.join(MAP_FILES_ROOT, "sorted_bboxes.p")
        with open(output_path, "wb") as f:
            pickle.dump(sorted_map_element_bboxes, f)

    def build_sorted_element_bboxes(self) -> Dict[str, Tuple[Tuple, Tuple, Tuple, Tuple]]:
        sorted_bbox_path = os.path.join(MAP_FILES_ROOT, "sorted_bboxes.p")
        with open(sorted_bbox_path, "rb") as f:
            sorted_map_element_bboxes = pickle.load(f)
    
        return sorted_map_element_bboxes 
        
    def save_elements_bbox(self, layer_name: str, output_path: str) -> None:
        if layer_name == "lane":
            all_lanes = self.get_elements_from_layer("lane")
            output = dict()  # lane_id -> np.ndarray [xmin, ymin, xmax, ymax]
            for lane in all_lanes:
                lane_id = self.id_as_str(lane.id)
                lane_bbox = self.get_bbox("lane", lane_id)
                output[lane_id] = copy.deepcopy(lane_bbox)
        elif layer_name == "junction":
            all_junctions = self.get_elements_from_layer("junction")
            output = dict()  # junction_id -> MapElement
            for junction in all_junctions:
                junction_id = self.id_as_str(junction.id)
                if self._is_valid_junction(junction_id):  # remove invalid junctions, which has no lanes
                    junction_bbox = self.get_bbox("junction", junction_id)
                    output[junction_id] = copy.deepcopy(junction_bbox)
        elif layer_name == "crosswalk":
            all_crosswalks = self.get_elements_from_layer("crosswalk")
            output = dict()  # crosswalk_id -> MapElement
            for crosswalk in all_crosswalks:
                crosswalk_id = self.id_as_str(crosswalk.id)
                crosswalk_bbox = self.get_bbox("crosswalk", crosswalk_id)
                output[crosswalk_id] = copy.deepcopy(crosswalk_bbox)
        elif layer_name == "parking_zone":
            all_parking_zones = self.get_elements_from_layer("parking_zone")
            output = dict()  # parking_zone_id -> MapElement
            for parking_zone in all_parking_zones:
                parking_zone_id = self.id_as_str(parking_zone.id)
                parking_zone_bbox = self.get_bbox("parking_zone", parking_zone_id)
                output[parking_zone_id] = parking_zone_bbox
        elif layer_name == "traffic_light":
            all_traffic_lights = self.get_elements_from_layer("traffic_light")
            output = dict()  # traffic_light_id -> MapElement
            for traffic_light in all_traffic_lights:
                traffic_light_id = self.id_as_str(traffic_light.id)
                traffic_light_bbox = self.get_bbox("traffic_light", traffic_light_id)
                output[traffic_light_id] = traffic_light_bbox
        else:
            raise ValueError(f"Map element {layer_name} has not been defined!")
        
        output_root = os.path.dirname(output_path) 
        os.makedirs(output_root, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(output, f)

    ######## Privater method ########
    def _element_of_type(self, elem: MapElement, layer_name: str) -> bool:
        return elem.element.HasField(layer_name)

    def _get_lane_bbox(self, lane_id: str) -> np.ndarray:
        lane = self[lane_id]
        bbox_coords = self._transform_bbox_coordinate(lane.bounding_box)[:, :2]  # lower left, upper right

        xmin, ymin = bbox_coords[0][0], bbox_coords[0][1]
        xmax, ymax = bbox_coords[1][0], bbox_coords[1][1]

        bbox = np.array([xmin, ymin, xmax, ymax])

        return bbox    
    
    def _transform_bbox_coordinate(self, bbox: LatLngBox) -> np.ndarray:
        """
        Transform the latitude and longitude of the two corners of the bounding box into world coordinate.
        """
        sw_corner = bbox.south_west
        sw_lat, sw_lon = self._undo_e7(sw_corner.lat_e7), self._undo_e7(sw_corner.lng_e7)
        sw_ecef = geodetic_to_ecef(np.array([sw_lat, sw_lon]))
        sw_xyz = transform_point(sw_ecef, self.ecef_to_world)

        ne_corner = bbox.north_east
        ne_lat, ne_lon = self._undo_e7(ne_corner.lat_e7), self._undo_e7(ne_corner.lng_e7)
        ne_ecef = geodetic_to_ecef([ne_lat, ne_lon])
        ne_xyz = transform_point(ne_ecef, self.ecef_to_world)

        return np.array([sw_xyz, ne_xyz])
    
    def _get_traffic_element_bbox(self, traffic_element_id: str) -> np.ndarray:
        coords = self.get_traffic_element_coords(traffic_element_id)

        xy = coords['xyz'][:, :2]
        xmin, ymin = min(xy[:, 0]), min(xy[:, 1])
        xmax, ymax = max(xy[:, 0]), max(xy[:, 1])
        bbox = np.array([xmin, ymin, xmax, ymax])

        return bbox
    
    def _get_junction_bbox(self, junction_id: str) -> np.ndarray:
        junction = self[junction_id]
        junction_elements = junction.element.junction
        lane_ids = junction_elements.lanes  # list of lane ids
        traffic_element_ids = junction_elements.traffic_control_elements  # list of traffic control element ids

        lane_bboxes = []
        for lane_id in lane_ids:
            lane_id_str = self.id_as_str(lane_id)
            lane_bbox = self._get_lane_bbox(lane_id_str) 
            lane_bboxes.append(lane_bbox)   
        lane_bboxes_np = np.array(lane_bboxes).reshape(-1, 4)  # shape = [num_lanes, 4]

        traffic_element_bboxes = []
        for traffic_element_id in traffic_element_ids:
            traffic_element_id_str = self.id_as_str(traffic_element_id)
            traffic_element_bbox = self._get_traffic_element_bbox(traffic_element_id_str)
            traffic_element_bboxes.append(traffic_element_bbox)
        traffic_element_bboxes_np = np.array(traffic_element_bboxes).reshape(-1, 4)  # shape = [num_traffic_elements, 4], here reshape to avoid the concatenation problem of empty array

        all_bboxes = np.concatenate([lane_bboxes_np, traffic_element_bboxes_np])
        xmin, ymin = min(all_bboxes[:, 0]), min(all_bboxes[:, 1])
        xmax, ymax = max(all_bboxes[:, 2]), max(all_bboxes[:, 3])

        bbox = np.array([xmin, ymin, xmax, ymax])

        return bbox
    
    
    def _is_valid_junction(self, junction_id: str) -> bool:
        """
        There exist some junctions that have no lane, so we need to remove them when saving.
        """
        junction = self[junction_id]
        return len(junction.element.junction.lanes) > 0
