import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from l5kit.data.map_api import MapAPI

from preprocess.vectornet.custom_map import CustomMapAPI

class MapRender:
    
    def __init__(self, map_api: CustomMapAPI):
        self._map_api = map_api
        self._color_map = dict(lane="#474747",                  # shallow black
                               traffic_control="#FFD700",       # gold
                               node="#0000CD",                  # blue
                               point_traffic_control="#f30000") # red
        
    def render_layer(self, layer_name: str, ax=None) -> None:
        if layer_name == "lane":
            self.render_lanes(ax)
        else:
            raise ValueError(f"Rendering {layer_name} has not been defined!")

    ###################################### Render different map elements ####################### 
    ######### Lane #########
    
    # Render all lanes.
    def render_lanes(self, ax=None) -> None:
        if ax is None:        
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_axes([0, 0, 1, 1])
            
        all_lanes = self._map_api.get_elements_from_layer("lane")
        for lane in all_lanes:
            lane_id = self._map_api.id_as_str(lane.id)
            self.render_lane(ax, lane_id)
            
    # Render one lane.
    def render_lane(self, ax, lane_id: str) -> None:
        coords = self._map_api.get_lane_coords(lane_id)
        self.render_boundary(ax, coords["xyz_left"])
        self.render_boundary(ax, coords["xyz_right"])
        
    def render_boundary(self, ax, boundary: np.ndarray) -> None:
        xs = boundary[:, 0]
        ys = boundary[:, 1]
        ax.plot(xs, ys, color=self._color_map["lane"])
    
    ######### Traffic control element ########
    
    # Render one traffic control element
    def render_traffic_control_element(self, ax, traffic_control_element_id: str):
        coords = self._map_api.get_traffic_element_coords(traffic_control_element_id)
        if coords["xyz"].shape[0] == 1:  # a point traffic control element like stop sign
            x = coords["xyz"][0, 0]
            y = coords["xyz"][0, 1]
            ax.scatter(x, y, s=50, color=self._color_map["point_traffic_control"])      
        elif coords["xyz"].shape[0] > 1:  # e.g., crosswalk, traffic light
            xy = coords['xyz'][:, :2]
            xy = np.r_[xy, xy[0].reshape(1, -1)]
            ax.plot(xy[:, 0], xy[:, 1], color=self._color_map["traffic_control"])
        else:
            raise ValueError("Traffic control element with no coordindate!")
        
    ######### Road network node #######
    def render_node(self, ax, node_id):
        coords = self._map_api.get_node_coords(node_id)
        ax.scatter(coords['xyz'][0], coords['xyz'][1], s=50, color=self._color_map["node"])
        
    ######### Junction ###########
    
    # Render one junction
    def render_junction(self, ax, junction_id: str):
        junction = self._map_api[junction_id].element.junction
        nodes = junction.road_network_nodes
        traffic_control_elements = junction.traffic_control_elements
        lanes = junction.lanes

        for node in nodes:
            node_id = self._map_api.id_as_str(node)
            self.render_node(ax, node_id)   

        for traffic_control_element in traffic_control_elements:
            traffic_control_element_id = self._map_api.id_as_str(traffic_control_element)
            self.render_traffic_control_element(ax, traffic_control_element_id)

        for lane in lanes:
            lane_id = self._map_api.id_as_str(lane)
            self.render_lane(ax, lane_id)

    ######################## Private draw functions ###################
    def draw_bbox(self, ax, bbox: np.ndarray) -> None:
        bbox = Rectangle(xy=(bbox[0], bbox[1]),
                         width=bbox[2] - bbox[0],
                         height=bbox[3] - bbox[1],
                         linewidth=2,
                         color='blue',
                         fill=False)
        ax.add_patch(bbox)

