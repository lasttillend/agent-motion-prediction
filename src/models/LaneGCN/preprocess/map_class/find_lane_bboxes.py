import os
import json

import numpy as np

from l5kit.data import LocalDataManager
from l5kit.data.map_api import MapAPI
from l5kit.rasterization.rasterizer_builder import _load_metadata
from l5kit.geometry import geodetic_to_ecef, transform_point


DATA_ROOT = "/home/han/study/projects/agent-motion-prediction/data/lyft_dataset/"
os.environ["L5KIT_DATA_FOLDER"] = DATA_ROOT

MAP_FILES_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_files")


def get_bbox_coordinate(bbox, map_api):
    """
    Transform the latitude and longitude of the two corners of the bounding box into world coordinate.
    """
    sw_corner = bbox.south_west
    sw_lat, sw_lon = map_api._undo_e7(sw_corner.lat_e7), map_api._undo_e7(sw_corner.lng_e7)
    sw_ecef = geodetic_to_ecef(np.array([sw_lat, sw_lon]))
    sw_xyz = transform_point(sw_ecef, map_api.ecef_to_world)
    
    ne_corner = bbox.north_east
    ne_lat, ne_lon = map_api._undo_e7(ne_corner.lat_e7), map_api._undo_e7(ne_corner.lng_e7)
    ne_ecef = geodetic_to_ecef([ne_lat, ne_lon])
    ne_xyz = transform_point(ne_ecef, map_api.ecef_to_world)
    
    return np.array([sw_xyz, ne_xyz])


def is_lane(elem, map_api):
    return elem.element.HasField("lane")


def get_lanes(map_api):
    return [elem for elem in map_api.elements if is_lane(elem, map_api)]


def main():
	# 1. Initial setups
	dm = LocalDataManager()

	cfg = {
	    "raster_params": {
	        "semantic_map_key": "semantic_map/semantic_map.pb",
	        "dataset_meta_key": "meta.json",
	    }
	}

	semantic_map_filepath = dm.require(cfg["raster_params"]["semantic_map_key"])
	dataset_meta = _load_metadata(cfg["raster_params"]["dataset_meta_key"], dm)
	world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)
	map_api = MapAPI(semantic_map_filepath, world_to_ecef)

	# 2. Find all lanes
	all_lanes = get_lanes(map_api)

	# 3. Find all bounding boxes
	bbox_list = []
	bbox_idx_to_lane_id_dict = {}  # bbox idx -> lane_id
	for idx in range(len(all_lanes)):
	    lane = all_lanes[idx]
	    lane_id = MapAPI.id_as_str(lane.id)
	    bbox = get_bbox_coordinate(lane.bounding_box, map_api)
	    bbox = bbox[:, :2]     # get rid of z coordinate
	    bbox = bbox.flatten()  # [xmin, ymin, xmax, ymax]
	    bbox_list.append(bbox)
	    bbox_idx_to_lane_id_dict[str(idx)] = lane_id    
	bbox_np = np.array(bbox_list)

	# 4. Save bbox_np and bbox_idx_to_lane_id_dict
	with open(os.path.join(MAP_FILES_ROOT, "lane_bbox.npy"), "wb") as f:
		np.save(f, bbox_np)

	with open(os.path.join(MAP_FILES_ROOT, "bbox_idx_to_lane_id.json"), "w") as f:
		json.dump(bbox_idx_to_lane_id_dict, f)

if __name__ == "__main__":
	main()