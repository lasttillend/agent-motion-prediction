import os
import pickle
from collections import defaultdict

import numpy as np

from l5kit.data import LocalDataManager
from l5kit.data.map_api import MapAPI
from l5kit.rasterization.rasterizer_builder import _load_metadata
from l5kit.geometry import geodetic_to_ecef, transform_point

from src.preprocess.lanegcn.map_class.lane_segment import LaneSegment

DATA_ROOT = "/home/han/study/projects/agent-motion-prediction/data/lyft_dataset/"
os.environ["L5KIT_DATA_FOLDER"] = DATA_ROOT

MAP_FILES_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_files")


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

	# 3. Construct LaneSegment objects
	suc = dict()

	for i in range(len(all_lanes)):
	    lane = all_lanes[i]
	    lane_id = MapAPI.id_as_str(lane.id)
	    successors = [MapAPI.id_as_str(lane) for lane in lane.element.lane.lanes_ahead]
	    suc[lane_id] = successors
	    
	pre = defaultdict(list)
	for p, suc_list in suc.items():
	    for s in suc_list:
	        pre[s].append(p)

	## Build all LaneSegments
	lane_objs = {}  # lane_id -> LaneSegment

	for i in range(len(all_lanes)):
	    lane = all_lanes[i]
	    lane_id = MapAPI.id_as_str(lane.id)
	    l_neighbor_id = MapAPI.id_as_str(lane.element.lane.adjacent_lane_change_left)
	    r_neighbor_id = MapAPI.id_as_str(lane.element.lane.adjacent_lane_change_right)
	    predecessors = pre[lane_id]
	    successors = suc[lane_id]

	    boundary = map_api.get_lane_coords(lane_id)
	    left_boundary = boundary['xyz_left']
	    right_boundary = boundary['xyz_right']
	    size = min(left_boundary.shape[0], right_boundary.shape[0])  # left_boundary and right_boundary may have different number of points
	    centerline = (left_boundary[:size] + right_boundary[:size]) / 2
	    
	    ls = LaneSegment(lane_id, l_neighbor_id, r_neighbor_id, predecessors, successors, centerline)
	    
	    lane_objs[lane_id] = ls

	# 4. Save lane_objs
	os.makedirs(MAP_FILES_ROOT, exist_ok=True)
	with open(os.path.join(MAP_FILES_ROOT, "lane_objs.p"), "wb") as f:
	    pickle.dump(lane_objs, f)


if __name__ == "__main__":
	main()
