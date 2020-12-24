from typing import List, Mapping, Tuple
import numpy as np
import os

from preprocess.lanegcn.utils.json_utils import read_json_file
from preprocess.lanegcn.utils.vector_map_loader import load_lane_segments_from_file
from preprocess.lanegcn.utils.manhattan_search import find_all_polygon_bboxes_overlapping_query_bbox
from preprocess.lanegcn.map_class.lane_segment import LaneSegment

from l5kit.data import MapAPI
from l5kit.data.proto.road_network_pb2 import MapElement

ROOT = os.path.dirname(os.path.abspath(__file__))
MAP_FILES_ROOT = os.path.join(ROOT, "map_files")


class LyftMap:
	
	def __init__(self):
		self.lane_centerlines_dict = self.build_centerline_index()
		self.lane_bbox, self.bbox_idx_to_lane_id = self.build_lane_bbox_index()
		
	def build_centerline_index(self) -> Mapping[str, LaneSegment]:
		"""
		Build dictionary of centerline for each lane, with lane_id as key.
		
		Returns:
			lane_centerlines_dict: Keys are lane_id, values are lane info, i.e., LaneSegment objects.
		"""
		fpath = os.path.join(MAP_FILES_ROOT, "lane_objs.p")  # lane_objs.p contains all LaneSegment objects
		lane_centerlines_dict = load_lane_segments_from_file(fpath)  # lane_id -> LaneSegment
		
		return lane_centerlines_dict
			
	def build_lane_bbox_index(self) -> Tuple[np.ndarray, Mapping[str, str]]:
		json_fpath = os.path.join(MAP_FILES_ROOT, "bbox_idx_to_lane_id.json")  # path to bbox_idx_to_lane_id.json file
		bbox_idx_to_lane_id = read_json_file(json_fpath)  
		npy_fpath = os.path.join(MAP_FILES_ROOT, "lane_bbox.npy")  # path to lane_bbox.npyy
		lane_bbox = np.load(npy_fpath)
		
		return lane_bbox, bbox_idx_to_lane_id
	
	def get_lane_ids_in_xy_bbox(
		self,
		query_x: float,
		query_y: float,
		query_search_range_manhattan: float = 5.0,
	) -> List[str]:
		query_min_x = query_x - query_search_range_manhattan
		query_max_x = query_x + query_search_range_manhattan        
		query_min_y = query_y - query_search_range_manhattan        
		query_max_y = query_y + query_search_range_manhattan        
		
		overlap_idcs = find_all_polygon_bboxes_overlapping_query_bbox(
				self.lane_bbox,
				np.array([query_min_x, query_min_y, query_max_x, query_max_y])
		)
		
		if len(overlap_idcs) == 0:
			return []
		
		neighborhood_lane_ids: List[str] = []
		for overlap_idx in overlap_idcs:
			lane_segment_id = self.bbox_idx_to_lane_id[str(overlap_idx)]
			neighborhood_lane_ids.append(lane_segment_id)
		
		return neighborhood_lane_ids
   

if __name__ == "__main__":
	lyft_map = LyftMap()
	# print("Successfully build lfyt map!")
