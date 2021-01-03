import json
import os

import numpy as np

from preprocess.vectornet.custom_map import CustomMapAPI

file_dir = os.path.dirname(os.path.abspath(__file__))
semantic_map_filepath = os.path.join(file_dir, "semantic_map.pb")

with open(os.path.join(file_dir, "meta.json"), "rb") as f:
    dataset_meta = json.load(f)
world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)

# Create custom map. 
custom_map_api = CustomMapAPI(semantic_map_filepath, world_to_ecef)

for element_name in ["lane", "crosswalk", "parking_zone", "traffic_light", "junction"]:
    custom_map_api.save_elements_bbox(element_name, os.path.join(file_dir, "map_element_bbox", f"{element_name}_bboxes.p"))

