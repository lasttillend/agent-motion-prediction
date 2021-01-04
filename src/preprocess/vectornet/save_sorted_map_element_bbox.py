import json
import os

import numpy as np

from preprocess.vectornet.custom_map import CustomMapAPI

MAP_FILES_ROOT = "/home/han/study/projects/agent-motion-prediction/src/preprocess/vectornet/map_elements"

file_dir = os.path.dirname(os.path.abspath(__file__))
semantic_map_filepath = os.path.join(file_dir, "semantic_map.pb")

with open(os.path.join(file_dir, "meta.json"), "rb") as f:
    dataset_meta = json.load(f)
world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)

# Create custom map. 
custom_map_api = CustomMapAPI(semantic_map_filepath, world_to_ecef)

# Save all sorted bbox.
custom_map_api.save_all_sorted_bbox()

# Load back for checking
sorted_map_element_bboxes = custom_map_api.build_sorted_element_bboxes()
print("sorted map elment bboxes:", sorted_/map_element_bboxes)
