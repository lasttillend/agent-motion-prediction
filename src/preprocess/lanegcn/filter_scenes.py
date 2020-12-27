# 场景预处理

# 思路：过滤场景（5秒），过滤条件为：该场景至少有MIN_FRAME_FOR_ONE_SCENE帧（默认为21帧，包括20帧过去，1帧未来），
#      并且至少存在一个符合以下两个条件的障碍物：
#			（1）在第 0～19帧中至少出现 MIN_FRAME_HISTORY帧；
#			（2）在第20～49帧中至少出现 MIN_FRAME_FUTURE 帧。
# Note: 这里过滤的是5秒场景，不是25秒的。

import os
import zarr

from src.preprocess.lanegcn.filter_scenes_helper import find_invalid_scenes, filter_invalid_scenes

print("Validate 5sec 0")
DATA_ROOT = "/home/han/study/projects/agent-motion-prediction/data/lyft_dataset_5sec_scene/"

dataset_path = os.path.join(DATA_ROOT, "scenes/splited/validate_zarr_splited/validate_5sec_0.zarr")
dataset_zarr = zarr.open(dataset_path, mode='r')

scenes = dataset_zarr["scenes"]
frames = dataset_zarr["frames"]
agents = dataset_zarr["agents"]
tl_faces = dataset_zarr["traffic_light_faces"]

# find invalid scenes
invalid_scenes_list = find_invalid_scenes(scenes, frames, agents)

# filter invalid scenes if exists
if len(invalid_scenes_list) > 0:
	print(f"{len(invalid_scenes_list)} scenes are invalid")
	output_path = os.path.join(DATA_ROOT, "scenes/splited/validate_zarr_splited_valid/validate_5sec_0_valid.zarr")
	filter_invalid_scenes(invalid_scenes_list, scenes, frames, agents, tl_faces, output_path)
else:
	print("All scenes are valid!")




