# 场景预处理辅助函数

from collections import Counter
from tqdm import tqdm
from typing import List
import zarr
from zarr.core import Array as ZArray
import numpy as np

MIN_FRAME_FOR_ONE_SCENE = 21  # minimum frames for a valid 5-sec scene  
MIN_FRAME_HISTORY = 10
MIN_FRAME_FUTURE = 1

#### find_invalid_scenes ####
def find_invalid_scenes(scenes: ZArray, frames: ZArray, agents: ZArray) -> List[int]:
	"""
	Find invalid scenes which either contains frames less than MIN_FRAME_HISTORY + MIN_FRAME_FUTURE
	or there is no agent (in the scene) which satisfies the following three conditions:
		1. the agent must appear in the 19-th frame;
		2. the agent must appear in at least MIN_FRAME_HISTORY frames in the 0-th to 19-th frame;
		3. the agent must appear in at least MIN_FRAME_FUTURE frames in the 20-th to 49-th frame.

	The indexes of invalid scenes are returned.
	"""
	invalid_scenes_list = []
	for scene_id in tqdm(range(len(scenes)), desc="find invalid scenes"):
		scene = scenes[scene_id]
		frame_start_id = scene["frame_index_interval"][0]
		frame_end_id = scene["frame_index_interval"][1]

		if (frame_end_id - frame_start_id) < MIN_FRAME_FOR_ONE_SCENE:
			invalid_scenes_list.append(scene_id)
			continue

		frame19 = frames[frame_start_id + 19]
		agents_in_frame19 = agents[frame19["agent_index_interval"][0]:frame19["agent_index_interval"][1]]["track_id"]

		history_frames = frames[frame_start_id:frame_start_id + 20]  # include the 19-th frame
		future_frames = frames[frame_start_id + 20:frame_end_id]

		history_agents_start_id = history_frames[0]["agent_index_interval"][0]
		history_agents_end_id = history_frames[-1]["agent_index_interval"][1]
		future_agents_start_id = future_frames[0]["agent_index_interval"][0]
		future_agents_end_id = future_frames[-1]["agent_index_interval"][1]

		history_agents = agents[history_agents_start_id:history_agents_end_id]["track_id"]
		history_agents_cnt = Counter(history_agents)
		future_agents = agents[future_agents_start_id:future_agents_end_id]["track_id"]
		future_agents_cnt = Counter(future_agents)

		cond1_satisfy = agents_in_frame19
		cond2_satisfy = np.array([track_id for track_id, time in history_agents_cnt.items() if time >= MIN_FRAME_HISTORY])
		cond3_satisfy = np.array([track_id for track_id, time in future_agents_cnt.items() if time >= MIN_FRAME_FUTURE])

		all_satisfy = np.intersect1d(np.intersect1d(cond1_satisfy, cond2_satisfy), cond3_satisfy)
		if len(all_satisfy) < 1:
			invalid_scene_list.append(scene_id)	

	return invalid_scenes_list

#### filter_invalid_scenes ####
# 1. 计算invalid scenes, frames，agents和tl_faces
# 2. 通过invalid计算valid
# 3. 构建一个新的zarr文件，将valid agents, tl_faces, frames, 和scenes分别做以下处理：
# 	（1）agents: 不用修改agent的任何属性值，直接拷贝到新的zarr文件中;
#	（2）tl_faces: 同agents
#	（3）frames: 平移"agent_index_interval"和"traffic_light_faces_index_interval"后再拷贝;
#	（4）scenes: 平移"frame_index_interval"后再拷贝。

def filter_invalid_scenes(invalid_scenes_list: List[int], scenes: ZArray, frames: ZArray, agents: ZArray, tl_faces: ZArray, output_path: str) -> None:
	# get invalid scenes, frames, agents, tl_faces
	invalid_frames_list = [scenes[scene_id]["frame_index_interval"] for scene_id in invalid_scenes_list]
	invalid_agents_list = []
	invalid_tl_faces_list = [] 
	for invalid_frames in invalid_frames_list:
		frame_start_id = invalid_frames[0]
		frame_end_id = invalid_frames[1] - 1

		agent_start_id = frames[frame_start_id]["agent_index_interval"][0]
		agent_end_id = frames[frame_end_id]["agent_index_interval"][1]
		invalid_agents_list.append((agent_start_id, agent_end_id))

		tl_faces_start_id = frames[frame_start_id]["traffic_light_faces_index_interval"][0]
		tl_faces_end_id = frames[frame_end_id]["traffic_light_faces_index_interval"][1]
		invalid_tl_faces_list.append((tl_faces_start_id, tl_faces_end_id))

	# get valid scenes, frames, agents, tl_faces
	# scenes
	valid_scenes_start_id = 0
	valid_scenes_end_id = -1
	valid_scenes_range_list = []
	for invalid_scene_id in invalid_scenes_list:
		valid_scenes_start_id = valid_scenes_end_id + 1
		valid_scenes_end_id = invalid_scene_id
		valid_scenes_range = (valid_scenes_start_id, valid_scenes_end_id)
		valid_scenes_range_list.append(valid_scenes_range)
	if valid_scenes_end_id < len(scenes):
		valid_scenes_range_list.append((valid_scenes_end_id + 1, len(scenes)))

	# frames
	valid_frames_start_id = 0
	valid_frames_end_id = 0
	valid_frames_range_list = []
	for invalid_frames_id in invalid_frames_list:
		valid_frames_start_id = valid_frames_end_id
		valid_frames_end_id = invalid_frames_id[0]
		valid_frames_range = (valid_frames_start_id, valid_frames_end_id)
		valid_frames_range_list.append(valid_frames_range)
		valid_frames_end_id = invalid_frames_id[1]
	if valid_frames_end_id < len(frames):
		valid_frames_range_list.append((valid_frames_end_id, len(frames)))

	# agents
	valid_agents_start_id = 0
	valid_agents_end_id = 0
	valid_agents_range_list = []
	for invalid_agents_id in invalid_agents_list:
		valid_agents_start_id = valid_agents_end_id
		valid_agents_end_id = invalid_agents_id[0]
		valid_agents_range = (valid_agents_start_id, valid_agents_end_id)
		valid_agents_range_list.append(valid_agents_range)
		valid_agents_end_id = invalid_agents_id[1]
	if valid_agents_end_id < len(agents):
		valid_agents_range_list.append((valid_agents_end_id, len(agents)))
	valid_agents_list = [agents[valid_agents_range[0]:valid_agents_range[1]] for valid_agents_range in valid_agents_range_list]

	# tl_faces
	valid_tl_faces_start_id = 0
	valid_tl_faces_end_id = 0
	valid_tl_faces_range_list = []
	for invalid_tl_faces_id in invalid_tl_faces_list:
		valid_tl_faces_start_id = valid_tl_faces_end_id
		valid_tl_faces_end_id = invalid_tl_faces_id[0]
		valid_tl_faces_range = (valid_tl_faces_start_id, valid_tl_faces_end_id)
		valid_tl_faces_range_list.append(valid_tl_faces_range)
		valid_tl_faces_end_id = invalid_tl_faces_id[1]
	if valid_tl_faces_end_id < len(tl_faces):
		valid_tl_faces_range_list.append((valid_tl_faces_end_id, len(tl_faces)))
	valid_tl_faces_list = [tl_faces[valid_tl_faces_range[0]:valid_tl_faces_range[1]] for valid_tl_faces_range in valid_tl_faces_range_list]

	# shift agent index and traffic light index for valid frames
	shift_size_agents = 0
	shift_size_tl_faces = 0 
	valid_chunks = len(valid_scenes_range_list)

	first_valid_frames_chunk_range = valid_frames_range_list[0]
	first_valid_frames_chunk = frames[first_valid_frames_chunk_range[0]:first_valid_frames_chunk_range[1]]
	valid_frames_list = [first_valid_frames_chunk]
	 
	for i in range(1, valid_chunks):  # no need to shift the first chunk
		valid_frames_range = valid_frames_range_list[i]
		valid_frames = frames[valid_frames_range[0]:valid_frames_range[1]]

		num_invalid_agents = invalid_agents_list[i - 1][1] - invalid_agents_list[i - 1][0]
		num_invalid_tl_faces = invalid_tl_faces_list[i - 1][1] - invalid_tl_faces_list[i - 1][0]

		shift_size_agents += num_invalid_agents
		shift_size_tl_faces += num_invalid_tl_faces

		# shift agent_index_interval and traffic_light_faces_index_interval for all valid frames
		valid_frames["agent_index_interval"] -= shift_size_agents
		valid_frames["traffic_light_faces_index_interval"] -= shift_size_tl_faces
		valid_frames_list.append(valid_frames)

	# shift frame index for valid scenes
	shift_size = 0

	first_valid_scenes_chunk_range = valid_scenes_range_list[0]
	first_valid_scenes_chunk = scenes[first_valid_scenes_chunk_range[0]:first_valid_scenes_chunk_range[1]] 
	valid_scenes_list = [first_valid_scenes_chunk]

	for i in range(1, valid_chunks):
		valid_scenes_range = valid_scenes_range_list[i]
		valid_scenes = scenes[valid_scenes_range[0]:valid_scenes_range[1]]
		num_invalid_frames = invalid_frames_list[i - 1][1] - invalid_frames_list[i - 1][0]
		shift_size += num_invalid_frames
		# shift frame_index_interval for all valid scenes
		valid_scenes["frame_index_interval"] -= shift_size
		valid_scenes_list.append(valid_scenes)

	# concatenate valid scenes, frames, agents, tl_faces 
	valid_scenes_np = np.concatenate(valid_scenes_list)
	valid_frames_np = np.concatenate(valid_frames_list) 
	valid_agents_np = np.concatenate(valid_agents_list)
	valid_tl_faces_np = np.concatenate(valid_tl_faces_list)

	# save to a new zarr file    
	valid_scenes = zarr.array(valid_scenes_np)
	valid_frames = zarr.array(valid_frames_np)
	valid_agents = zarr.array(valid_agents_np)
	valid_tl_faces = zarr.array(valid_tl_faces_np)

	print(f"Start to save to new zarr: {output_path}")
	zarr.save(output_path, scenes=valid_scenes, frames=valid_frames, agents=valid_agents, traffic_light_faces=valid_tl_faces)
	print("Done!")


