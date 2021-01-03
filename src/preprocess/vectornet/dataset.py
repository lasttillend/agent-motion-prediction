import bisect
from collections import Counter, defaultdict
from typing import List
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset

from l5kit.data import ChunkedDataset, LocalDataManager, get_frames_slice_from_scenes
from l5kit.rasterization.rasterizer_builder import _load_metadata
from l5kit.geometry.transform import rotation33_as_yaw
from l5kit.sampling.agent_sampling import get_agent_context, get_relative_poses

from preprocess.vectornet.custom_map import CustomMapAPI
from preprocess.vectornet.vectorizer import Vectorizer
from preprocess.vectornet.polyline import Vector, Polyline
from preprocess.vectornet.lane import Lane

MIN_FRAME_HISTORY = 10
MIN_FRAME_FUTURE = 1

class LyftDataset(Dataset):
    
    def __init__(self, 
                 cfg: dict,
                 zarr_dataset: ChunkedDataset,
                 map_api: CustomMapAPI,
                 vectorizer: Vectorizer):
        self.cfg = cfg
        self.dataset = zarr_dataset
        self.map_api = map_api
        self.vectorizer = vectorizer

        self.history_num_frames = self.cfg["model_params"]["history_num_frames"]
        self.future_num_frames = self.cfg["model_params"]["future_num_frames"]
        
        # Mask out invalid agents.
        agents_mask = mask_agents(self.dataset, self.history_num_frames, self.future_num_frames)
        self.agents_indices = np.nonzero(agents_mask)[0]
        
        self.cumulative_sizes = self.dataset.scenes["frame_index_interval"][:, 1]
        self.cumulative_sizes_agents = self.dataset.frames["agent_index_interval"][:, 1]
        
        
    def __len__(self):
        return len(self.agents_indices)
    
    def __getitem__(self, index: int):
        index = self.agents_indices[index]
        track_id = self.dataset.agents[index]["track_id"]
        frame_index = bisect.bisect_right(self.cumulative_sizes_agents, index)
        scene_index = bisect.bisect_right(self.cumulative_sizes, frame_index)
        
        if scene_index == 0:
            state_index = frame_index
        else:
            state_index = frame_index - self.cumulative_sizes[scene_index - 1]
        return self.get_frame(scene_index, state_index, track_id=track_id)
    
    def get_frame(self, scene_index: int, state_index: int, track_id: int) -> dict:        
        frames = self.dataset.frames[get_frames_slice_from_scenes(self.dataset.scenes[scene_index])]
        agents = self.dataset.agents
        tl_faces = self.dataset.tl_faces
        (
            history_frames,
            future_frames,
            history_agents,
            future_agents,
            history_tl_faces,
            future_tl_faces,
        ) = get_agent_context(state_index, frames, agents, tl_faces, self.history_num_frames, self.future_num_frames)
             
        cur_frame = history_frames[0]
        cur_agents = history_agents[0]
        
        ## 1. Find all map elements around the ego car (in history frames), which will be vectorized.
        elements_need_vectorized = defaultdict(set)  # element type (str) -> a set of element ids (str)
        element_types = ["lane"]
        all_types_elements = {element_type: self.map_api.get_elements_from_layer(element_type) for element_type in element_types}

        for frame in tqdm(history_frames, desc="Find map elements needing vectorized"): 
            ego_position = frame["ego_translation"][:2]  # (x, y) coordinate of the ego car in current frame, note this is in world coordinate
            ego_yaw = rotation33_as_yaw(frame["ego_rotation"])  # yaw in radian
            # We define a new coordinate system called "vector coordinate system" where the origin is placed at ego car's current position and the x-axis is the moving direction of the ego car.
            world_from_vector = calc_world_from_vector_matrix(ego_position, ego_yaw)
            # vector_from_world = np.linalg.inv(world_from_vector)
            ego_search_box = calc_ego_car_search_box(self.cfg["vector_params"]["vector_range"], self.cfg["vector_params"]["ego_center"])  # search bbox is aligned in the moving direction of ego car

            for element_type in element_types:
                all_elements = all_types_elements[element_type]
                for element in all_elements:
                    element_id = self.map_api.id_as_str(element.id)
                    element_bbox = self.map_api.get_bbox(element_type, element_id)  # [xmin, ymin, xmax, ymax] in world coordinate
                    element_bbox_transformed = transform_bbox(element_bbox, world_from_vector)  # transform the bounding box to be aligned with the search box and change coordinate into agent coordinate
                    if is_overlapping2D(ego_search_box, element_bbox_transformed) and element_id not in elements_need_vectorized[element_type]:
                        elements_need_vectorized[element_type].add(element_id) 

        ## 2. Find history trajectories for agents in the current frame, which will be vectorized as well.
        cur_agents_track_ids = cur_agents["track_id"]
        cur_agents_history_trajs = dict()  # track_id -> Tuple[history_trajs (np.ndarray of shape=[num_times, 2]), timestamps]
        for agent_track_id in tqdm(cur_agents_track_ids, desc="Find agent history trajectories needing vectorized"):
            history_trajs = []
            timestamps = []
            for i in range(len(history_agents)):
                tmp_frame = history_frames[i]
                tmp_agents = history_agents[i] 
                agent_occur_in_this_frame = tmp_agents[tmp_agents["track_id"] == agent_track_id]
                if len(agent_occur_in_this_frame) > 0:
                    agent_centroid = agent_occur_in_this_frame[0]["centroid"]
                    timestamp = tmp_frame["timestamp"]
                    history_trajs.append(agent_centroid)
                    timestamps.append(timestamp)
            history_trajs_np = np.array(history_trajs[::-1])  # reverse the trajectory so that it starts from past to current
            timestamps = timestamps[::-1]
            cur_agents_history_trajs[agent_track_id] = (history_trajs_np, timestamps) 

        ## 3. Vectorize map elements to prepare for creating map features.
        map_vector_sets = defaultdict(list)  # element_type -> List[Polyline] 
        for element_type, element_ids in tqdm(elements_need_vectorized.items(), desc="Vectorizing map element"):
            # Transform to custom map element, which has a vectorize method.
            element_objs = []
            for element_id in element_ids:
                if element_type == "lane":
                    element_objs.append(Lane(self.map_api[element_id]))
                else:
                    raise ValueError(f"Custom {element_type} map element is not defined!")
            # Vectorize each custom map element by invoking the vectorize method.            
            for element_obj in element_objs:
                polyline_id = self.vectorizer.polyline_ids[element_type]
                polylines = element_obj.vectorize(self.map_api, polyline_id)
                num_polylines = element_obj.num_polylines()
                self.vectorizer.polyline_ids["element_type"] += num_polylines 

                map_vector_sets[element_type].extend(polylines) 
        
        for k, v in map_vector_sets.items():
            print(f"Number of {k} map elements vectorized: {len(v)}")
            

        ## 4. Vectorize trajectories to prepare for creating trajectory features.
        trajs_vector_sets = []  # List[Polyline]
        for agent_track_id, (history_trajs, timestamps) in tqdm(cur_agents_history_trajs.items(), desc="Vectorize trajectories"):
            lines = []
            for i in range(len(history_trajs) - 1):
                start = history_trajs[i] 
                end = history_trajs[i + 1]
                timestamp = timestamps[i]
                trajs_attributes = dict(OBJECT_TYPE="trajectory", timestamp=timestamp)
                polyline_id = self.vectorizer.polyline_ids["trajectory"]
                vector = Vector(start, end, trajs_attributes, polyline_id)
                lines.append(vector)
            history_trajs_polyline = Polyline(lines, self.vectorizer.polyline_ids["trajectory"], "trajectory")
            self.vectorizer.polyline_ids["trajectory"] += 1
            trajs_vector_sets.append(history_trajs_polyline)
        print(f"Number of agent history trajectories vectorized {len(trajs_vector_sets)}")

        ## 5. Find the position of the selected agent, and create agent coordinate system by setting this target agent's current position as origin, and current moving direction as x-axis.

        ##  Extract future positions of selected agent and set the offsets as our prediction target. 
        ##  Create a target avaliability list to record whether the target agent occurred in each frame of future_num_frames, set each element to 1 if it was and 0 otherwise. 
        selected_track_id = track_id
        selected_agent = cur_agents[cur_agents["track_id"] == selected_track_id][0] 
        selected_agent_position = selected_agent["centroid"]
        print(f"selected agent centroid: {selected_agent_position}")
        selected_agent_yaw = float(selected_agent["yaw"])
        selected_agent_extend = selected_agent["extent"]

        world_from_agent = calc_world_from_agent_matrix(selected_agent_position, selected_agent_yaw)
        agent_from_world = np.linalg.inv(world_from_agent)
        
        future_positions_m, future_yaws_rad, future_extents, future_avaliabilities = get_relative_poses(self.future_num_frames, future_frames, selected_track_id, future_agents, agent_from_world, selected_agent_yaw) 

        ## 6. Create map features and trajectory features. 
        # Map feature
        map_features = [] 
        for element_type, element_polylines in tqdm(map_vector_sets.items(), desc="Creating map feature:"):
            for polyline in element_polylines:
                vector_features = []
                for vector in polyline:
                    start = vector.start
                    end = vector.end
                    new_start = transform_point(start, agent_from_world)
                    new_end = transform_point(end, agent_from_world)
                    all_attributes = vector.attributes
                    if element_type == "lane":
                        turn_type = all_attributes["turn_type"]
                        vector_feature = np.r_[new_start, new_end, turn_type] # [x_start, y_start, x_end, y_end, turn_type]
                    else:
                        raise ValueError(f"{element_type} map element has not been defined")
                    vector_features.append(vector_feature)
                vector_features_tensor = torch.from_numpy(np.array(vector_features)).float()
                map_features.append(vector_features_tensor)

        # Trajectory feature 
        trajs_features = [] 
        for trajs_polyline in tqdm(trajs_vector_sets, desc="Creating trajectory feature:"):
            vector_features = []
            for vector in trajs_polyline:
                start = vector.start
                end = vector.end
                new_start = transform_point(start, agent_from_world)
                new_end = transform_point(end, agent_from_world)
                all_attributes = vector.attributes
                timestamp = all_attributes["timestamp"]
                vector_feature = np.r_[new_start, new_end, timestamp]  # [x_start, y_start, x_end, y_end, timestamp]
                vector_features.append(vector_feature)
            vector_features_tensor = torch.from_numpy(np.array(vector_features)).float()
            trajs_features.append(vector_features_tensor)
            
        data = dict()
        data["map_feature"] = map_features  # List[tensor]
        data["traj_feature"] = trajs_features  # List[tensor]

        data["scene_index"] = scene_index
        data["frame_index"] = state_index
        data["timestamp"] = frames[state_index]["timestamp"]
        data["track_id"] = track_id       

        data["target_positions"] = future_positions_m
        data["target_yaws"] = future_yaws_rad
        data["target_availabilities"] = future_avaliabilities
        data["agent_from_world"] = agent_from_world
        data["centroid"] = selected_agent_position
        data["yaw"] = selected_agent_yaw
        data["extent"] = selected_agent_extend
        
        print("target positions shape:", future_positions_m.shape) 
        print("target avails: ", np.sum(future_avaliabilities != 0))
        print("target positions[:10]:", future_positions_m[:10, :])
        
        return data
            
################## Utils ##############################            
            
def mask_agents(zarr_dataset: ChunkedDataset, history_num_frames: int, future_num_frames: int) -> List[int]:
    scenes = zarr_dataset.scenes
    frames = zarr_dataset.frames
    agents = zarr_dataset.agents
    
    agent_kept_ids = [] 
    for scene in tqdm(scenes, desc="Masking agents!"):
        middle_start = scene["frame_index_interval"][0] + history_num_frames
        middle_end = scene["frame_index_interval"][1] - future_num_frames  
        # Mask out invalid agents in middle. Here a valid agent must satisfy three conditions:
        # 1. it is selected from the current frame;
        # 2. it must appear at least MIN_FRAME_HISTORY times from current_frame - history_num_frames to current_frame (excluded);
        # 3. it must appear at least MIN_FRAME_FUTURE times from current_frame + 1 to current_frame + future_num_frames (included).
        for frame_id in range(middle_start, middle_end):
            current_frame = frames[frame_id]
            current_frame_agents_start_id = current_frame["agent_index_interval"][0]
            current_frame_agents_end_id = current_frame["agent_index_interval"][1]
            agents_in_current_frame = agents[current_frame_agents_start_id:current_frame_agents_end_id]["track_id"]
            
            history_frames = frames[frame_id - history_num_frames:frame_id]
            future_frames = frames[frame_id + 1:frame_id + future_num_frames + 1]
            
            history_agents_start_id = history_frames[0]["agent_index_interval"][0]
            history_agents_end_id = history_frames[-1]["agent_index_interval"][1]
            future_agents_start_id = future_frames[0]["agent_index_interval"][0]
            future_agents_end_id = future_frames[-1]["agent_index_interval"][1]
            
            history_agents = agents[history_agents_start_id:history_agents_end_id]["track_id"]
            history_agents_cnt = Counter(history_agents)
            future_agents = agents[future_agents_start_id:future_agents_end_id]["track_id"]
            future_agents_cnt = Counter(future_agents)
            
            cond1_satisfy = agents_in_current_frame
            cond2_satisfy = np.array([track_id for track_id, time in history_agents_cnt.items() if time >= MIN_FRAME_HISTORY])
            cond3_satisfy = np.array([track_id for track_id, time in future_agents_cnt.items() if time >= MIN_FRAME_FUTURE])
            
            all_satisfy = np.intersect1d(np.intersect1d(cond1_satisfy, cond2_satisfy), cond3_satisfy)
            
            for track_id in all_satisfy:
                agent_ids = np.arange(current_frame_agents_start_id, current_frame_agents_end_id)
                agent_kept_id = agent_ids[agents_in_current_frame == track_id]
                agent_kept_ids.append(agent_kept_id)
    
    agent_mask = np.zeros(len(agents))
    agent_mask[agent_kept_ids] = 1
    
    return agent_mask
    
    
def calc_bbox_of_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """
    Calculate the bounding box of the given trajectory.
    
    Args:
        trajectory: np.ndarray, shape = [num_points, 2]
    
    Returns:
        bbox: np.ndarray, shape = [4, ]
    """
    xmin, ymin = min(trajectory[:, 0]), min(trajectory[:, 1])
    xmax, ymax = max(trajectory[:, 0]), max(trajectory[:, 1])
    
    bbox = np.array([xmin, ymin, xmax, ymax])
    
    return bbox        

def is_overlapping2D(box1: np.ndarray, box2: np.ndarray) -> bool:
    """
    Two axes aligned boxes (of any dimension) overlap if and only if the projections to all axes overlap. The projection to an axis is simply the coordinate range for that axis.
    See https://stackoverflow.com/questions/20925818/algorithm-to-check-if-two-boxes-overlap.
    
    Args:
        box1: np.ndarray of shape = [4, ], more specifically, [xmin, ymin, xmax, ymax]
        box2: np.ndarray of shape = [4, ]

    Returns:
        bool
    """
    return is_overlapping1D(box1[[0, 2]], box2[[0, 2]]) and is_overlapping1D(box1[[1, 3]], box2[[1, 3]])
    
def is_overlapping1D(interval1: np.ndarray, interval2: np.ndarray) -> bool:
    xmin1 = interval1[0]
    xmax1 = interval1[1]
    
    xmin2 = interval2[0]
    xmax2 = interval2[1]

    return xmax1 >= xmin2 and xmax2 >= xmin1 

def calc_world_from_vector_matrix(ego_position: np.ndarray, ego_yaw: float) -> np.ndarray:
    """
    Return the ego car's pose as a 3x3 matrix. This corresponds to world_from_vector matrix.
    
    Args:
        ego_position (np.ndarray): 2D coordinates of the ego car
        ego_yaw (float): yaw of the ego car
    
    Returns:
        (np.ndarray): 3x3 world_from_vector matrix
    """
    return np.array(
        [
            [np.cos(ego_yaw), -np.sin(ego_yaw), ego_position[0]],
            [np.sin(ego_yaw),  np.cos(ego_yaw), ego_position[1]],
            [0, 0, 1],
        ]
    )

def calc_ego_car_search_box(vector_range: List[int], ego_center: List[float]) -> np.ndarray:
    xmin = -ego_center[0] * vector_range[0]
    ymin = -ego_center[1] * vector_range[1]
    xmax = (1 - ego_center[0]) * vector_range[0]
    ymax = (1 - ego_center[1]) * vector_range[1]

    search_box = np.array([xmin, ymin, xmax, ymax])
    return search_box

def transform_point(point: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
    """
    Transform point coordinate into a new coordinate system.
    
    Args:
        point (np.ndarray): [x, y]
        transformation_matrix (np.ndarray): 3x3 matrix, coordinate change matrix, e.g., world_from_vector.

    Returns:
        (np.ndarray): [new_x, new_y]
    """
    point = np.r_[point, 1].reshape(-1, 1)  # append 1 to the point and make it a column vector so that it has shape = [3, 1]
    new_point = (transformation_matrix @ point).flatten()[:2]  # extract new_x, new_y
    return new_point


def rotate_point(point: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Args:
        point (np.ndarray): [x, y]
        rotation_matrix (np.ndarray): 2x2 rotation matrix
    
    Returns:
        (np.ndarray): [rot_x, rot_y]
    """
    point = point.reshape(-1, 1)  # column vector
    return (rotation_matrix @ point).flatten()

def transform_bbox(bbox: np.ndarray, world_from_vector: np.ndarray) -> np.ndarray:
    """
    Rotate the bounding box to be aligned with the ego car's search box and then change coordinate system from world into vector.
    
    Args:
        bbox (np.ndarray): [xmin, ymin, xmax, ymax]
        world_from_vector (np.ndarray): 3x3 coordinate change matrix

    returns:
        (np.ndarray): [new_xmin, new_yim, new_xmax, new_ymax]
    """
    # 1. Find the coordinates of 4 corners of bounding box
    lower_left = np.array([bbox[0], bbox[1]])
    lower_right = np.array([bbox[2], bbox[1]])
    upper_right = np.array([bbox[2], bbox[3]])
    upper_left = np.array([bbox[0], bbox[3]])
    corners = [lower_left, lower_right, upper_right, upper_left]
    
    # 2. Rotate the bounding box by rotating the 4 corners.
    rot_corners = []
    rotation_matrix = world_from_vector[:2, :2]
    for point in corners:
        rot_point = rotate_point(point, rotation_matrix)
        rot_corners.append(rot_point)
      
    # 3. Transform the coordinates of these corners using the vector_from_world matrix.
    vector_from_world = np.linalg.inv(world_from_vector)
    new_corners = []
    for rot_point in rot_corners:
        new_point = transform_point(rot_point, vector_from_world)
        new_corners.append(new_point)
    new_corners_np = np.array(new_corners)

    # 4. Recalculate minimum/maximum value of new x and new y.
    new_x_min = np.min(new_corners_np[:, 0])
    new_y_min = np.min(new_corners_np[:, 1])
    new_x_max = np.max(new_corners_np[:, 0])
    new_y_max = np.max(new_corners_np[:, 1])

    return np.array([new_x_min, new_y_min, new_x_max, new_y_max])
         
     
def calc_world_from_agent_matrix(agent_position: np.ndarray, agent_raw: float) -> np.ndarray:
    """
    Return the agent's pose as a 3x3 matrix. This corresponds to world_from_agent matrix.
    
    Args:
        ego_position (np.ndarray): 2D coordinates of the ego car
        ego_yaw (float): yaw of the ego car
    
    Returns:
        (np.ndarray): 3x3 world_from_vector matrix
    """
    return calc_world_from_vector_matrix(agent_position, agent_raw) 
