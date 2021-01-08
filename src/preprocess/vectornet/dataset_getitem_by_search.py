"""
Get map elment around ego car by search.
"""
import bisect
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import warnings

import numpy as np
from zarr import convenience

import torch
from torch.utils.data import Dataset

from l5kit.data import ChunkedDataset, LocalDataManager, get_frames_slice_from_scenes
from l5kit.rasterization.rasterizer_builder import _load_metadata
from l5kit.geometry.transform import rotation33_as_yaw
from l5kit.sampling.agent_sampling import get_agent_context, get_relative_poses
from l5kit.dataset.select_agents import TH_DISTANCE_AV, TH_EXTENT_RATIO, TH_YAW_DEGREE, select_agents

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
                 vectorizer: Vectorizer,
                 agents_mask: Optional[np.ndarray]=None,
                 min_frame_history: int=MIN_FRAME_HISTORY,
                 min_frame_future: int=MIN_FRAME_FUTURE):
        self.cfg = cfg
        self.dataset = zarr_dataset
        self.map_api = map_api
        self.vectorizer = vectorizer
        self.map_element_types = ["lane"]
        
        self.vector_map = self.generate_vector_map()
        self.all_vectors, self.vector_middle_points, self.vector_polyline_ids, self.vector_polylines = self.generate_vectors_info()

        self.history_num_frames = self.cfg["model_params"]["history_num_frames"]
        self.future_num_frames = self.cfg["model_params"]["future_num_frames"]
        
        self.cumulative_sizes = self.dataset.scenes["frame_index_interval"][:, 1]
        self.cumulative_sizes_agents = self.dataset.frames["agent_index_interval"][:, 1]   
            
        # Mask out invalid agents.
        if agents_mask is None:  # if not provided try to load it from the zarr
            agents_mask = self.load_agents_mask()
            past_mask = agents_mask[:, 0] >= min_frame_history
            future_mask = agents_mask[:, 1] >= min_frame_future
            agents_mask = past_mask * future_mask
            
        self.agents_indices = np.nonzero(agents_mask)[0]
        
    def load_agents_mask(self) -> np.ndarray:
        """
        Loads a boolean mask of the agent availability stored into the zarr. Performs some sanity check against cfg.
        Returns: a boolean mask of the same length of the dataset agents
        """
        agent_prob = self.cfg["vector_params"]["filter_agents_threshold"]

        agents_mask_path = Path(self.dataset.path) / f"agents_mask/{agent_prob}"
        if not agents_mask_path.exists():  # don't check in root but check for the path
            warnings.warn(
                f"cannot find the right config in {self.dataset.path},\n"
                f"your cfg has loaded filter_agents_threshold={agent_prob};\n"
                "but that value doesn't have a match among the agents_mask in the zarr\n"
                "Mask will now be generated for that parameter.",
                RuntimeWarning,
                stacklevel=2,
            )

            select_agents(
                self.dataset,
                agent_prob,
                th_yaw_degree=TH_YAW_DEGREE,
                th_extent_ratio=TH_EXTENT_RATIO,
                th_distance_av=TH_DISTANCE_AV,
            )

        agents_mask = convenience.load(str(agents_mask_path))  # note (lberg): this doesn't update root
        return agents_mask      
                
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
        ) = get_agent_context(state_index, frames, agents, tl_faces, self.history_num_frames, self.future_num_frames)  # history_frames includes current frame
             
        cur_frame = history_frames[0]
        cur_agents = history_agents[0]
        ego_position = cur_frame["ego_translation"][:2]  # (x, y) coordinate of the ego car in current frame, note this is in world coordinate
        ego_yaw = rotation33_as_yaw(cur_frame["ego_rotation"])  # yaw in radian
 
        # 1. Find map elements around the ego car in current frame.
        total_num_polylines = self.cfg["vector_params"]["num_polylines"]  
        map_polylines = self.get_map_polylines_around_ego_car(ego_position, total_num_polylines)  # List[Polyline]

        # 2. Find agents around the ego car, whose history trajectory will be vectorized.
        total_num_agents = self.cfg["vector_params"]["num_agents"]
        cur_agents_track_ids = cur_agents["track_id"]
        trajectory_polylines = self.get_trajectory_polylines_around_ego_car(ego_position, total_num_agents, cur_agents_track_ids, cur_agents,  history_frames, history_agents)

        ## 3. Find the position of the selected agent, and create agent coordinate system by setting this target agent's current position as origin, and current moving direction as x-axis.

        ##  Extract future positions of selected agent and set the offsets as our prediction target. 
        ##  Create a target avaliability list to record whether the target agent occurred in each frame of future_num_frames, set each element to 1 if it was and 0 otherwise. 
        selected_track_id = track_id
        selected_agent = cur_agents[cur_agents["track_id"] == selected_track_id][0] 
        selected_agent_position = selected_agent["centroid"]
        selected_agent_yaw = float(selected_agent["yaw"])
        selected_agent_extend = selected_agent["extent"]

        world_from_agent = calc_world_from_agent_matrix(selected_agent_position, selected_agent_yaw)
        agent_from_world = np.linalg.inv(world_from_agent)
        
        future_positions_m, future_yaws_rad, future_extents, future_avaliabilities = get_relative_poses(self.future_num_frames, future_frames, selected_track_id, future_agents, agent_from_world, selected_agent_yaw) 

        ## 4. Create map features and trajectory features. 
        # Map feature
        map_features = [] 
        for polyline in map_polylines:
            vector_features = []
            for vector in polyline:
                start = vector.start
                end = vector.end
                new_start = transform_point(start, agent_from_world)
                new_end = transform_point(end, agent_from_world)
                all_attributes = vector.attributes
                element_type = polyline.object_type
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
        for trajs_polyline in trajectory_polylines:
            vector_features = []
            for vector in trajs_polyline:
                start = vector.start
                end = vector.end
                if not trajs_polyline.is_padded:
                    new_start = transform_point(start, agent_from_world)
                    new_end = transform_point(end, agent_from_world)
                    is_padded = 0
                else:
                    new_start = start.copy()
                    new_end = start.copy()
                    is_padded = 1
                all_attributes = vector.attributes
                timestamp = all_attributes["timestamp"]
                vector_feature = np.r_[new_start, new_end, timestamp, is_padded]  # [x_start, y_start, x_end, y_end, timestamp, is_padded]
                vector_features.append(vector_feature)
           
            vector_features_tensor = torch.from_numpy(np.array(vector_features)).float()
            trajs_features.append(vector_features_tensor)
            
        data = dict()
        data["map_feature"]  = torch.stack(map_features)     #  [num_map_polylines, num_points_on_polyline, num_map_attributes]
        data["traj_feature"] = torch.stack(trajs_features)   #  [num_trajectory_polylines, history_num_frames, num_trajectory_attribute]

        data["scene_index"] = scene_index
        data["frame_index"] = state_index
        data["timestamp"] = frames[state_index]["timestamp"]
        data["track_id"] = np.int64(track_id)  # always a number to avoid crasing torch     

        data["target_positions"] = future_positions_m
        data["target_yaws"] = future_yaws_rad
        data["target_availabilities"] = future_avaliabilities
        data["agent_from_world"] = agent_from_world
        data["centroid"] = selected_agent_position
        data["yaw"] = selected_agent_yaw
        data["extent"] = selected_agent_extend
        
        return data
    
    def generate_vector_map(self) -> Dict[str, List[Polyline]]:
        vector_map = defaultdict(list)  # element_type -> List[Polyline]
        
        for element_type in self.map_element_types:
            if element_type == "lane":
                lanes = self.map_api.get_elements_from_layer("lane")
                element_objs = [Lane(lane) for lane in lanes]
            else:
                raise ValueError(f"Custom {element_type} map element is not defined!")
            for element_obj in tqdm(element_objs, desc="Generating vector map..."):
                polyline_id = self.vectorizer.polyline_ids[element_type]
                polylines = element_obj.vectorize(self.map_api, polyline_id)
                num_polylines = element_obj.num_polylines()
                self.vectorizer.polyline_ids[element_type] += num_polylines
                vector_map[element_type].extend(polylines)
        
        return vector_map
        
    def generate_vectors_info(self) -> Tuple[List[Vector], np.ndarray, List[int], List[Polyline]]:
        """
        Returns all vectors, their middle points array, corresponding polyline and id.
        
        Note:
            middle_pts_np: np.ndarray (shape=[num_all_vectors, 2])
        """
        vectors = []
        middle_pts = []
        vector_polyline_ids = []
        vector_polylines = []
        
        for _, polylines in self.vector_map.items():
            for polyline in tqdm(polylines, desc="Generating vector info..."):
                for vector in polyline:
                    start = vector.start
                    end = vector.end
                    middle = (start + end) / 2

                    vectors.append(vector)
                    middle_pts.append(middle)
                    vector_polyline_ids.append(polyline.ids)
                    vector_polylines.append(polyline)
                    
        middle_pts_np = np.array(middle_pts)
        
        return vectors, middle_pts_np, vector_polyline_ids, vector_polylines
    
    def get_map_polylines_around_ego_car(self, ego_position: np.ndarray, num_polylines: int) -> List[Polyline]:
        # Use Einsum to calculate distance between ego position and each vector.
        # Ref: https://stackoverflow.com/questions/40996957/calculate-distance-between-numpy-arrays
        subs = self.vector_middle_points[:, None] - ego_position
        dist_squared = np.einsum("ijk,ijk->ij", subs, subs).flatten()
        sorted_ids = np.argsort(dist_squared)
        
        polyline_cnt = 0
        polylines_around = []
        polylines_ids = defaultdict(set)  # element_type -> polyline ids
        for i in sorted_ids:
            vector = self.all_vectors[i]
            object_type = vector.object_type
            polyline_id = self.vector_polyline_ids[i]
            polyline = self.vector_polylines[i]
            if polyline_id not in polylines_ids[object_type]:
                polylines_around.append(polyline)
                polylines_ids[object_type].add(polyline_id)
                polyline_cnt += 1

            if polyline_cnt >= num_polylines:
                break

        return polylines_around

    def get_trajectory_polylines_around_ego_car(self, ego_position: np.ndarray, num_agents: int, cur_agents_track_ids: List[int], cur_agents: np.ndarray, history_frames: np.ndarray, history_agents: np.ndarray) -> List[Polyline]:
        cur_agents_history_trajs = dict()  # track_id -> Tuple[history_trajs (np.ndarray of shape=[num_history_frames, 2]), timestamps]
        for agent_track_id in cur_agents_track_ids:
            history_trajs = []
            timestamps = []
            for i in range(len(history_agents)):
                tmp_frame = history_frames[i]
                tmp_agents = history_agents[i]
                agent_occur_in_this_frame = tmp_agents[tmp_agents["track_id"] == agent_track_id]
                if len(agent_occur_in_this_frame) > 0:
                    agent_centroid = agent_occur_in_this_frame[0]["centroid"]
                    timestamp = tmp_frame["timestamp"]
                else:
                    agent_centroid = np.array([0, 0])
                    timestamp = 0
                history_trajs.append(agent_centroid)
                timestamps.append(timestamp)
            # Agent may occur less than history_num_frames times in the past, so we need to pad.
            for _ in range(len(history_trajs), self.history_num_frames + 1):
                agent_centroid = np.array([0, 0])
                timestamp = 0
                history_trajs.append(agent_centroid)
                timestamps.append(timestamp)
            history_trajs_np = np.array(history_trajs[::-1])  # reverse the trajectory so that it starts from past to current
            timestamps = timestamps[::-1]
            cur_agents_history_trajs[agent_track_id] = (history_trajs_np, timestamps)
                    
        # Use Einsum to calculate distance between ego position and agent.
        subs = cur_agents["centroid"][:, None] - ego_position
        dist_squared = np.einsum("ijk,ijk->ij", subs, subs).flatten()
        sorted_ids = np.argsort(dist_squared)
        sorted_agents_track_ids = list(np.array(cur_agents_track_ids)[sorted_ids])

        # Vectorize trajectories.
        polyline_cnt = 0
        trajs_polylines = []  # List[Polyline]
        for agent_track_id in sorted_agents_track_ids:
            history_trajs, timestamps = cur_agents_history_trajs[agent_track_id]
            lines = []
            for i in range(len(history_trajs) - 1):
                start = history_trajs[i]
                end = history_trajs[i + 1]
                timestamp = timestamps[i]
                trajs_attributes = dict(OBJECT_TYPE="trajectory", timestamp=timestamp, is_padded=0)
                polyline_id = self.vectorizer.polyline_ids["trajectory"]
                vector = Vector(start, end, trajs_attributes, polyline_id)
                lines.append(vector)
            history_trajs_polyline = Polyline(lines, self.vectorizer.polyline_ids["trajectory"], "trajectory")
            self.vectorizer.polyline_ids["trajectory"] += 1
            trajs_polylines.append(history_trajs_polyline)

            polyline_cnt += 1
            if polyline_cnt >= num_agents:
                break

        # Not enough agents in current frame, pad some void trajectories by setting is_padded attribute to 1 and all other attributes to 0.
        if polyline_cnt < num_agents:
            for _ in range(num_agents - polyline_cnt):
                for _ in range(len(history_trajs) - 1):
                    start = np.array([0, 0])
                    end = np.array([0, 0])
                    timestamp = 0
                    trajs_attributes = dict(OBJECT_TYPE="trajectory", timestamp=timestamp, is_padded=1)
                    polyline_id = self.vectorizer.polyline_ids["trajectory"]
                    vector = Vector(start, end, trajs_attributes, timestamp=timestamp, is_padded=1)
                padded_polyline = Polyline(lines, self.vectorizer.polyline_ids["trajcetory"], "trajectory")
                padded_polyline.is_padded = 1
                self.vectorizer.polyline_ids["trajectory"] += 1
                trajs_polylines.append(padded_polyline)
        
        return trajs_polylines    
    
    
################## Utils ##############################            
            
# def mask_agents(zarr_dataset: ChunkedDataset, history_num_frames: int, future_num_frames: int) -> List[int]:
#     scenes = zarr_dataset.scenes
#     frames = zarr_dataset.frames
#     agents = zarr_dataset.agents
    
#     agent_kept_ids = [] 
#     for scene in tqdm(scenes, desc="Masking agents!"):
#         middle_start = scene["frame_index_interval"][0] + history_num_frames
#         middle_end = scene["frame_index_interval"][1] - future_num_frames  
#         # Mask out invalid agents in middle. Here a valid agent must satisfy three conditions:
#         # 1. it is selected from the current frame;
#         # 2. it must appear at least MIN_FRAME_HISTORY times from current_frame - history_num_frames to current_frame (excluded);
#         # 3. it must appear at least MIN_FRAME_FUTURE times from current_frame + 1 to current_frame + future_num_frames (included).
#         for frame_id in range(middle_start, middle_end):
#             current_frame = frames[frame_id]
#             current_frame_agents_start_id = current_frame["agent_index_interval"][0]
#             current_frame_agents_end_id = current_frame["agent_index_interval"][1]
#             agents_in_current_frame = agents[current_frame_agents_start_id:current_frame_agents_end_id]["track_id"]
            
#             history_frames = frames[frame_id - history_num_frames:frame_id]
#             future_frames = frames[frame_id + 1:frame_id + future_num_frames + 1]
            
#             history_agents_start_id = history_frames[0]["agent_index_interval"][0]
#             history_agents_end_id = history_frames[-1]["agent_index_interval"][1]
#             future_agents_start_id = future_frames[0]["agent_index_interval"][0]
#             future_agents_end_id = future_frames[-1]["agent_index_interval"][1]
            
#             history_agents = agents[history_agents_start_id:history_agents_end_id]["track_id"]
#             history_agents_cnt = Counter(history_agents)
#             future_agents = agents[future_agents_start_id:future_agents_end_id]["track_id"]
#             future_agents_cnt = Counter(future_agents)
            
#             cond1_satisfy = agents_in_current_frame
#             cond2_satisfy = np.array([track_id for track_id, time in history_agents_cnt.items() if time >= MIN_FRAME_HISTORY])
#             cond3_satisfy = np.array([track_id for track_id, time in future_agents_cnt.items() if time >= MIN_FRAME_FUTURE])
            
#             all_satisfy = np.intersect1d(np.intersect1d(cond1_satisfy, cond2_satisfy), cond3_satisfy)
            
#             for track_id in all_satisfy:
#                 agent_ids = np.arange(current_frame_agents_start_id, current_frame_agents_end_id)
#                 agent_kept_id = agent_ids[agents_in_current_frame == track_id]
#                 agent_kept_ids.append(agent_kept_id)
    
#     agent_mask = np.zeros(len(agents))
#     agent_mask[agent_kept_ids] = 1
    
#     return agent_mask
    
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

def calc_world_from_ego_matrix(ego_position: np.ndarray, ego_yaw: float) -> np.ndarray:
    """
    Return the ego car's pose as a 3x3 matrix. This corresponds to world_from_ego matrix.
    
    Args:
        ego_position (np.ndarray): 2D coordinates of the ego car
        ego_yaw (float): yaw of the ego car
    
    Returns:
        (np.ndarray): 3x3 world_from_ego matrix
    """
    return np.array(
        [
            [np.cos(ego_yaw), -np.sin(ego_yaw), ego_position[0]],
            [np.sin(ego_yaw),  np.cos(ego_yaw), ego_position[1]],
            [0, 0, 1],
        ]
    )

def calc_ego_car_search_box(world_from_ego: np.ndarray, xy_range: List[int], ego_center: List[float]) -> np.ndarray:
    """
    We first find the local coordinates of the four corners of search box, i.e., in ego coordinate system which is centered around ego car with xy range given by xy_range list, and aligned to the moving direction of ego car.
    Then transform these 4 corner coordinates into world coordinate system and find min/max of x/y, i.e., the bounding box of the search box.
    """ 
    # Find search box in ego coordinate system.
    xmin = -ego_center[0] * xy_range[0]
    ymin = -ego_center[1] * xy_range[1]
    xmax = (1 - ego_center[0]) * xy_range[0]
    ymax = (1 - ego_center[1]) * xy_range[1]
    lower_left = np.array([xmin, ymin])
    lower_right = np.array([xmax, ymin])
    upper_right = np.array([xmax, ymax])
    upper_left = np.array([xmin, ymax])
    corners = [lower_left, lower_right, upper_right, upper_left]
    
    # Transform coordinates of the corners into world coordinate system.  
    new_corners = []
    for corner in corners:
        new_corner = transform_point(corner, world_from_ego)
        new_corners.append(new_corner)
    new_corners_np = np.array(new_corners)    
    
    # Calculate min/max values of new x/y.
    new_x_min = np.min(new_corners_np[:, 0])
    new_y_min = np.min(new_corners_np[:, 1])
    new_x_max = np.max(new_corners_np[:, 0])
    new_y_max = np.max(new_corners_np[:, 1])
    
    return np.array([new_x_min, new_y_min, new_x_max, new_y_max])

def transform_point(point: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
    """
    Transform point coordinate into a new coordinate system.
    
    Args:
        point (np.ndarray): [x, y]
        transformation_matrix (np.ndarray): 3x3 matrix, coordinate change matrix, e.g., world_from_ego.

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

# def transform_bbox(bbox: np.ndarray, world_from_ego: np.ndarray) -> np.ndarray:
#     """
#     Rotate the bounding box to be aligned with the ego car's search box and then change coordinate system from world into vector.

#     Note: the coordinate of bbox is in world coordinate system.f
    
#     Args:
#         bbox (np.ndarray): [xmin, ymin, xmax, ymax]
#         world_from_ego (np.ndarray): 3x3 coordinate change matrix

#     returns:
#         (np.ndarray): [new_xmin, new_yim, new_xmax, new_ymax]
#     """
#     # 1. Find coordinates of the 4 corners of bounding box in its local coordinate system (origin at its center).
#     center_x = (bbox[0] + bbox[2]) / 2
#     center_y = (bbox[1] + bbox[3]) / 2
#     center = np.array([center_x, center_y])
#     lower_left = np.array([bbox[0], bbox[1]])  - center
#     lower_right = np.array([bbox[2], bbox[1]]) - center
#     upper_right = np.array([bbox[2], bbox[3]]) - center
#     upper_left = np.array([bbox[0], bbox[3]])  - center
#     corners = [lower_left, lower_right, upper_right, upper_left]
    
#     # 2. Rotate the bounding box around its center to align with the ego car's moving direction. 
#     rot_corners = []
#     rotation_matrix = world_from_ego[:2, :2]
#     for point in corners:
#         rot_point = rotate_point(point, rotation_matrix)
#         rot_corners.append(rot_point)
#     rot_corners_np = np.array(rot_corners)
    
#     # 3. Change back to world coordinate system by adding center.
#     rot_corners_np = rot_corners_np + center

#     # 4. Transform coordinates of these rotated corners to vector coordinate system.
#     ego_from_world = np.linalg.inv(world_from_ego)
#     new_corners = []
#     for rot_point in rot_corners_np:
#         new_point = transform_point(rot_point, ego_from_world)
#         new_corners.append(new_point)
#     new_corners_np = np.array(new_corners)

#     # 4. Recalculate minimum/maximum value of new x and new y.
#     new_x_min = np.min(new_corners_np[:, 0])
#     new_y_min = np.min(new_corners_np[:, 1])
#     new_x_max = np.max(new_corners_np[:, 0])
#     new_y_max = np.max(new_corners_np[:, 1])

#     return np.array([new_x_min, new_y_min, new_x_max, new_y_max])
         
     
def calc_world_from_agent_matrix(agent_position: np.ndarray, agent_raw: float) -> np.ndarray:
    """
    Return the agent's pose as a 3x3 matrix. This corresponds to world_from_agent matrix.
    
    Args:
        ego_position (np.ndarray): 2D coordinates of the ego car
        ego_yaw (float): yaw of the ego car
    
    Returns:
        (np.ndarray): 3x3 world_from_ego matrix
    """
    return calc_world_from_ego_matrix(agent_position, agent_raw) 
