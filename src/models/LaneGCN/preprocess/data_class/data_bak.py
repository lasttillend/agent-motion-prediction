# LyftDataset
import os
import copy
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import numpy as np
from scipy import sparse

from lyft_map import LyftMap
from lane_segment import LaneSegment

from l5kit.data import ChunkedDataset, LocalDataManager

# set env variable for input data
DATA_ROOT = "/home/han/study/projects/agent-motion-prediction/data/lyft_dataset/"
os.environ["L5KIT_DATA_FOLDER"] = DATA_ROOT

MIN_FRAME_HISTORY = 10  # minimum number of frames an agents must have in the past to be picked
MIN_FRAME_FUTURE = 1    # minimum number of frames an agents must have in the future to be picked

class LyftDataset(Dataset):
    
    def __init__(self, config, mode="train"):
        self.config = config
        dm = LocalDataManager()        
        if "preprocess" in config and config["preprocess"]:
            if mode == "train":
                dataset_path = dm.require(self.config["preprocess_train"])
                self.split = np.load(dataset_path, allow_pickle=True)
            else:
                dataset_path = dm.require(self.config["preprocess_val"])               
                self.split = np.load(dataset_path, allow_pickle=True)
        else:
            if mode == "train":
                dataset_path = dm.require(self.config["train_data_loader"])
            elif mode == "val":
                dataset_path = dm.require(self.config["val_data_loader"])
            elif mode == "test":
                dataset_path = dm.require(self.config["test_data_loader"])
            else:
                raise ValueError("Dataset must be train, val, or test!")
            zarr_dataset = ChunkedDataset(dataset_path)
            zarr_dataset.open()

            self.zarr_dataset = zarr_dataset
            self.scenes = self.zarr_dataset.scenes
            self.frames = self.zarr_dataset.frames        
            self.agents = self.zarr_dataset.agents

            self.lyft_map = LyftMap()

            self.scenes = split_scenes(self.scenes, mode)
    #         print("Start filtering scenes!")
    #         self.scenes = filter_scenes(self.scenes, self.frames, self.agents)        
    #         print("Finished filtering scenes!")

            self.scene_track_ids = self.compute_scene_track_ids()
            self.cumulative_size = self.scenes["frame_index_interval"][:, 0]
        
    def __len__(self):
        if "preprocess" in self.config and self.config["preprocess"]:
            return len(self.split)
        else:
            return len(self.scenes)
    
    def __getitem__(self, idx):
        if "preprocess" in self.config and self.config["preprocess"]:
            data = self.split[idx]
            new_data = dict()
            
            for key in ["city", "orig", "gt_preds", "has_preds", "theta", "agent_from_world", "feats", "ctrs", "graph"]:
                if key in data:
                    new_data[key] = ref_copy(data[key])
            data = new_data
            
            return data

        data = self.read_lyft_data(idx)
        data = self.filter_agents(data)
        data = self.get_obj_feats(data)  # output data is a list of dictionaries
        data['idx'] = idx
        
        data['graph'] = self.get_lane_graph(data)
        
        return data
        
    def compute_scene_track_ids(self):
        scene_track_ids = dict()  # scene_id->num_unique_agents
        for scene_id in range(len(self.scenes)):
            scene = self.scenes[scene_id]
            frame_start_id = scene["frame_index_interval"][0]
            frame_end_id = scene["frame_index_interval"][1] - 1
            agent_start_id = self.frames[frame_start_id]["agent_index_interval"][0]
            agent_end_id = self.frames[frame_end_id]["agent_index_interval"][1]

            track_ids = np.unique(self.agents[agent_start_id:agent_end_id]["track_id"])
            scene_track_ids[scene_id + 1] = track_ids
            
        return scene_track_ids

    def filter_agents(self, data):
        steps = data["steps"]
        consecutive_frames_range = np.arange(19 - MIN_FRAME_HISTORY, 19 + MIN_FRAME_FUTURE + 1)
        agents_to_remove = []

        for i in range(len(steps) - 1):  # do not include ego car
            # consecutive_frames_range is contained in steps[i] <==> consecutive_frames_range intersect steps[i] = consecutive_frames_range
            intersection = np.intersect1d(consecutive_frames_range, steps[i])
            if len(intersection) != len(consecutive_frames_range):
                # different length, intersection does not equal consecutive_frames_range
                agents_to_remove.append(i)
            elif not np.all(consecutive_frames_range == intersection):
                # same length, but intersection still does not equal consecutive_frames_range
                agents_to_remove.append(i)
            else:
                # this is a valid agent, do not filter
                continue
                
        for idx in sorted(agents_to_remove, reverse=True):
            del data["trajs"][idx]
            del data["steps"][idx]   
            
        return data
        
    def read_lyft_data(self, idx):
#         print("Enter read lyft data!")
        scene_id = idx
        scene = self.scenes[scene_id]

        data = dict()  # keys are trajs and steps

        frame_start_id = scene["frame_index_interval"][0]
        frame_end_id = scene["frame_index_interval"][1] - 1
        agent_start_id = self.frames[frame_start_id]["agent_index_interval"][0]
        agent_end_id = self.frames[frame_end_id]["agent_index_interval"][1]
        agents_np = np.array(self.agents)
        agents_in_current_scene_np = agents_np[agent_start_id:agent_end_id]

        frame_id_for_all_agents_in_current_scene_list = []
        for frame_id in range(frame_start_id, frame_end_id + 1):
            agt_start_id, agt_end_id = self.frames[frame_id]["agent_index_interval"]
            frame_id_for_all_agents_in_current_scene_list += [frame_id] * (agt_end_id - agt_start_id)
        frame_id_for_all_agents_in_current_scene_np = np.array(frame_id_for_all_agents_in_current_scene_list)

        trajs_list = []
        steps_list = []
        # Extract trajectories and steps of all agents
        for track_id in self.scene_track_ids[scene_id + 1]:
            masks = agents_in_current_scene_np["track_id"] == track_id

            # trajectory
            agt = agents_in_current_scene_np[masks]
            trajs = agt["centroid"]
            trajs_list.append(trajs)

            # step
            steps = frame_id_for_all_agents_in_current_scene_np[masks]
            steps -= self.cumulative_size[idx]
            steps_list.append(steps)

        # add ego car's trajectory and step
        ego_trajs = self.frames[frame_start_id:frame_end_id + 1]["ego_translation"][:, :2]
        ego_steps = np.arange(frame_start_id, frame_end_id + 1) - self.cumulative_size[idx]
        trajs_list.append(ego_trajs)
        steps_list.append(ego_steps)

        data["trajs"] = trajs_list
        data["steps"] = steps_list        
    
#         print("Leave read lyft data!")
        return data

    def get_obj_feats(self, data):
#         print("Enter get_obj_feats!")
        # all the agents in 19-th frame        
        agents_in_19th_frame_idx = list(range(len(data['trajs']) - 1))  # not include ego car
        agents_in_19th_frame_list = []  # store coordinates of all agents in 19-th frame
        for idx in agents_in_19th_frame_idx:
            is_frame_19 = data['steps'][idx] == 19
            agents_in_19th_frame = data['trajs'][idx][is_frame_19].flatten()
            agents_in_19th_frame_list.append(agents_in_19th_frame)
        agents_in_19th_frame_np = np.array(agents_in_19th_frame_list)    
    
        ego_car_19th_frame = data['trajs'][-1][19] # ego car is the last element
        dist = np.linalg.norm(agents_in_19th_frame_np - ego_car_19th_frame, axis=1)
        min_dist_idx = np.argmin(dist)
        closest_agent = agents_in_19th_frame_np[min_dist_idx]
        orig = closest_agent        
        
        is_frame_18 = data['steps'][min_dist_idx] == 18
        frame_18_coords = data['trajs'][min_dist_idx][is_frame_18].flatten()    
        pre = orig - frame_18_coords
        theta = np.arctan2(pre[1], pre[0])  # theta range [-pi, pi)    
        # world_from_agent
        world_from_agent = np.array([[np.cos(theta), -np.sin(theta), orig[0]],
                                     [np.sin(theta),  np.cos(theta), orig[1]],
                                     [0, 0, 1]])
        # agent_from_world
        agent_from_world = np.linalg.inv(world_from_agent)

        feats, ctrs, gt_preds, has_preds = [], [], [], []
        for traj, step in zip(data['trajs'], data['steps']):
            # Note: all steps contain 19-th frame because we have done filtering using `filter_agents` method.
            #       So no need to check whether 19-th frame is in step any more.
            gt_pred = np.zeros((30, 2), np.float32)
            has_pred = np.zeros(30, np.bool)    
            future_mask = np.logical_and(step >= 20, step < 50)
            post_step = step[future_mask] - 20
            post_traj = traj[future_mask]
            post_traj_add_column_one = np.c_[(post_traj, np.ones(len(post_traj)))]  # shape = [len(post_traj), 3]
            post_traj = np.matmul(agent_from_world, post_traj_add_column_one.T).T
            post_traj = post_traj[:, :2]
            gt_pred[post_step] = post_traj  # gt_pred is in agent coordinate
            has_pred[post_step] = 1

            obs_mask = step < 20
            step = step[obs_mask]
            traj = traj[obs_mask]
            idcs = step.argsort()
            step = step[idcs]
            traj = traj[idcs]

            for i in range(len(step)):
                if step[i] == 19 - (len(step) - 1) + i:
                    break
        #     print("i = ", i)  # i should be equal to 0
            step = step[i:]
            traj = traj[i:]

            feat = np.zeros((20, 3), np.float32)
            traj_add_column_one = np.c_[(traj, np.ones(len(traj)))]  # shape = [len(traj), 3]
            feat[step, :] = np.matmul(agent_from_world, traj_add_column_one.T).T  # feat is in agent coordinate

            x_min, x_max, y_min, y_max = self.config["pred_range"]
            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue

            ctrs.append(feat[-1, :2].copy())  # 19-th frame coordinate 
            feat[1:, :2] -= feat[:-1, :2]  # every step's displacement
            feat[step[0], :2] = 0  # the first displacement is 0
            feats.append(feat)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

        feats = np.asarray(feats, np.float32)
        ctrs = np.asarray(ctrs, np.float32)
        gt_preds = np.asarray(gt_preds, np.float32)
        has_preds = np.asarray(has_preds, np.bool)
        
        data2 = {}
        data2['feats'] = feats
        data2['ctrs'] = ctrs
        data2['orig'] = orig
        data2['theta'] = theta
        data2['agent_from_world'] = agent_from_world
        data2['gt_preds'] = gt_preds
        data2['has_preds'] = has_preds  

#         print("Leave get_obj_feats!")
        return data2
    
    def get_lane_graph(self, data):
        """Get a rectangle area defined by pred_range."""
#         print("Enter get_lane_graph!")

        orig = data['orig']
        agent_from_world = data['agent_from_world']
        x_min, x_max, y_min, y_max = self.config['pred_range']
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        lane_ids = self.lyft_map.get_lane_ids_in_xy_bbox(orig[0], orig[1], query_search_range_manhattan=radius)        
        lane_ids = copy.deepcopy(lane_ids)
        
        lanes = dict()
        for lane_id in lane_ids:
            lane = self.lyft_map.lane_centerlines_dict[lane_id]
            lane = copy.deepcopy(lane)
            centerline = lane.centerline
            centerline_add_column_one = np.c_[(centerline[:, :2], np.ones(len(centerline)))]
            centerline = np.matmul(agent_from_world, centerline_add_column_one.T).T
            centerline = centerline[:, :2]   
            x, y = centerline[:, 0], centerline[:, 1]

            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue # lane segment is out of prediction range, so do not include it
            else:
                lane.centerline = centerline
                lanes[lane_id] = lane
        
        lane_ids = list(lanes.keys())
#         print("lane_ids after removing out of prediciton range:", lane_ids) 
        ctrs, feats = [], []
        for lane_id in lane_ids:
            lane = lanes[lane_id]
            ctrln = lane.centerline
            num_segs = len(ctrln) - 1
            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32)) # middle point btw two centerline points
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))  # displacement of centerline points
            
        node_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            node_idcs.append(range(count, count + len(ctr)))
            count += len(ctr)
        num_nodes = count

        pre, suc = dict(), dict()
        for key in ['u', 'v']:
            pre[key], suc[key] = [], []

        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lane_id]
            idcs = node_idcs[i]

            pre['u'] += idcs[1:]
            pre['v'] += idcs[:-1]

            if lane.predecessors is not None:
                for nbr_id in lane.predecessors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre['u'].append(idcs[0])
                        pre['v'].append(node_idcs[j][-1])         

            suc['u'] += idcs[:-1]
            suc['v'] += idcs[1:]

            if lane.successors is not None:
                for nbr_id in lane.successors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc['u'].append(idcs[-1])
                        suc['v'].append(node_idcs[j][0])    

        lane_idcs = []
        for i, idcs in enumerate(node_idcs):
            lane_idcs.append(i * np.ones(len(idcs), np.int64))
        lane_idcs = np.concatenate(lane_idcs, 0)

        pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lane_id]

            nbr_ids = lane.predecessors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre_pairs.append([i, j])

            nbr_ids = lane.successors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc_pairs.append([i, j])

            nbr_id = lane.l_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    left_pairs.append([i, j])

            nbr_id = lane.r_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    right_pairs.append([i, j])  

        pre_pairs = np.asarray(pre_pairs, np.int64)
        suc_pairs = np.asarray(suc_pairs, np.int64)
        left_pairs = np.asarray(left_pairs, np.int64)
        right_pairs = np.asarray(right_pairs, np.int64)     

        graph = dict()
        graph['ctrs'] = np.concatenate(ctrs, 0)
        graph['num_nodes'] = num_nodes
        graph['feats'] = np.concatenate(feats, 0)
        graph['pre'] = [pre]
        graph['suc'] = [suc]
        graph['lane_idcs'] = lane_idcs
        graph['pre_pairs'] = pre_pairs
        graph['suc_pairs'] = suc_pairs
        graph['left_pairs'] = left_pairs
        graph['right_pairs'] = right_pairs    

        for k1 in ['pre', 'suc']:
            for k2 in ['u', 'v']:
                graph[k1][0][k2] = np.asarray(graph[k1][0][k2], np.int64)  # convert list to numpy array

        for key in ['pre', 'suc']:
            graph[key] += dilated_nbrs(graph[key][0], graph['num_nodes'], self.config['num_scales'])        

#         print("Leave get_lane_graph!")
        return graph

    
###################### Utility Functions ######################################   
# 1. split_scenes
import zarr

SCENE_DTYPE = [
    ("frame_index_interval", np.int64, (2,)),
    ("host", "<U16"),  # Unicode string up to 16 chars
    ("start_time", np.int64),
    ("end_time", np.int64),
]

def split_scenes(scenes, mode):
    """
    Split each scene (25s or 10s) into 5-second parts.
    
    Args:
        scenes: zarr.core.Array, scenes to split.
        mode: str, if mode is 'train' or 'val', then each scene lasts 25 secs, if mode is 'test', then it lasts 10 secs.
    Returns:
        scenes_splited: zarr.core.Array
    """
    num_scenes = len(scenes)
    if mode == "train" or mode == "val":
        EACH_SCENE_TIME = 25
    elif mode == "test":
        EACH_SCENE_TIME = 10
    else:
        raise ValueError("Scenes to be split must be among train, val, or test!")
    chunks = EACH_SCENE_TIME // 5
    chunk_frames = 50
    chunk_time = 5
    SEC_TO_NANO_SEC = 1_000_000_000

    scenes_splited_list = []    
    for scene_id in range(num_scenes):
        scene = scenes[scene_id]    
        for chunk_id in range(chunks):
            frame_start_id = scene["frame_index_interval"][0] + chunk_id * chunk_frames
            frame_end_id = min(frame_start_id + chunk_frames, scene["frame_index_interval"][1]) 

            frame_index_interval = (frame_start_id, frame_end_id)
            host = scene['host']       
            start_time = scene['start_time'] + chunk_id * chunk_time * SEC_TO_NANO_SEC
            end_time = start_time + chunk_time * SEC_TO_NANO_SEC

            scene_splited = np.array((frame_index_interval, host, start_time, end_time), dtype=SCENE_DTYPE)
            scenes_splited_list.append(scene_splited)
            
    scenes_splited = zarr.array(scenes_splited_list)
    return scenes_splited

# 2. filter_scenes
def filter_scenes(scenes, frames, agents):
    """
    Filter scenes which has no agents satisfying the two conditions.
    """
#     print("Enter filter_scenes!")
    scenes_kept_list = []
    consecutive_frames_range = np.arange(19 - MIN_FRAME_HISTORY, 19 + MIN_FRAME_FUTURE + 1)            
    for scene_id in tqdm(range(len(scenes))):
        scene = scenes[scene_id]
        frame_start_id = scene["frame_index_interval"][0]
        
        consecutive_frames_agents = dict()
        for idx in consecutive_frames_range:
            frame_id = frame_start_id + idx
            agents_start_id, agents_end_id = frames[frame_id]["agent_index_interval"] 
            track_ids = agents[agents_start_id:agents_end_id]["track_id"]
            consecutive_frames_agents[frame_id] = track_ids

        agents_in_all_consecutive_frames = list(consecutive_frames_agents.values())[0]
        for agents_in_current_frame in consecutive_frames_agents.values():
            agents_in_all_consecutive_frames = np.intersect1d(agents_in_all_consecutive_frames, agents_in_current_frame)   
            
        if len(agents_in_all_consecutive_frames) > 0:
            scenes_kept_list.append(scene)
        
    scenes_kept = zarr.array(scenes_kept_list)
    print("Number of scenes kept:", len(scenes_kept))
#     print("Leave filter_scenes!")    
    return scenes_kept

# 3. dilated_nbrs
from scipy import sparse

def dilated_nbrs(nbr, num_nodes, num_scales):
    data = np.ones(len(nbr['u']), np.bool)
    csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

    mat = csr
    nbrs = []
    for i in range(1, num_scales):
        mat = mat * mat

        nbr = dict()
        coo = mat.tocoo()
        nbr['u'] = coo.row.astype(np.int64)
        nbr['v'] = coo.col.astype(np.int64)
        nbrs.append(nbr)
    return nbrs    

# 4. collate_fn
def collate_fn(batch):
    batch = from_numpy(batch)
    return_batch = dict()
    # Batching by use a list for non-fixed size
    for key in batch[0].keys():
        return_batch[key] = [x[key] for x in batch]
    return return_batch

# 5. from_numpy
def from_numpy(data):
    """Recursively transform numpy.ndarray to torch.Tensor.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = from_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [from_numpy(x) for x in data]
    if isinstance(data, np.ndarray):
        """Pytorch now has bool type."""
        data = torch.from_numpy(data)
    return data

# 6. ref_copy
def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data


