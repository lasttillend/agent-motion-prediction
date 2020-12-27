import os
import time
import pickle
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from l5kit.data import ChunkedDataset, LocalDataManager

from src.preprocess.lanegcn.data_class.data import LyftDataset as Dataset, split_scenes, collate_fn, from_numpy, ref_copy
from src.preprocess.lanegcn.map_class.lane_segment import LaneSegment
from src.preprocess.lanegcn.utils.gpu_utils import gpu

# root directory to save the preprocssed data
PREPROCESSED_ROOT = "/home/han/study/projects/agent-motion-prediction/data/preprocess/lanegcn/"
# ROOT = os.path.dirname(os.path.abspath(__file__))

# config
cfg = {
    'pred_range': [-100.0, 100.0, -100.0, 100.0],
    'num_scales': 6,
    'cross_dist': 6,
    'batch_size': 64,
    'workers': 0,
    
    'train_data_loader': 'scenes/splited/train_zarr_splited_valid/train_5sec_7_valid.zarr',
    'val_data_loader': 'scenes/splited/validate_zarr_splited_valid/validate_5sec_6_valid.zarr',
    # 'test_data_loader': 'scenes/splited/test_zarr_splited/test_zarr_0.zarr',  

    # "train_data_loader": "scenes/train.zarr",
    # "val_data_loader": "scenes/validate.zarr",
    # "test_data_loader": "scenes/test.zarr",


    "preprocess_train": os.path.join(PREPROCESSED_ROOT, "preprocessed_train", "train_crs_dist6_7.p"), # train_crs_dist6_3.p, 
    "preprocess_val": os.path.join(PREPROCESSED_ROOT, "preprocessed_val", "val_crs_dist6_6.p"),  # val_crs_dist6_7.p, val_crs_dist6_2.p 
    # "preprocess_test": os.path.join(PREPROCESSED_ROOT, "preprocessed_test", "test_test.p"),
}


def main():
    
    os.makedirs(os.path.dirname(cfg['preprocess_train']), exist_ok=True)
    os.makedirs(os.path.dirname(cfg['preprocess_val']), exist_ok=True)
    # os.makedirs(os.path.dirname(cfg['preprocess_test']), exist_ok=True)
    
    print("Preprocessing training data! train_5sec_7_valid")
    train(cfg)
    print("training data finished!")  
    # print("Preprocessing validation data! validate_5sec_6_valid")    
    # val(cfg)
    # print("Validation data finished!")
    # print("Preprocessing test data!")
    # test(cfg)
    # print("Test data finished!")
    # print("Successfully preprocessed data!")


def train(config):
    print("Start loading training dataset!")
    dataset = Dataset(config, mode='train')
    print("Training dataset loaded!")
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    stores = [None for x in range(len(dataset))]
    t = time.time()
    for i, data in enumerate(tqdm(train_loader)):
        for j in range(len(data["idx"])):
            store = dict()
            for key in [
                "idx",
                "feats",
                "ctrs",
                "orig",
                "theta",
                "agent_from_world",
                "gt_preds",
                "has_preds",
                "graph",
            ]:
                store[key] = to_numpy(data[key][j])
                if key == "graph":
                    store[key] = to_int16(store[key])
            stores[store["idx"]] = store
            
        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()
            
    dataset = PreprocessDataset(stores, config)
    data_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=False,
        drop_last=False
    )
    
    modify(config, data_loader, config["preprocess_train"])


def val(config):
    # Data loader for validation set
    print("Start load validation dataset!")    
    dataset = Dataset(config, mode='val')
    print("Validation set loaded!")
    val_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=False
    )

    stores = [None for x in range(len(dataset))]
    t = time.time()
    for i, data in enumerate(tqdm(val_loader)):
        data = dict(data)
        for j in range(len(data["idx"])):
            store = dict()
            for key in [
                "idx",
                "feats",
                "ctrs",
                "orig",
                "theta",
                "agent_from_world",
                "gt_preds",
                "has_preds",
                "graph",
            ]:
                store[key] = to_numpy(data[key][j])
                if key in ["graph"]:
                    store[key] = to_int16(store[key])
            stores[store["idx"]] = store

        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()

    dataset = PreprocessDataset(stores, config)
    data_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=False
    )

    modify(config, data_loader, config["preprocess_val"])


def test(config):
    # Data loader for validation set
    print("Start loading test set!")
    dataset = Dataset(config, mode='test')
    print("Test set loaded!")
    test_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=False
    )

    stores = [None for x in range(len(dataset))]
    t = time.time()
    for i, data in enumerate(tqdm(test_loader)):
        data = dict(data)
        for j in range(len(data["idx"])):
            store = dict()
            for key in [
                "idx",
                "feats",
                "ctrs",
                "orig",
                "theta",
                "agent_from_world",
                "gt_preds",
                "has_preds",
                "graph",
            ]:
                store[key] = to_numpy(data[key][j])
                if key in ["graph"]:
                    store[key] = to_int16(store[key])
            stores[store["idx"]] = store

        if (i + 1) % 100 == 0:
            print(i, time.time() - t)
            t = time.time()

    dataset = PreprocessDataset(stores, config)
    data_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        shuffle=False,
        collate_fn=from_numpy,
        pin_memory=False
    )

    modify(config, data_loader, config["preprocess_test"])


#####################################################
class PreprocessDataset():
    def __init__(self, split, config):
        self.split = split  # list[store], where store is a dict
        self.config = config

    def __getitem__(self, idx):
#         from lyft_data import from_numpy, ref_copy  # commented out by Tang Han
        data = self.split[idx]
        graph = dict()
        for key in ['lane_idcs', 'ctrs', 'pre_pairs', 'suc_pairs', 'left_pairs', 'right_pairs', 'feats']:
            graph[key] = ref_copy(data['graph'][key])
        graph['idx'] = idx
        return graph

    def __len__(self):
        return len(self.split)


def preprocess(graph, cross_dist, cross_angle=None):
    left, right = dict(), dict()

    lane_idcs = graph['lane_idcs']
    num_nodes = len(lane_idcs)
    num_lanes = lane_idcs[-1].item() + 1

    dist = graph['ctrs'].unsqueeze(1) - graph['ctrs'].unsqueeze(0) 
    dist = torch.sqrt((dist ** 2).sum(2))

    hi = torch.arange(num_nodes).long().to(dist.device).view(-1, 1).repeat(1, num_nodes).view(-1)
    wi = torch.arange(num_nodes).long().to(dist.device).view(1, -1).repeat(num_nodes, 1).view(-1)
    row_idcs = torch.arange(num_nodes).long().to(dist.device)

    if cross_angle is not None:
        f1 = graph['feats'][hi]
        f2 = graph['ctrs'][wi] - graph['ctrs'][hi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = t2 - t1
        m = dt > 2 * np.pi
        dt[m] = dt[m] - 2 * np.pi
        m = dt < -2 * np.pi
        dt[m] = dt[m] + 2 * np.pi
        mask = torch.logical_and(dt > 0, dt < config['cross_angle'])
        left_mask = mask.logical_not()
        mask = torch.logical_and(dt < 0, dt > -config['cross_angle'])
        right_mask = mask.logical_not()

    pre = graph['pre_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()

    #### Added by Tang Han ####
    # Originally, there was no check condition
    if len(graph['pre_pairs']) > 0:
        pre[graph['pre_pairs'][:, 0], graph['pre_pairs'][:, 1]] = 1

    suc = graph['suc_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
    #### Added by Tang Han ####
    # Originally, there was no check condition
    if len(graph['suc_pairs']) > 0:
        suc[graph['suc_pairs'][:, 0], graph['suc_pairs'][:, 1]] = 1

    # For each node u on the current lane, find the nearest node v on the left side lanes if there exist such lanes including the left lane, left behind lane and left forward lane. After this procedure, we form a node pair (u, v).
    pairs = graph['left_pairs']
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        # mat @ pre: left behind lane
        # mat @ suc: left forward lane
        # mat @ pre + mat @ suc + mat: all three left relationships
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5  # filter non-zero element
        left_dist = dist.clone()  # shape = [num_nodes, num_nodes]
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()  # shape = [num_nodes x num_nodes, ]
        left_dist[hi[mask], wi[mask]] = 1e6  # mask out distance for non-left relationships between nodes
        if cross_angle is not None:
            left_dist[hi[left_mask], wi[left_mask]] = 1e6

        min_dist, min_idcs = left_dist.min(1)
        mask = min_dist < cross_dist  # only consider left nodes within the given cross_dist distance
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = graph['feats'][ui]
        f2 = graph['feats'][vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)  # rotation angle
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)  # angle is between -pi to pi
        m = dt < 0.25 * np.pi  # only consider angles between nodes within -45 degree to 45 degree

        ui = ui[m]
        vi = vi[m]

        left['u'] = ui.cpu().numpy().astype(np.int16)
        left['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        left['u'] = np.zeros(0, np.int16)
        left['v'] = np.zeros(0, np.int16)

    pairs = graph['right_pairs']
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        right_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        right_dist[hi[mask], wi[mask]] = 1e6
        if cross_angle is not None:
            right_dist[hi[right_mask], wi[right_mask]] = 1e6

        min_dist, min_idcs = right_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = graph['feats'][ui]
        f2 = graph['feats'][vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        right['u'] = ui.cpu().numpy().astype(np.int16)
        right['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        right['u'] = np.zeros(0, np.int16)
        right['v'] = np.zeros(0, np.int16)

    out = dict()
    out['left'] = left
    out['right'] = right
    out['idx'] = graph['idx']

    return out


def modify(config, data_loader, save):
    t = time.time()
    store = data_loader.dataset.split

    for i, data in enumerate(data_loader):
#         data = [dict(x) for x in data]  # commented out by Tang Han because x has already been a dict
        out = []
        for j in range(len(data)):
            out.append(preprocess(to_long(gpu(data[j])), config['cross_dist']))

        for j, graph in enumerate(out):
            idx = graph['idx']
            store[idx]['graph']['left'] = graph['left']
            store[idx]['graph']['right'] = graph['right']

        if (i + 1) % 100 == 0:
            print((i + 1) * config['batch_size'], time.time() - t)
            t = time.time()

    f = open(os.path.join(PREPROCESSED_ROOT, save), 'wb')
    pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def to_numpy(data):
    """Recursively transform torch.Tensor to numpy.ndarray.
    """
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_numpy(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_numpy(x) for x in data]
    if torch.is_tensor(data):
        data = data.numpy()
    return data


def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data


def to_int16(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_int16(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_int16(x) for x in data]
    if isinstance(data, np.ndarray) and data.dtype == np.int64:
        data = data.astype(np.int16)
    return data


if __name__ == "__main__":
    main()

