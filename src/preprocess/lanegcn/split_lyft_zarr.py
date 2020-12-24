import os
from pathlib import Path
from tqdm import tqdm

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.data.zarr_utils import zarr_split

DATA_ROOT = "/home/han/study/projects/agent-motion-prediction/data/lyft_dataset/"
os.environ["L5KIT_DATA_FOLDER"] = DATA_ROOT

OUTPUT_ROOT = "/home/han/study/projects/agent-motion-prediction/data/lyft_dataset/scenes/splited"
GIGABYTE = 1 * 1024 * 1024 * 1024
SPLITED_SIZE = 1  # split size is 1GB

def main():
    dm = LocalDataManager()
    # dataset = ["train", "validate", "test"]
    dataset = ["train", "validate"]
    for name in dataset:
        dataset_path = dm.require(f"scenes/{name}.zarr")
        print(f"{name} dataset path:", dataset_path)
        path_size = compute_path_size(dataset_path) / GIGABYTE
        print("path_size:", path_size)
        num_splited = int(path_size / SPLITED_SIZE)
        print("num to split:", num_splited)
        output_path = os.path.join(OUTPUT_ROOT, f"{name}_zarr_splited")
        print("output path:", output_path)
        split_zarr(dataset_path, output_path, num_splited, name)  

def split_zarr(zarr_path, output_path, num_splited, name):
    split_infos = []
    for i in range(num_splited):
        split = {"name": f"{name}_{i}.zarr", "split_size_GB": SPLITED_SIZE}
        split_infos.append(split)
    split_infos.append({"name": f"{name}_zarr_{i}.zarr", "split_size_GB": -1})   
    
    zarr_split(zarr_path, output_path, split_infos)

def compute_path_size(path: str) -> int:
    """
    Compute the total size of the folder, considering also nested elements.
    Can be run to get zarr total size
    Args:
        path (str): base path
    Returns:
        (int): total size in bytes
    """
    root_directory = Path(path)
    return sum(f.stat().st_size for f in root_directory.glob("**/*") if f.is_file())

if __name__ == "__main__":
    main()
    
