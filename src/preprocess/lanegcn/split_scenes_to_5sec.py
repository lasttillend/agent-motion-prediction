# 对场景进行预处理，将1个25秒场景分割成5个5秒场景并存储

import argparse
import zarr

import numpy as np

SCENE_DTYPE = [
    ("frame_index_interval", np.int64, (2,)),
    ("host", "<U16"),  # Unicode string up to 16 chars
    ("start_time", np.int64),
    ("end_time", np.int64),
]

#### 分割场景 ####
def split_scenes(scenes, mode):
    """
    Split each scene (25s or 10s) into 5-second parts.
    
    Args:
        scenes: zarr.core.Array, scenes to split.
        mode: str, if mode is 'train' or 'validate', then each scene lasts 25 secs, if mode is 'test', then it lasts 10 secs.
    Returns:
        scenes_splited: zarr.core.Array
    """
    num_scenes = len(scenes)
    if mode == "train" or mode == "validate":
        EACH_SCENE_TIME = 25
    elif mode == "test":
        EACH_SCENE_TIME = 10
    else:
        raise ValueError("Scenes to be split must be among train, validate, or test!")
    chunks = EACH_SCENE_TIME // 5
    chunk_frames = 50
    chunk_time = 5
    SEC_TO_NANO_SEC = 1_000_000_000

    scenes_splited_list = []  

    for scene_id in range(num_scenes):
        scene = scenes[scene_id]    
        cur_scene_start_frame_id = scene["frame_index_interval"][0]
        cur_scene_end_frame_id = scene["frame_index_interval"][1]

        small_scene_start_frame_id = cur_scene_start_frame_id
        small_scene_end_frame_id = cur_scene_start_frame_id
        for chunk_id in range(chunks):
            small_scene_start_frame_id = small_scene_end_frame_id
            small_scene_end_frame_id = min(small_scene_end_frame_id + chunk_frames, cur_scene_end_frame_id)
            small_scene_frame_index_interval = (small_scene_start_frame_id, small_scene_end_frame_id)

            host = scene['host']       
            start_time = scene['start_time'] + chunk_id * chunk_time * SEC_TO_NANO_SEC
            end_time = start_time + chunk_time * SEC_TO_NANO_SEC

            scene_splited = np.array((small_scene_frame_index_interval, host, start_time, end_time), dtype=SCENE_DTYPE)
            scenes_splited_list.append(scene_splited)
            

    # for scene_id in range(num_scenes):
    #     scene = scenes[scene_id]    
    #     for chunk_id in range(chunks):
    #         frame_start_id = scene["frame_index_interval"][0] + chunk_id * chunk_frames
    #         frame_end_id = min(frame_start_id + chunk_frames, scene["frame_index_interval"][1]) 
    #         frame_index_interval = (frame_start_id, frame_end_id)
    #         host = scene['host']       
    #         start_time = scene['start_time'] + chunk_id * chunk_time * SEC_TO_NANO_SEC
    #         end_time = start_time + chunk_time * SEC_TO_NANO_SEC

    #         scene_splited = np.array((frame_index_interval, host, start_time, end_time), dtype=SCENE_DTYPE)
    #         scenes_splited_list.append(scene_splited)
            
    scenes_splited = zarr.array(scenes_splited_list)
    return scenes_splited

def parse_args():
    parser = argparse.ArgumentParser(description="Split each 25-sec scene into five 5-sec scenes.")
    parser.add_argument("input_path", help="Absolute path of your zarr file")
    parser.add_argument("output_path", help="Absolute path of your output splited zarr file")
    parser.add_argument("mode", help="Type of your zarr file, must be among train/validate/test")

    args = parser.parse_args()
    return args
    
def main():
    args = parse_args()
    input_path = args.input_path
    output_path = args.output_path
    mode = args.mode

    dataset_zarr = zarr.open(input_path, 'r')

    scenes = dataset_zarr.scenes
    print("Number of scenes before spliting:", len(scenes))
    scenes = split_scenes(scenes, mode)
    print("Number of scenes after spliting:", len(scenes))

    frames = dataset_zarr.frames
    agents = dataset_zarr.agents
    traffic_light_faces = dataset_zarr.traffic_light_faces

    zarr.save(output_path, scenes=scenes, frames=frames, agents=agents, traffic_light_faces=traffic_light_faces)

if __name__ == "__main__":
    main()
