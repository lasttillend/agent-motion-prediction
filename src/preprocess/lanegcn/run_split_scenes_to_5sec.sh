# Example: 
# input_path = /home/han/study/projects/agent-motion-prediction/data/lyft_dataset/scenes/splited/train_zarr_splited/train_0.zarr
# output_path = /home/han/study/projects/agent-motion-prediction/data/lyft_dataset_5sec_scene/scenes/splited/train_zarr_splited/train_5sec_0.zarr
# mode = train

input_path=/home/han/study/projects/agent-motion-prediction/data/lyft_dataset/scenes/splited/validate_zarr_splited/validate_7.zarr
output_path=/home/han/study/projects/agent-motion-prediction/data/lyft_dataset_5sec_scene/scenes/splited/validate_zarr_splited/validate_5sec_7.zarr
mode=validate
python split_scenes_to_5sec.py $input_path $output_path $mode