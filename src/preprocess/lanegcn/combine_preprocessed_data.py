# Combine small preprocessed data (e.g., train_crs_dist6_0.p ~ train_crs_dist6_7.p) to a complete file.
import os
import pickle
from typing import List, Dict

ROOT = "/home/han/study/projects/agent-motion-prediction/"
PREPROCESSED_ROOT = os.path.join(ROOT, "data/preprocess/lanegcn/preprocessed_val/")  # train, val
COMBINED_ROOT = os.path.join(ROOT, "data/preprocess/lanegcn/combined_val/")  # train, val        

Element = Dict
Data = List[Element]
DataList = List[Data]

# The following two functions can be  used to check whether the index in the preprocessed data starts from 0
def get_preprocessed_indices_from_file(preprocessed_root: str, filename: str) -> List[int]:
    data = read_one_preprocessed_data(preprocessed_root, filename)
    return get_preprocessed_indices_from_data_list(data)

def get_preprocessed_indices_from_data(data: Data) -> List[int]:
    indices_list = []
    for i in range(len(data)):
        indices_list.append(data[i]["idx"])
    return indices_list

def read_one_preprocessed_data(preprocessed_root: str, filename: str) -> Data:
    with open(os.path.join(preprocessed_root, filename), "rb") as f:
        data = pickle.load(f)
    return data                                                               

def read_all_preprocessed_data(preprocessed_root: str) -> DataList:
    files = os.listdir(preprocessed_root)
    data_list = []
    for filename in files:
        with open(os.path.join(preprocessed_root, filename), "rb") as f:
            data = pickle.load(f)
            data_list.append(data)
    return data_list

def combine_data(data_list: DataList) -> Data:
    cnt = 0
    combined_data = []
    for data in data_list:
        for i in range(len(data)):
            data[i]["idx"] = cnt
            cnt += 1
            combined_data.append(data[i])
    return combined_data

def main():
    print("Start combine preprocessed data!")
    data_list = read_all_preprocessed_data(PREPROCESSED_ROOT)
    combined_data = combine_data(data_list)

    print(f"Save combined data to {COMBINED_ROOT}")
    output_filename = "val_crs_dist6_combined.p"
    with open(os.path.join(COMBINED_ROOT, output_filename), "wb") as f:
        pickle.dump(combined_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done!")

if __name__ == "__main__":
    main()
