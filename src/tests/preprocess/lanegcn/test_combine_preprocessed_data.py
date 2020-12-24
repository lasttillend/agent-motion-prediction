import os
import pickle
from typing import List, Dict
from preprocess.lanegcn.combine_preprocessed_data import read_all_preprocessed_data, combine_data

Element = Dict
Data = List[Element]
DataList = List[Data]

ROOT = "/home/han/study/projects/agent-motion-prediction/"
PREPROCESSED_ROOT = os.path.join(ROOT, "data/preprocess/lanegcn/preprocessed_val/")

def test_read_all_preprocessed_data() -> None:
    data_list = read_all_preprocessed_data(PREPROCESSED_ROOT)
    assert len(data_list) == len(os.listdir(PREPROCESSED_ROOT))

def test_combine_data() -> None:
    data_list = read_all_preprocessed_data(PREPROCESSED_ROOT)
    combined_data = combine_data(data_list)
    cnt = 0
    for i in range(len(combined_data)):
        data = combined_data[i]
        assert data["idx"] == cnt, f"Ooops, combined data{i} indexing is wrong!"
        cnt += 1
