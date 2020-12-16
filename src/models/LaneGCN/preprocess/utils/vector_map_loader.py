import pickle

def load_lane_segments_from_file(fpath):
    """
    Load lane segment objects from a binary file.
    """
    with open(fpath, 'rb') as f:
        lane_segments = pickle.load(f)

    return lane_segments
