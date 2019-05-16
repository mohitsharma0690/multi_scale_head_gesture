import argparse
import numpy as np
import h5py
import pdb
import os

OpenFace_Valid_Frame_Threshold = 100
OpenFace_Columns = 431
Max_Frames = 20000
Tick_Annotation_Value = 5

def create_h5(h5_fname, openface_fname, annot_dir_name=None):
    feats = read_openface_data(openface_fname)
    annot = np.zeros(feats.shape[0])

    if annot_dir_name is not None:
        tick_frames = read_annotations(annot_dir_name)
        for tick_seq in tick_frames:
            annot[tick_seq[0]:tick_seq[1]+1] = Tick_Annotation_Value

    with h5py.File(h5_fname, 'w') as hf:
        hf.create_dataset('annotations', data=np.array(annot,dtype=int))
        hf.create_dataset('features', data=feats)

def read_annotations(dirname):
    '''
    Returns a list of list. Each inner list has two numbers, the starting frame
    and the end frame for the tick.
    '''
    tick_frames = []
    for name in os.listdir(dirname):
        # file name format nod_08_2924-to-3044.mp4
        if name.endswith("mp4"):
            frame_nums = name.split('_')[2].split('-to-')
            assert(len(frame_nums) == 2)
            frame_nums[1] = frame_nums[1].split('.')[0]
            tick_frames.append([int(f) for f in frame_nums])
    return tick_frames

def read_openface_data(fname):
    with open(fname) as f:
        num_cols = len(f.readline().split())
        f.seek(0)
        features = np.genfromtxt(f, delimiter=',', skip_header=1)
        # We only have 412 features saved in the original h5 files starting from
        # gaze. Thus we only choose that many from here. Also there are some
        # Action Units (AU) more in the open face results I got. We can avoid
        # them for now.
        features = features[:,4:416]
        return features
            
def main(h5_name, csv_path, fdir):
    create_h5(h5_name, csv_path, fdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Convert openface csv output to h5.')
    parser.add_argument('--h5_name', nargs='?', type=str, const=1,
            required=True, help='Name for the h5 file to store openface output.')
    parser.add_argument('--csv', nargs='?', type=str, const=1,
            required=True, help='Path to the openface annotations.')
    args = parser.parse_args()
    print(args)
    main(args.h5_name, args.csv, None)

