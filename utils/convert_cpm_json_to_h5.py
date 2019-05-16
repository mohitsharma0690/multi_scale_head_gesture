import argparse
import h5py
import json
import numpy as np
import os
import pdb
import data_utils

CPM_INDEX_TO_JOINT_TYPE = {
   0:  "Nose",
   1:  "Neck",
   2:  "RShoulder",
   3:  "RElbow",
   4:  "RWrist",
   5:  "LShoulder",
   6:  "LElbow",
   7:  "LWrist",
   8:  "RHip",
   9:  "RKnee",
   10: "RAnkle",
   11: "LHip",
   12: "LKnee",
   13: "LAnkle",
   14: "REye",
   15: "LEye",
   16: "REar",
   17: "LEar",
   # 18: "Bkg",  # CPM output doesn't seem to emit this value
}

def get_joint_index_to_type(idx):
  return CPM_INDEX_TO_JOINT_TYPE[idx]

def get_joint_type_to_idx(joint):
  for (idx, joint_type) in CPM_INDEX_TO_JOINT_TYPE.iteritems():
    if joint == joint_type:
      return idx
  assert(False)
  return -1

def get_joint_type_to_h5_idx(joint):
  cpm_idx = get_joint_type_to_idx(joint)
  return cpm_idx * 3

def get_angle_between_neck_and_nose(X, frame_num):
  if frame_num == 0:
    return 0
  nose_idx = get_joint_type_to_h5_idx('Nose')
  neck_idx = get_joint_type_to_h5_idx('Neck')
  prev_nose_val = X[frame_num-1][nose_idx:nose_idx+3]
  prev_neck_val = X[frame_num-1][neck_idx:neck_idx+3]
  curr_nose_val = X[frame_num][nose_idx:nose_idx+3]
  curr_neck_val = X[frame_num][neck_idx:neck_idx+3]
  prob_th = 0.5  # prob threshold
  if (prev_nose_val[2] < prob_th or prev_neck_val[2] < prob_th or
      curr_nose_val[2] < prob_th or curr_neck_val[2] < prob_th):
    return -100
  # Get the angle from the math
  # TODO(Mohit): Do the math to get the angle. One caveat with the math will be
  # since the points are not coplanar and can move in any direction independent
  # of each other we will never get the actual angular velocity since for that
  # we need to find the 3D point estimates.
  return 0

def is_cpm_json_file(json_filename):
  return json_filename.endswith('json') and json_filename.startswith('frame')

def create_cpm_h5(json_path, h5_path):
  frame_count = 0
  for f in os.listdir(json_path):
    if is_cpm_json_file(f):
      frame_count = frame_count + 1

  # each joint has (x, y, conf) and we also store the angle between nose and
  # neck at each frame
  X = np.zeros((frame_count, len(CPM_INDEX_TO_JOINT_TYPE)*3 + 1))
  frame_bitset = np.zeros(frame_count).astype(int)
  for f in sorted(os.listdir(json_path)):
    if is_cpm_json_file(f):
      with open(os.path.join(json_path, f)) as json_fp:
        cpm_output = json.load(json_fp)
        frame_num = int(f.split('frame')[-1].split('.')[0])
        joints = cpm_output['bodies'][0]['joints']
        for i in range(len(joints)):
          X[frame_num][i] = joints[i]
        theta = get_angle_between_neck_and_nose(X, frame_num)
        X[frame_num][len(joints)] = theta
        frame_bitset[frame_num] = 1
  assert(np.sum(frame_bitset[frame_bitset == 0]) == 0)
  h5_file = h5py.File(h5_path, 'w')
  data_utils.recursively_save_dict_contents_to_group(h5_file, '/', 
    {'joints': X})
  h5_file.flush()
  h5_file.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Convert json predictions to h5.')
  parser.add_argument('--json_dir', nargs='?', type=str, const=1,
      required=True, default='../data', help='h5 files directory')
  parser.add_argument('--h5_name', nargs='?', type=str, const=1,
      required=True, help='h5 files directory')
  args = parser.parse_args()
  print(args)

  h5_path = os.path.join(os.path.dirname(args.json_dir), args.h5_name)
  create_cpm_h5(args.json_dir, h5_path)

