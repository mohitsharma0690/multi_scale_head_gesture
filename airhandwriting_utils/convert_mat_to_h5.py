import argparse
import csv
import h5py
import numpy as np
import os
import scipy.io as sio
import sys
import pdb

if __name__ == '__main__' and  __package__ is None:
  print('appending to sys path')
  print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils as global_utils

class MatFileInfo(object):
  '''
  Sample Name: 'lower_a_C1_t01.mat' or 'MAIL_I2_t03.mat'
  '''
  def __init__(self, filepath, is_char_mat=True):
    assert(os.path.exists(filepath))
    self.filepath = filepath
    self.filename = os.path.basename(filepath)
    start_idx = 0 if is_char_mat else -1
    filename_splits = self.filename.split('_')
    if start_idx > 0:
      self.label_type = filename_splits[start_idx]
    else:
      # Only used as a placeholder to maintain consistency between both char
      # and word representations.
      self.label_type = 'upper'
    self.label = filename_splits[start_idx+1]
    self.user = filename_splits[start_idx+2]
    self.trial = filename_splits[start_idx+3]

  def get_gesture(self):
    mat_f = sio.loadmat(self.filepath)
    return np.array(mat_f['gest'])
  
  def get_noise(self):
    mat_f = sio.loadmat(self.filepath)
    return np.array(mat_f['noise'])

  def get_bias(self):
    mat_f = sio.loadmat(self.filepath)
    return np.array(mat_f['bias'])

def get_info_from_mat_filename(mat_filename):
  pass

def main(data_dir, h5_name, csv_name, is_char_mat=True):
  mat_files = filter(lambda(f): f.endswith('mat'), os.listdir(data_dir))
  data_dict = {}
  for mat_f in mat_files:
    mat_info = MatFileInfo(
            os.path.join(data_dir, mat_f), is_char_mat=is_char_mat)
    if data_dict.get(mat_info.label_type) is None:
      data_dict[mat_info.label_type] = {}
    if data_dict[mat_info.label_type].get(mat_info.label) is None:
      data_dict[mat_info.label_type][mat_info.label] = {}
    if data_dict[mat_info.label_type][mat_info.label].get(mat_info.user) is None:
      data_dict[mat_info.label_type][mat_info.label][mat_info.user] = {}
    user_dict = data_dict[mat_info.label_type][mat_info.label][mat_info.user]
    user_dict[mat_info.trial] = {}
    user_dict[mat_info.trial]['gest'] = mat_info.get_gesture() 
    if is_char_mat:
        user_dict[mat_info.trial]['bias'] = mat_info.get_bias() 
        user_dict[mat_info.trial]['noise'] = mat_info.get_noise() 

  h5_path = os.path.join(os.path.dirname(data_dir), h5_name)
  print("Will write matlab data to {}".format(h5_path))
  h5_f = h5py.File(h5_path, 'w')
  global_utils.recursively_save_dict_contents_to_group(h5_f, '/', data_dict)
  h5_f.flush()
  h5_f.close()

  all_keys, curr_keys = [], []
  global_utils.recursively_get_list_of_all_keys(data_dict, all_keys, curr_keys) 
  csv_path = os.path.join(os.path.dirname(data_dir), csv_name)
  with open(csv_path, 'w') as csv_f:
    csv_writer = csv.writer(csv_f, delimiter=',')
    csv_writer.writerow(['label_type','label','user','trial','type'])
    for row in all_keys:
      if row[-1] == 'gest':
        row.pop()
        row.append('train')
        csv_writer.writerow(row)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Convert airhandwriting mat files to h5.')
  parser.add_argument('--data_dir', nargs='?', type=str, const=1,
      required=True, default='../data_airhandwriting/matR_char',
      help='Mat files directory.')
  parser.add_argument('--h5_name', nargs='?', type=str, const=1,
      required=True, help='Filename to save h5 file.')
  parser.add_argument('--csv_name', nargs='?', type=str, const=1,
      required=True, help='Filename to save csv file.')
  parser.add_argument('--char_mat', nargs='?', type=int, const=1,
      required=True, help='1 if the character data 0 if word data')
  args = parser.parse_args()
  print(args)
  main(args.data_dir, args.h5_name, args.csv_name, args.char_mat)

