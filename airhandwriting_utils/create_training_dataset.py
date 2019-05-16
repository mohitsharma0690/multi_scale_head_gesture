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

def main(h5_path, csv_path, dataset_h5, label_type='upper', is_char_data=True):
  h5_f = h5py.File(h5_path, 'r')
  dataset_h5_dict = {'train': {}, 'test': {}, 
      'train_noise': {}, 'test_noise': {},
      'train_bias': {}, 'test_bias': {}}
  with open(csv_path, 'r') as csv_f:
    csv_reader = csv.DictReader(csv_f)
    for i, row in enumerate(csv_reader):
      data = h5_f[row['label_type']][row['label']][row['user']][row['trial']]
      data_type = row['type']
      if dataset_h5_dict[data_type].get(row['label']) is None:
        dataset_h5_dict[data_type][row['label']] = {}
      if is_char_data and \
          dataset_h5_dict[data_type+'_noise'].get(row['label']) is None:
        dataset_h5_dict[data_type+'_noise'][row['label']] = {}
      if is_char_data and \
          dataset_h5_dict[data_type+'_bias'].get(row['label']) is None:
        dataset_h5_dict[data_type+'_bias'][row['label']] = {}

      idx = len(dataset_h5_dict[data_type][row['label']])
      dataset_h5_dict[data_type][row['label']][str(idx)] = np.array(data['gest'])
      if is_char_data:
        dataset_h5_dict[data_type+'_noise'][row['label']][str(idx)] = np.array(
            data['noise'])
        dataset_h5_dict[data_type+'_bias'][row['label']][str(idx)] = np.array(
            data['bias'])

  dataset_h5_f = h5py.File(dataset_h5, 'w')
  global_utils.recursively_save_dict_contents_to_group(
      dataset_h5_f, '/', dataset_h5_dict)
  dataset_h5_f.flush()
  dataset_h5_f.close()
  h5_f.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Create h5 dataset for airhandwriting training.')
  parser.add_argument('--h5', nargs='?', type=str, const=1,
      required=True, default='../data_airhandwriting/matR_char.h5',
      help='h5 file containing matR_char data.')
  parser.add_argument('--dataset_h5', nargs='?', type=str, const=1,
      required=True, default='../data_airhandwriting/matR_train_data.h5',
      help='h5 filename containing matR_char train/test data.')
  parser.add_argument('--csv', nargs='?', type=str, const=1,
      required=True, default='../data_airhandwriting/matR_char_datatype.csv',
      help='CSV to use to create the dataset.')
  parser.add_argument('--char_data', nargs='?', type=int, const=1, required=True,
      help='1 if char data 0 for word data.')
  args = parser.parse_args()
  print(args)
  main(args.h5, args.csv, args.dataset_h5, is_char_data=args.char_data) 

