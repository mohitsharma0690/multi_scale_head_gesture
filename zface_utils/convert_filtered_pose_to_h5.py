import numpy as np
import h5py
import sys
import os
import json
import csv
import types
import argparse

if __name__ == '__main__' and  __package__ is None:
    print('appending to sys path')
    print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils as global_utils

def process_mat(mat_dir, h5_dir):
  pass

def process_csv(csv_dir, h5_dir):
  for f in os.listdir(csv_dir):
    if f.endswith("csv"):
      h5_dict = {}
      with open(os.path.join(csv_dir, f), 'r') as csv_f:
        csv_reader = csv.reader(csv_f)
        all_rows = []
        for row in csv_reader:
          all_rows.append(np.array(row, dtype=float))
        a = np.array(all_rows).T
        assert(a.shape[1] == 6)
        h5_dict['features'] = a
    
      h5_file_path = os.path.join(h5_dir, f[:-4]+'.h5')
      h5_f = h5py.File(h5_file_path, 'w')
      global_utils.recursively_save_dict_contents_to_group(h5_f, '/', h5_dict)
      h5_f.flush()
      h5_f.close()

def main(h5_dir, csv_dir, mat_dir):
  if len(csv_dir) > 0:
    process_csv(csv_dir, h5_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Convert zface results to h5.')
  parser.add_argument('--csv_dir', nargs='?', type=str, const=1,
      required=True, default='', help='Directory containing csv files')
  parser.add_argument('--h5_dir', nargs='?', type=str, const=1,
      default='', help='Directory to save h5 files.')
  parser.add_argument('--mat_dir', nargs='?', type=str, const=1,
      default='', help='Directory containing mat files.')
  args = parser.parse_args()
  print(args)

  main(args.h5_dir, args.csv_dir, args.mat_dir)
