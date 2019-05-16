import argparse
import os
import numpy as np
import data_utils

# Example usage:
# data_utils.calculate_mean_std('../openface_data/cpm_output', 'joints', 'cpm_mean_std.h5', suffix='static.mp4.txt.h5')

def main(h5_dir, h5_group, save_h5_name, prefix=None, suffix=None):
  data_utils.calculate_mean_std(h5_dir, h5_group, save_h5_name, 
      prefix=prefix, suffix=suffix)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
          description='Convert all json predictions to h5.')
  parser.add_argument('--h5_dir', nargs='?', type=str, const=1,
          required=True, default='', help='h5 files directory.')
  parser.add_argument('--h5_group', nargs='?', type=str, const=1,
          required=True, default='', help='Group to use for calculating stats.')
  parser.add_argument('--save_h5_name', nargs='?', type=str, const=1,
          required=True, help='h5 filename to save statistics in.')
  parser.add_argument('--prefix', nargs='?', type=str, const=1,
          help='h5 file prefix to use in h5_dir')
  parser.add_argument('--suffix', nargs='?', type=str, const=1,
          help='h5 file suffix to use in h5_dir.')
  args = parser.parse_args()
  print(args)

  main(args.h5_dir, args.h5_group, args.save_h5_name, args.prefix, args.suffix)
