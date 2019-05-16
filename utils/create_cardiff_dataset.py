import argparse
import csv
import h5py
import json
import math
import numpy as np
import os
import sys
import types

import data_utils as global_utils

VIDEO_FPS = 30.0

def get_gesture_type_from_text(txt):
  if "Nod" in txt:
    return 1
  elif "Tilt" in txt:
    return 6
  elif "Shake" in txt:
    return 8
  else:
    assert(false)
  return -1


def get_all_annotations(txt_dir):
  '''
  txt_dir: Directory containing txt files exported from ELAN. Each txt file is
  tab delimited and contains the start and end timestamps
  '''
  file_to_annotations = {}
  for f in os.listdir(txt_dir):
    if f.endswith("txt"):
      file_to_annotations[f[:-4]] = []
      with open(os.path.join(txt_dir, f), 'r') as txt_f:
        csv_reader = csv.reader(txt_f, delimiter='\t')
        for row in csv_reader:
          gest_type = get_gesture_type_from_text(row[0])
          # Time is in ss:msss format
          start_time, end_time = float(row[2].strip()), float(row[3].strip())
          start_frame = math.floor(VIDEO_FPS*start_time)
          end_frame = round(VIDEO_FPS*end_time)
          # Remove the file extension from file name
          file_to_annotations[f[:-4]].append([gest_type, start_frame, end_frame])

  return file_to_annotations 

def get_openface_features(openface_dir):
  '''
  Read the openface features into a dictionary.
  '''
  feat_st_idx, feat_end_idx = 4, 151
  file_to_openface_feats = {}
  for f in os.listdir(openface_dir):
    with open(os.path.join(openface_dir, f), 'r') as openface_f:
      csv_reader = csv.reader(openface_f, delimiter=',')
      row_count = sum(1 for row in csv_reader)  

      feats = np.zeros((row_count-1, feat_end_idx-feat_st_idx+1))
      # Reinstantiate csv reader
      csv_reader = csv.reader(openface_f, delimiter=',')
      for idx, row in enumerate(csv_reader):
        if idx > 0:  # first row is column names
          for i in range(feat_st_idx, feat_end_idx+1):
            feats[idx-1, i] = float(row[i].strip())
      # Remove file extension from file name
      file_to_openface_feats[f[:-4]] = feats 
  return file_to_openface_feats

def main(openface_dir, txt_dir, save_path):
  file_to_annotations = get_all_annotations(txt_dir)
  file_to_feats = get_openface_features(openface_dir)

  # Merge annotations and features together
  for f, feat in file_to_feats.iteritems():
    file_to_data = {}
    annotations = file_to_annotations[f]
    y = np.zeros(feat.shape[0])
    for (gest, start_frame, end_frame) in annotations:
      # TODO(Mohit): Multiple gestures at same time instance
      # There could be multiple gestures here at the same time not sure how to
      # deal with that.
      start_frame, end_frame = int(start_frame), int(end_frame)
      y[start_frame:end_frame+1] = gest

    # save file_to_annotations
    p = os.path.join(save_path, f+'.h5')
    h5_f = h5py.File(p, 'a')
    del h5_f['annotations']
    h5_f['annotations'] = y.astype(int)
    # global_utils.recursively_save_dict_contents_to_group(
    #    h5_f, '/', {'features': np.array(h5_f['features']), 'annotations': y})
    h5_f.flush()
    h5_f.close()
    print('Did save h5 file to {}'.format(p))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
          description='Convert all annotated ELAN exported files to h5/csv.')
  parser.add_argument('--openface_dir', nargs='?', type=str, const=1,
      required=True, default='', help='Directory containing openface csvs.')
  parser.add_argument('--txt_dir', nargs='?', type=str, const=1, required=True,
      default='', help='Directory containing ELAN exported files.')
  parser.add_argument('--save_dir', nargs='?', type=str, const=1, required=True,
      help='Dir to store h5 file file and csv.')
  args = parser.parse_args()
  print(args)

  main(args.openface_dir, args.txt_dir, args.save_dir)

