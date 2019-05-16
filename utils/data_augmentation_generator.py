import time
import os
import h5py
import math
import sys
import pdb
import json
import itertools
import csv
import types
import argparse

import numpy as np
import cPickle as cp
import pandas as pd
from scipy.interpolate import splev, splrep
from scipy.interpolate.rbf import Rbf
from collections import namedtuple
from copy import deepcopy

import convert_cpm_json_to_h5
import data_utils
from data_utils import GestureListUtils
from video_type_csv_info import VideoTypeCSVInfo

WIN_STEP = 3

# correct_gesture.csv header definition
CorrectGestureItem = namedtuple('CorrectGestureItem',
    'filename,old_label,new_label,start_frame,end_frame,desc')

class UpdateGestureCSVItem(object):

  @classmethod
  def item_from_csv_row(cls, csv_row):
    return cls(csv_row['filename'],
        int(csv_row['start_frame']),
        int(csv_row['end_frame']),
        int(csv_row['old_label']),
        csv_row['new_label'])

  @staticmethod
  def parse_new_labels(new_labels_str, start_frame, end_frame):
    ''' Get a list of tuples in sequential order
    new_labels_str: The format for the new labels string in csv is
    <end_frame_1>:<new_label_1>$<end_frame_2>:<new_label_2>$...
    '''
    frame_splits = new_labels_str.split('$')
    st = start_frame
    new_labels = []
    if frame_splits[0] == '-1':
      return []
    for i, s in enumerate(frame_splits):
      end, label = s.split(':')
      end, label = int(end), int(label)
      new_labels.append((st, end, label)) 
      st = end + 1
    assert st == end_frame + 1, 'Incorrect last frames of old, new gesture'

    return new_labels

  @staticmethod
  def write_dict_to_csv(item_map, csv_path):
    ''' Write dict with UpdateGestureCSVItem as values to csv
    '''
    field_names = ['filename', 'old_label', 'new_label', 'start_frame', \
        'end_frame', 'desc']
    with open(csv_path, 'w') as csv_f:
      csv_writer = csv.DictWriter(csv_f, field_names, lineterminator='\n')
      csv_writer.writeheader()
      for k in sorted(item_map.keys()):
        v = item_map[k]
        d = v.get_dict_repr()
        csv_writer.writerow(d)
    print("Did save csv at {}".format(csv_path))

  def __init__(self, filename, start_frame, end_frame, old_label,
      new_labels_str, desc=''):
    self._filename = filename 
    self._start_frame = start_frame
    self._end_frame = end_frame
    self._old_label = old_label
    self._new_labels = UpdateGestureCSVItem.parse_new_labels(
        new_labels_str, start_frame, end_frame)
    self._desc = desc

  @property
  def filename(self):
    return self._filename

  @property
  def start_frame(self):
    return self._start_frame

  @property
  def end_frame(self):
    return self._end_frame

  @property
  def old_label(self):
    return self._old_label

  @property
  def new_labels(self):
    return self._new_labels
  
  @property
  def new_labels_str(self):
    s = ''
    for start, end, label in self.new_labels:
      s += '{}:{}$'.format(end, label)
    return s[:-1]  # Remove last $

  def get_dict_repr(self):
    return {
        'filename': self.filename,
        'old_label': self.old_label,
        'new_label': self.new_labels_str,
        'start_frame': self.start_frame,
        'end_frame': self.end_frame,
        'desc': '',
    }

class GestureListWriter(object):
  ''' Update Gestures list using CSV file with rows as UpdateGestureCSVItem.
  '''
  def __init__(self, gest_list_h5):
    self._gest_list_h5 = gest_list_h5

  @staticmethod
  def read_update_gest_csv(update_gest_csv):
    ''' Reads the update gest csv.
    Return: A dictionary with keys as '<user_file_name>_<gest_start_frame>'
    and value as UpdateGestureCSVItem.
    '''
    csv_info = {}
    with open(update_gest_csv, 'r') as csv_f:
      csv_reader = csv.DictReader(csv_f)
      for row in csv_reader:
        item = UpdateGestureCSVItem.item_from_csv_row(row)
        key = item.filename + '_' + str(item.start_frame)
        assert csv_info.get(key) is None, \
            'Duplicate start frame rows in csv {}: {}'.format(
                update_gest_csv, key)
        csv_info[key] = item
    return csv_info
  
  def get_new_gesture_list(self, update_gest_csv): 
    gest_list_h5 = self._gest_list_h5

    # Read the CSV
    update_csv_info = GestureListWriter.read_update_gest_csv(update_gest_csv)

    # We edit the gest_list_data while using org_gest_list_data
    gest_list_data = GestureListUtils.read_gesture_list(gest_list_h5)
    org_gest_list_data = GestureListUtils.read_gesture_list(gest_list_h5)
    for group in org_gest_list_data.keys():
      for user in org_gest_list_data[group].keys():
        for label, gest_list in org_gest_list_data[group][user].iteritems():
          if len(gest_list) == 0:
            continue
          for j in xrange(len(gest_list)):
            start_frame = gest_list[j][0]
            key = user + '_' + str(start_frame)
            if update_csv_info.get(key) is not None:
              # Need to update this CSV item
              csv_item = update_csv_info[key]

              # NOTE: Although we append new labels at the end while we remove
              # from the front the indexes here might not correspond correctly
              # since we might delete from the back of the gest list first

              # Remove old item
              for i in xrange(len(gest_list_data[group][user][label])):
                if gest_list_data[group][user][label][i][0] == start_frame:
                  gest_list_data[group][user][label].pop(i)
                  break

              # Append new items
              for st, end, new_label in csv_item.new_labels:
                if new_label >= 0:
                  gest_list_data[group][user][str(new_label)].append([st, end])

    return gest_list_data
  
  def gest_list_to_array(self, gest_list):
    for group in gest_list.keys():
      for user in gest_list[group].keys():
        for label in gest_list[group][user].keys():
          gest_list[group][user][label] = np.array(
              gest_list[group][user][label])
    return gest_list

  def save_gesture_list(self, gest_list, new_gest_list_h5):
    new_gest_list_path = os.path.join(os.path.dirname(self._gest_list_h5),
        new_gest_list_h5)
    f = h5py.File(new_gest_list_path, 'w')
    data_utils.recursively_save_dict_contents_to_group(f, '/', gest_list)
    f.flush()
    f.close()
    print("Did write h5 file {}".format(new_gest_list_path))


class GestureListAugmentationUtils(object):
  ''' Util methods for data augmentation. '''

  @staticmethod
  def copy_augmentations_using_csv(aug_h5, new_aug_h5, video_type_csv):
    aug_h5_f = h5py.File(aug_h5, 'r')
    new_aug_h5_f = h5py.File(new_aug_h5, 'w')

    csv_info = VideoTypeCSVInfo(video_type_csv)
    for g in ['train', 'test']:
      new_group = new_aug_h5_f.create_group(g)
      files = csv_info.train_files() if g == 'train' else csv_info.test_files()
      for f in files:
        org_group = 'train' if f in aug_h5_f['train'].keys() else 'test'
        file_group = new_group.create_group(f)

        for t in aug_h5_f[org_group][f].keys():
          t_group = file_group.create_group(t)
          for win_size in aug_h5_f[org_group][f][t].keys():
            win_group = t_group.create_group(win_group)
            data_path = '/{}/{}/{}/{}'.format(org_group, f, t, win_size)
            aug_h5_f.copy(data_path, win_group)

    new_aug_h5_f.flush()
    new_aug_h5_f.close()
    aug_h5_f.close()

class OpenfaceAugmentationType:
  LANDMARKS_ONLY=1
  LANDMARKS_AND_VELOCITY=2
  ALL_LANDMARKS_AND_POSE=3

  def __init__(self, aug_type):
    assert(aug_type == OpenfaceAugmentationType.LANDMARKS_ONLY or
        aug_type == OpenfaceAugmentationType.LANDMARKS_AND_VELOCITY or
        aug_type == OpenfaceAugmentationType.ALL_LANDMARKS_AND_POSE)
    self.aug_type = aug_type


def get_all_seq_augmentations_3(X, frame, win_size):
  '''
  Do data augmentation on the given frame in the h5fname for a given
  win_size. This is different than `get_all_seq_augmentations` since it
  returns fixed number of augmentations for all window sizes.
  K:number of augmentations in total. This uses a different algorithm to
  generate augmentations i.e. it involves both compression and expansion
  of given signal.
  '''
  K = 32
  start_f = int(frame-win_size/2)
  end_f = start_f + win_size
  start_ext_f = int(frame-3*win_size/2.0)
  end_ext_f = start_ext_f + 3*win_size
  X_org = X[start_f:end_f, :]
  X_ext, can_compress = None, False
  if start_ext_f >= 0 and end_ext_f < X.shape[0]:
    can_compress = True
    X_ext = X[start_ext_f:end_ext_f]

  x_axis = np.linspace(1, win_size, win_size)
  all_aug = np.zeros((K, X_org.shape[1], win_size))

  # Loop through all the different feature values used from openface signal
  for i in range(X_org.shape[1]):
    all_aug_idx = 0
    # Loop through all window sizes (for now we just have 1 win_size)
    y = X_org[:, i]
    x = np.linspace(1, win_size, win_size)

    # expansion augmentation
    # increase interpolation scale by 2 times. This will calculate the
    # interpolating signal at the half values i.e. 1.5, 2.5 etc.
    x2 = np.linspace(1, win_size, 2*win_size)
    for poly_order in [3]:
      tck = splrep(x, y, k=poly_order)
      # y2 has 3*win_size points now
      y2 = splev(x2, tck)

      step_size, final_K = 1, 16
      if win_size == 32:
        step_size, final_K = 2, 32
      elif win_size == 64:
        step_size, final_K = 4, 64
      for i1 in range(0, final_K, step_size):
        all_aug[all_aug_idx, i, :] = y2[i1:i1+win_size]
        all_aug_idx = all_aug_idx + 1
        if not can_compress:
          # If we can't compress the signal just add the reversed signal
          all_aug[all_aug_idx, i, :] = y2[i1+win_size-1:i1-1:-1]
          all_aug_idx = all_aug_idx + 1

    # compression augmentation
    if can_compress:
      step_size, final_K = 1, 16
      if win_size == 32:
        step_size, final_K = 2, 32
      elif win_size == 64:
        step_size, final_K = 4, 64
      for i1 in range(0,final_K, step_size):
        y = X_ext[i1:i1+2*win_size, i]
        x = np.linspace(1, 2*win_size, 2*win_size)
        tck = splrep(x, y, k=3)
        x2 = np.linspace(1, 2*win_size, win_size)
        y2 = splev(x2, tck)
        all_aug[all_aug_idx, i, :] = y2
        all_aug_idx = all_aug_idx + 1
  assert(all_aug_idx == K)
  return all_aug

def get_all_seq_augmentations_4(X, frame, win_size, num_augmentations=32):
  '''
  Do data augmentation on the given frame in the h5fname for a given
  win_size. This is different than `get_all_seq_augmentations` since it
  returns fixed number of augmentations for all window sizes.
  K:number of augmentations in total. This uses a different algorithm to
  generate augmentations i.e. it involves both compression and expansion
  of given signal.
  '''
  K = num_augmentations
  speed_up = [1.125, 1.25, 1.375]
  all_win_sizes = [int(round(win_size * x)) for x in speed_up]
  points_per_win_size = {
      # 16:[2, 3, 3], 32:[2, 3, 3], 64:[2, 3, 3]
      16:[1, 1, 2], 32:[1, 1, 2], 64:[1, 1, 2]
  }
  org_win_size = win_size
  all_aug = np.zeros((K, X.shape[1], org_win_size))

  for feat_idx in range(X.shape[1]):
    all_aug_idx = 0
    win_idx = 0
    for win_size in all_win_sizes:
      start_f = int(frame - org_win_size/2)
      end_f = start_f + org_win_size
      start_ext_f = int(frame-win_size/2.0)
      end_ext_f = start_ext_f + win_size
      X_org = X[start_f:end_f, :]
      X_ext, can_compress = None, False
      if start_ext_f >= 0 and end_ext_f < X.shape[0]:
        can_compress = True
        X_ext = X[start_ext_f:end_ext_f]
      else:
        can_compress = True
        X_ext = X[start_ext_f:, :]


      # Loop through all the different feature values used from openface
      # signal
      # Loop through all window sizes (for now we just have 1 win_size)
      y = X_org[:, feat_idx]
      x = np.linspace(1, org_win_size, org_win_size)

      # expansion augmentation
      # increase interpolation scale by 2 times. This will calculate the
      # interpolating signal at the half values i.e. 1.5, 2.5 etc.
      x2 = np.linspace(1, org_win_size, win_size)

      # range(2) since we do 2 kinds of interpolation for now.
      for j in range(2):
        if j == 0:
          # Cubic interpolation
          tck = splrep(x, y, k=3)
          y2 = splev(x2, tck)
        else:
          rbf_adj = Rbf(x, y, function='gaussian')
          y2 = rbf_adj(x2)

        if np.abs(np.mean(y) - np.mean(y2)) > 100:
          assert(False)

        num_interpolations = points_per_win_size[org_win_size][win_idx]
        if win_size - org_win_size == num_interpolations:
          for i in range(win_size-org_win_size):
            all_aug[all_aug_idx, feat_idx, :] = y2[i:i+org_win_size]
            all_aug_idx = all_aug_idx + 1
        else:
          for i in range(num_interpolations):
            idx = np.random.randint(0, win_size - org_win_size)
            all_aug[all_aug_idx, feat_idx, :] = y2[idx:idx+org_win_size]
            all_aug_idx = all_aug_idx + 1

      assert(can_compress)
      # compression augmentation
      if can_compress:
        y = X_ext[:, feat_idx]
        x = np.linspace(1, win_size, win_size)
        tck = splrep(x, y, k=3)
        rbf_adj = Rbf(x, y, function='gaussian')

        num_interpolations = points_per_win_size[org_win_size][win_idx]
        assert(win_size - org_win_size >= num_interpolations)
        for i in range(num_interpolations):
          x2 = np.linspace(1+i, win_size, org_win_size)
          y2_cubic = splev(x2, tck)
          # For egocentric videos the mean change is a lot greater than 100px
          # easily. But for static camera case it is ok.
          if np.abs(np.mean(y) - np.mean(y2_cubic)) > 500:
            print("NOTE!! Very high mean difference during interpolation. " \
                "Most likely some value is inf. Frame# {}".format(
                  frame))
          all_aug[all_aug_idx, feat_idx, :] = y2_cubic
          all_aug_idx = all_aug_idx + 1
          y2_rbf = rbf_adj(x2)
          if np.abs(np.mean(y) - np.mean(y2_rbf)) > 500:
            print("NOTE!! Very high mean difference during interpolation. " \
                "Most likely some value is inf. Frame# {}".format(
                  frame))
          all_aug[all_aug_idx, feat_idx, :] = y2_rbf
          all_aug_idx = all_aug_idx + 1

      win_idx = win_idx + 1

    assert(all_aug_idx == K)

  return all_aug

def create_zface_data_augmentation_2(fdir, gest_seq_h5, new_filepath,
    aug_features, win_sizes=[16,32,64], labels=[6,7,8,9,10]):
  h5_f = h5py.File(gest_seq_h5, 'r')
  h5_train = h5_f['train']
  new_h5 = h5py.File(new_filepath, 'w')
  all_aug_map = {}
  count = 0
  for f in h5_train.keys():
    all_aug_map[f] = {}
    v = h5_train[f]
    fpath = os.path.join(fdir, f)
    currf_h5 = h5py.File(fpath, 'r')
    X, feats = currf_h5['features'], []
    assert(len(aug_features) > 0)
    for j in aug_features:
      feats.append(X[:, j])
    X = np.vstack(feats).T

    for i in labels:
      str_i = str(i)
      gest_seq = v[str(i)]
      all_aug_map[f][str_i] = {}
      for seq in gest_seq:
        if type(seq) != type(np.array([])):
          continue
        gest_len = seq[1] - seq[0]
        gest_start = seq[0] + (gest_len // 5)
        gest_end = seq[1] - (gest_len // 5)
        for t in range(gest_start, gest_end+1, WIN_STEP):
          str_t = str(t)
          all_aug_map[f][str_i][str_t] = {}
          for win_size in win_sizes:
            seq_augmentation = get_all_seq_augmentations_4(X, t, win_size)
            all_aug_map[f][str_i][str_t][str(win_size)] = seq_augmentation
            count = count + 1
            if count % 300 == 0:
              print('Did get seq augmentation for file: {}, label: {}, ' \
                    't: {}, win_size: {}'.format(f, i, t, win_size))

    print('Did process file {}'.format(f))
    data_utils.recursively_save_dict_contents_to_group(
        new_h5, str('/'+f+'/'), all_aug_map[f])
    print('Did write {} augmentations'.format(f))
    new_h5.flush()
    all_aug_map[f] = {}

  new_h5.flush()
  new_h5.close()


def create_cpm_data_augmentation_2(fdir, gest_seq_h5, new_filepath,
    win_sizes=[16,32,64], labels=[6,7,8,9,10], aug_pose_only=True):
  h5_f = h5py.File(gest_seq_h5, 'r')
  h5_train = h5_f['train']
  new_h5 = h5py.File(new_filepath, 'w')
  all_aug_map = {}
  count = 0
  for f in h5_train.keys():
    all_aug_map[f] = {}
    v = h5_train[f]
    fpath = os.path.join(fdir, f)
    currf_h5 = h5py.File(fpath, 'r')
    assert(aug_pose_only)
    X, cpm_feats = currf_h5['joints'], []
    for j in ['Nose', 'Neck', 'RShoulder', 'LShoulder']:
      j_idx = convert_cpm_json_to_h5.get_joint_type_to_h5_idx(j)
      cpm_feats.append(X[:, j_idx:j_idx+2])
    X = np.hstack(cpm_feats)

    for i in labels:
      str_i = str(i)
      gest_seq = v[str(i)]
      all_aug_map[f][str_i] = {}
      for seq in gest_seq:
        if type(seq) != type(np.array([])):
          continue
        gest_len = seq[1] - seq[0]
        gest_start = seq[0] + (gest_len // 5)
        gest_end = seq[1] - (gest_len // 5)
        for t in range(gest_start, gest_end+1, WIN_STEP):
          str_t = str(t)
          all_aug_map[f][str_i][str_t] = {}
          for win_size in win_sizes:
            seq_augmentation = get_all_seq_augmentations_4(X, t, win_size)
            all_aug_map[f][str_i][str_t][str(win_size)] = seq_augmentation
            count = count + 1
            if count % 1000 == 0:
              print('Did get seq augmentation for file: {}, label: {}, ' \
                    't: {}, win_size: {}'.format(f, i, t, win_size))

    print('Did process file {}'.format(f))
    data_utils.recursively_save_dict_contents_to_group(
        new_h5, str('/'+f+'/'), all_aug_map[f])
    print('Did write {} augmentations'.format(f))
    new_h5.flush()
    all_aug_map[f] = {}

  new_h5.flush()
  new_h5.close()


def create_data_augmentation_2(fdir, gest_seq_h5, new_filepath,
    win_sizes=[16,32,64], labels=[6,7,8,9,10],
    aug_type=OpenfaceAugmentationType.LANDMARKS_ONLY):
  h5_f = h5py.File(gest_seq_h5, 'r')
  h5_train = h5_f['train']
  new_h5 = h5py.File(new_filepath, 'w')
  all_aug_map = {}
  count = 0
  for f in h5_train.keys():
    all_aug_map[f] = {}
    v = h5_train[f]
    fpath = fdir + '/' + f
    currf_h5 = h5py.File(fpath, 'r')
    num_augmentations = 16
    if aug_type.aug_type == OpenfaceAugmentationType.LANDMARKS_ONLY:
      X = data_utils.trim_extra_landmarks(currf_h5['features'])
    elif aug_type.aug_type == OpenfaceAugmentationType.LANDMARKS_AND_VELOCITY:
      X, _ = data_utils.process_data(
          currf_h5['features'], currf_h5['annotations'])
    elif aug_type.aug_type == OpenfaceAugmentationType.ALL_LANDMARKS_AND_POSE:
      X = np.array(currf_h5['features'])
      X = X[:, :148]
      num_augmentations = 16
    else:
      assert(False)


    for i in labels:
      str_i = str(i)
      gest_seq = v[str(i)]
      all_aug_map[f][str_i] = {}
      for seq in gest_seq:
        if type(seq) != type(np.array([])):
          continue
        gest_len = seq[1] - seq[0]
        gest_start = seq[0] + (gest_len // 5)
        gest_end = seq[1] - (gest_len // 5)
        for t in range(gest_start, gest_end+1, WIN_STEP):
          str_t = str(t)
          all_aug_map[f][str_i][str_t] = {}
          for win_size in win_sizes:
            seq_augmentation = get_all_seq_augmentations_4(
                X, t, win_size, num_augmentations=num_augmentations)
            all_aug_map[f][str_i][str_t][str(win_size)] = seq_augmentation
            count = count + 1
            if count % 300 == 0:
              print('Did get seq augmentation for file: {}, label: {}, ' \
                    't: {}, win_size: {}'.format(f, i, t, win_size))

    print('Did process file {}'.format(f))
    data_utils.recursively_save_dict_contents_to_group(
        new_h5, str('/'+f+'/'), all_aug_map[f])
    print('Did write {} augmentations'.format(f))
    new_h5.flush()
    all_aug_map[f] = {}

  new_h5.flush()
  new_h5.close()

def read_correct_gestures(correct_gest_csv, num_classes):
  '''
  Return: Dict of Dict where the outer dict has the filename as the key and a
  dict as the value. The inner dict has the frame number as the key and a named
  tuple for the csv row.
  '''
  correct_gest_by_file = {}
  if correct_gest_csv is None:
    return None

  items = map(CorrectGestureItem._make,
      csv.reader(open(correct_gest_csv, 'r')))
  items = items[1:]  # drop the first row (headers)

  # convert to integers
  items = [CorrectGestureItem(i.filename,int(i.old_label),int(i.new_label),
    int(i.start_frame),int(i.end_frame),i.desc) for i in items]

  for item in items:
    if correct_gest_by_file.get(item.filename, None) is None:
      correct_gest_by_file[item.filename] = {}
    file_gest = correct_gest_by_file[item.filename]
    assert(file_gest.get(item.start_frame, None) is None)
    file_gest[item.start_frame] = item

  return correct_gest_by_file

def read_incorrect_gestures(incorrect_gest_csv, num_classes):
  incorrect_gest_by_file = {}
  with open(incorrect_gest_csv, 'r') as csvfile:
    r = csv.reader(csvfile, delimiter=',')
    for row in r:
      if row[0] == 'filename':
        continue  # First row
      file_name = row[0]
      if incorrect_gest_by_file.get(file_name) is None:
        incorrect_gest_by_file[file_name] = \
            [[] for _ in range(num_classes)]
      label = int(row[1])
      start_idx = int(row[2])
      incorrect_gest_by_file[file_name][label].append(start_idx)
  return incorrect_gest_by_file

def read_video_type(vid_type_csv):
  type_by_filename = {}
  with open(vid_type_csv, 'r') as csvfile:
    r = csv.reader(csvfile, delimiter=',')
    for row in r:
      if row[0] == 'filename':
        continue  # First row
      type_by_filename[row[0]] = row[1]
  return type_by_filename

def save_all_gest_dict_2(fsave_path, gest_list_by_type):
  '''
  Save both train and test data in one h5 file.
  '''
  hf = h5py.File(fsave_path, 'w')
  for (data_type, gest_list_by_file) in gest_list_by_type.iteritems():
    data_grp = hf.create_group(data_type)
    for (gest_file_name, gest_list) in gest_list_by_file.iteritems():
      grp = data_grp.create_group(gest_file_name)
      for i in range(len(gest_list)):
        if len(gest_list[i]):
          grp.create_dataset(str(i), data=np.array(gest_list[i]))
        else:
          grp.create_dataset(str(i), data=np.array([0, 0]))

  hf.flush()
  hf.close()
  print 'Write dataset to {}'.format(fsave_path)


def update_correct_invalid_gestures(gests_by_label, correct_gests):
  '''
  correct_gests: Map with correct gesture start frame as key and correct gesture
  as value.
  '''
  new_gests_by_label = deepcopy(gests_by_label)
  for label in range(len(gests_by_label)):
    curr_gests = gests_by_label[label]
    i = 0
    while i < len(curr_gests):
      gest = curr_gests[i]
      if gest[0] in correct_gests:
        # print('Found correct gesture {}: {}'.format(
        #   gest, correct_gests[gest[0]]))
        # We found a gesture that needs to be replaced
        all_correct_gests = [correct_gests[gest[0]]]
        needs_update = False
        if correct_gests[gest[0]].new_label >= 0:
          assert(label == correct_gests[gest[0]].new_label)
        else:
          # The new label is -1 in this case, hence we need to update i.e.
          # remove this
          assert(label == correct_gests[gest[0]].old_label)
          needs_update = True

        if not needs_update:
          i = i + 1
          continue

        # Now remove this incorrect gesture. We do the loop here since we split
        # gestures as we group them by labels. So one gesture can be represented
        # across multiple list values which need to be removed here.
        new_gests_by_label[label].remove(gest)
        # gest_end = gest[1]
        #while i+1 < len(curr_gests) and curr_gests[i][1] == curr_gests[i+1][0]:
        #  new_gests_by_label[label].remove(curr_gests[i+1])
        #  gest_end = curr_gests[i+1][1]
        #  i = i + 1

      i = i + 1

  for i in range(len(new_gests_by_label)):
    new_gests_by_label[i].sort(key=lambda(seq): seq[0])
  return new_gests_by_label

def add_correct_gestures(gests_by_label, correct_gests):
  new_gests_by_label = deepcopy(gests_by_label)
  for label in range(len(gests_by_label)):
    curr_gests = gests_by_label[label]
    i = 0
    while i < len(curr_gests):
      gest = curr_gests[i]
      if gest[0] in correct_gests:
        # print('Found correct gesture {}: {}'.format(
        #   gest, correct_gests[gest[0]]))
        # We found a gesture that needs to be replaced
        all_correct_gests = [correct_gests[gest[0]]]
        assert(label == correct_gests[gest[0]].old_label)

        # Now remove this incorrect gesture. We do the loop here since we split
        # gestures as we group them by labels. So one gesture can be represented
        # across multiple list values which need to be removed here.
        new_gests_by_label[label].remove(gest)
        gest_end = gest[1]
        while i+1 < len(curr_gests) and curr_gests[i][1] == curr_gests[i+1][0]:
          new_gests_by_label[label].remove(curr_gests[i+1])
          gest_end = curr_gests[i+1][1]
          i = i + 1

        for t in range(gest[0]+1, gest_end):
          # Skip new labels marked with -1 since we aren't clear about their
          # labels.
          if t in correct_gests and correct_gests[t].new_label != -1:
            assert(label == correct_gests[t].old_label)
            all_correct_gests.append(correct_gests[t])

        # Replace it with the correct gestures for appropriate labels
        for correct_item in all_correct_gests:
          # Since the last frame is not included we do a +1.
          new_gests_by_label[correct_item.new_label].append(
              [correct_item.start_frame, correct_item.end_frame+1])
      i = i + 1

  for i in range(len(new_gests_by_label)):
    new_gests_by_label[i].sort(key=lambda(seq): seq[0])
  return new_gests_by_label


def get_gest_list_file(fdir, fname, win_len, num_classes=11,
    incorrect_gests=None, correct_gests=None):
  '''
  Returns a list of list of all the gestures in a given h5file. The class
  type is used as an index into the top level list. The inner list is a list
  of start idx, end idx, gest_type for the respective gesture.
  incorrect_gests: List of list. The outer list represents each gesture
  indexed from 0 to num_classes. Each of these lists is a list of
  incorrectly labeled start frames.
  correct_gests: Dictionary with the start frame of the correct gesture as it's
  key and CorrectGestureItem tuple as it's value.
  '''
  fpath = fdir + '/' + fname
  gest_idx = [[] for _ in range(num_classes)]
  hf = h5py.File(fpath, 'r')
  d_annotations = hf.get('annotations')
  a = np.array(d_annotations).astype(int)

  #prepare train sequence based on class
  c, min_gesture_threshold = 0, 6
  min_none_threshold = 10
  # First few frames can be skipped they are mostly incorrect since camera,
  # hand and other artifacts cover it.
  i = 100
  last_idx = len(a)-50
  while i < last_idx:
    if a[i] != a[i+1] or (c == win_len-1):
      #annotations differ or max length reached
      if ((a[i] == 0 and c > min_none_threshold) or
        (a[i] > 0 and c > min_gesture_threshold)):
        gest_idx[a[i]].append([i-c, i+1]) # Add as relevant gesture

        # Gaussian noise can be added at runtime for other gestures.
        # since batch length is decided on nods we add random noise
        # here itself for nods.
        if a[i] == -11:
          r_mean, r_var, r_samples = 5, 10, 5
          for r in range(r_samples):
            r_val = int(np.random.normal(r_mean, r_var, 1))
            left_idx, right_idx = i-c, i+1
            while left_idx-r_val < 0 or right_idx - left_idx + r_val < 10:
              r_val = int(np.random.normal(r_mean, r_var, 1))
            gest_idx[a[i]].append([left_idx-r_val, right_idx])

      # reset to a new sequence.
      c=0
      if incorrect_gests is not None and i+1 in incorrect_gests[a[i+1]]:
        # Found incorrect gesture. Don't include this gesture
        j = i+1
        while j < last_idx and a[j] == a[i+1]:
          j = j + 1
        i = j
        continue
    else:
      c+=1
    i = i + 1

  if correct_gests is not None:
    #return add_correct_gestures(gest_idx, correct_gests)
    return update_correct_invalid_gestures(gest_idx, correct_gests)
  return gest_idx

def get_all_gest(fdir, win_len, flist=None, incorrect_gests=None,
    correct_gests=None):
  '''
  Returns a dictionary of all the gestures for each file. The dict key
  is the h5 filename and the value is a list of lists as returned by
  get_gest_list_file.
  incorrect_gests: If not None this is a dictionary with the filename as the
  key and a list of all gestures where each gesture is again a list of the
  incorrect gestures start frame in the video.
  correct_gests: If not None this is a dictionary of dictionary. The outer dict
  has the filename as the key and value as the inner dict. The inner dict has
  the start frame as the key and the named tuple CorrectGestureItem as it's
  value.
  '''
  num_classes = 11
  gest_list_by_file = {}
  for npfile in os.listdir(fdir):
    # Add file only if either flist is None which means we want to add all
    # files.
    # Or if flist is Not None only add files in flist since we only want to
    # add specific files
    if flist is None or (npfile in flist):
      incorrect_gests_file = None
      if incorrect_gests is not None:
        incorrect_gests_file = incorrect_gests.get(npfile, None)

      correct_gests_file = None
      if correct_gests is not None:
        correct_gests_file = correct_gests.get(npfile, None)

      gest_list = get_gest_list_file(fdir, npfile, win_len,
          incorrect_gests=incorrect_gests_file,
          correct_gests=correct_gests_file)

      gest_list_by_file[npfile] = gest_list
      g_len = [len(gest_list[i]) for i in xrange(len(gest_list))]
      print('{} stats: {}'.format(npfile, g_len))
  return gest_list_by_file

def save_gesture_list(h5_save_name, vid_type_csv, incorrect_gest_csv,
    correct_gest_csv, win_len,
    fdir='../openface_data/face_gestures/dataseto_text'):
  '''
  This is similar to save_custom_batch_data but in here we respect the
  incorrectly marked gestures as well as the type of the i.e. train and
  test. Note that the h5 files created using this should be used for k-fold
  cross validation since no explicit validation set is created.
  '''
  train_file, invalid_file, test_file = [], [], []
  type_by_filename = read_video_type(vid_type_csv)
  incorrect_gest_by_file = read_incorrect_gestures(incorrect_gest_csv, 11)
  correct_gest_by_file = read_correct_gestures(correct_gest_csv, 11)
  for npfile in os.listdir(fdir):
    if npfile.endswith("static.mp4.txt.h5") or True:
      dataset_for_file = type_by_filename[npfile]
      if dataset_for_file == 'train':
        train_file.append(npfile)
      elif dataset_for_file == 'test':
        test_file.append(npfile)
      elif dataset_for_file == 'incorrect':
        invalid_file.append(npfile)
      else:
        assert(False)

  # Get gesture dict as list
  train_gest = get_all_gest(fdir, win_len, flist=train_file,
      incorrect_gests=incorrect_gest_by_file,
      correct_gests=correct_gest_by_file)

  test_gest = get_all_gest(fdir, win_len, flist=test_file,
      incorrect_gests=incorrect_gest_by_file,
      correct_gests=correct_gest_by_file)

  save_all_gest_dict_2(
      h5_save_name,
      {'train': train_gest, 'test': test_gest})

def save_update_gest_list_h5(gest_list_h5, update_gest_csv, new_gest_list_h5):
  ''' Update gesture list using update gesture CSV.
  '''
  gest_list_writer = GestureListWriter(gest_list_h5)
  new_gest_list = gest_list_writer.get_new_gesture_list(update_gest_csv)
  new_gest_list = gest_list_writer.gest_list_to_array(new_gest_list)
  gest_list_writer.save_gesture_list(new_gest_list, new_gest_list_h5)
  print("Did udpate gest list {} using {}".format(
    gest_list_h5, update_gest_csv))


if __name__ == '__main__':

  # If code is executed as script add the above directory (root directory of
  # repo) to sys path
  if __package__ is None:
    print('appending to sys path')
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


  parser = argparse.ArgumentParser(
          description='Save gesture list as h5 and create augmented gestures.')
  parser.add_argument('--h5_dir', nargs='?', type=str, const=1,
      default='../openface_data/face_gestures/dataseto_text',
      help='h5 files directory')
  parser.add_argument('--save', nargs='?', type=int, const=1,
      default=0, help='1 if want to save new gest list as h5 file else 0')
  parser.add_argument('--gest_list_h5', nargs='?', type=str, const=1,
      default='', help='Path to save the gesture list h5 file.')
  parser.add_argument('--video_type_csv', nargs='?', type=str, const=1,
      default='../utils/video_type_1.csv',
      help='Video type csv filename.')
  parser.add_argument('--incorrect_gesture_csv', nargs='?', type=str, const=1,
      default='../utils/incorrect_gesture.csv',
      help='Incorrect gesture csv filename.')
  parser.add_argument('--correct_gesture_csv', nargs='?', type=str, const=1,
      default='', help='Correct gesture csv filename.')
  parser.add_argument('--save_aug_gest_h5', nargs='?', type=str, const=1,
      default='', help='Augmented gesture h5 filename.')
  parser.add_argument('--cpm_h5_dir', nargs='?', type=str, const=1,
      default='', help='CPM h5 files directory')
  parser.add_argument('--save_cpm_aug_gest_h5', nargs='?', type=str, const=1,
      default='', help='Augmented ConvPoseMachien gesture h5 filename.')
  parser.add_argument('--zface_h5_dir', nargs='?', type=str, const=1,
      default='', help='Directory with zface h5 files.')
  parser.add_argument('--save_zface_aug_gest_h5', nargs='?', type=str, const=1,
      default='', help='Augmented Z-face gesture h5 filename.')

  # Update gesture list
  parser.add_argument('--update_gest_csv', nargs='?', type=str, const=1,
      default='', help='CSV filepath to update gesture list.')
  parser.add_argument('--new_gest_list_h5', nargs='?', type=str, const=1,
      default='', help='h5 Filename for new gesture list.')
  parser.add_argument('--save_update_gest_list_h5',
      dest='save_update_gest_list_h5',
      action='store_true',
      help='Should save new updated gesture CSV')
  parser.add_argument('--no-save_update_gest_list_h5',
      dest='save_update_gest_list_h5',
      action='store_false',
      help="Don't save new updated gesture CSV")
  parser.set_defaults(save_update_gest_list_h5=False)
      

  args = parser.parse_args()
  print(args)

  gest_list_h5 = args.gest_list_h5
  vid_type_csv = args.video_type_csv
  incorrect_gest_csv = args.incorrect_gesture_csv
  correct_gest_csv = args.correct_gesture_csv \
      if len(args.correct_gesture_csv) > 0 else None
  update_gest_csv = args.update_gest_csv \
      if len(args.update_gest_csv) else None

  if args.save_update_gest_list_h5:
    save_update_gest_list_h5(
        gest_list_h5, update_gest_csv, args.new_gest_list_h5)

  if args.save:
    save_gesture_list(gest_list_h5, vid_type_csv, incorrect_gest_csv,
        correct_gest_csv, 120, fdir=args.h5_dir)  # win_len = 120

  if args.save_cpm_aug_gest_h5 is not None and \
      len(args.save_cpm_aug_gest_h5) > 0:
    create_cpm_data_augmentation_2(
        args.cpm_h5_dir, gest_list_h5, args.save_cpm_aug_gest_h5)

  if args.save_zface_aug_gest_h5 is not None and \
      len(args.save_zface_aug_gest_h5) > 0:
    create_zface_data_augmentation_2(
        args.zface_h5_dir, gest_list_h5, args.save_zface_aug_gest_h5,
        aug_features=[0,1,2,3,4,5])

  if args.save_aug_gest_h5 is not None and len(args.save_aug_gest_h5) > 0:
    create_data_augmentation_2(
        args.h5_dir, gest_list_h5, args.save_aug_gest_h5,
        win_sizes=[16,32,64],
        aug_type=OpenfaceAugmentationType(
          OpenfaceAugmentationType.LANDMARKS_ONLY),
        labels=range(1,11))
