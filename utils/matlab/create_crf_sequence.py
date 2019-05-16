import argparse
import h5py
import numpy as np
import os
import pdb
import scipy as sp
import sys 

if __name__ == '__main__' and  __package__ is None:
    print('appending to sys path')
    print(os.path.dirname(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.append(
        os.path.dirname(
          os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))

from utils import data_utils as global_utils
from simple_lstm import data_augmentation_generator as data_generator

def get_zface_gesture(h5_filepath, start, end):
  f = h5py.File(h5_filepath, 'r')
  X = np.array(f['features'])
  f.close()
  new_X = np.hstack((X[start:end, 0:1], X[start:end, 3:]))
  return new_X

def get_joint_gestures_labels(gest_map, num_feats_expected=55):
  gests, labels = [], []
  for gest_type, gest_X_list in gest_map.iteritems():
    for gest_X in gest_X_list:
      gest_X = gest_X.transpose()
      assert(gest_X.shape[0] == num_feats_expected)
      gests.append(gest_X)
      l = np.ones((1, gest_X.shape[1]), dtype=int)
      l = l * gest_type
      labels.append(l)
  return gests, labels

def get_real_features(
        X_file_to_openface,
        X_file_to_cpm,
        X_file_to_zface,
        file_name,
        start_frame,
        end_frame,
        openface_mean_std,
        cpm_mean_std):
  '''
  Return each original feature. Process each feature as required
  '''
  curr_X_openface, curr_X_cpm, curr_X_zface = None, None, None
  if X_file_to_openface is not None:
    X_openface = X_file_to_openface[file_name]
    curr_X_openface = X_openface[start_frame:end_frame+1]
    curr_X_openface, _ = global_utils.process_single_data(
            curr_X_openface,
            None,
            mean=openface_mean_std['mean'][file_name],
            std=openface_mean_std['std'][file_name])

  if X_file_to_cpm is not None:
    X_cpm = X_file_to_cpm[file_name]
    curr_X_cpm = X_cpm[start_frame:end_frame+1]
    curr_X_cpm = global_utils.process_cpm_data(curr_X_cpm)

  if X_file_to_zface is not None:
    X_zface = X_file_to_zface[file_name]
    curr_X_zface = X_zface[start_frame:end_frame+1]
    curr_X_zface = global_utils.process_zface_data(curr_X_zface)

  # curr_X = [curr_X_openface, curr_X_cpm, curr_X_zface]
  curr_X = [curr_X_openface]
  # Remove 'None' from list
  curr_X = [x for x in curr_X if x is not None]
  curr_X = np.hstack(curr_X)
  
  return curr_X

def get_aug_features(
        openface_aug_h5,
        cpm_aug_h5,
        zface_aug_h5,
        file_name,
        gest_type,
        start_frame,
        end_frame,
        openface_mean_std,
        cpm_mean_std,
        max_aug=5,
        openface_trimmed_aug=False):
  '''
  return the aug feature from respective aug h5
  '''
  curr_X = []
  if openface_aug_h5 is not None:
    augs_openface = global_utils.get_all_aug_in_seq(openface_aug_h5, file_name,
        gest_type, start_frame, end_frame, win_len=32)
    augs_openface = augs_openface[len(augs_openface)//2]

    if cpm_aug_h5 is not None:
      augs_cpm = global_utils.get_all_aug_in_seq(
          cpm_aug_h5,
          file_name,
          gest_type,
          start_frame,
          end_frame,
          win_len=32)
      augs_cpm = augs_cpm[len(augs_cpm)//2]

    if zface_aug_h5 is not None:
      augs_zface = global_utils.get_all_aug_in_seq(
          zface_aug_h5,
          file_name,
          gest_type,
          start_frame,
          end_frame,
          win_len=32)
      augs_zface = augs_zface[len(augs_zface)//2]

    for i in range(max_aug):
      idx = np.random.randint(augs_openface.shape[0])
      X_openface, _ = global_utils.process_aug_data(
              augs_openface[idx,:,:].T,
              None,
              openface_trimmed_aug=openface_trimmed_aug,
              mean=openface_mean_std['mean'][file_name],
              std=openface_mean_std['std'][file_name])
                

      # X_cpm = global_utils.process_aug_cpm_data(augs_cpm[idx,:,:].T)
      # X_zface = global_utils.process_aug_zface_data(augs_zface[idx,:,:].T)

      curr_X.append(np.hstack([X_openface]))

  return curr_X

def main(openface_h5_dir, cpm_h5_dir, zface_h5_dir, openface_aug_h5,
    cpm_aug_h5, zface_aug_h5, openface_mean_h5, cpm_mean_h5,
    gest_list_h5, data_mat_file, label_mat_file, num_features,
    openface_trimmed_aug):

  # file_filter = lambda(f): "static.mp4" in f and "mohit" not in f
  file_filter = lambda(f): True
  X_file_to_openface, X_file_to_cpm, X_file_to_zface, _ = \
      global_utils.load_all_face_body_features(openface_h5_dir, cpm_h5_dir,
          zface_h5_dir, file_filter)
  openface_mean_std, cpm_mean_std = global_utils.read_mean_files_list(
      [openface_mean_h5, cpm_mean_h5])

  max_samples = 500
  for group_name in ['train', 'test']:
    gest_list_h5f = h5py.File(gest_list_h5, 'r')
    gest_map = {0:[], 1: [], 2: [], 3: [], 4: []} 
    for f in gest_list_h5f[group_name].keys():
      max_gest_type = [0]*5
      for gest_type in range(11):
        gest_seq = np.array(gest_list_h5f[group_name][f][str(gest_type)])
        final_gest_type = global_utils.get_classif_class_for_gest_5(gest_type)

        if len(gest_seq.shape) <= 1: continue
        elif gest_seq.shape[0] > max_samples: gest_seq = gest_seq[:max_samples]
        for i in range(gest_seq.shape[0]):
          seq = gest_seq[i]
          if seq[0] > seq[1]: 
            print "WTF!! should happen only once"
            continue
          # X = get_zface_gesture(os.path.join(h5_dir, f), seq[0], seq[1])
          curr_X = get_real_features(
                  X_file_to_openface,
                  None, #X_file_to_cpm,
                  None, #X_file_to_zface,
                  f,
                  seq[0],
                  seq[1],
                  openface_mean_std,
                  cpm_mean_std)
          gest_map[final_gest_type].append(curr_X)

          # Get aug gestures
          if group_name == 'train' and gest_type >= 6:
            curr_aug_X = get_aug_features(openface_aug_h5,
                    None, #cpm_aug_h5,
                    None, #zface_aug_h5,
                    f,
                    gest_type,
                    seq[0],
                    seq[1],
                    openface_mean_std,
                    cpm_mean_std,
                    max_aug=10,
                    openface_trimmed_aug=openface_trimmed_aug)

            before_len = len(gest_map[final_gest_type])
            gest_map[final_gest_type] += curr_aug_X
            assert(before_len+len(curr_aug_X) == len(gest_map[final_gest_type]))
      print("Did process file: {}".format(f))
          
  

    gest_list_h5f.close()
    sys.stdout.write('{} stats: '.format(group_name))
    for i in gest_map.itervalues():
        sys.stdout.write(' {}, '.format(len(i)))
    sys.stdout.write('\n')

    # TODO(Mohit): Add augmented gestures to make the fewer classes equal.

    num_gests = [len(v) for k,v in gest_map.iteritems() if len(v) > 0]
    print(num_gests)
    min_gests = min(num_gests)
    for k,v in gest_map.iteritems():
      gest_map[k] = v[:min_gests]

    gests, labels = get_joint_gestures_labels(gest_map, num_features)

    global_utils.save_list_as_matlab_cell_array(
        gests, group_name+'_'+data_mat_file, 'seq')
    global_utils.save_list_as_matlab_cell_array(
        labels, group_name+'_'+label_mat_file, 'labels')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Create cell array from data files where each gesture is a '
        'separate sequence')
  parser.add_argument('--openface_h5_dir', nargs='?', type=str, const=1,
      required=True, help='Directory containing openface h5 files.')
  parser.add_argument('--zface_h5_dir', nargs='?', type=str, const=1,
      required=True, help='Directory containing zface h5 files.')
  parser.add_argument('--cpm_h5_dir', nargs='?', type=str, const=1,
      required=True, help='Directory containing CPM h5 files.')
  parser.add_argument('--openface_aug_h5', nargs='?', type=str, const=1,
      required=True, help='openface aug h5.')
  parser.add_argument('--zface_aug_h5', nargs='?', type=str, const=1,
      required=True, help='zface aug h5.')
  parser.add_argument('--cpm_aug_h5', nargs='?', type=str, const=1,
      required=True, help='CPM aug h5.')
  parser.add_argument('--gesture_list_h5', nargs='?', type=str, const=1,
      required=True, help='h5 file containing the list of gestures.')
  parser.add_argument('--label_mat_file', nargs='?', type=str, const=1,
          required=True, help='Name of matlab file with ')
  parser.add_argument('--openface_mean_h5', nargs='?', type=str, const=1,
          required=True, help='h5 file containing mean/std for openface feats.')
  parser.add_argument('--cpm_mean_h5', nargs='?', type=str, const=1,
          required=True, help='h5 file containing mean/std for CPM feats.')
  parser.add_argument('--data_mat_file', nargs='?', type=str, const=1,
          required=True, help='Name of the mat file to save cells in.')
  # This is used for verification purpose only so that the user knows what
  # actually we are saving
  parser.add_argument('--num_features', nargs='?', type=int, const=1,
          default=55, help='Number of features to be saved in mat file.')

  parser.add_argument('--openface_trimmed_aug',
          dest='openface_trimmed_aug',
          action='store_true',
          help='The augmentation h5 passed in contains trimmed augmentations.')
  parser.add_argument('--no-openface_trimmed_aug',
          dest='openface_trimmed_aug',
          action='store_false')
  parser.set_defaults(openface_trimmed_aug=False)

  args = parser.parse_args()
  print(args)

  main(args.openface_h5_dir, args.cpm_h5_dir, args.zface_h5_dir,
      args.openface_aug_h5, args.cpm_aug_h5, args.zface_aug_h5,
      args.openface_mean_h5, args.cpm_mean_h5, args.gesture_list_h5,
      args.data_mat_file, args.label_mat_file, args.num_features,
      args.openface_trimmed_aug)

