import numpy as np
import h5py
import json
import argparse
import seaborn as sns
import os
import matplotlib.pyplot as plt
import pdb

import data_utils
import plot_conf_mat

PLOT_WIDTH = 2
PLOT_HEIGHT = 2

def filter_features(X):
  '''
  Remove all features that we don't want to use to find variance.
  '''
  sel_feats = range(12,18) # use pose and pose rotation velocity
  sel_feats += range(34,38) # use forehead and nosetip velocity
  sel_feats += range(50,56) # Face difference vectors
  sel_feats.append(56) # nose-forehead angle
  X_new = X[:, sel_feats]
  feat_names = [
      'pose_x',
      'pose_y',
      'pose_z',
      'pose_x_rot',
      'pose_y_rot',
      'pose_z_rot',
      'forhead_x',
      'forhead_y',
      'nose_x',
      'nose_y',
      'top_face_x',
      'top_face_y',
      'left_face_x',
      'left_face_y',
      'right_face_x',
      'right_face_y',
      'nose_head_angle'
      ]

  return X_new, feat_names


def calculate_feat_variance(X, feat_names, gest_seq, segment_len=None, top_K=3):
  '''
  gest_seq: Contains all gesture sequences for the given gesture
  '''
  var_stats = [0] * X.shape[1]
  max_vel_stats = [0] * X.shape[1]
  for i in range(gest_seq.shape[0]):
    seq_st, seq_end = gest_seq[i,0], gest_seq[i,1]
    if segment_len is not None and segment_len > 0:
      for j in range(seq_st, seq_end, segment_len):
        new_seq_end = min(seq_end, j+segment_len)
        seg_X = X[j:new_seq_end, :]

        var_X = np.var(seg_X, 0)
        max_X = np.max(seg_X, 0)
        top_var = np.argsort(var_X)[::-1]
        top_max = np.argsort(max_X)[::-1]
        for k in range(top_K):
          var_stats[top_var[k]] += 1
          max_vel_stats[top_max[k]] += 1
    else:
      seg_X = X[seq_st:seq_end, :]

      var_X = np.var(seg_X, 0)
      max_X = np.max(seg_X, 0)
      top_var = np.argsort(var_X)[::-1]
      top_max = np.argsort(max_X)[::-1]
      for k in range(top_K):
        var_stats[top_var[k]] += 1
        max_vel_stats[top_max[k]] += 1

  return var_stats, max_vel_stats
      
def plot_histograms(per_user_var_stats, feat_names, users=None):
  if users is None:
    users = per_user_var_stats.keys()[:9]

  fig, all_axs = plt.subplots(PLOT_WIDTH, PLOT_HEIGHT)
  for i in range(len(users)):
    hist = per_user_var_stats[users[i]]
    if type(all_axs) != type(np.array(0)):
      axs = all_axs
    elif len(all_axs.shape) == 2:
      r, c = i // PLOT_WIDTH, i % PLOT_WIDTH
      axs = all_axs[r, c]
    else:
      axs = all_axs[i]

    axs.bar(range(len(hist)), hist) 
    axs.set_title(users[i])
    axs.set_xlabel('Gesture')
    axs.set_ylabel('Frequency')
    axs.set_xticks(np.linspace(0,10,len(feat_names)))
    axs.set_xticklabels(feat_names, rotation='vertical')
    
  plt.show()


def create_most_informative_hist(gest_seq_h5_filepath, openface_dir,
    cpm_dir, user, gesture):
  gest_seq_h5 = h5py.File(gest_seq_h5_filepath, 'r') 
  per_user_var_stats, per_user_max_vel_stats = {}, {}
  final_feat_names = None

  for g in range(10):
    for target_group in ['train', 'test']:
      for target_user in gest_seq_h5[target_group].keys():
        if len(gest_seq_h5[target_group][target_user][str(g)].shape) <= 1:
          continue
        gest_seq = np.array(gest_seq_h5[target_group][target_user][str(g)])

        target_openface_h5_path = os.path.join(openface_dir, target_user) 
        target_cpm_h5_path = os.path.join(cpm_dir, target_user)
        X, y, cpm_X = data_utils.get_all_features(
            target_openface_h5_path, target_cpm_h5_path)
        X, _ = data_utils.process_data(X, y, cpm_X)
        X_filt, feat_names = filter_features(X)
        if final_feat_names is None:
          final_feat_names = feat_names

        var_stats, max_vel_stats = calculate_feat_variance(
            X_filt, feat_names, gest_seq)

        if per_user_var_stats.get(str(g)) is None:
          per_user_var_stats[str(g)] = {}
          per_user_max_vel_stats[str(g)] = {}

        per_user_var_stats[str(g)][target_user] = var_stats
        per_user_max_vel_stats[str(g)][target_user] = max_vel_stats
  gest_seq_h5.close()


  create_feature_gesture_distribution(per_user_var_stats, final_feat_names)
  
  # Plot the histgram for a given gesture
  sorted_filenames = sorted(per_user_var_stats[gesture].keys())

  start_idx = 4 * PLOT_WIDTH*PLOT_HEIGHT
  plot_histograms(per_user_var_stats[gesture], feat_names,
      sorted_filenames[start_idx:start_idx+PLOT_WIDTH*PLOT_HEIGHT])

def create_feature_gesture_distribution(per_user_var_stats, feat_names):
  final_stats_per_gest = {}
  for g in range(10):
    g_str = str(g)
    final_stats_per_gest[g_str] = [0] * len(feat_names)
    for k, v in per_user_var_stats[g_str].iteritems():
      for i in range(len(v)):
        final_stats_per_gest[g_str][i] += v[i]
  # Now we have all the results for all gestures and features
  print(final_stats_per_gest)

  conf = np.zeros((len(final_stats_per_gest.keys()), len(feat_names)))
  for k, v in sorted(final_stats_per_gest.iteritems()):
    conf[k] = v
  norm_conf = plot_conf_mat.normalize_conf(conf)
  plot_conf_mat.plot_conf(
      norm_conf, '../meetings/next', 'feature_gest_dist_2.png')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Find information about most informative features/joints.')
  parser.add_argument('--gest_seq_h5', nargs='?', type=str, const=1,
      required=True, help='Gest sequence file to use.')
  parser.add_argument('--openface_dir', nargs='?', type=str, const=1,
      required=True, help='Directory containing openface h5 files.')
  parser.add_argument('--cpm_dir', nargs='?', type=str, const=1,
      default='', help='Directory containging Conv Pose Machine h5 files.')
  parser.add_argument('--user', nargs='?', type=str, const=1,
      default='', help='User to find value for. If None we find for all.')
  parser.add_argument('--gesture', nargs='?', type=str, const=1,
      default='6', help='Gesture to use. If None we use it for all.')
  args = parser.parse_args()
  print(args)

  user = None if len(args.user) == 0 else args.user
  create_most_informative_hist(args.gest_seq_h5, args.openface_dir,
      args.cpm_dir, user, args.gesture)

