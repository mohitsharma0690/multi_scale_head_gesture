import numpy as np
import os
import json
import h5py
import sys
import pdb
import argparse
from sklearn.metrics import confusion_matrix

def print_conf_matrix(conf):
  opt = np.get_printoptions()
  np.set_printoptions(threshold='nan')
  print('\n===== Confusion matrix =====')
  print(conf)
  np.set_printoptions(**opt)

def get_class_5(label):
  if label == 0:
    return 0
  elif label >= 1 and label <= 5:
    return 1
  elif label == 6:
    return 2
  elif label == 7 or label == 8:
    return 3
  elif label == 9 or label == 10:
    return 4
  else:
    assert(False)
    return -1

def get_all_user_conf_mat(fdir, pred_h5_f):
  all_conf = {}
  pred_h5 = h5py.File(pred_h5_f, 'r')
  pred_dir = os.path.dirname(pred_h5_f)
  total_conf = np.zeros((5, 5))
  for npfile in pred_h5.keys():
    preds = np.array(pred_h5[npfile]['pred']).astype(int)
    org_h5_file = os.path.join(fdir, npfile)
    org_h5 = h5py.File(org_h5_file)
    targets = np.array(org_h5['annotations']).astype(int)
    org_h5.close()

    # Construct a confusion matrix of both h5's
    assert(preds.shape[0] == targets.shape[0])
    # Since we skipped some frames during classification
    final_preds, final_targets = [], []
    for i in range(preds.shape[0]):
      if preds[i] >= 0:
        final_preds.append(preds[i])
        final_targets.append(get_class_5(targets[i]))
    conf = confusion_matrix(final_targets, final_preds, [0, 1, 2, 3, 4])
    all_conf[npfile] = conf
    total_conf = total_conf + conf

  pred_h5.close()
  all_conf['all'] = total_conf
  return all_conf

def get_all_user_predictions(fdir, pred_h5_f):
  '''Get probability distribution for each data point prediction.
  Return:
    probs: Dictionary with probability distribution for each prediction keyed
    by the user h5 file.
    preds: Dictionary with predictions for each data input keyed by the user
    h5 file.
    targets: Dictionary with targets for each data input keyed by the user
    h5 file.
  ''' 
  probs, preds, targets = {}, {}, {}
  pred_h5 = h5py.File(pred_h5_f, 'r')
  pred_dir = os.path.dirname(pred_h5_f)
  for npfile in pred_h5.keys():
    user_prob = np.array(pred_h5[npfile]['prob']).astype(float)
    user_pred = np.array(pred_h5[npfile]['pred']).astype(int)
    org_h5_file = os.path.join(fdir, npfile)
    org_h5 = h5py.File(org_h5_file)
    user_target = np.array(org_h5['annotations']).astype(int)
    org_h5.close()

    # Construct a confusion matrix of both h5's
    assert(user_prob.shape[0] ==user_target.shape[0])
    # Since we skipped some frames during classification
    final_probs, final_preds, final_targets = [], [], []
    for i in xrange(user_prob.shape[0]):
      if user_pred[i] >= 0:
        final_preds.append(user_pred[i])
        final_probs.append(user_prob[i, :].tolist())
        final_targets.append(get_class_5(user_target[i]))
    probs[npfile] = final_probs
    preds[npfile] = final_preds
    targets[npfile] = final_targets

  pred_h5.close()
  return probs, preds, targets

    
def main(fdir, pred_h5_f):
  all_conf = get_all_user_conf_mat(fdir, pred_h5_f)
  for npfile, conf in all_conf.iteritems():
    print('File {} stats:'.format(npfile))
    conf_file = os.path.join(pred_dir, npfile + '_conf.txt')
    print_conf_matrix(conf)
    np.savetxt(conf_file, conf, fmt='%d', delimiter=', ')
    print('\n')
     
if __name__ == '__main__':
  parser = argparse.ArgumentParser(
          description='Print per user stats for given pred.h5')
  parser.add_argument('--h5_dir', nargs='?', type=str, const=1,
          required=True, default='../data', help='h5 files directory')
  parser.add_argument('--pred_h5', nargs='?', type=str, const=1,
          required=True, help='h5 files directory')
  args = parser.parse_args()
  print(args)

  fdir = args.h5_dir
  pred_h5 = args.pred_h5
  main(fdir, pred_h5)

