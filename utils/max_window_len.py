import numpy as np
import h5py
import math
import os
import pdb
import sys

NUM_CLASSES = 11
MAX_SLIDING_WINDOW = 180

def max_win_len(a):
  """
  Find the maximum window length in the annotations of any gesture.
  """
  l, i, max_len, max_len_idx = a.shape[0], 0, 0, 0
  gest_stats = {}
  for i in xrange(1,NUM_CLASSES+1):
    gest_stats[i] = []

  while i < l:
    if a[i] != 0:
      j = i+1
      while j < l and a[j] == a[i]:
        j = j + 1

      if j-i >= 10:
          gest_stats[a[i]].append(j-i)

      if j-i > max_len:
        max_len = j-i
        max_len_idx = i
      i = j
    else:
      i = i+1
   
  return max_len, max_len_idx, gest_stats

def get_gesture_seq_count(a):
  i, gest_seq_count, seq_len = 0, [0]*NUM_CLASSES, len(a)

  while i < seq_len:
    j, target_gest = i+1, a[i]
    while j < seq_len and a[j] == a[i] and j-i < MAX_SLIDING_WINDOW:
      j = j + 1
    # define a minimum length for a gesture
    if j - i > 5:
      gest_seq_count[a[i]] = gest_seq_count[a[i]] + 1
    i = j

  return gest_seq_count

def get_mean_var(gest_stats):
  mean, var = [], []
  for (key,val) in gest_stats.iteritems():
    gest = np.array(val)
    mean.append(np.nanmean(gest))
    var.append(np.sqrt(np.nanvar(gest)))
  return mean, var

def load_h5_static_data():
  """
  The h5 file contains the 412 features taken out of the raw text files. They
  don't contain the frame, time number and some other columns. The dimension of
  the features array is Nx412.
  """
  fdir = '../openface_data/face_gestures/dataseto_text'

  for npfile in os.listdir(fdir):
    if npfile.endswith("_static.mp4.txt.h5"):
      hf = h5py.File(fdir + "/" + npfile,'r')
      annot = np.array(hf.get('annotations')).astype(int)
      max_len, max_len_idx, gest_stats = max_win_len(annot)
      mean, var = get_mean_var(gest_stats)
      print('File {0}, Max sequence len: {1}, gesture: {2}, idx: {3}'.format(
        npfile, max_len, annot[max_len_idx], max_len_idx))
      sys.stdout.write('Mean ')
      for i in mean[1:]:
        if math.isnan(i):
          sys.stdout.write('0  ')
        else:
          sys.stdout.write('{:.2f} '.format(i))
      sys.stdout.write('\nVar: ')
      for i in var[1:]:
        if math.isnan(i):
          sys.stdout.write('0  ')
        else:
          sys.stdout.write('{:.2f} '.format(i))
      sys.stdout.write('\n')
      sys.stdout.flush()

      gest_seq_count = get_gesture_seq_count(annot)
      print('{}:{}'.format(npfile, gest_seq_count))

def main():
    load_h5_static_data()

if __name__ == '__main__':
    main()
