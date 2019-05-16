import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

NUM_CLASSES = 11
MAX_SLIDING_WINDOW = 500

def get_gesture_seq_len(a):
  i, seq_len = 0, len(a)
  gest_seq_len = [[] for i in range(NUM_CLASSES)]

  while i < seq_len:
    j, target_gest = i+1, a[i]
    while j < seq_len and a[j] == a[i] and j-i < MAX_SLIDING_WINDOW:
      j = j + 1
    # define a minimum length for a gesture
    if j - i > 5:
      gest_seq_len[a[i]].append(j-i)
    i = j

  return gest_seq_len

def get_file_stats(gest_label):
  fdir = '../data/open_face_h5'
  gest_seq_len_by_file = {}

  for npfile in os.listdir(fdir):
    if npfile.endswith("_static.mp4.txt.h5"):
      hf = h5py.File(fdir + "/" + npfile,'r')
      a = np.array(hf.get('annotations')).astype(int)
      f = np.array(hf.get('features'))
      gest_seq_len = get_gesture_seq_len(a)
      gest_seq_len_by_file[npfile] = gest_seq_len

  plt.plot([1,2,3])
  all_gests = [[] for i in range(NUM_CLASSES)]
  for i in range(NUM_CLASSES):
    for (f, gest_lens) in gest_seq_len_by_file.iteritems():
      all_gests[i] = all_gests[i] + gest_lens[i]

    plt.subplot(3, 4, i+1)
    plt.scatter(range(len(all_gests[i])), all_gests[i])
  plt.show()

  return all_gests

def main():
  all_gests = get_file_stats(2)
  #plt.scatter(range(len(all_gests)), all_gests)
  #plt.show()

if __name__ == '__main__':
    main()
