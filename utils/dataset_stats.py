import numpy as np
import h5py
import os
import pdb
import sys

NUM_CLASSES = 11

def get_file_stats():
  fdir = '../openface_data/face_gestures/dataseto_text'
  d = {}

  for npfile in os.listdir(fdir):
    if npfile.endswith("_static.mp4.txt.h5"):
      d[npfile] = [0]*NUM_CLASSES
      hf = h5py.File(fdir + "/" + npfile,'r')
      a = np.array(hf.get('annotations')).astype(int)
      f = np.array(hf.get('features'))
      for i in range(NUM_CLASSES):
        d[npfile][i] = np.sum(a == i)

  for (file, gest_counts) in d.iteritems():
    sys.stdout.write('File: {}, Gestures: '.format(file))
    for i in xrange(NUM_CLASSES):
      sys.stdout.write('{} '.format(d[file][i]))
    sys.stdout.write('\n')
  sys.stdout.flush()
  return d

def main():
    get_file_stats() 

if __name__ == '__main__':
    main()

