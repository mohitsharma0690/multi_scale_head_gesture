#!/usr/bin/python

# Plot confusion matrix
# David Butterworth

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import os

'''
conf_arr = np.array([[585,  19,  43,   1,   1],
                [ 51,   5,   5,   0,   0],
                [ 54,   3,  78,   1,   2],
                [  3,   1,  16,   8,   2],
                [  1,   0,   4,   3,  13]])
'''
def read_conf_mat(conf_txt):
  conf = np.loadtxt(conf_txt, delimiter=', ') 
  return conf

def normalize_conf(conf):
  # Normalize:
  norm_conf = []
  for i in conf:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
      if float(a):
        tmp_arr.append(float(j)/float(a))
      else:
        tmp_arr.append(float(j))
    norm_conf.append(tmp_arr)
  return norm_conf

def plot_conf(norm_conf, save_dir, save_fname):
  # Plot using seaborn
  # (this is style I used for ResNet matrix)
  df_cm = pd.DataFrame(norm_conf)
  plt.figure(figsize = (10,7))
  sns.heatmap(df_cm, annot=True)
  plt.savefig(save_dir + '/' + save_fname)
  plt.show()

def main(conf_txt, save_dir, save_fname):
  conf_mat = read_conf_mat(conf_txt)
  conf_mat = normalize_conf(conf_mat)
  plot_conf(conf_mat, save_dir, save_fname)

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print('incorrect arguments no file specified')
    assert(False)
  
  conf_txt_file=sys.argv[1]
  save_fname = os.path.basename(conf_txt_file)[:-4] + '.png' 
  save_dir = os.path.dirname(conf_txt_file)
  main(conf_txt_file, save_dir, save_fname)

