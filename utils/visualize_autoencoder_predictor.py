import argparse
import cv2
import data_utils
import h5py
import json
import numpy as np
import os
import pdb
import sys

import matplotlib.pyplot as plt

import data_utils
    
 
def get_original_from_normalized(X, stats_dict, normalization):
  '''
  X: 2d array as (Time, Features)
  '''
  X_new = np.array(X)
  if normalization == '0-1':
    for i in range(12, 12+X.shape[1]):
      min_val, max_val = stats_dict['min'][i], stats_dict['max'][i]
      X_new[:,i-12] = (max_val - min_val) * X[:,i-12] + min_val 
  elif normalization == 'z':
    for i in range(12, 12+X.shape[1]):
      mean_val, std_val = stats_dict['mean'][i], stats_dict['std'][i]
      X_new[:,i-12] = (std_val * X[:,i-12]) + mean_val
  else:
    assert(False)
  return X_new

def get_stats_for_file(stats_h5, file_name):
  stats_h5_f = h5py.File(stats_h5, 'r')
  stats= {}
  stats['mean'] = np.array(stats_h5_f['mean'][file_name])
  stats['std'] = np.array(stats_h5_f['std'][file_name])
  stats['min'] = np.array(stats_h5_f['min'][file_name])
  stats['max'] = np.array(stats_h5_f['max'][file_name])
  stats_h5_f.close()
  return stats

def get_input_output(h5_file, gest_idx):
  h5_f = h5py.File(h5_file, 'r')
  inp = np.array(h5_f['input'][str(gest_idx)])
  op = np.array(h5_f['output'][str(gest_idx)])
  h5_f.close()
  return inp, op

def display_predict_8_vid(inp_landmarks, op_landmarks):
  inp_X = np.array(inp_landmarks)
  op_X = np.array(op_landmarks)
  inp_X[:,:68] = inp_X[:,:68] - 700
  op_X[:,:68] = op_X[:,:68] - 700
  inp_X[:,68:] = inp_X[:,68:] - 300
  op_X[:,68:] = op_X[:,68:] - 300
  assert(np.sum(inp_X<0) == 0)
  assert(np.sum(op_X<0) == 0)
  # Original video size is too much

  # Since we use 24 frames as input and predict next 8 frames
  new_op_X = np.array(inp_X)
  new_op_X[10:20,:] = op_X
  op_X = new_op_X

  # Draw
  inp_img = np.zeros((600,600,3), np.float32)
  op_img = np.zeros(inp_img.shape, np.float32)


  # Shift 
  shift_amt = 5
  shifted_inp_X = inp_X * (2**shift_amt)
  shifted_op_X = op_X * (2**shift_amt)

  # pdb.set_trace()
  for i in range(0,inp_X.shape[0]):
    inp_img[:,:,:] = 0
    for j in range(inp_X.shape[1]//2):
      # print((int(shifted_inp_X[i,j]), int(shifted_inp_X[i,j+68])))
      cv2.circle(inp_img, (int(shifted_inp_X[i,j]), int(shifted_inp_X[i,j+68])),
        radius=3, color=(0,0,255), thickness=2, shift=shift_amt)
      cv2.circle(inp_img, (int(shifted_op_X[i,j]), int(shifted_op_X[i,j+68])),
        radius=3, color=(0,255,0), thickness=2, shift=shift_amt)
    cv2.imshow('Input', inp_img)
    key_pressed = cv2.waitKey(1000)
    if key_pressed == 27:
      break
  cv2.destroyAllWindows()

def main(h5_file, json_file, stats_h5, normalization, gest_idx, vid_idx):
  '''
  h5_file with input and output groups.
  '''
  inp_info = None
  with open(json_file, 'r') as json_f:
    inp_info = json.load(json_f)
  # Curr_file will be [<filename>, <gest start frame>, <gest end frame>]
  curr_file = inp_info[gest_idx][vid_idx]
  curr_file_stats = get_stats_for_file(stats_h5, curr_file[0])
  # Since input output is saved using Lua which is 1-indexed the starting index
  # will be 2 in our h5file
  curr_inp, curr_op = get_input_output(h5_file, gest_idx+1)

  curr_inp_vid, curr_op_vid = curr_inp[vid_idx], curr_op[vid_idx]
  org_inp_vid = get_original_from_normalized(
    curr_inp_vid, curr_file_stats, normalization)
  org_op_vid = get_original_from_normalized(
    curr_op_vid, curr_file_stats, normalization)

  display_predict_8_vid(org_inp_vid, org_op_vid) 

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Run autoencoder prediction movements.')
  parser.add_argument('--inp_h5', nargs='?', type=str, const=1,
      required=True, help='h5 file with inputs and results as outputs.')
  parser.add_argument('--inp_json', nargs='?', type=str, const=1,
      required=True, help='json file with info about inputs.')
  parser.add_argument('--stats_h5', nargs='?', type=str, const=1,
      required=True, help='h5 file with statistics about how input, output '
      'were normalized.')
  parser.add_argument('--normalization', nargs='?', type=str, const=1,
      help='Input Normalization used during training.')
  parser.add_argument('--gest_index', nargs='?', type=int, const=1,
      help='Gesture index to use for display. Nods start with 1.')
  parser.add_argument('--vid_index', nargs='?', type=int, const=1,
      help='index in the h5 file to use for display')
  args = parser.parse_args()
  print(args)

  main(args.inp_h5, args.inp_json, args.stats_h5, args.normalization, 
      args.gest_index, args.vid_index)

