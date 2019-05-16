import numpy as np
import h5py
import os
import pdb
import random

from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

from scipy.signal import savgol_filter

class BatchGenerator(object):

  def __init__(self, h5file_path, batch_size, num_classes, time_step, num_features,
          h5dir='../openface_data/face_gestures/dataseto_text', num_classify=11):
    assert(np.mod(batch_size, num_classify) == 0)
    self.f_path = h5file_path
    self.batch_size = batch_size
    self.num_classes = num_classes
    self.num_classify = num_classify 
    self.win_len = time_step
    self.h5dir = h5dir
    self.num_features = num_features

  def normalize_data_each_sensor_signal(self, X, y):
    '''
    Normalize data X and y.
    '''
    mean = np.mean(X, 0)
    std = np.std(X, 0)
    norm_X = (X - mean) / std 
    # norm_X = X - mean 
    return norm_X, y

  def load_all_features(self):
    X_by_file = {}
    y_by_file = {}

    for npfile in os.listdir(self.h5dir):
      if npfile.endswith('static.mp4.txt.h5'):
        hf = h5py.File(self.h5dir + '/' + npfile)
        a = np.array(hf.get('annotations')).astype(int)
        f = np.array(hf.get('features'))
        X1, y1 = np.copy(f), np.copy(a)
        X, y = self.process_data(X1, y1)
        X_by_file[npfile] = X
        y_by_file[npfile] = y
    return X_by_file, y_by_file


  def get_all_gest_by_type(self):
    gest_by_type = [[] for _ in range(self.num_classes)]
    h5_file = h5py.File(self.f_path, 'r')
    # assert(self.num_classes == len(h5_file[h5_file.keys()[0]]))
    for f_name in h5_file.keys():
      file_gest_by_type = h5_file[f_name]
      assert(self.num_classes == len(file_gest_by_type))
      for i in xrange(self.num_classes):
          gest = file_gest_by_type[str(i)]
          for j in xrange(gest.shape[0]):
            # NOTE: We add the filename as the first argument of the list here.
            gest_by_type[i].append([f_name, gest[j, 0], gest[j, 1]])

    h5_file.close()
    # Shuffle the data order so that for each batch the network sees data in 
    # different order.
    for i in xrange(len(gest_by_type)):
      random.shuffle(gest_by_type[i]) 

    return gest_by_type

  def smooth_data(self, X):
    window_len, poly_order = 11, 2
    for i in xrange(X.shape[1]):
      X_data = X[:,i]
      X[:, i] = savgol_filter(X_data, window_len, poly_order)
    return X

  def process_data(self, X, y):
    """
    Process the data set to do normalization and other clean up techniques.
    """
    #TODO(Mohit): Normalize all sensor signals in X.
    if X.shape[1] > self.num_features:
      X = X[:, :148]
      X1 = np.copy(X)
      X = self.smooth_data(X1)

      # Includes pose as well as gaze.
      X_pose = X[:, :12]

      X_pose_diff = X[:-1, 6:8] - X[1:, 6:8]
      X_pose_diff = np.vstack((np.array([0, 0]), X_pose_diff))

      # Add specific landmarks. First landmark is indexed as 1.
      landmarks = [
          28, 28 + 68, # forehead
          34, 34 + 68, # nosetip
          2,   2 + 68, # left side of face
          4,   4 + 68,
          8,   8 + 68, # bottom (right)
          10, 10 + 68,
          14, 14 + 68, # top
          16, 16 + 68]
      l = [l1 + 11 for l1 in landmarks]
      X_landmarks = X[:, l]

      # Maybe take a difference for these vectors
      X_landmarks_diff = X[:-1, l] - X[1:, l]
      X_landmarks_diff = np.vstack((np.zeros(16), X_landmarks_diff))

      # Take 4 direction vectors on face which might change as we move
      X_face_vec_1 = np.array(
         [X[:, 28+11] - X[:, 34+11], X[:, 28+68+11] - X[:, 34+68+11]]).T
      X_face_vec_2 = np.array(
         [X[:, 3+11] - X[:, 34+11], X[:, 3+68+11] - X[:, 34+68+11]]).T
      X_face_vec_3 = np.array(
         [X[:, 15+11] - X[:, 34+11], X[:, 15+68+11] - X[:, 34+68+11]]).T

      X = np.hstack(([X_pose, X_pose_diff, X_landmarks, X_landmarks_diff,
        X_face_vec_1, X_face_vec_2, X_face_vec_3]))

    # Let's only classify ticks
    # We shouldn't do 1-vs-all classification here since we lose the relevant
    # spatial info i.e. we won't know what gesture it really was while
    # grouping things in windows. We should do this as a post processing
    # step.  X, y = do_one_vs_all(X, y, 5)
    X, y = self.normalize_data_each_sensor_signal(X, y)
    return X, y

  def get_features_for_batch(self, batch, noise_mask, X_by_file, y_by_file):
    '''
    Returns features for batch.
    '''
    X = np.zeros((self.batch_size, self.win_len, 52))
    y = np.zeros(self.batch_size)
    i = 0
    for gest in batch:
      X_gest, y_gest = X_by_file[gest[0]], y_by_file[gest[0]]
      # Assign the label first since random noise might change it
      # Also always use gest[2] since gest[1] (i.e. left side of
      # gesture is randomly changed for nods (class 1)
      y[i] = self.get_classif_class_for_gest(y_gest[gest[2]-1])

      # Add gaussian noise based on noise mask
      if noise_mask[i]: 
        rand_mean, rand_var = 5, 10
        rand_noise = int(np.random.normal(rand_mean, rand_var, 1))
        while gest[1]-rand_noise < 0 or \
                (gest[2] - gest[1] + rand_noise) < 10 or \
                (gest[2] - gest[1] + rand_noise) > self.win_len:
          rand_noise = int(np.random.normal(rand_mean, rand_var, 1))
        gest[1] = gest[1]-rand_noise
        
      x = X_gest[gest[1]:gest[2]]
      pad_top = (self.win_len - (gest[2] - gest[1])) // 2
      pad_bottom = self.win_len - pad_top - (gest[2] - gest[1])
      x = np.pad(x, ((pad_top, pad_bottom), (0, 0)), 'constant',
          constant_values=0)
      X[i, :, :] = x

      i = i + 1

    y = y.astype(int)
    # Verify we are populating things correctly
    print(np.bincount(y))
    # for c1 in np.bincount(y):
    # assert(c1 <= 24 and c1 >= 20)

    return X, y

  def get_classif_class_for_gest(self, gest_type):
    if gest_type == 0:
      return 0
    elif gest_type >= 1 and gest_type <= 5:
      return 1
    elif gest_type == 6:
      return 2
    elif gest_type == 7 or gest_type == 8:
      return 3
    elif gest_type == 9 or gest_type == 10:
      return 4
    else:
      raise ValueError

  def group_gest_into_classes(self, gest_by_type):
    '''
    Groups gestures into number of classes we want to classify in. gest_by_type will contain alltu
    '''
    gest_by_classify = [[] for _ in xrange(self.num_classify)]
    for i in xrange(len(gest_by_type)):
      new_class = self.get_classif_class_for_gest(i)
      gest_by_classify[new_class] += gest_by_type[i]
    return gest_by_classify

  def generate(self):
    '''
    Need to go through all data in generate. This is called multiple times in
    every epoch and each time we want to return a uniform collection of
    classes.
    '''
    h5_file = h5py.File(self.f_path, 'r')
    num_files = len(h5_file.keys())
    gest_by_type = self.get_all_gest_by_type()
    gest_by_type = self.group_gest_into_classes(gest_by_type)
    X_by_file, y_by_file = self.load_all_features()

    '''
    Note: num_classify is the number of classes we want our network to classify
    in the end after softmax.  num_classes is the total number of classes in
    our data. Here we have to sample from num_classify
    '''
    # Class 0 should always represent None or the major type. We will
    # undersample this class.
    total_samples_in_epoch = len(gest_by_type[1]) * self.num_classify
    num_batches = int(total_samples_in_epoch / self.batch_size)
    print('Wasting {} samples in epoch'.format(np.mod(total_samples_in_epoch,
      self.batch_size)))

    # Generate samples per epoch
    idx_by_type = [0] * self.num_classify
    assert(self.batch_size % self.num_classify == 0)
    for i in xrange(num_batches):
      class_samples_in_batch = self.batch_size / self.num_classify
      ''' Create one batch '''
      # NOTE: Noise mask is used to choose which samples should noise be
      # added to. Right now we always add noise when adding duplicates.
      batch, noise_mask = [], np.zeros(self.num_classify*class_samples_in_batch)

      for c in range(self.num_classify):
        for s in range(class_samples_in_batch):
          if len(gest_by_type[c]) > idx_by_type[c]:
            # The first argument is the file name.
            batch.append(gest_by_type[c][idx_by_type[c]])
            idx_by_type[c] = idx_by_type[c] + 1
          else:
            # Select some gesture randomly since we have already exhausted all
            # the gestures from here.
            rand_idx = random.randint(0, len(gest_by_type[c])-1)
            batch.append(gest_by_type[c][rand_idx])
            noise_mask[c*class_samples_in_batch + s] = 1

      random.shuffle(batch)
      X_batch, y_batch = self.get_features_for_batch(
              batch, noise_mask, X_by_file, y_by_file)
      yield (X_batch, np_utils.to_categorical(y_batch, self.num_classify))
    h5_file.close()

