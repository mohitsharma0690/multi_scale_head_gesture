import lasagne
import theano
import time
import os
import h5py
import math
import sys
import pdb
import json
import keras
import itertools

import numpy as np
import cPickle as cp
import pandas as pd

# Add Keras dependencies.
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

from sliding_window import sliding_window

USE_ONE_VS_ALL = 'use_one_vs_all'

class DataLoader(object):

    def __init__(self, config, num_classes, num_features,
        sliding_window_length):

        self.config = config
        self.num_classes = num_classes
        # DataLoader should decide this number
        self.num_features = num_features 
        self.win_len = sliding_window_length

    def load_dataset(self, filename):

      f = file(filename, 'rb')
      data = cp.load(f)
      f.close()

      X_train, y_train = data[0]
      X_test, y_test = data[1]

      print(" ..from file {}".format(filename))
      print(" ..reading instances: train {0}, test {1} from file {}".format(
        X_train.shape, X_test.shape, filename))

      X_train = X_train.astype(np.float32)
      X_test = X_test.astype(np.float32)

      # The targets are casted to int8 for GPU compatibility.
      y_train = y_train.astype(np.uint8)
      y_test = y_test.astype(np.uint8)

      return X_train, y_train, X_test, y_test

    def normalize_data_each_sensor_signal(self, X, y):
      '''
      Normalize data X and y.
      '''
      mean_signal = sum(X, 0) / float(X.shape[0])
      zero_mean_X = X - mean_signal
      var_X = sum((zero_mean_X ** 2), 0) / float(X.shape[0])
      # norm_X = zero_mean_X / var_X
      norm_X = zero_mean_X
      return norm_X, y

    def classify_into_classes(self, trainX, valX, testX):
      '''
      TrainX: List of list of training gesture np arrays keyed by gesture id.
      TestX: List of list of test gesture np arrays keyed by gesture id.
      '''
      target_label = 5
      if self.config[USE_ONE_VS_ALL]:
        new_trainX, new_valX, new_testX = [[]]*5, [[]]*5, [[]]*5
        for i in xrange(self.num_classes):
          new_idx = 0
          if i > 0 and i <= 5:
            new_idx = 1
          elif i == 6:
            new_idx = 2
          elif i == 7 or i == 8:
            new_idx = 3
          elif i == 9 or i == 10:
            new_idx = 4
          elif i != 0:
            assert(False)

          # Classify all vertical movements as one type of gesture
          new_trainX[new_idx] = list(itertools.chain(new_trainX[new_idx],
            trainX[i]))
          new_valX[new_idx] = list(itertools.chain(new_valX[new_idx],
            valX[i]))
          new_testX[new_idx] = list(itertools.chain(new_testX[new_idx],
            testX[i]))

        return new_trainX, new_valX, new_testX

      return trainX, valX, testX

    def post_process_categorized_gestuers(self, trainX, valX, testX):
      '''
      TrainX: List of list of training gesture np arrays keyed by gesture id.
      TestX: List of list of test gesture np arrays keyed by gesture id.
      '''
      target_label = 5
      if self.config[USE_ONE_VS_ALL]:
        new_trainX, new_valX, new_testX = [[]]*2, [[]]*2, [[]]*2
        for i in xrange(self.num_classes):
          if i == 0 or i > target_label:
            new_trainX[0] = list(itertools.chain(new_trainX[0], trainX[i]))
            new_valX[0] = list(itertools.chain(new_valX[0], valX[i]))
            new_testX[0] = list(itertools.chain(new_testX[0], testX[i]))
          else:
            # Classify all vertical movements as one type of gesture
            new_trainX[1] = list(itertools.chain(new_trainX[1], trainX[i]))
            new_valX[1] = list(itertools.chain(new_valX[1], valX[i]))
            new_testX[1] = list(itertools.chain(new_testX[1], testX[i]))

        return new_trainX, new_valX, new_testX

      return trainX, valX, testX

    def save_data(self, X_train, y_train, X_val, y_val, X_test, y_test):
      '''
      Creates one h5 file with relevant sections (train, val, test). Each with it's
      own X and y
      '''
      file_name = '../openface_data/mohit_data_all.h5'
      hf = h5py.File(file_name, 'w')
      train_grp = hf.create_group('train')
      val_grp = hf.create_group('val')
      test_grp = hf.create_group('test')

      train_grp.create_dataset('X', data=X_train)
      train_grp.create_dataset('y', data=y_train)

      val_grp.create_dataset('X', data=X_val)
      val_grp.create_dataset('y', data=y_val)

      test_grp.create_dataset('X', data=X_test)
      test_grp.create_dataset('y', data=y_test)

      hf.close()
      print 'Write dataset to {}'.format(file_name)

    def process_data(self, X, y):
      """
      Process the data set to do normalization and other clean up techniques.
      """
      #TODO(Mohit): Normalize all sensor signals in X.
      if X.shape[1] > self.num_features:
        X = X[:, :148]

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
        X_landmarks_diff = X[:-1, landmarks] - X[1:, landmarks]
        X_landmarks_diff = np.vstack((np.zeros(16), X_landmarks_diff))

        # TODO(Mohit): Maybe take a difference for these vectors
        # Take 4 direction vectors on face which might change as we move
        X_face_vec_1 = np.array(
           [X[:, 28] - X[:, 34], X[:, 28+68] - X[:, 34+68]]).T
        X_face_vec_2 = np.array(
           [X[:, 3] - X[:, 34], X[:, 3+68] - X[:, 34+68]]).T
        X_face_vec_3 = np.array(
           [X[:, 15] - X[:, 34], X[:, 15+68] - X[:, 34+68]]).T

        X = np.hstack(([X_pose, X_pose_diff, X_landmarks, X_landmarks_diff,
          X_face_vec_1, X_face_vec_2, X_face_vec_3]))

      # Let's only classify ticks
      # We shouldn't do 1-vs-all classification here since we lose the relevant
      # spatial info i.e. we won't know what gesture it really was while
      # grouping things in windows. We should do this as a post processing
      # step.  X, y = do_one_vs_all(X, y, 5)
      X, y = self.normalize_data_each_sensor_signal(X, y)
      return X, y

    def process_keras_data(self, X_train, y_train, X_test, y_test):
      '''
      X input for keras should be in the form of N x WINDOW_LENGTH x NB_SENSORS
      Keras requires the input targets to be in one-hot format.
      '''
      X_train = X_train.reshape((-1, self.win_len, self.num_features))
      X_test = X_test.reshape((-1, self.win_len, self.num_features))
      y_train = y_train.reshape((-1, self.win_len))
      y_test = y_test.reshape((-1, self.win_len))
      y_train = y_train[:, 0]  # All other columns have the same value
      y_test = y_test[:, 0]  # All other columns have the same value
      y_train = np_utils.to_categorical(y_train,
          __NUM_CLASSES(in_real_data=False))
      y_test = np_utils.to_categorical(y_test,
          __NUM_CLASSES(in_real_data=False))
      return X_train, y_train, X_test, y_test

    def __NUM_CLASSES(self, in_real_data=True):
      '''
      in_real_data: Number of classes in real data. If we are doing one-vs-all
      classification this might be false
      '''
      return self.num_classes if in_real_data else 2

    def load_h5_static_data(self, test_size=0.3):
      """
      The h5 file contains the 412 features taken out of the raw text files. They
      don't contain the frame, time number and some other columns. The dimension of
      the features array is Nx412.
      """
      fdir = '../openface_data/face_gestures/dataseto_text'
      X, y = [], []
      d = {}
      for i in range(self.num_classes):
        d[i] = 0

      for npfile in os.listdir(fdir):
        if npfile.endswith("_static.mp4.txt.h5"):
          hf = h5py.File(fdir + "/" + npfile,'r')
          a = np.array(hf.get('annotations'))
          f = np.array(hf.get('features'))
          X_file, y_file = process_data(f, a)
          X.append(X_file)
          y.append(y_file)

      X = np.vstack(X)
      y = np.concatenate(y)
      train_size = int(X.shape[0] * (1.0-test_size))
      return X[:train_size], y[:train_size], X[train_size:], y[train_size:]

    def load_sequential_data_headgestures(self, test_size=0.2):
      """
      Load the data sequentially. We don't perform any preprocessing on the data
      but rather convert it into the right format i.e. (TIME x No Of Sensors) for
      the training input. We choose the last timestamp label as the label for
      this dataset.
      """
      fdir = "../openface_data/staticsetb"

      X, y = [], []

      for npfile in os.listdir(fdir):
        if npfile.endswith(".h5"):
          hf = h5py.File(fdir + "/" + npfile,'r')
          a = np.array(hf.get('annotations'))
          f = np.array(hf.get('features'))
          X_file, y_file = process_data(f, a)
          X.append(X_file)
          y.append(y_file)

      X = np.vstack(X)
      y = np.concatenate(y)
      train_size = int(X.shape[0] * (1.0-test_size))
      return X[:train_size], y[:train_size], X[train_size:], y[train_size:]

    def load_data_centered_not_padded(self):
      trainX = [[] for _ in range(self.num_classes)]
      valX = [[] for _ in range(self.num_classes)]
      tesX = [[] for _ in range(self.num_classes)]
      fdir = '../openface_data/face_gestures/dataseto_text'

      for npfile in os.listdir(fdir):
        if npfile.endswith("static.mp4.txt.h5"):
          tmpX = [[] for _ in range(self.num_classes)]
          print 'Processing file ', npfile

          hf = h5py.File(fdir + "/" + npfile,'r')
          d_annotations = hf.get('annotations')
          d_features = hf.get('features')
          a = np.array(d_annotations).astype(int)
          f = np.array(d_features)
          f, a = process_data(f, a)

          #prepare train sequence based on class
          win_len = 0
          SWL = self.win_len # Notational convenience
          for i in range(len(a)-1):
            if (not np.array_equal(a[i],a[i+1])) or \
                (win_len==self.win_len-1):
              #annotations differ or max length reached

              # Get padding to center the given signal.
              pad = (self.win_len - win_len - 1)/2
              left_pad, right_pad = 0, 0
              if (self.win_len - win_len - 1) % 2 == 0:
                left_pad, right_pad = pad, pad
              else:
                left_pad, right_pad = pad+1, pad

              assert(win_len+1+left_pad+right_pad == self.win_len)

              # TODO(Mohit): Fix this.
              # We don't have required padding to complete this gesture so skip
              # it for now.
              if i-win_len-left_pad < 0 or i+1+right_pad >= len(a):
                continue

              train_seq = f[i-win_len-left_pad:i+1+right_pad]
              tmpX[a[i]].append(train_seq)  # train sequence
              # Maybe reversing doesn't make a lot of sense for gesture signals.
              # tmpX[a[i]].append(train_seq[::-1]) # reverse train sequence

              # Add gaussian noise
              rand_mean, rand_var, num_rand_samples = 0, 15, 5
              rand_noise = np.array(np.random.normal(rand_mean, rand_var,
                  num_rand_samples), dtype=np.int32)
              gest_left_idx, gest_right_idx = i-win_len-left_pad, i+1+right_pad
              for n in rand_noise:
                if gest_left_idx+n < 0 or gest_right_idx+n >= len(a):
                  continue
                tmpX[a[i]].append(f[gest_left_idx+n:gest_right_idx+n])

              win_len = 0

            win_len += 1

          # TODO(Mohit): Last sequence can't be centered so just avoid it for now.
          # Mohit: Interestingly the i carries over or this is a bug?
          # tmpX[a[i]].append(f[i+1-win_len:i+2]) #train sequence

          for i in range(self.num_classes):
            if 'mohit' in npfile: # int(npfile[:3]) == 14:
              tesX[i] += tmpX[i][:]
            else:
              insert_in_set = None
              if int(npfile[:3]) <= 10:
                insert_in_set = valX
              else:
                insert_in_set = trainX
              if i == 0:
                # Too much gestures for None. Just take a few of them.
                insert_in_set[i] += tmpX[i][:1000]
              else:
                insert_in_set[i] += tmpX[i][:]

      # Do any post processing e.g. doing a one-vs-all classification we change
      # the elements of this array.
      # trainX, valX, tesX = post_process_categorized_gestuers(trainX, valX, tesX)
      trainX = []

      __num_classes = 5 if config[USE_ONE_VS_ALL] else self.num_classes

      presX = np.empty((0, self.win_len, self.num_features))
      presY = np.empty((0))

      ptstX = np.empty((0, self.win_len, self.num_features))
      ptstY = np.empty((0))

      final_valX = np.empty((0, self.win_len, self.num_features))
      final_valY = np.empty((0))

      # Mohit: The total number of nod sequences in the training dataset
      # We should chooose the most appropriate class here e.g. ticks since they are
      # large in number proportinately to other gestures.

      nodlen = len(trainX[1]) #length of nods class

      # Mohit: Interesting we don't take a lot of 0's but rather take the same
      # amount as the number of classes
      trainX[0] = trainX[0][0:nodlen] #clip 0 class to same size

      # Clone and clip other classes to have the same size
      for i in range(2, __num_classes):
        oldlen = len(trainX[i])
        reps = int(math.ceil(float(nodlen)/float(oldlen)))
        # Below does not work on fucking padded seqs!
        trainX[i] = np.tile(trainX[i], (reps, 1, 1))
        trainX[i] = trainX[i][0:nodlen]

      for i in range(__num_classes):
        # Mohit: Padding happens here.
        # No padding will happen since they are all of self.win_len but
        # I need to use this to convert a list of list of Numpy Arrays to 3d
        # arrays
        trainX[i] = pad_sequences(trainX[i], maxlen=self.win_len,
            dtype='float32')
        valX[i] = pad_sequences(valX[i], maxlen=self.win_len,
            dtype='float32')
        tesX[i] = pad_sequences(tesX[i], maxlen=self.win_len,
           dtype='float32')

      for i in range(__num_classes):
        print "Shapes: | train {0} | test {1} | ".format(ptstX.shape,
            tesX[i].shape)
        presX = np.concatenate((presX,trainX[i]))
        presY = np.concatenate((presY,np.full(nodlen, i).astype(int)))

        if valX[i].shape[0]:
          final_valX = np.concatenate((final_valX, valX[i]))
          final_valY = np.concatenate((final_valY,
            np.full(len(valX[i]), i).astype(int)))

        if tesX[i].shape[0]:
          ptstX = np.concatenate((ptstX,tesX[i]))
          ptstY = np.concatenate((ptstY,np.full(len(tesX[i]), i).astype(int)))

      sys.stdout.write('Final dataset stats:\n')
      for i in xrange(__num_classes):
        gest_i = np.sum(presY==i)
        sys.stdout.write('{0} '.format(gest_i))
      sys.stdout.write('\n')
      sys.stdout.flush()

      presY = np_utils.to_categorical(presY, __num_classes)
      final_valY = np_utils.to_categorical(final_valY, __num_classes)
      ptstY = np_utils.to_categorical(ptstY, __num_classes)

      # Mohit: Takeaway reduce the dataset for no gesture that's kind of important.
      # Collect some stats from your code itself to see how many classes have how
      # much data for training.
      return presX, presY, final_valX, final_valY, ptstX, ptstY

    def get_gest_list_file(self, fpath):
      '''
      Returns a list of list of all the gestures in a given h5file. The class
      type is used as an index into the top level list. The inner list is a list
      of start idx, end idx, gest_type for the respective gesture.
      '''
      gest_idx = [[] for _ in range(self.num_classes)]
      hf = h5py.File(fpath, 'r')
      d_annotations = hf.get('annotations')
      a = np.array(d_annotations).astype(int)
      # We don't do any processing on the annotations so no need to call this
      # f, a = self.process_data(f, a)

      #prepare train sequence based on class
      c, min_gesture_threshold = 0, 10
      for i in range(len(a)-1):
        if a[i] != a[i+1] or (c == self.win_len-1):
          #annotations differ or max length reached
          if c > min_gesture_threshold:
            gest_idx[a[i]].append([i-c, i+1]) # Add as relevant gesture

            # Gaussian noise can be added at runtime for other gestures.
            # since batch length is decided on nods we add random noise
            # here itself for nods.
            if a[i] == -1:
              r_mean, r_var, r_samples = 5, 10, 5
              for r in range(r_samples):
                r_val = int(np.random.normal(r_mean, r_var, 1))
                left_idx, right_idx = i-c, i+1
                while left_idx-r_val < 0 or right_idx - left_idx + r_val < 10:
                  r_val = int(np.random.normal(r_mean, r_var, 1))
                gest_idx[a[i]].append([left_idx-r_val, right_idx])

          # reset to a new sequence.
          c=0
        else:
          c+=1

      # Mohit: Interestingly the i carries over or this is a bug?
      gest_idx[a[i]].append([i+1-c, i+2]) #train sequence
      return gest_idx

    def get_all_gest(self, fdir, flist=None):
      '''
      Returns a dictionary of all the gestures for each file. The dict key
      is the h5 filename and the value is a list of lists as returned by
      get_gest_list_file.
      '''
      gest_list_by_file = {}
      for npfile in os.listdir(fdir):
        # Add file only if either flist is None which means we want to add all
        # files.
        # Or if flist is Not None only add files in flist since we only want to
        # add specific files
        if not flist or (npfile in flist):
          gest_list = self.get_gest_list_file(fdir + '/' + npfile)
          gest_list_by_file[npfile] = gest_list
          g_len = [len(gest_list[i]) for i in xrange(len(gest_list))]
          print('{} stats: {}'.format(npfile, g_len))
      return gest_list_by_file
        
    def save_all_gest_dict(self, fsave_path, gest_list_by_file):
      hf = h5py.File(fsave_path, 'w')
      for (gest_file_name, gest_list) in gest_list_by_file.iteritems():
        grp = hf.create_group(gest_file_name)
        for i in range(len(gest_list)):
          grp.create_dataset(str(i), data=np.array(gest_list[i]))
      hf.flush()
      hf.close()
      print 'Write dataset to {}'.format(fsave_path)

    def save_custom_batch_data(self, fdir='../openface_data/face_gestures/dataseto_text'):
      train_file_path, val_file_path, test_file_path = [], [], []
      for npfile in os.listdir(fdir):
        if npfile.endswith("static.mp4.txt.h5"):
          dataset_for_file = self.dataset_for_file(npfile)
          if dataset_for_file == 0:
            train_file_path.append(npfile)
          elif dataset_for_file == 1:
            val_file_path.append(npfile)
          else:
            test_file_path.append(npfile)

      # Get gesture dict as list
      train_gest = self.get_all_gest(fdir, flist=train_file_path)
      val_gest = self.get_all_gest(fdir, flist=val_file_path)
      test_gest = self.get_all_gest(fdir, flist=test_file_path)

      # Save gesture dict 
      self.save_all_gest_dict('../openface_data/train_gesture_by_file.h5', train_gest)
      self.save_all_gest_dict('../openface_data/val_gesture_by_file.h5', val_gest)
      self.save_all_gest_dict('../openface_data/test_gesture_by_file.h5', test_gest)

    def dataset_for_file(self, npfile):
      '''
      Return 0 if file is train, 1 for validation data, 2 for test data
      '''
      train, val, test = 0, 1, 2
      if int(npfile[:3]) >= 12 and int(npfile[:3]) <= 15:
        return test
      else:
        if int(npfile[:3]) <= 11:
          return val
        else:
          return train

    def load_data_centered_padded(self):

      f_data = np.empty((0, self.num_features))
      a_data = np.empty((0))
      c=0
      trainX = [[] for _ in range(self.num_classes)]
      valX = [[] for _ in range(self.num_classes)]
      tesX = [[] for _ in range(self.num_classes)]
      fdir = '../openface_data/face_gestures/dataseto_text/'

      # Any gesture (even none) needs to be > min_gesture_threshold
      min_gesture_threshold = 30

      for npfile in os.listdir(fdir):
        if npfile.endswith("static.mp4.txt.h5"):
          tmpX = [[] for _ in range(self.num_classes)]

          hf = h5py.File(fdir + "/" + npfile,'r')
          d_annotations = hf.get('annotations')
          d_features = hf.get('features')
          a = np.array(d_annotations).astype(int)
          f = np.array(d_features)
          f, a = self.process_data(f, a)

          #prepare train sequence based on class
          for i in range(len(a)-1):
            if (not np.array_equal(a[i],a[i+1])) or (c==self.win_len-1):
              #annotations differ or max length reached
              if c > min_gesture_threshold:
                tmpX[a[i]].append(f[i-c:i+1]) #train sequence
                # Add gaussian noise
                rand_mean, rand_var, num_rand_samples = 0, 10, 5
                rand_noise = np.array(np.random.normal(rand_mean, rand_var,
                    num_rand_samples), dtype=np.int32)
                gest_left_idx, gest_right_idx = i-c, i+1
                for n in rand_noise:
                  if (gest_left_idx-n < 0 or gest_right_idx+n >= len(a) or
                          gest_right_idx-gest_left_idx+n > self.win_len):
                    continue
                  tmpX[a[i]].append(f[gest_left_idx+n:gest_right_idx])

              c=0
              continue
            c+=1

          # Mohit: Interestingly the i carries over or this is a bug?
          tmpX[a[i]].append(f[i+1-c:i+2]) #train sequence

          for i in range(self.num_classes):
            if int(npfile[:3]) >= 12 and int(npfile[:3]) <= 15:
              tesX[i] += tmpX[i][:]
            else:
              insert_in_set = None
              if int(npfile[:3]) <= 11:
                insert_in_set = valX
              else:
                insert_in_set = trainX
              if i == 0:
                # Too much gestures for None. Just take a few of them?
                insert_in_set[i] += tmpX[i][:]
              else:
                insert_in_set[i] += tmpX[i][:]

      # Do any post processing e.g. doing a one-vs-all classification we change
      # the elements of this array.
      # trainX, valX, tesX = self.post_process_categorized_gestuers(trainX, valX, tesX)
      trainX, valX, tesX = self.classify_into_classes(trainX, valX, tesX)


      __num_classes = 5 if self.config[USE_ONE_VS_ALL] else self.num_classes

      presX = np.empty((0, self.win_len, self.num_features))
      presY = np.empty((0))

      ptstX = np.empty((0, self.win_len, self.num_features))
      ptstY = np.empty((0))

      final_valX = np.empty((0, self.win_len, self.num_features))
      final_valY = np.empty((0))

      # Mohit: The total number of nod sequences in the training dataset
      # length of nods class
      #nodlen = len(trainX[1]) if len(trainX[1]) < len(trainX[0]) else len(trainX[0])

      # Mohit: Interesting we don't take a lot of 0's but rather take the same
      # amount as the number of classes
      # trainX[0] = trainX[0][0:nodlen] #clip 0 class to same size
      # trainX[1] = trainX[1][:nodlen] # clip 1 class to same size

      # Clone and clip other classes to have the same size
      #for i in range(2, __num_classes):
      #  oldlen = len(docX[i])
      #  reps = int(math.ceil(float(nodlen)/float(oldlen)))
      #  trainX[i] = np.tile(trainX[i], reps) #does not work onfpadded seqs!
      #  trainX[i] = trainX[i][0:nodlen]

      for i in range(__num_classes):
        # Mohit: Padding happens here.
        trainX[i] = pad_sequences(trainX[i], maxlen=self.win_len,
            dtype='float32')
        valX[i] = pad_sequences(valX[i], maxlen=self.win_len,
            dtype='float32')
        tesX[i] = pad_sequences(tesX[i], maxlen=self.win_len,
            dtype='float32')

      for i in range(__num_classes):
        print 'Shapes: train {0}| test {1}| '.format(ptstX.shape,tesX[i].shape)
        presX = np.concatenate((presX,trainX[i]))
        presY = np.concatenate((presY, 
          np.full(trainX[i].shape[0], i).astype(int)))
        if valX[i].shape[0]:
          final_valX = np.concatenate((final_valX, valX[i]))
          final_valY = np.concatenate((final_valY, 
            np.full(len(valX[i]), i).astype(int)))

        if tesX[i].shape[0]:
          ptstX = np.concatenate((ptstX,tesX[i]))
          ptstY = np.concatenate((ptstY,
            np.full(len(tesX[i]), i).astype(int)))

      sys.stdout.write('Final dataset stats:\n')
      for i in xrange(__num_classes):
        gest_i = np.sum(presY==i)
        sys.stdout.write('{0} '.format(gest_i))
      sys.stdout.write('\n')
      sys.stdout.flush()

      presY = np_utils.to_categorical(presY, __num_classes)
      final_valY = np_utils.to_categorical(final_valY, __num_classes)
      ptstY = np_utils.to_categorical(ptstY, __num_classes)

      # Mohit: Takeaway reduce the dataset for no gesture that's kind of
      # important.  Collect some stats from your code itself to see how many
      # classes have how much data for training.
      return presX, presY, final_valX, final_valY, ptstX, ptstY

    def add_centered_windows(self, X, y, X_c, y_c, st, end, c_idx, current_label,
        max_perturbations=-1):
      """
      Fills the centered windows in X_c and corresponding label in y_c. The start
      and end are the locations of the gesture in X. Note since we do data
      augmentation there can be instances wherein this is not true i.e.  X[st] and
      X[end] might not have current_label as their corresponding label.
      """
      SWL = self.win_len  # For convenience
      gest_len, center = end-st, (st+end)/2
      X_c[c_idx:c_idx+SWL, :] = X[center - SWL/2:center - SWL/2 + SWL, :]
      y_c[c_idx:c_idx+SWL] = current_label
      c_idx = c_idx + SWL

      if current_label == 0:
        return c_idx

      g, shift_window = np.ceil(np.sqrt(gest_len)), 1
      num_windows = g if g % 2 else g-1

      while shift_window <= num_windows/2:
        for new_center in [center-shift_window, center+shift_window]:
          X_c[c_idx:c_idx+SWL,:] = X[new_center-SWL/2:new_center+SWL/2,:]
          y_c[c_idx:c_idx+SWL] = current_label
          c_idx = c_idx + SWL
        shift_window += 1
      return c_idx


    def create_centered_data(self, X, y):
      # Centered data
      X_c = np.zeros((X.shape[0]*self.win_len, X.shape[1]))
      y_c = np.zeros(X_c.shape[0])
      c_idx = 0

      # Avoid edge cases with indexes by choosing [10:len-10] frames only for
      # processing
      N = X.shape[0] - self.win_len / 2
      i, NONE_LABEL = self.win_len / 2, 0
      SWL = self.win_len # just for short variable name
      while i < N:
        # If there was "some" gesture we need to center it.
        # Find the end of this gesture and then based on the length of this
        # gesture create multiple windows of length self.win_len each
        # with a peack on a part of this gesture.
        # Here we do this by making sqrt(N) windows for an N length gesture.
        j, current_label = i+1, y[i]
        while j < N and y[j] == current_label and j-i<SWL:
          j = j+1

        c_idx = add_centered_windows(X, y, X_c, y_c, i, j, c_idx, current_label)
        # Add random noise. We generate random numbers for a normal distribution
        # and change the start and end timestamp of our signals.
        noise = np.array(np.random.normal(0,3,0), dtype=np.int32)
        for n in noise:
          if (abs(n) > 0 and abs(n) < 5 and (i+j)/2+n-SWL/2 >= 0 and
            (i+j)/2+n+SWL/2 < X.shape[0]):
            c_idx = add_centered_windows(X, y, X_c, y_c, i+n, j,
                c_idx, current_label)

        # Increment i to the gesture end.
        i = j

      return X_c[:c_idx, :], y_c[:c_idx]


    def load_data_centered_headgestures(self, test_size=0.2):
      """
      Loads the headgestures centered on each frame. Given a frame we choose the
      window length such that the selected frame is at the center of this window.
      """
      fdir = '../openface_data/face_gestures/dataseto_text'

      X, y = [], []

      for npfile in os.listdir(fdir):
        if npfile.endswith("static.mp4.txt.h5"):
          hf = h5py.File(fdir + "/" + npfile,'r')
          a = np.array(hf.get('annotations')).astype(int)
          f = np.array(hf.get('features'))
          X_file, y_file = process_data(f, a)
          X_file, y_file = create_centered_data(X_file, y_file)
          print('File: {0}, windows: {1}'.format(npfile,
            X_file.shape[0]/self.win_len))
          X.append(X_file)
          y.append(y_file)

      X = np.vstack(X)
      y = np.concatenate(y)
      sys.stdout.write('Final dataset stats:\n')
      for i in xrange(self.num_classes):
        gest_i = np.sum(y==i) / self.win_len
        sys.stdout.write('{0} '.format(gest_i))
      sys.stdout.write('\n')
      sys.stdout.flush()

      assert(X.shape[0] % self.win_len == 0)
      train_size = int(int(X.shape[0]/self.win_len) * (1.0-test_size))
      train_size = train_size * self.win_len
      return X[:train_size], y[:train_size], X[train_size:], y[train_size:]

    def load_data_headgestures(self, test_size=0.3):
      fdir = "../openface_data/staticsetb"

      f_data = np.empty((0, self.num_features))
      a_data = np.empty((0))

      docX = [[] for _ in range(self.num_classes)]
      tesX = [[] for _ in range(self.num_classes)]

      for npfile in os.listdir(fdir):
        if npfile.endswith(".h5"):
          tmpdocX = [[] for _ in range(self.num_classes)]
          tmptesX = [[] for _ in range(self.num_classes)]

          #print("training with: " + npfile)
          hf = h5py.File(fdir + "/" + npfile,'r')
          a = hf.get('annotations')
          f = hf.get('features')
          #a = np.array(d_annotations)
          #f = np.array(d_features)

          #prepare train sequence based on class
          for i in range(len(a)-self.win_len):
            pos = i+(self.win_len/2)
            aclass = a[pos]

            # segment is a 2D array of size (window_len, no_of_sensors)
            segment = f[i:i+self.win_len]

            #create binary mask
            mask = a[i:i+self.win_len]
            mask[mask==aclass] = 100
            mask[mask!=aclass] = 0
            mask[mask==100] = 1

            maskedsegment = segment
            maskedsegment[mask!=1] = 0

            # This should be a list of 2d numpy arrays each of length
            # (window_len, no_of_sensors)
            tmptesX[a[pos]].append(segment)
            # tmpdocX[a[pos]].append(segment[mask==1]) #have to try this
            tmpdocX[a[pos]].append(maskedsegment)

          for i in range(self.num_classes):
            ntrn = int(round(len(tmptesX[i]) * (1 - test_size)))
            docX[i] += tmpdocX[i][:ntrn]
            tesX[i] += tmptesX[i][ntrn:]
            # print '===== mohit ====='
            # print type(docX[i][0])
            # print len(docX[i][0])
            # sys.exit(0)
            # print '===== mohit ====='

      for i in range(self.num_classes):
        print(" size train {0}, test {1}".format(len(docX[i]), len(tesX[i])))

      presX = np.empty((0, self.win_len, self.num_features))
      presY = np.empty((0))

      ptstX = np.empty((0, self.win_len, self.num_features))
      ptstY = np.empty((0))

      resX = np.empty((0, self.num_features))
      resY = np.empty((0))

      tstX = np.empty((0, self.num_features))
      tstY = np.empty((0))

      nodlen = len(docX[0]) #length of nods class
      '''
      for i in range(1,self.num_classes): #clone and clip other classes to have the same size
        oldlen = len(docX[i])
        reps = int(math.ceil(float(nodlen)/float(oldlen)))
        docX[i] = np.tile(docX[i], reps) #does not work on fucking padded seqs!
        docX[i] = docX[i][0:nodlen]
      '''

      #for i in range(self.num_classes):
      #	docX[i] = pad_sequences(docX[i], maxlen=self.win_len, dtype='float64')
      #	tesX[i] = pad_sequences(tesX[i], maxlen=self.win_len, dtype='float64')

      for i in xrange(self.num_classes):
        presX = np.concatenate((presX,docX[i]))
        # presX = np.vstack([presX np.vstack(docX[i])
        presY = np.concatenate((presY,np.full(nodlen, i)))
        ptstX = np.concatenate((ptstX,tesX[i]))
        # presY = np.vstack(tesX[i])
        ptstY = np.concatenate((ptstY,np.full(len(tesX[i]), i)))

      print 'Done concat arrays 1'

      # TODO(Mohit): Maybe the memory constraints for this fitting in memory are
      # huge. presX is a 365269x60x26 array
      print len(presX)
      # Preallocate to save time.
      resX = np.zeros((len(presX)*presX[0].shape[0], presX[0].shape[1]))
      resY = np.zeros(resX.shape[0])
      for i in xrange(len(presX)):
        # print presX[i].shape
        # sys.exit(0)
        resX[i*self.win_len:(i+1)*self.win_len] = presX[i]
        resY[i*self.win_len:(i+1)*self.win_len] = presY[i]
        # resX = np.concatenate((resX,presX[i]))
        # resY = np.concatenate((resY,np.full(self.win_len, presY[i])))

      print 'Done concat arrays 2'

      tstX = np.zeros((len(ptstX)*self.win_len, self.num_features))
      tstY = np.zeros(tstX.shape[0])
      for i in xrange(len(ptstX)):
        tstX[i*self.win_len:(i+1)*self.win_len] = ptstX[i]
        tstY[i*self.win_len:(i+1)*self.win_len] = ptstY[i]
        # tstX = np.concatenate((tstX,ptstX[i]))
        # tstY = np.concatenate((tstY,np.full(self.win_len, ptstY[i])))

      resX, resY = process_data(resX, resY)
      tstX, tstY = process_data(tstX, tstY)
      print 'Created training and test set'
      return resX, resY, tstX, tstY

    def get_data(self, use_cached=True):
      """
      Load the data and format it accordingly.
      use_cached: Load data from the cache i.e. saved in the appropriate format on
      disk.
      use_sequential: Simpler data access.
      """
      use_fulL_data = self.config.get(USE_FULL_DATA, False)
      use_sequential_data = self.config[USE_SEQUENTIAL_DATA]
      use_centered_data = self.config[USE_CENTERED_DATA]

      if use_cached and os.path.isfile('dclstm_ss3.h5'):
        print "Loading dataset"
        hf = h5py.File('dclstm_ss3.h5','r')
        X_train = np.array(hf.get('X_train'))
        y_train = np.array(hf.get('y_train'))
        X_test = np.array(hf.get('X_test'))
        y_test = np.array(hf.get('y_test'))
      else:
        print "Creating dataset"
        X_train, y_train, X_test, y_test = None, None, None, None
        if use_sequential_data:
          if use_fulL_data:
            X_train, y_train, X_test, y_test = load_h5_static_data()
          else:
            X_train, y_train, X_test, y_test = load_sequential_data_headgestures()
        elif use_centered_data:
          X_train, y_train, X_test, y_test = load_data_centered_headgestures()
        else:
          X_train, y_train, X_test, y_test = load_data_headgestures()

      if not use_cached and config[CACHE_DATA]:
        hf = h5py.File(CACHE_DATA_FILE_NAME, 'w')
        hf.create_dataset('X_train', data=X_train)
        hf.create_dataset('y_train', data=y_train)
        hf.create_dataset('X_test', data=X_test)
        hf.create_dataset('y_test', data=y_test)
        print 'Write dataset to {}'.format(CACHE_DATA_FILE_NAME)

      X_train = X_train.astype(np.float32)
      # y_train = y_train.astype(np.float32)
      X_test = X_test.astype(np.float32)
      # y_test = y_test.astype(np.float32)
      return X_train, y_train, X_test, y_test

