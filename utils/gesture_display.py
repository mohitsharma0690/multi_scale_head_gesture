import numpy as np
#import matplotlib.pyplot as plt
import os
import sys
import h5py
import pdb
import types

from scipy.signal import savgol_filter

def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]

def draw_X(X):
    x = range(X.shape[1])
    y = range(X.shape[0])
    #data = np.random.randn(y.size,x.size)
    #data = np.array([
    #    [0, 1, 2, 3, 3, 2, 1, 0], 
    #    [1, 2, 3, 4, 4, 3, 2, 1]])             
    data = X

    plt.imshow(data, aspect='auto', interpolation='none',
            extent=extents(x) + extents(y), origin='lower')
    plt.show()

def __recursively_save_dict_contents_to_group__(h5file, path, dic):
    """
    Take an already open HDF5 file and insert the contents of a dictionary
    at the current path location. Can call itself recursively to fill
    out HDF5 files with the contents of a dictionary.
    """
    assert type(dic) is types.DictionaryType, "must provide a dictionary"
    assert type(path) is types.StringType, "path must be a string"
    assert type(h5file) is h5py._hl.files.File, "must be an open h5py file"
    for key in dic:
      assert type(key) is types.StringType, \
                'dict keys must be strings to save to hdf5'
      if type(dic[key]) in (np.int64, np.float64, types.StringType):
        h5file[path + key] = dic[key]
        assert h5file[path + key].value == dic[key], \
                'The data representation in the HDF5 file does not match the ' \
                'original dict.'
      if type(dic[key]) is np.ndarray:
        h5file[path + key] = dic[key]
        assert np.array_equal(h5file[path + key].value, dic[key]), \
                'The data representation in the HDF5 file does not match the ' \
                'original dict.'
      elif type(dic[key]) is types.DictionaryType:
        self.__recursively_save_dict_contents_to_group__(
                h5file, path + key + '/', dic[key])

def smooth_data(X):
  window_len, poly_order = 11, 2
  for i in xrange(X.shape[1]):
    X_data = X[:,i]
    X[:, i] = savgol_filter(X_data, window_len, poly_order)
  return X

def process_data(X, y):
  """
  Process the data set to do normalization and other clean up techniques.
  """
  #TODO(Mohit): Normalize all sensor signals in X.
  if X.shape[1] > 148:
    X = X[:, :148]
    X1 = np.copy(X)
    X = smooth_data(X1)

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
  # X, y = normalize_data_each_sensor_signal(X, y)
  return X, y

def get_index_for_gest_type(gest_type):
    if gest_type == 'forehead_x':
        return 14
    elif gest_type == 'forehead_y':
        return 15
    elif gest_type == 'nosetip_x':
        return 16
    elif gest_type == 'nosetip_y':
        return 17
    elif gest_type == 'forehead_x_diff':
        return 30 
    elif gest_type == 'forehead_y_diff':
        return 31
    elif gest_type == 'nosetip_x_diff':
        return 32
    elif gest_type == 'nosetip_y_diff':
        return 33
    else:
        assert(False)
        return -1

def main(fdir, gest_by_file, save_path, gest_label=1):
    gest_by_file_contents = h5py.File(gest_by_file, 'r')
    # TODO(Mohit): We can group gestures here.
    save_h5_dict = {}
    for group in gest_by_file_contents.keys():
      for target_file in gest_by_file_contents[group].keys():
        gest_seq = np.array(
                gest_by_file_contents[group][target_file][str(gest_label)])
        target_path = os.path.join(fdir, target_file) 
        target_file_f = h5py.File(target_path, 'r')  
        X = np.array(target_file_f['features'])
        X, _ = process_data(X, None)
        target_file_f.close()

        final_gest_width = 0
        # Get the maximum width for some gesture (should be <= 120)
        for i in range(gest_seq.shape[0]):
            seq = gest_seq[i]
            seq_len = seq[1] - seq[0]
            final_gest_width = max(final_gest_width, seq_len)

        final_X = np.zeros((gest_seq.shape[0], final_gest_width)) 
        gest_idx = get_index_for_gest_type('nosetip_y')

        for i in range(gest_seq.shape[0]):
            seq = gest_seq[i]
            gest = X[seq[0]:seq[1], gest_idx]
            gest = np.pad(gest, (0, final_gest_width-gest.shape[0]), 'constant',
                    constant_values=(0, 0))
            final_X[i, :] = gest

        save_h5_dict[str(target_file)] = final_X

    save_f = h5py.File(save_path, 'w')
    __recursively_save_dict_contents_to_group__(save_f, '/', save_h5_dict)
    print('Did save file {} with gesture {}'.format(save_path, gest_label))
    save_f.close()
    gest_by_file_contents.close()


if __name__ == '__main__':
    fdir = sys.argv[1]
    gest_by_file_path = sys.argv[2]
    save_path = sys.argv[3]
    main(fdir, gest_by_file_path, save_path)

