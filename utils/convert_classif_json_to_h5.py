import numpy as np
import h5py
import sys
import os
import json
import csv
import types
import argparse

# Sample Usage:
#
# python convert_classif_json_to_h5.py \
#   --h5_dir <Dir where h5 files for videos with annotations are stored> \
#   --pred_dir <Directory containing the predictions json> \

# python convert_classif_json_to_h5.py \
#   --h5_dir ../data \
#   --pred ../conv_lstm/torch/exp1
#

FLAGS = None

def get_num_annotations(fdir, h5_file):
    f = h5py.File(fdir + '/' + h5_file)
    return len(f['annotations'])

def recursively_save_dict_contents_to_group(h5file, path, dic):
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
            recursively_save_dict_contents_to_group(
                    h5file, path + key + '/', dic[key])

def main(fdir, h5_dir, batch_json, test_pred_json, test_scores_json):
    batch_json_f = open(batch_json, 'r')
    test_pred_f = open(test_pred_json, 'r')
    scores_json_f = open(test_scores_json, 'r')
    beta_json_f = None
    print(os.path.join(os.path.dirname(batch_json), 'test_beta.json'))
    if os.path.exists(os.path.join(
        os.path.dirname(batch_json), 'test_beta.json')):
        beta_json = os.path.join(os.path.dirname(batch_json), 'test_beta.json')
        beta_json_f = open(beta_json, 'r')

    # Dict with h5filename as key and a numpy array of predictions from the
    # model. For frames that have no predictions we write -1 for all other we
    # write the classifications.
    test_preds = {}
    batch = json.load(batch_json_f)
    pred = json.load(test_pred_f)
    scores = json.load(scores_json_f)
    beta = json.load(beta_json_f) if beta_json_f is not None else None
    batch_json_f.close()
    test_pred_f.close()
    scores_json_f.close()
    if beta_json_f is not None: beta_json_f.close()

    assert(len(batch) == len(pred))

    for i in range(len(batch)):
        if test_preds.get(batch[i][0], None) is None:
            num_annotations = get_num_annotations(h5_dir, batch[i][0])
            test_preds[str(batch[i][0])] = {}
            test_preds[str(batch[i][0])]['pred'] = np.ones(num_annotations) * -1
            test_preds[str(batch[i][0])]['prob'] = np.zeros((num_annotations, 5))
            test_preds[str(batch[i][0])]['beta'] = np.zeros(num_annotations)

        # Set frame as given by batch[i][0] to what we predicted as given by
        # pred[i][0]. Note Torch predictions are based on 1 as the starting
        # index. Therefore to convert it into the standard format (for python)
        # and h5 in general we subtract 1 from it here.
        if type(pred[i]) == type(1):
            test_preds[str(batch[i][0])]['pred'][int(batch[i][1])] = pred[i] - 1
        else:
            test_preds[str(batch[i][0])]['pred'][int(batch[i][1])] = \
                    pred[i][0] - 1
        if type(scores[i][0]) == type([]):
            test_preds[str(batch[i][0])]['prob'][int(batch[i][1])] = scores[i][0]
        else:
            test_preds[str(batch[i][0])]['prob'][int(batch[i][1])] = scores[i]
        if beta is not None:
            test_preds[str(batch[i][0])]['beta'][int(batch[i][1])] = beta[i][0]
            # test_preds[str(batch[i][0])]['beta'][int(batch[i][1])] = beta[i]

    print('Will save keys')
    print(test_preds.keys())
    # Save test preds
    fname = fdir + '/' + 'pred.h5'
    h5_pred = h5py.File(fname, 'w')
    recursively_save_dict_contents_to_group(h5_pred, '/', test_preds)
    print('Did save test predictions to {}'.format(fname))
    h5_pred.flush()
    h5_pred.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Convert json predictions to h5.')
    parser.add_argument('--h5_dir', nargs='?', type=str, const=1,
            required=True, default='../data', help='h5 files directory')
    parser.add_argument('--pred_dir', nargs='?', type=str, const=1,
            default='', help='pred files directory continaing json files',)
    parser.add_argument('--pred_super_dir', nargs='?', type=str, const=1,
            default='', help='Directory containing all pred dir',)
    args = parser.parse_args()
    print(args)

    if len(args.pred_super_dir) > 0:
      for d in os.listdir(args.pred_super_dir):
        if d.startswith('pred_cp_'):
          pred_dir = os.path.join(args.pred_super_dir, d)
          batch_json = os.path.join(pred_dir, 'test_batch.json')
          test_pred_json = os.path.join(pred_dir, 'test_preds.json')
          test_scores_json = os.path.join(pred_dir, 'test_scores.json')
          fdir = pred_dir
          main(fdir, args.h5_dir, batch_json, test_pred_json, test_scores_json)
    else:
      batch_json = os.path.join(args.pred_dir, 'test_batch.json')
      test_pred_json = os.path.join(args.pred_dir, 'test_preds.json')
      test_scores_json = os.path.join(args.pred_dir, 'test_scores.json')
      fdir = os.path.dirname(batch_json)
      main(fdir, args.h5_dir, batch_json, test_pred_json, test_scores_json)

