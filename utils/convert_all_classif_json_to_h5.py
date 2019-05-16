import numpy as np
import h5py
import sys
import os
import json
import csv
import types
import argparse

import convert_classif_json_to_h5

def save_all_h5(h5_dir, pred_dir, pred_dir_prefix='pred_cp'):
    for f in os.listdir(pred_dir):
        if f.startswith(pred_dir_prefix):
            target_pred_dir = os.path.join(pred_dir, f)
            batch_json = os.path.join(target_pred_dir, 'test_batch.json')
            test_pred_json = os.path.join(target_pred_dir, 'test_preds.json')
            test_scores_json = os.path.join(target_pred_dir, 'test_scores.json')
            target_pred_h5 = os.path.join(target_pred_dir, 'pred.h5')
            if (os.path.exists(batch_json) and
                os.path.exists(test_pred_json) and
                os.path.exists(test_scores_json) and
                os.path.exists(target_pred_h5) is False):
                convert_classif_json_to_h5.main(
                        target_pred_dir,
                        h5_dir,
                        batch_json,
                        test_pred_json,
                        test_scores_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Convert all json predictions to h5.')
    parser.add_argument('--h5_dir', nargs='?', type=str, const=1,
            required=True, default='../data', help='h5 files directory.')
    parser.add_argument('--pred_dir', nargs='?', type=str, const=1,
            required=True, help='directory with all predictions dirs.')
    args = parser.parse_args()
    print(args)

    batch_json = os.path.join(args.pred_dir, 'test_batch.json')
    test_pred_json = os.path.join(args.pred_dir, 'test_preds.json')
    test_scores_json = os.path.join(args.pred_dir, 'test_scores.json')
    fdir = os.path.dirname(batch_json)
    save_all_h5(args.h5_dir, args.pred_dir)
