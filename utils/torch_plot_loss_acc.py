import numpy as np
import pandas as pd
import seaborn as sns
import pdb
import json
import sys
import os
import matplotlib.pyplot as plt

def get_conf(json_file, num_classes=5):
    with open(json_file, 'r') as fp:
        data = json.load(fp)
        conf = data.get('conf', None)
    if conf is None:
        return
    # c1 = conf.split('\n')[1].split("]")[0].split("[ ")[1].split(" ")
    c1 = conf.split('\n')
    conf_mat, row_idx = np.zeros((num_classes, num_classes)), 0
    for i in c1:
        #pdb.set_trace()
        is_conf_row = False
        if ']' in i and '[[' in i:
            val = i.split(']')[0].split('[[')[1].split(' ')
            is_conf_row = True
        elif ']' in i and '[' in i:
            val = i.split(']')[0].split('[')[1].split(' ')
            is_conf_row = True
        if is_conf_row:
            col_idx = 0
            for v in val:
                if not len(v):
                    continue
                try:
                    conf_mat[row_idx, col_idx] = int(v)
                    col_idx = col_idx + 1
                except:
                    continue
            row_idx = row_idx + 1

    assert(row_idx == num_classes)
    conf_mat = conf_mat.astype(int)
    fdir = os.path.dirname(json_file)
    json_name = os.path.basename(json_file)[:-5]
    conf_file_name = fdir + '/' + 'conf_' + json_name + '.txt'
    np.savetxt(conf_file_name, conf_mat, fmt='%d', delimiter=', ')
    return conf_mat

def get_f1_score(conf, weights):
    prec = np.zeros(conf.shape[0])
    recall = np.zeros(conf.shape[0])
    
    f1 = np.zeros(conf.shape[0])
    for i in xrange(conf.shape[0]):
        if np.sum(conf[:, i]) != 0:
            prec[i] = float(conf[i,i])/np.sum(conf[:, i])
        else:
            prec[i] = 0

        recall[i] = float(conf[i,i])/np.sum(conf[i,:])
        if (prec[i] + recall[i] == 0):
            f1[i] = 0
        else:
            f1[i] = (2.0 * prec[i] * recall[i]) / (prec[i] + recall[i])
    weights = weights / np.sum(weights)
    return np.sum(weights * f1)

def best_f_scores(fdir, num_classes=5):
    best_checkpoints = [None, None, None]
    best_3_fscores = [0, 0, 0]
    best_confs = [np.array(()), np.array(()), np.array(())]
    f1_weight_list = [1.0] * num_classes
    f1_weights = np.array(f1_weight_list)
    for f in os.listdir(fdir):
        if f.endswith('json'):
            json_file = fdir + '/' + f
            conf = get_conf(json_file, num_classes)
            f1 = get_f1_score(conf, f1_weights)
            print('file: {}, f1: {:4f}'.format(f, f1))
            max_idx = -1
            for i in range(len(best_3_fscores)):
                if best_3_fscores[i] > f1:
                    break
                max_idx = i
            for j in range(max_idx):
                best_3_fscores[j] = best_3_fscores[j+1]
                best_confs[j] = best_confs[j+1]
                best_checkpoints[j] = best_checkpoints[j+1]

            best_3_fscores[max_idx] = f1
            best_confs[max_idx] = conf
            best_checkpoints[max_idx] = f

    return best_3_fscores, best_confs, best_checkpoints

def main(json_file, fdir):
    with open(json_file, 'r') as fp:
        data = json.load(fp)
    # Loss history might not be of equal length.
    train_loss_hist = data['train_loss_history']
    val_loss_hist = data['val_loss_history']

    plt.close('all')
    plt.figure(figsize=(20,10))
    plt.subplot(2, 1, 1)
    plt.plot(train_loss_hist)
    plt.title('Train loss')
    plt.ylabel('loss')
    plt.xlabel('time')
    plt.subplot(2, 1, 2)
    plt.plot(val_loss_hist)
    plt.title('Val loss')
    plt.ylabel('loss')
    plt.xlabel('time')
    plt.savefig(fdir + '/' + os.path.basename(json_file)[:-4] + 'png')
    plt.show()

if __name__ == '__main__':
    assert(len(sys.argv) >= 2)
    json_file = sys.argv[1]
    fdir = os.path.dirname(json_file)
    if (len(sys.argv) >= 2):
        num_classes = int(sys.argv[2])
    else:
        num_classes = 5
    f_score, confs, checkpoints = best_f_scores(fdir, num_classes)
    print '======'
    print('Best f scores', f_score)
    print '======'
    print('Best confusion matrixs')
    print(confs[0])
    print '======'
    print(confs[1])
    print '======'
    print(confs[2])
    print '======'
    print('Best checkpoints ', checkpoints)

    main(json_file, fdir)

