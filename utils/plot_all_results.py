import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import os
import json

def get_json_list_data(data):
    '''
    Each list item is a dictionary of two keys 'loss' and 'acc'.
    '''
    assert(type(data) == type([]))
    plt.close('all')
    data_dict = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []} 
    for d in data:
        for k, v in data_dict.iteritems():
            if d.get(k, None):
                v.append(d[k])
    return data_dict

def plot_data(data, save_dir):
    if data.get('test_acc') is not None:
        plot_data_3(data, save_dir)
    else:
        plot_data_2(data, save_dir)

def plot_data_2(data, save_dir):
    '''
    data is a dictionary containing 'loss', 'acc', 'val_acc', 'val_loss'
    '''
    plt.close('all')
    plt.figure(figsize=(20,10))
    print(data.keys())

    # summarize history for accuracy
    plt.subplot(2,2,1)
    plt.plot(data['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    plt.legend(['train'], loc='upper left')
    plt.subplot(2,2,2)
    plt.plot(data['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['val'], loc='upper left')

    # summarize history for loss
    plt.subplot(2, 2, 3)
    plt.plot(data['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.subplot(2, 2, 4)
    plt.plot(data['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['validation'], loc='upper left')
    plt.savefig(save_dir + '/' + 'loss_hist.png')

def plot_data_3(data, save_dir):
    plt.close('all')
    plt.figure(figsize=(20,10))

    # summarize history for accuracy
    plt.subplot(2,3,1)
    plt.plot(data['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    plt.legend(['train'], loc='upper left')
    plt.subplot(2,3,2)
    plt.plot(data['val_acc'])
    plt.title('Val accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['val'], loc='upper left')
    plt.subplot(2,3,3)
    plt.plot(data['test_acc'])
    plt.title('Test accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['test'], loc='upper left')

    # summarize history for loss
    plt.subplot(2, 3, 4)
    plt.plot(data['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.subplot(2, 3, 5)
    plt.plot(data['val_loss'])
    plt.title('Val loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['val'], loc='upper left')
    plt.subplot(2, 3, 6)
    plt.plot(data['test_loss'])
    plt.title('Test loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['test'], loc='upper left')
    plt.savefig(save_dir + '/' + 'loss_acc_hist.png')

def get_best_matrix_fpath(fdir, fname_prefix, idx):
    all_files = []
    for f in os.listdir(fdir):
        if f.startswith(fname_prefix):
            f_path = fdir + '/' + f
            all_files.append(f_path)
    all_files.sort()
    return all_files[idx]

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

def plot_conf(norm_conf, fdir, fname):
    # Plot using seaborn
    # (this is style I used for ResNet matrix)
    df_cm = pd.DataFrame(norm_conf)
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True)
    plt.savefig(fdir + '/' + fname)

def main(fdir, json_file):
    data = None
    with open(json_file, 'r') as fp:
        data = json.load(fp)
        if type(data) == type({}):
            plot_data(data, fdir)
        elif type(data) == type([]):
            data = get_json_list_data(data)
            plot_data(data, fdir)

    if data is None or type(data) != type({}):
        print('data not in correct format (dict) will not plot conf mat')
        return

    # Find index for best accuracy in validation and test sets from data
    best_val_idx, best_val_acc = -1, 0
    best_test_idx, best_test_acc = -1, 0
    for i in range(len(data['val_acc'])):
        if data['val_acc'][i] > best_val_acc:
            best_val_acc = data['val_acc'][i]
            best_val_idx = i
    for i in range(len(data['test_acc'])):
        if data['test_acc'][i] > best_test_acc:
            best_test_acc = data['test_acc'][i]
            best_test_idx = i

    # Get best confusion matrix for val and test
    best_val_mat_txt = get_best_matrix_fpath(fdir, 'conf_val', 
            best_val_idx)
    best_test_mat_txt = get_best_matrix_fpath(fdir, 'conf_test',
            best_test_idx)

    val_conf_mat = read_conf_mat(best_val_mat_txt)
    val_conf_mat = normalize_conf(val_conf_mat)
    plot_conf(val_conf_mat, fdir, 'val_conf_mat.png')

    test_conf_mat = read_conf_mat(best_test_mat_txt)
    test_conf_mat = normalize_conf(test_conf_mat)
    plot_conf(test_conf_mat, fdir, 'test_conf_mat.png')

if __name__ == '__main__':
    plt.close('all')
    json_file = sys.argv[1]
    fdir = os.path.dirname(json_file)
    main(fdir, json_file)
    plt.show()
