import numpy as np
import matplotlib.pyplot as plt
import json
import pdb
import sys
import seaborn as sns
import os

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
    plt.show()

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

    plt.show()


def main(json_file, save_dir):
    with open(json_file, 'r') as fp:
        data = json.load(fp)
    if type(data) == type({}):
        plot_data(data, save_dir)
    elif type(data) == type([]):
        plot_data(get_json_list_data(data), save_dir)
    else:
        assert(False)

if __name__ == '__main__':
    assert(len(sys.argv) >= 2)
    json_file = sys.argv[1]
    save_dir = os.path.dirname(json_file)
    main(json_file, save_dir)

