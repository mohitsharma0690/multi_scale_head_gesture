import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
import pdb
import sys
import os

from scipy.signal import savgol_filter

'''
Given a gesture video and an h5 file with the gesture annotations plot a graph
for a particular tracked node for the entire gesture sequence length and another
plot for the predictions as given by the prediction h5 file.
'''
def normalize_data_each_sensor_signal(X, y):
    '''
    Normalize data X and y.
    '''
    mean_signal = sum(X, 0) / float(X.shape[0])
    zero_mean_X = X - mean_signal
    var_X = sum((zero_mean_X ** 2), 0) / float(X.shape[0])
    norm_X = zero_mean_X / var_X
    return norm_X, y

def smooth_data(X):
    window_len, poly_order = 11, 2
    for i in xrange(X.shape[1]):
        X_data = X[:,i]
        X[:, i] = savgol_filter(X_data, window_len, poly_order)
    return X

def main_temp(json_file):
    with open(json_file, 'r') as fp:
        data = json.load(fp)
    assert(len(data.keys()) >= 2)
    plt.close('all')
    print(data.keys())

    # summarize history for accuracy
    plt.plot(data['acc'])
    plt.plot(data['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('acc_hist.png')
    plt.show()

    # summarize history for loss
    plt.plot(data['loss'])
    plt.plot(data['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss_hist.png')
    plt.show()

def load_data(fname, axis=[45], keep_label=[]):
    f = h5py.File(fname)
    X = np.array(f['features'])
    y = np.array(f['annotations']).astype(int)
    filt_X = np.zeros((X.shape[0], len(axis)))
    for i in xrange(len(axis)):
        filt_X[:,i] = X[:, axis[i]]
    if len(keep_label):
        for label in keep_label:
            y[y==label] = 1001 + label
        y[y < 1000] = 0
        y[y > 1000] -= 1001
    return filt_X, y

def classify_labels_5(y):
    y_new = np.copy(y)
    for i in range(len(y_new)):
        if y_new[i] == -1 or y_new[i] == 0:
            continue
        elif y_new[i] >= 1 and y_new[i] <= 5:
            y_new[i] = 1
        elif y_new[i] == 6:
            y_new[i] = 2
        elif y_new[i] == 7 or y_new[i] == 8:
            y_new[i] == 3
        elif y_new[i] == 9 or y_new[i] == 10:
            y_new[i] = 4
        else:
            assert(False)
    return y_new

def split_data_gestures(X, y, start_ts, end_ts):
    '''
    return a list of arrays with splits according to the type of gesture
    '''
    gest_x, gest_y = [], []
    t = start_ts
    while t < end_ts:
        j = t+1
        while j<end_ts and y[j]==y[t]:
            j = j + 1
        gest_x.append(X[t:j])
        gest_y.append(y[t])
        t = j
    return gest_x, gest_y

def main(h5_file_name, h5_pred_filename, start_ts=0, end_ts=1000):
    save_plot_fname = None
    # Important landmarks (index starting at 1)
    # 28 (forehead), 34 (nose tip), 2, 4 (left face edge), 8, 10 (bottom face)
    # 16, 14 (right face edge)

    # 12 (pose, gaze), 68 landmarks each (x, y)
    plot_landmarks_axis_y = [12 + 68 + 34 - 1]
    plot_landmarks_axis_x = [34 - 1]
    #plot_pose_gaze_axis = [7]
    #h5_file_name = '..//024_static.mp4.txt.h5'
    #h5_file_name = '../TrainingData/mohit_tick_1.h5'
    if len(sys.argv) >= 2:
        save_plot_fname = sys.argv[1]
    X, y = load_data(h5_file_name, axis=plot_landmarks_axis_y,
            keep_label=range(0,11))
    y = classify_labels_5(y)
    #X = X[:,0] - X[:,1]

    # Preprocess data
    X, y = normalize_data_each_sensor_signal(X, y)
    # This is the difference between the signal at each timestamp
    X = X[0:-1] - X[1:]
    # X = smooth_data(X)

    T = end_ts - start_ts
    gest_seq_x, gest_seq_y = split_data_gestures(X, y, start_ts, end_ts)
    plt.close('all')
    target_label, t = range(0,11), 0

    ####
    ##### 0 = none
    ##### 1 = Nod
    ##### 2 = Jerk
    ##### 3 = Up
    ##### 4 = Down
    ##### 5 = Ticks
    ####6 = Tilt
    ##### 7 = Shake
    ##### 8 = Turn
    ##### 9 = Forward
    ##### 10 = Backward
    ####
    colors = ['#0000FF', '#FF0000', '#00FF00', '#00FFFF', '#FF00FF', '#FFFF00',
            '#FFFFFF', '#ABCDEF', '#BBCCFF', '#DDABCE', '#FFABCD']
    gest_labels = ['None', 'Nod', 'Jerk', 'Up', 'Down', 'Ticks', 'Tilt',
            'Shake', 'Turn', 'Forward', 'Backward']
    # To prevent matplotlib from labeling things twice
    gest_labeled = np.zeros(len(gest_labels))
    plt.subplot(211)
    for i in xrange(len(gest_seq_x)):
        if gest_seq_y[i] in target_label:
            if gest_labeled[gest_seq_y[i]]:
                plt.plot(range(t,t+len(gest_seq_x[i])), gest_seq_x[i],
                        color=colors[gest_seq_y[i]], marker='o')
            else:
                plt.plot(range(t,t+len(gest_seq_x[i])), gest_seq_x[i],
                        color=colors[gest_seq_y[i]], marker='o',
                        label=gest_labels[gest_seq_y[i]])
            gest_labeled[gest_seq_y[i]] = 1
        else:
            plt.plot(range(t,t+len(gest_seq_x[i])), gest_seq_x[i], 'bo',
                    label='Other')
        t = t + len(gest_seq_x[i])
    plt.legend(loc='upper left')


    # Plot predictions
    pred_h5 = h5py.File(h5_pred_filename, 'r')
    target_h5 = os.path.basename(h5_file_name)
    pred = pred_h5['/' + str(target_h5)]
    # new_X = X[start_ts:end_ts]
    # new_pred = pred[start_ts:end_ts]
    new_X, new_pred = split_data_gestures(X, pred, start_ts, end_ts)
    plt.subplot(212)
    t = 0
    for i in xrange(len(new_X)):
        label = int(new_pred[i])
        if new_pred[i] == -1:
            plt.plot(range(t, t+len(new_X[i])), new_X[i], color='k', marker='o')
        else:
            plt.plot(range(t, t+len(new_X[i])), new_X[i], color=colors[label],
                    marker='o')
        t = t + len(new_X[i])

    pred_h5.close()
    plt.show()


if __name__ == '__main__':
    assert(len(sys.argv) == 5)
    h5_file_name = sys.argv[1]
    start_frames = int(sys.argv[2])
    end_frames = int(sys.argv[3])
    h5_pred_filename = sys.argv[4]
    main(h5_file_name, h5_pred_filename, start_frames, end_frames)

