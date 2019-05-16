import numpy as np
import matplotlib.pyplot as plt
import h5py

g_train_gests = {
        '007_static.mp4.txt.h5 stats': [152, 21, 8, 1, 3, 22, 9, 6, 25, 7, 10],
        '008_static.mp4.txt.h5 stats': [125, 15, 2, 1, 15, 15, 37, 2, 31, 2, 2],
        '009_static.mp4.txt.h5 stats': [162, 89, 4, 6, 7, 40, 37, 0, 9, 6, 1],
        '011_static.mp4.txt.h5 stats': [160, 103, 16, 2, 15, 16, 6, 1, 9, 0, 0],
        '013_static.mp4.txt.h5 stats': [130, 53, 5, 2, 25, 44, 35, 1, 12, 6, 0],
        '014_static.mp4.txt.h5 stats': [145, 77, 12, 7, 3, 60, 16, 0, 1, 0, 0],
        '015_static.mp4.txt.h5 stats': [134, 32, 1, 0, 5, 35, 4, 4, 11, 12, 0],
        '017_static.mp4.txt.h5 stats': [156, 61, 14, 6, 9, 31, 40, 5, 37, 4, 2],
        '019_static.mp4.txt.h5 stats': [183, 157, 5, 8, 19, 19, 8, 1, 4, 4, 1],
        '022_static.mp4.txt.h5 stats': [204, 125, 43, 5, 30, 4, 4, 2, 3, 2, 2],
        '024_static.mp4.txt.h5 stats': [141, 31, 30, 14, 12, 115, 15, 0, 3, 1, 2],
        '025_static.mp4.txt.h5 stats': [147, 66, 4, 1, 0, 77, 14, 0, 0, 0, 0],
        '026_static.mp4.txt.h5 stats': [171, 104, 6, 2, 16, 2, 0, 0, 18, 1, 1],
        '031_static.mp4.txt.h5 stats': [186, 55, 24, 3, 34, 11, 7, 2, 15, 6, 2],
        '032_static.mp4.txt.h5 stats': [159, 35, 16, 0, 2, 52, 6, 1, 0, 1, 0],
        '034_static.mp4.txt.h5 stats': [133, 39, 4, 0, 11, 30, 30, 0, 20, 2, 1],
        '035_static.mp4.txt.h5 stats': [152, 19, 0, 0, 1, 8, 14, 0, 34, 13, 6],
        '036_static.mp4.txt.h5 stats': [142, 49, 5, 8, 25, 15, 25, 5, 26, 15, 8],
        '037_static.mp4.txt.h5 stats': [158, 70, 4, 8, 7, 4, 3, 0, 7, 2, 1],
        }
g_test_gests = {
        '038_static.mp4.txt.h5 stats': [159, 74, 1, 7, 15, 3, 5, 0, 22, 4, 2],
        '040_static.mp4.txt.h5 stats': [134, 12, 1, 1, 7, 5, 23, 0, 91, 3, 3],
        '041_static.mp4.txt.h5 stats': [153, 15, 19, 4, 10, 30, 7, 9, 54, 0, 12],
        '042_static.mp4.txt.h5 stats': [160, 26, 16, 0, 35, 39, 2, 0, 24, 0, 3],
        '043_static.mp4.txt.h5 stats': [132, 6, 0, 0, 5, 3, 3, 3, 18, 12, 5],
        '044_static.mp4.txt.h5 stats': [146, 24, 19, 8, 5, 22, 18, 1, 17, 2, 9],
    }

def read_from_h5(fname):
    f = h5py.File(fname)
    f_train = f['train']
    f_test = f['test']
    train_gests = {}
    test_gests = {}
    for vid_file in f_train.keys():
        train_gests[vid_file] = []
        for i in range(11):
            train_gests[vid_file].append(len(f_train[vid_file][str(i)]))
    for vid_file in f_test.keys():
        test_gests[vid_file] = []
        for i in range(11):
            test_gests[vid_file].append(len(f_test[vid_file][str(i)]))
    return train_gests, test_gests

def plot_gests(train_gests, test_gests):
    final_train_values = [0] * 11
    final_test_values = [0] * 11
    for (k, v) in train_gests.iteritems():
        for i in range(len(v)):
            final_train_values[i] += v[i]

    for (k, v) in test_gests.iteritems():
        for i in range(len(v)):
            final_test_values[i] += v[i]

    ind = np.arange(11)  # the x locations for the groups
    width = 0.35         # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, final_train_values, width, color='r')
    rects2 = ax.bar(ind + width, final_test_values, width, color='y')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Number of gestures performed')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('None', 'Nod', 'Jerk', 'Up', 'Down', 'Tick',
        'Tilt', 'Shake', 'Turn', 'Forward', 'Backward'))

    ax.legend((rects1[0], rects2[0]), ('Train', 'Test'))

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height), ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.plot()
    plt.savefig('curated_data_stats.png')
    plt.show()

def main():
    train_gests, test_gests = read_from_h5('../data/main_gest_by_file.h5')
    plot_gests(train_gests, test_gests) 

if __name__ == '__main__':
    main()
