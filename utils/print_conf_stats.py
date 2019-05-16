import numpy as np
import os
import json
import sys

def get_conf_stats(conf, weights):
    prec = np.zeros(conf.shape[0])
    recall = np.zeros(conf.shape[0])
    f1 = np.zeros(conf.shape[0])
    true_pos = 0
    for i in xrange(conf.shape[0]):
        if np.sum(conf[:, i]) != 0:
            prec[i] = float(conf[i,i])/np.sum(conf[:, i])
        else:
            prec[i] = 0
        true_pos = true_pos + conf[i, i]

        recall[i] = float(conf[i,i])/np.sum(conf[i,:])
        if (prec[i] + recall[i]) != 0:
            f1[i] = (2.0 * prec[i] * recall[i]) / (prec[i] + recall[i])
        else:
            f1[i]  = 0
    weights = weights / np.sum(weights)
    stats = { 
            'F1-score' : np.sum(weights * f1), 
            'acc': float(true_pos)/np.sum(conf),
            'F1': f1.tolist(),
            'precision': prec.tolist(),
            'recall': recall.tolist(),
            'confusion': conf.tolist()
            }
    return stats


def main(conf_txt, fdir, fname, num_classify):
    #conf = np.array([[2608, 284, 39, 5, 0], [206, 4, 7, 1, 0],
    #    [175, 157, 183, 50, 0], [4, 6, 40, 43, 2], [4, 8, 10, 24, 14]])
    conf = np.loadtxt(conf_txt, delimiter=', ')
    if num_classify == 11:
        weights = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    elif num_classify == 5:
        weights = np.array([1., 1., 1., 1., 1.]) 
    stats = get_conf_stats(conf, weights)
    print(stats)
    json_name = fdir + '/' + 'results_' + fname + '.json'
    with open(json_name, 'w') as fp:
        json.dump(stats, fp, sort_keys=True, indent=4) 

if __name__ == '__main__':
    conf_txt = sys.argv[1] 
    num_classify = int(sys.argv[2])
    fdir = os.path.dirname(conf_txt)
    fname = os.path.basename(conf_txt)[:-4]
    main(conf_txt, fdir, fname, num_classify)

