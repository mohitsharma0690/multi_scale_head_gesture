import numpy as np
import json
import pdb
import os
import sys

def main(fdir):
    for f in os.listdir(fdir):
        if f.endswith('json') and '_pp.json' not in f:
            # read json
            json_f = open(fdir+'/'+f, 'r')
            data = json.load(json_f)
            # write json pretty printed
            # remove '.json'; append '_pp' (pretty print); add .json
            new_json_fname = f[:-5] + '_pp' + '.json'
            new_json_path = fdir + '/' + new_json_fname
            new_json_f = open(new_json_path, 'w')
            json.dump(data, new_json_f, indent=4)
            json_f.close()
            new_json_f.close()
            print('Did write {}'.format(new_json_fname))


if __name__ == '__main__':
    fdir = sys.argv[1]
    main(fdir)
