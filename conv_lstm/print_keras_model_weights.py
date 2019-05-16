from __future__ import print_function

import h5py
import sys
import pdb

def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
    weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return 

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
               param = g[p_name]
               print("      {}: ".format(p_name))
               pdb.set_trace()
    finally:
        f.close()

if __name__ == '__main__':
    if len(sys.argv) >= 1:
        fname = sys.argv[1]
        print_structure(fname)
    else:
        print('No file path entered')

