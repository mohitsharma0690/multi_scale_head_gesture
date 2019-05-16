import time
import os
import h5py
import math
import sys
import pdb
import json
import itertools
import csv
import types
import argparse
import copy
import pprint

import numpy as np
import cPickle as cp
import pandas as pd
from scipy.interpolate import splev, splrep
from scipy.interpolate.rbf import Rbf
from collections import namedtuple
from copy import deepcopy

import data_utils as global_utils
from data_utils import GestureListUtils
from data_augmentation_generator import UpdateGestureCSVItem

if __name__ == '__main__' and  __package__ is None:
    print('appending to sys path')
    print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from simple_lstm import data_augmentation_generator

FLAGS = None

class EasyGestureGenerator(object):

  @staticmethod
  def get_user_gesture_filter_values():
    d = {}
    d['007_static.mp4.txt.h5'] = {}
    d['007_static.mp4.txt.h5'][0] = [0.5, 0.5, 0.3]
    d['007_static.mp4.txt.h5'][1] = [0.6, 1.0, 0.35]
    d['007_static.mp4.txt.h5'][2] = [0.7, 0.9, 0.3]
    d['007_static.mp4.txt.h5'][3] = [0.0, 0.0, 0.0]
    d['007_static.mp4.txt.h5'][4] = [1.1, 1.0, 0.4]
    d['007_static.mp4.txt.h5'][5] = [0.5, 0.8, 0.2]
    d['007_static.mp4.txt.h5'][6] = [2.0, 2.0, 0.0]
    d['007_static.mp4.txt.h5'][7] = [0.8, 0.6, 0.3]
    d['007_static.mp4.txt.h5'][8] = [1.0, 0.5, 0.3]
    d['007_static.mp4.txt.h5'][9] = [1.6, 1.6, 0]
    d['007_static.mp4.txt.h5'][10] = [1.6, 1.6, 0]

    d['008_static.mp4.txt.h5'] = {}
    d['008_static.mp4.txt.h5'][0] = [0.4, 0.4, 0.2]
    d['008_static.mp4.txt.h5'][1] = [0.6, 1.0, 0.35]
    d['008_static.mp4.txt.h5'][2] = [0.0, 0.0, 0.0]
    d['008_static.mp4.txt.h5'][3] = [0.0, 0.0, 0.0]
    d['008_static.mp4.txt.h5'][4] = [0.6, 1.0, 0.4]
    d['008_static.mp4.txt.h5'][5] = [0.45, 0.55, 0.2]
    d['008_static.mp4.txt.h5'][6] = [0.8, 0.8, 0.0]
    d['008_static.mp4.txt.h5'][7] = [0.8, 0.6, 0.3]
    d['008_static.mp4.txt.h5'][8] = [1.0, 0.6, 0.3]
    d['008_static.mp4.txt.h5'][9] = [1.5, 1.5, 0]
    d['008_static.mp4.txt.h5'][10] = [1.2, 1.2, 0]

    d['009_static.mp4.txt.h5'] = {}
    d['009_static.mp4.txt.h5'][0] = [0.6, 0.6, 0.2]
    d['009_static.mp4.txt.h5'][1] = [0.5, 0.9, 0.4]
    d['009_static.mp4.txt.h5'][2] = [1.0, 1.0, 0.0]  # Reject everything
    d['009_static.mp4.txt.h5'][3] = [0.0, 0.0, 0.0]  # None present
    d['009_static.mp4.txt.h5'][4] = [1.0, 1.5, 0.4]  # Accept everyone
    d['009_static.mp4.txt.h5'][5] = [0.6, 0.8, 0.2]
    d['009_static.mp4.txt.h5'][6] = [0.4, 0.4, 0.0]
    d['009_static.mp4.txt.h5'][7] = [2.0, 2.0, 2.0]  # None present
    d['009_static.mp4.txt.h5'][8] = [0.6, 0.5, 0.2]  # very few present
    d['009_static.mp4.txt.h5'][9] = [1.5, 1.5, 0]
    d['009_static.mp4.txt.h5'][10] = [1.2, 1.2, 0]  # None present

    d['011_static.mp4.txt.h5'] = {}
    d['011_static.mp4.txt.h5'][0] = [0.6, 0.6, 0.2]
    d['011_static.mp4.txt.h5'][1] = [1.0, 1.5, 0.4]
    d['011_static.mp4.txt.h5'][2] = [0.5, 1.2, 0.4]  # Reject everything
    d['011_static.mp4.txt.h5'][3] = [0.0, 0.0, 0.0]  # None present
    d['011_static.mp4.txt.h5'][4] = [1.2, 1.5, 0.4]  # Accept everyone
    d['011_static.mp4.txt.h5'][5] = [0.6, 1.0, 0.2]
    d['011_static.mp4.txt.h5'][6] = [0.4, 0.4, 0.0]
    d['011_static.mp4.txt.h5'][7] = [2.0, 2.0, 2.0]  # None present
    d['011_static.mp4.txt.h5'][8] = [1.5, 0.6, 0.4]  # very few present
    d['011_static.mp4.txt.h5'][9] = [1.5, 1.5, 0]    # None present
    d['011_static.mp4.txt.h5'][10] = [1.2, 1.2, 0]   # None present

    d['012_static.mp4.txt.h5'] = {}
    d['012_static.mp4.txt.h5'][0] = [0.6, 0.6, 0.2]
    d['012_static.mp4.txt.h5'][1] = [1.0, 1.5, 0.4]
    d['012_static.mp4.txt.h5'][2] = [0.5, 1.2, 0.4]  # Reject everything
    d['012_static.mp4.txt.h5'][3] = [0.0, 0.0, 0.0]  # None present
    d['012_static.mp4.txt.h5'][4] = [1.2, 1.5, 0.4]  # Accept everyone
    d['012_static.mp4.txt.h5'][5] = [0.6, 1.0, 0.2]
    d['012_static.mp4.txt.h5'][6] = [0.4, 0.4, 0.0]
    d['012_static.mp4.txt.h5'][7] = [2.0, 2.0, 2.0]  # None present
    d['012_static.mp4.txt.h5'][8] = [1.5, 0.6, 0.4]  # very few present
    d['012_static.mp4.txt.h5'][9] = [1.5, 1.5, 0]    # None present
    d['012_static.mp4.txt.h5'][10] = [1.2, 1.2, 0]   # None present

    d['013_static.mp4.txt.h5'] = {}
    d['013_static.mp4.txt.h5'][0] = [0.6, 0.6, 0.2]
    d['013_static.mp4.txt.h5'][1] = [0.6, 1.2, 0.3]
    d['013_static.mp4.txt.h5'][2] = [0.5, 1.2, 0.4]  # Reject everything
    d['013_static.mp4.txt.h5'][3] = [0.0, 0.0, 0.0]  # None present
    d['013_static.mp4.txt.h5'][4] = [0.6, 0.8, 0.3]  # Accept everyone
    d['013_static.mp4.txt.h5'][5] = [0.5, 0.8, 0.2]
    d['013_static.mp4.txt.h5'][6] = [0.4, 0.4, 0.0]
    d['013_static.mp4.txt.h5'][7] = [2.0, 2.0, 2.0]  # None present
    d['013_static.mp4.txt.h5'][8] = [0.8, 0.5, 0.3]  # very few present
    d['013_static.mp4.txt.h5'][9] = [1.5, 1.5, 0]    # High values
    d['013_static.mp4.txt.h5'][10] = [1.2, 1.2, 0]   # None present

    d['014_static.mp4.txt.h5'] = {}
    d['014_static.mp4.txt.h5'][0] = [0.5, 0.5, 0.2]
    d['014_static.mp4.txt.h5'][1] = [0.6, 0.8, 0.3]
    d['014_static.mp4.txt.h5'][2] = [0.5, 0.8, 0.3]  # seems ok
    d['014_static.mp4.txt.h5'][3] = [0.0, 0.0, 0.0]  # None present
    d['014_static.mp4.txt.h5'][4] = [0.6, 1.0, 0.3]  # Accept everyone
    d['014_static.mp4.txt.h5'][5] = [0.4, 0.6, 0.2]
    d['014_static.mp4.txt.h5'][6] = [0.4, 0.4, 0.0]
    d['014_static.mp4.txt.h5'][7] = [2.0, 2.0, 2.0]  # None present
    d['014_static.mp4.txt.h5'][8] = [0.8, 0.5, 0.3]  # None present
    d['014_static.mp4.txt.h5'][9] = [1.5, 1.5, 0]    # None present
    d['014_static.mp4.txt.h5'][10] = [1.2, 1.2, 0]   # None present

    d['015_static.mp4.txt.h5'] = {}
    d['015_static.mp4.txt.h5'][0] = [0.4, 0.4, 0.2]
    d['015_static.mp4.txt.h5'][1] = [0.4, 0.6, 0.3]
    d['015_static.mp4.txt.h5'][2] = [0.5, 0.8, 0.3]  # None present
    d['015_static.mp4.txt.h5'][3] = [0.0, 0.0, 0.0]  # None present
    d['015_static.mp4.txt.h5'][4] = [0.8, 1.2, 0.4]  # Accept one
    d['015_static.mp4.txt.h5'][5] = [0.5, 0.8, 0.25]
    d['015_static.mp4.txt.h5'][6] = [0.8, 0.8, 0.0]
    d['015_static.mp4.txt.h5'][7] = [0.7, 0.4, 0.2]  # None present
    d['015_static.mp4.txt.h5'][8] = [1.0, 0.8, 0.3]  # None present
    d['015_static.mp4.txt.h5'][9] = [1.2, 1.2, 0]    # Usually small values
    d['015_static.mp4.txt.h5'][10] = [1.2, 1.2, 0]   # None present

    d['017_static.mp4.txt.h5'] = {}
    d['017_static.mp4.txt.h5'][0] = [0.5, 0.5, 0.2]
    d['017_static.mp4.txt.h5'][1] = [0.6, 1.0, 0.3]
    d['017_static.mp4.txt.h5'][2] = [0.6, 0.8, 0.3]  # None present
    d['017_static.mp4.txt.h5'][3] = [1.0, 1.0, 0.0]  # None present
    d['017_static.mp4.txt.h5'][4] = [1.0, 1.0, 0.4]  # Accept one
    d['017_static.mp4.txt.h5'][5] = [0.6, 1.2, 0.3]
    d['017_static.mp4.txt.h5'][6] = [1.2, 1.2, 0.0]
    d['017_static.mp4.txt.h5'][7] = [1.0, 1.0, 0.2]
    d['017_static.mp4.txt.h5'][8] = [0.8, 0.6, 0.2]
    d['017_static.mp4.txt.h5'][9] = [1.0, 1.0, 0]    # Usually small values
    d['017_static.mp4.txt.h5'][10] = [1.0, 1.0, 0]   # None accepted

    d['019_static.mp4.txt.h5'] = {}
    d['019_static.mp4.txt.h5'][0] = [0.5, 0.5, 0.2]
    d['019_static.mp4.txt.h5'][1] = [0.4, 0.8, 0.3]
    d['019_static.mp4.txt.h5'][2] = [0.4, 0.8, 0.3]  # None present
    d['019_static.mp4.txt.h5'][3] = [0.3, 0.6, 0.3]
    d['019_static.mp4.txt.h5'][4] = [0.4, 0.8, 0.4]  # Accept one
    d['019_static.mp4.txt.h5'][5] = [0.4, 0.6, 0.3]
    d['019_static.mp4.txt.h5'][6] = [1.2, 1.2, 0.0]
    d['019_static.mp4.txt.h5'][7] = [1.0, 1.0, 0.2]  # None present
    d['019_static.mp4.txt.h5'][8] = [0.6, 0.4, 0.3]
    d['019_static.mp4.txt.h5'][9] = [2.0, 2.0, 0]    # Usually small values
    d['019_static.mp4.txt.h5'][10] = [1.0, 1.0, 0]   # None present

    d['022_static.mp4.txt.h5'] = {}
    d['022_static.mp4.txt.h5'][0] = [0.4, 0.4, 0.2]
    d['022_static.mp4.txt.h5'][1] = [0.6, 1.2, 0.5]  # High y movement
    d['022_static.mp4.txt.h5'][2] = [0.6, 1.2, 0.5]
    d['022_static.mp4.txt.h5'][3] = [0.8, 1.0, 0.4]
    d['022_static.mp4.txt.h5'][4] = [0.6, 1.0, 0.3]  # Accept many
    d['022_static.mp4.txt.h5'][5] = [1.6, 2.0, 0.3]
    d['022_static.mp4.txt.h5'][6] = [1.2, 1.2, 0.0]  # Accept all (3 only)
    d['022_static.mp4.txt.h5'][7] = [1.0, 1.0, 0.2]  # None present
    d['022_static.mp4.txt.h5'][8] = [0.6, 0.4, 0.3]  # None present
    d['022_static.mp4.txt.h5'][9] = [2.0, 2.0, 0]    # Usually small values
    d['022_static.mp4.txt.h5'][10] = [1.0, 1.0, 0]   # None present

    d['024_static.mp4.txt.h5'] = {}
    d['024_static.mp4.txt.h5'][0] = [0.6, 0.6, 0.2]
    d['024_static.mp4.txt.h5'][1] = [0.6, 1.0, 0.4]
    d['024_static.mp4.txt.h5'][2] = [0.6, 1.4, 0.5]
    d['024_static.mp4.txt.h5'][3] = [0.6, 1.2, 0.4]
    d['024_static.mp4.txt.h5'][4] = [0.6, 1.2, 0.4]
    d['024_static.mp4.txt.h5'][5] = [0.6, 1.1, 0.3]
    d['024_static.mp4.txt.h5'][6] = [1.4, 1.4, 0.0]  # Accept all (3 only)
    d['024_static.mp4.txt.h5'][7] = [1.0, 1.0, 0.2]  # None present
    d['024_static.mp4.txt.h5'][8] = [0.6, 0.4, 0.3]  # None present
    d['024_static.mp4.txt.h5'][9] = [2.0, 2.0, 0]    # None present
    d['024_static.mp4.txt.h5'][10] = [1.0, 1.0, 0]   # None present

    d['025_static.mp4.txt.h5'] = {}
    d['025_static.mp4.txt.h5'][0] = [0.5, 0.5, 0.2]
    d['025_static.mp4.txt.h5'][1] = [0.5, 0.9, 0.4]
    d['025_static.mp4.txt.h5'][2] = [0.6, 1.0, 0.5]
    d['025_static.mp4.txt.h5'][3] = [0.6, 1.2, 0.4]  # None present
    d['025_static.mp4.txt.h5'][4] = [0.6, 1.2, 0.4]  # None present
    d['025_static.mp4.txt.h5'][5] = [0.5, 0.7, 0.4]
    d['025_static.mp4.txt.h5'][6] = [1.4, 1.4, 0.0]  # Accept all (5 only)
    d['025_static.mp4.txt.h5'][7] = [1.0, 1.0, 0.2]  # None present
    d['025_static.mp4.txt.h5'][8] = [0.6, 0.4, 0.3]  # None present
    d['025_static.mp4.txt.h5'][9] = [2.0, 2.0, 0]    # None present
    d['025_static.mp4.txt.h5'][10] = [1.0, 1.0, 0]   # None present

    d['026_static.mp4.txt.h5'] = {}
    d['026_static.mp4.txt.h5'][0] = [0.6, 0.6, 0.2]
    d['026_static.mp4.txt.h5'][1] = [0.6, 1.0, 0.4]
    d['026_static.mp4.txt.h5'][2] = [0.6, 1.0, 0.3]
    d['026_static.mp4.txt.h5'][3] = [0.6, 1.2, 0.4]  # None present
    d['026_static.mp4.txt.h5'][4] = [0.6, 0.8, 0.3]
    d['026_static.mp4.txt.h5'][5] = [0.7, 0.8, 0.4]
    d['026_static.mp4.txt.h5'][6] = [1.4, 1.4, 0.0]  # None present
    d['026_static.mp4.txt.h5'][7] = [1.0, 1.0, 0.2]  # None present
    d['026_static.mp4.txt.h5'][8] = [0.8, 0.5, 0.3]
    d['026_static.mp4.txt.h5'][9] = [2.0, 2.0, 0]    # None present
    d['026_static.mp4.txt.h5'][10] = [1.0, 1.0, 0]   # None present

    d['031_static.mp4.txt.h5'] = {}
    d['031_static.mp4.txt.h5'][0] = [0.6, 0.6, 0.2]
    d['031_static.mp4.txt.h5'][1] = [0.7, 1.2, 0.4]
    d['031_static.mp4.txt.h5'][2] = [0.7, 1.1, 0.3]
    d['031_static.mp4.txt.h5'][3] = [0.6, 1.2, 0.4]
    d['031_static.mp4.txt.h5'][4] = [0.8, 1.2, 0.4]
    d['031_static.mp4.txt.h5'][5] = [0.8, 1.0, 0.3]
    d['031_static.mp4.txt.h5'][6] = [2.0, 2.0, 0.0]  # Large values
    d['031_static.mp4.txt.h5'][7] = [1.0, 0.6, 0.3]  # None present
    d['031_static.mp4.txt.h5'][8] = [0.8, 0.6, 0.3]
    d['031_static.mp4.txt.h5'][9] = [1.4, 1.4, 0]    # None present
    d['031_static.mp4.txt.h5'][10] = [1.0, 1.0, 0]   # None present

    d['032_static.mp4.txt.h5'] = {}
    d['032_static.mp4.txt.h5'][0] = [0.6, 0.6, 0.2]
    d['032_static.mp4.txt.h5'][1] = [0.5, 0.9, 0.4]
    d['032_static.mp4.txt.h5'][2] = [0.6, 1.0, 0.3]
    d['032_static.mp4.txt.h5'][3] = [0.6, 1.2, 0.4]  # None present
    d['032_static.mp4.txt.h5'][4] = [0.5, 0.7, 0.3]  # Accept one
    d['032_static.mp4.txt.h5'][5] = [0.5, 1.0, 0.4]
    d['032_static.mp4.txt.h5'][6] = [1.6, 1.6, 0.0]  # Small values
    d['032_static.mp4.txt.h5'][7] = [1.0, 0.6, 0.3]  # None present
    d['032_static.mp4.txt.h5'][8] = [0.8, 0.6, 0.3]  # None present
    d['032_static.mp4.txt.h5'][9] = [1.4, 1.4, 0]    # None present
    d['032_static.mp4.txt.h5'][10] = [1.0, 1.0, 0]   # None present

    d['034_static.mp4.txt.h5'] = {}
    d['034_static.mp4.txt.h5'][0] = [0.5, 0.5, 0.2]
    d['034_static.mp4.txt.h5'][1] = [1.0, 1.2, 0.3]
    d['034_static.mp4.txt.h5'][2] = [0.6, 0.8, 0.3]
    d['034_static.mp4.txt.h5'][3] = [0.8, 1.0, 0.3]  # None present
    d['034_static.mp4.txt.h5'][4] = [0.8, 1.0, 0.3]  # Accept one
    d['034_static.mp4.txt.h5'][5] = [0.6, 0.8, 0.3]
    d['034_static.mp4.txt.h5'][6] = [1.5, 1.5, 0.0]  # Small values
    d['034_static.mp4.txt.h5'][7] = [1.0, 0.6, 0.3]  # None present
    d['034_static.mp4.txt.h5'][8] = [0.8, 0.6, 0.3]
    d['034_static.mp4.txt.h5'][9] = [0.8, 0.8, 0]
    d['034_static.mp4.txt.h5'][10] = [1.0, 1.0, 0]   # None present

    d['035_static.mp4.txt.h5'] = {}
    d['035_static.mp4.txt.h5'][0] = [0.5, 0.5, 0.2]
    d['035_static.mp4.txt.h5'][1] = [0.6, 0.8, 0.3]
    d['035_static.mp4.txt.h5'][2] = [0.6, 0.8, 0.3]  # None present
    d['035_static.mp4.txt.h5'][3] = [0.8, 1.0, 0.3]  # None present
    d['035_static.mp4.txt.h5'][4] = [0.6, 0.8, 0.3]  # None present
    d['035_static.mp4.txt.h5'][5] = [0.6, 0.8, 0.3]
    d['035_static.mp4.txt.h5'][6] = [1.5, 1.5, 0.0]  # Only 2 values
    d['035_static.mp4.txt.h5'][7] = [1.0, 0.6, 0.3]  # None present
    d['035_static.mp4.txt.h5'][8] = [1.0, 0.6, 0.3]
    d['035_static.mp4.txt.h5'][9] = [2.0, 2.0, 0]
    d['035_static.mp4.txt.h5'][10] = [1.5, 1.5, 0]

    d['036_static.mp4.txt.h5'] = {}
    d['036_static.mp4.txt.h5'][0] = [0.6, 0.6, 0.2]
    d['036_static.mp4.txt.h5'][1] = [0.8, 1.2, 0.4]
    d['036_static.mp4.txt.h5'][2] = [0.8, 1.2, 0.4]
    d['036_static.mp4.txt.h5'][3] = [1.0, 1.2, 0.3]
    d['036_static.mp4.txt.h5'][4] = [0.8, 1.5, 0.4]
    d['036_static.mp4.txt.h5'][5] = [0.8, 1.2, 0.4]
    d['036_static.mp4.txt.h5'][6] = [1.5, 1.5, 0.0]  # Many tilts
    d['036_static.mp4.txt.h5'][7] = [1.0, 0.7, 0.3]
    d['036_static.mp4.txt.h5'][8] = [1.5, 0.8, 0.3]
    d['036_static.mp4.txt.h5'][9] = [2.0, 2.0, 0]
    d['036_static.mp4.txt.h5'][10] = [1.5, 1.5, 0]

    d['037_static.mp4.txt.h5'] = {}
    d['037_static.mp4.txt.h5'][0] = [0.5, 0.5, 0.2]
    d['037_static.mp4.txt.h5'][1] = [0.6, 1.0, 0.4]
    d['037_static.mp4.txt.h5'][2] = [0.6, 1.0, 0.3]
    d['037_static.mp4.txt.h5'][3] = [0.5, 0.8, 0.3]
    d['037_static.mp4.txt.h5'][4] = [0.5, 0.6, 0.3]
    d['037_static.mp4.txt.h5'][5] = [0.5, 0.8, 0.3]
    d['037_static.mp4.txt.h5'][6] = [1.0, 1.0, 0.0]  # Three tilts
    d['037_static.mp4.txt.h5'][7] = [1.0, 0.7, 0.3]  # None present
    d['037_static.mp4.txt.h5'][8] = [0.8, 0.6, 0.3]
    d['037_static.mp4.txt.h5'][9] = [1.4, 1.4, 0]
    d['037_static.mp4.txt.h5'][10] = [1.5, 1.5, 0]   # None present

    d['038_static.mp4.txt.h5'] = {}
    d['038_static.mp4.txt.h5'][0] = [0.3, 0.3, 0.2]
    d['038_static.mp4.txt.h5'][1] = [0.6, 1.0, 0.4]
    d['038_static.mp4.txt.h5'][2] = [0.6, 1.0, 0.3]
    d['038_static.mp4.txt.h5'][3] = [0.5, 0.8, 0.3]
    d['038_static.mp4.txt.h5'][4] = [0.5, 1.0, 0.3]
    d['038_static.mp4.txt.h5'][5] = [0.5, 0.8, 0.3]
    d['038_static.mp4.txt.h5'][6] = [1.0, 1.0, 0.0]  # None present
    d['038_static.mp4.txt.h5'][7] = [1.0, 0.7, 0.3]  # None present
    d['038_static.mp4.txt.h5'][8] = [1.0, 0.8, 0.3]
    d['038_static.mp4.txt.h5'][9] = [1.4, 1.4, 0]
    d['038_static.mp4.txt.h5'][10] = [1.2, 1.2, 0]

    d['040_static.mp4.txt.h5'] = {}
    d['040_static.mp4.txt.h5'][0] = [0.4, 0.4, 0.2]
    d['040_static.mp4.txt.h5'][1] = [0.5, 0.9, 0.3]
    d['040_static.mp4.txt.h5'][2] = [0.6, 0.8, 0.3]
    d['040_static.mp4.txt.h5'][3] = [0.5, 0.8, 0.3]
    d['040_static.mp4.txt.h5'][4] = [0.5, 1.0, 0.3]
    d['040_static.mp4.txt.h5'][5] = [0.5, 0.8, 0.3]
    d['040_static.mp4.txt.h5'][6] = [0.6, 0.6, 0.0]  # Many tilts
    d['040_static.mp4.txt.h5'][7] = [1.0, 0.7, 0.3]  # None present
    d['040_static.mp4.txt.h5'][8] = [0.8, 0.5, 0.3]
    d['040_static.mp4.txt.h5'][9] = [1.4, 1.4, 0]
    d['040_static.mp4.txt.h5'][10] = [0.9, 0.9, 0]

    d['041_static.mp4.txt.h5'] = {}
    d['041_static.mp4.txt.h5'][0] = [0.3, 0.3, 0.2]
    d['041_static.mp4.txt.h5'][1] = [0.6, 1.0, 0.3]
    d['041_static.mp4.txt.h5'][2] = [0.6, 1.0, 0.3]
    d['041_static.mp4.txt.h5'][3] = [0.5, 1.2, 0.3]
    d['041_static.mp4.txt.h5'][4] = [0.5, 1.2, 0.3]
    d['041_static.mp4.txt.h5'][5] = [0.6, 1.0, 0.3]
    d['041_static.mp4.txt.h5'][6] = [0.5, 0.5, 0.0]  # Few tilts
    d['041_static.mp4.txt.h5'][7] = [1.2, 0.8, 0.3]
    d['041_static.mp4.txt.h5'][8] = [1.0, 0.5, 0.3]  # Many turns
    d['041_static.mp4.txt.h5'][9] = [1.4, 1.4, 0]    # None present
    d['041_static.mp4.txt.h5'][10] = [0.8, 0.8, 0]

    d['042_static.mp4.txt.h5'] = {}
    d['042_static.mp4.txt.h5'][0] = [0.5, 0.5, 0.2]
    d['042_static.mp4.txt.h5'][1] = [0.8, 1.2, 0.3]
    d['042_static.mp4.txt.h5'][2] = [0.6, 1.2, 0.3]
    d['042_static.mp4.txt.h5'][3] = [0.5, 1.2, 0.3]  # None present
    d['042_static.mp4.txt.h5'][4] = [0.8, 1.2, 0.3]
    d['042_static.mp4.txt.h5'][5] = [0.8, 1.5, 0.4]
    d['042_static.mp4.txt.h5'][6] = [1.0, 1.0, 0.0]  # Two tilts
    d['042_static.mp4.txt.h5'][7] = [1.2, 0.8, 0.3]  # None present
    d['042_static.mp4.txt.h5'][8] = [1.2, 0.8, 0.4]  # Many turns
    d['042_static.mp4.txt.h5'][9] = [1.4, 1.4, 0]    # None present
    d['042_static.mp4.txt.h5'][10] = [1.4, 1.4, 0]

    d['043_static.mp4.txt.h5'] = {}
    d['043_static.mp4.txt.h5'][0] = [0.5, 0.5, 0.2]
    d['043_static.mp4.txt.h5'][1] = [0.4, 0.6, 0.2]  # Very few nods
    d['043_static.mp4.txt.h5'][2] = [0.6, 1.2, 0.3]  # None present
    d['043_static.mp4.txt.h5'][3] = [0.5, 1.2, 0.3]  # None present
    d['043_static.mp4.txt.h5'][4] = [0.5, 0.7, 0.2]
    d['043_static.mp4.txt.h5'][5] = [0.5, 0.6, 0.3]
    d['043_static.mp4.txt.h5'][6] = [0.6, 0.6, 0.0]  # Two tilts
    d['043_static.mp4.txt.h5'][7] = [0.8, 0.6, 0.2]  # None present
    d['043_static.mp4.txt.h5'][8] = [0.9, 0.6, 0.3]  # Many turns
    d['043_static.mp4.txt.h5'][9] = [1.5, 1.5, 0]    # None present
    d['043_static.mp4.txt.h5'][10] = [1.0, 1.0, 0]

    d['044_static.mp4.txt.h5'] = {}
    d['044_static.mp4.txt.h5'][0] = [0.5, 0.5, 0.2]
    d['044_static.mp4.txt.h5'][1] = [0.8, 1.2, 0.4]
    d['044_static.mp4.txt.h5'][2] = [0.6, 1.2, 0.4]
    d['044_static.mp4.txt.h5'][3] = [1.0, 1.2, 0.4]
    d['044_static.mp4.txt.h5'][4] = [0.5, 1.0, 0.2]
    d['044_static.mp4.txt.h5'][5] = [0.6, 0.8, 0.3]
    d['044_static.mp4.txt.h5'][6] = [0.8, 0.8, 0.0]  # Two tilts
    d['044_static.mp4.txt.h5'][7] = [1.2, 0.8, 0.3]  # None present
    d['044_static.mp4.txt.h5'][8] = [0.9, 0.6, 0.3]
    d['044_static.mp4.txt.h5'][9] = [1.5, 1.5, 0]
    d['044_static.mp4.txt.h5'][10] = [1.5, 1.5, 0]


    # CCDB files
    d['001_P1_P2_1402_C1.h5'] = {}
    d['001_P1_P2_1402_C1.h5'][0] = [0.5, 0.5, 0.2]
    d['001_P1_P2_1402_C1.h5'][1] = [1.0, 1.2, 0.2]  # Accept all (4)
    d['001_P1_P2_1402_C1.h5'][6] = [0.8, 0.8, 0.0]  # Accept all (2)
    d['001_P1_P2_1402_C1.h5'][8] = [0.9, 0.8, 0.3]

    d['002_P1_P2_1402_C2.h5'] = {}
    d['002_P1_P2_1402_C2.h5'][0] = [0.5, 0.5, 0.2]
    d['002_P1_P2_1402_C2.h5'][1] = [1.0, 1.2, 0.2]  # None present
    d['002_P1_P2_1402_C2.h5'][6] = [0.0, 0.0, 0.0]  # Accept all (3)
    d['002_P1_P2_1402_C2.h5'][8] = [0.6, 0.6, 0.3]  # Accept all (2)

    d['003_P1_P3_1502_C1.h5'] = {}
    d['003_P1_P3_1502_C1.h5'][0] = [0.5, 0.5, 0.2]
    d['003_P1_P3_1502_C1.h5'][1] = [1.0, 1.2, 0.2]  # None present
    d['003_P1_P3_1502_C1.h5'][6] = [0.8, 0.8, 0.0]  # Accept all (3)
    d['003_P1_P3_1502_C1.h5'][8] = [1.2, 1.0, 0.3]  # Accept all (2)

    d['004_P1_P3_1502_C2.h5'] = {}
    d['004_P1_P3_1502_C2.h5'][0] = [0.4, 0.4, 0.2]
    d['004_P1_P3_1502_C2.h5'][1] = [0.4, 0.6, 0.2]
    d['004_P1_P3_1502_C2.h5'][6] = [0.5, 0.5, 0.0]  # Many tilts
    d['004_P1_P3_1502_C2.h5'][8] = [1.2, 1.0, 0.3]  # None present

    d['005_P3_P4_1502_C1.h5'] = {}
    d['005_P3_P4_1502_C1.h5'][0] = [0.4, 0.4, 0.2]
    d['005_P3_P4_1502_C1.h5'][8] = [1.0, 1.0, 0.2]

    d['006_P3_P4_1502_C2.h5'] = {}
    d['006_P3_P4_1502_C2.h5'][0] = [0.5, 0.5, 0.2]
    d['006_P3_P4_1502_C2.h5'][1] = [0.5, 1.0, 0.4]  # Many nods
    d['006_P3_P4_1502_C2.h5'][6] = [0.4, 0.4, 0.0]  # 3 tilts
    d['006_P3_P4_1502_C2.h5'][8] = [1.5, 1.0, 0.3]  # Two present

    d['007_P5_P2_1003_C1.h5'] = {}
    d['007_P5_P2_1003_C1.h5'][0] = [0.5, 0.5, 0.2]
    d['007_P5_P2_1003_C1.h5'][1] = [0.4, 0.6, 0.4]  # Many nods

    d['008_P5_P2_1003_C2.h5'] = {}
    d['008_P5_P2_1003_C2.h5'][0] = [0.4, 0.4, 0.2]
    d['008_P5_P2_1003_C2.h5'][1] = [0.6, 0.9, 0.4]  # Many nods
    d['008_P5_P2_1003_C2.h5'][6] = [0.4, 0.4, 0.0]  # 3 tilts
    d['008_P5_P2_1003_C2.h5'][8] = [0.8, 0.5, 0.3]  # Two present

    d['009_P5_P3_2202_C1.h5'] = {}
    d['009_P5_P3_2202_C1.h5'][0] = [0.4, 0.4, 0.2]
    d['009_P5_P3_2202_C1.h5'][1] = [0.6, 0.8, 0.3]  # Many nods

    d['010_P5_P3_2202_C2.h5'] = {}
    d['010_P5_P3_2202_C2.h5'][0] = [0.4, 0.4, 0.2]
    d['010_P5_P3_2202_C2.h5'][1] = [0.45, 0.6, 0.2]  # Many nods
    d['010_P5_P3_2202_C2.h5'][6] = [0.5, 0.5, 0.2]  # Many nods

    d['011_P6_P2_1602_C1.h5'] = {}
    d['011_P6_P2_1602_C1.h5'][0] = [0.4, 0.4, 0.2]
    d['011_P6_P2_1602_C1.h5'][1] = [0.6, 0.8, 0.2]  # Many nods

    d['012_P6_P2_1602_C2.h5'] = {}
    d['012_P6_P2_1602_C2.h5'][0] = [0.4, 0.4, 0.2]
    d['012_P6_P2_1602_C2.h5'][1] = [0.5, 0.8, 0.3]  # Many nods

    d['013_P6_P3_1602_C1.h5'] = {}
    d['013_P6_P3_1602_C1.h5'][0] = [0.5, 0.5, 0.2]
    d['013_P6_P3_1602_C1.h5'][1] = [0.5, 0.9, 0.3]  # Many nods

    d['014_P6_P3_1602_C2.h5'] = {}
    d['014_P6_P3_1602_C2.h5'][0] = [0.5, 0.5, 0.2]
    d['014_P6_P3_1602_C2.h5'][1] = [0.6, 1.0, 0.4]  # Many nods
    d['014_P6_P3_1602_C2.h5'][6] = [0.6, 0.6, 0.0]  # 4 tilts
    d['014_P6_P3_1602_C2.h5'][8] = [0.9, 0.5, 0.3]  # 3 tilts

    d['015_P6_P4_1602_C1.h5'] = {}
    d['015_P6_P4_1602_C1.h5'][0] = [0.5, 0.5, 0.2]
    d['015_P6_P4_1602_C1.h5'][1] = [0.8, 1.2, 0.4]  # Many nods
    d['015_P6_P4_1602_C1.h5'][8] = [1.0, 1.2, 0.4]  # 4 tilts

    d['016_P6_P4_1602_C2.h5'] = {}
    d['016_P6_P4_1602_C2.h5'][0] = [0.5, 0.5, 0.2]
    d['016_P6_P4_1602_C2.h5'][1] = [1.0, 1.5, 0.4]  # Many nods
    d['016_P6_P4_1602_C2.h5'][6] = [1.0, 1.0, 0.0]  # 4 tilts
    d['016_P6_P4_1602_C2.h5'][8] = [0.9, 0.6, 0.3]  # 4 tilts

    return d

  def __init__(self, openface_h5_dir, cpm_h5_dir, zface_h5_dir, gest_list_h5,
      save_name='easy_gest_list.h5'):
    self.openface_h5_dir = openface_h5_dir
    self.cpm_h5_dir = cpm_h5_dir
    self.zface_h5_dir = zface_h5_dir
    self.gest_list_h5 = gest_list_h5
    self.save_name = save_name
    self.save_path = os.path.join(os.path.dirname(gest_list_h5), save_name)
    self.X_by_file, self.y_by_file = \
            global_utils.load_all_features_with_file_filter(
                    openface_h5_dir,
                    file_filter=lambda x: True,
                    process=False)
    assert len(self.X_by_file.keys()) > 0, 'No h5 files loaded'
    self.user_gesture_filter_values = \
        EasyGestureGenerator.get_user_gesture_filter_values()

  def get_pose_landmark_velocity(self, f_name):
    X = self.X_by_file[f_name]
    X_pose = np.array(X[:,:12])
    X_pose_vel = X_pose[1:, 6:12] - X_pose[:-1, 6:12]
    landmarks = [
      28+11, 34+11,
      28+11+68, 34+11+68,
      2+11, 34+11,
      2+11+68, 34+11+68,
      14+11, 34+11,
      14+11+68, 34+11+68
    ]
    X_landmarks = np.array(X[:, landmarks])
    X_landmarks_vel = np.zeros(X_landmarks.shape)
    X_landmarks_vel[1:,:] = X_landmarks[1:,:] - X_landmarks[:-1,:]
    return X_pose, X_pose_vel, X_landmarks, X_landmarks_vel

  def get_mean_per_col(X):
    return np.mean(np.abs(X), 0)


  def get_gesture_filter(self, gest_type, user):
    """ Get filter for gesture type for certain user.
    TODO(Mohit): Maybe include user specific features using lambda
    """
    filter_values = self.user_gesture_filter_values[user]
    assert filter_values.get(gest_type, None) is not None, "Missing filter " \
            "values for user {}, gest {}".format(user, gest_type)

    if gest_type == 0: return self.filter_none(filter_values[0])
    elif gest_type == 1: return self.filter_nod(filter_values[1])
    elif gest_type == 2: return self.filter_jerk(filter_values[2])
    elif gest_type == 3: return self.filter_up(filter_values[3])
    elif gest_type == 4: return self.filter_down(filter_values[4])
    elif gest_type == 5: return self.filter_tick(filter_values[5])
    elif gest_type == 6: return self.filter_tilt(filter_values[6])
    elif gest_type == 7: return self.filter_shake(filter_values[7])
    elif gest_type == 8: return self.filter_turn(filter_values[8])
    elif gest_type == 9: return self.filter_forward(filter_values[9])
    elif gest_type == 10: return self.filter_backward(filter_values[10])
    else: raise(ValueError('Undefined gesture type : {}'.format(gest_type)))

  def get_stats_for_gest(self, user_file, gest_type):
    ''' Get stats for gesture in user_file
    Return: A dictionary with different stats as keys and corresponding values
    for each gesture as list.
    '''
    gest_h5 = h5py.File(self.gest_list_h5, 'r')
    train_group = 'train' if user_file in gest_h5['train'].keys() else 'test'
    gestures = np.array(gest_h5[train_group][user_file][str(gest_type)])
    # We use 1 as a placeholder?
    # Also this is not statistically significant to tell anything
    if gestures.shape[0] <= 1 or len(gestures.shape) == 1:
      return None

    # Get the average x,y,z motion from pose and nosecenter
    pose, pose_vel, landmarks, landmarks_vel =self.get_pose_landmark_velocity(
        user_file)

    mean_vel = {
        'nosetip_x': [], 'nosetip_y': [], 'pose_x': [], 'pose_y':[],
        'pose_z': [], 'disp_nosetip_x': [], 'disp_nosetip_y': [],
        'nosetip_x_std': [], 'nosetip_y_std': [], 'pose_x_std': [],
        'pose_y_std': [], 'pose_z_std': []
        }
    for i in xrange(gestures.shape[0]):
      st, end = gestures[i, 0], gestures[i, 1]
      nosetip_vel = np.array(landmarks_vel[st:end+1, [1, 3]])
      mean_vel['nosetip_x'].append(np.mean(np.abs(nosetip_vel[:, 0])))
      mean_vel['nosetip_x_std'].append(np.std(np.abs(nosetip_vel[:, 0])))
      mean_vel['nosetip_y'].append(np.mean(np.abs(nosetip_vel[:, 1])))
      mean_vel['nosetip_y_std'].append(np.std(np.abs(nosetip_vel[:, 1])))
      pose_xy_vel = np.array(pose_vel[st:end+1, :3])
      mean_vel['pose_x'].append(np.mean(np.abs(pose_xy_vel[:, 0])))
      mean_vel['pose_x_std'].append(np.std(np.abs(pose_xy_vel[:, 0])))
      mean_vel['pose_y'].append(np.mean(np.abs(pose_xy_vel[:, 1])))
      mean_vel['pose_y_std'].append(np.std(np.abs(pose_xy_vel[:, 1])))
      mean_vel['pose_z'].append(np.mean(np.abs(pose_xy_vel[:, 2])))
      mean_vel['pose_z_std'].append(np.std(np.abs(pose_xy_vel[:, 2])))

      # There is one edge case where st > end in gest list because of an error
      # in the gesture_correction.csv
      if st >= end:
        mean_vel['disp_nosetip_x'].append(0)
        mean_vel['disp_nosetip_y'].append(0)
        continue

      nosetip_xy = np.array(landmarks[st:end+1, [1, 3]])
      max_nosetip_xy = np.max(nosetip_xy, 0)
      min_nosetip_xy = np.min(nosetip_xy, 0)
      mean_vel['disp_nosetip_x'].append(max_nosetip_xy[0]-min_nosetip_xy[0])
      mean_vel['disp_nosetip_y'].append(max_nosetip_xy[1]-min_nosetip_xy[1])

    for k, v in mean_vel.iteritems():
      assert len(v) == gestures.shape[0], "Uneqaul number of gesture stats " \
          "{} vs number of gestures {}".format(len(v), gestures.shape[0])

    for k, v in mean_vel.iteritems():
      for i in xrange(len(v)):
        if v[i] != v[i]: # Nan detected
          mean_vel[k][i] = 0

    return mean_vel

  def filter_easy_gestures(self, user_file, gest_type, mean_vel_stats):
    ''' Filter easy gestures and return list of accepted gestures and removed
    gestures. Each list item is a tuple of (start_frame, end_frame).
    '''
    gest_h5 = h5py.File(self.gest_list_h5, 'r')
    train_group = 'train' if user_file in gest_h5['train'].keys() else 'test'
    gestures = np.array(gest_h5[train_group][user_file][str(gest_type)])
    # We use 1 as a placeholder?
    # Also this is not statistically significant to tell anything
    if gestures.shape[0] <= 1 or len(gestures.shape) == 1:
      return None, None

    gest_list = []
    removed_gest = []
    gest_filt = self.get_gesture_filter(gest_type, user_file)
    for i in xrange(gestures.shape[0]):
      nosetip_x = mean_vel_stats['nosetip_x'][i]
      nosetip_y = mean_vel_stats['nosetip_y'][i]
      st, end = gestures[i, 0], gestures[i, 1]
      if gest_filt(nosetip_x, nosetip_y) and end - st > 4:
        gest_list.append([st, end])
      else:
        removed_gest.append([st, end])
    return gest_list, removed_gest

  @staticmethod
  def write_gest_list_h5(save_path, new_gest_list_map, removed_gest_list_map):
    print("Will write easy gesture list to {}".format(save_path))

    new_gest_list_map = GestureListUtils.convert_gesture_list_to_array(
        new_gest_list_map)

    save_h5 = h5py.File(save_path, 'w')
    global_utils.recursively_save_dict_contents_to_group(
        save_h5, '/', new_gest_list_map)
    save_h5.flush()
    save_h5.close()
    print("Did write easy gesture list to {}".format(save_path))

    removed_gest_map = {}
    for group, val in removed_gest_list_map.iteritems():
      for user, user_dict in val.iteritems():
        for label in user_dict.keys():
          removed_gest = removed_gest_list_map[group][user][label]
          if removed_gest is None: continue
          for s, e in removed_gest:
            if s == 0: continue
            item = UpdateGestureCSVItem(user, s, e, label, '{}:-1'.format(e))
            k = user + '_' + str(s)
            removed_gest_map[k] = item

    # Save CSV to save_path as well
    csv_path = save_path[:-2] + 'csv'
    UpdateGestureCSVItem.write_dict_to_csv(removed_gest_map, csv_path)

    for group in ['train', 'test']:
      for user in new_gest_list_map[group].keys():
        # if '038' not in user: continue
        gest_count = [new_gest_list_map[group][user][str(label)].shape[0] \
            for label in xrange(11)]
        print('{}: {}'.format(user, gest_count))


  @staticmethod
  def get_sigma_range(total_stats, key, sigma_factor=1):
    ''' Returns a range of values that lie within sigma_factor*sigma times.'''
    assert total_stats.get(key) is not None, \
        "Invalid key: {} not found in stats".format(key)
    mean_val, sigma_value = total_stats[key]['mean'], total_stats[key]['std']
    return (mean_val-sigma_factor*sigma_value, mean_val+sigma_factor*sigma_value)

  # =========================================================== #
  # ====== Filter specific gestures based on sigma values ===== #
  def filter_sigma_nods(
      self, current_gest_stats, total_stats, sigma_factor=1):
    return self.filter_sigma_nods_displacement_nosetip(
        current_gest_stats, total_stats, sigma_factor=sigma_factor)

  def filter_sigma_nods_nosetip(
      self, current_gest_stats, total_stats, sigma_factor=1):
    nosetip_x_range = EasyGestureGenerator.get_sigma_range(
        total_stats, 'nosetip_x')
    nosetip_y_range = EasyGestureGenerator.get_sigma_range(
        total_stats, 'nosetip_y')
    if (current_gest_stats['nosetip_x'] >= nosetip_x_range[0] and
        current_gest_stats['nosetip_x'] <= nosetip_x_range[1] and
        current_gest_stats['nosetip_y'] >= nosetip_y_range[0] and
        current_gest_stats['nosetip_y'] <= nosetip_y_range[1]):
      return True

    return False

  def filter_sigma_nods_pose(
      self, current_gest_stats, total_stats, sigma_factor=1):
    ''' Filter Nods based on pose values instead of nosetip velocity.
    Returns: True if valid gesture i.e. lies within sigma_factor*sigma of mean
    else False.
    '''
    pose_x_range = EasyGestureGenerator.get_sigma_range(total_stats, 'pose_x')
    pose_y_range = EasyGestureGenerator.get_sigma_range(total_stats, 'pose_y')
    if (current_gest_stats['pose_x'] >= pose_x_range[0] and
        current_gest_stats['pose_x'] <= pose_x_range[1] and
        current_gest_stats['pose_y'] >= pose_y_range[0] and
        current_gest_stats['pose_y'] <= pose_y_range[1]):
      return True

    return False

  def filter_sigma_nods_displacement_nosetip(
      self, current_gest_stats, total_stats, sigma_factor=1):
    ''' Filter gestures based on displacement in nosetip values.
    Returns: True if valid gesture i.e. lies within sigma_factor*sigma of mean
    else False.
    '''
    disp_x_range = EasyGestureGenerator.get_sigma_range(
        total_stats, 'disp_nosetip_x')
    disp_y_range = EasyGestureGenerator.get_sigma_range(
        total_stats, 'disp_nosetip_y')
    if (current_gest_stats['disp_nosetip_x'] >= disp_x_range[0] and
        current_gest_stats['disp_nosetip_x'] <= disp_x_range[1] and
        current_gest_stats['disp_nosetip_y'] >= disp_y_range[0] and
        current_gest_stats['disp_nosetip_y'] <= disp_y_range[1]):
      return True

    return False

  def filter_sigma_nods_global(
      self, current_gest_stats, total_stats, sigma_factor=1):
    pass


  def filter_sigma_none(self, current_gest_stats, total_stats, sigma_factor=1):
    return self.filter_sigma_nods(current_gest_stats, total_stats, sigma_factor)

  def filter_sigma_tilts(self, current_gest_stats, total_stats, sigma_factor=1):
    return self.filter_sigma_nods(current_gest_stats, total_stats, sigma_factor)

  def filter_sigma_shakes(self, current_gest_stats, total_stats, sigma_factor=1):
    return self.filter_sigma_nods(current_gest_stats, total_stats, sigma_factor)

  def filter_sigma_forward(self, current_gest_stats, total_stats, sigma_factor=1):
    return self.filter_sigma_nods(current_gest_stats, total_stats, sigma_factor)

  # =========================================================== #

  def filter_sigma_gestures(self, user_file, gest_type, total_stats,
      user_stats, sigma_factor=1):
    ''' Filter gestures for a user based on standard metrics of mean and std
    calculated for that gesture on the entire training data.

    gest_type: Label of the geture
    total_stats: Stats (mean, std for different features) on entire train data
    user_stats:  Stats (mean, std, ...) for current user per gesture. This is
    indexed as user_stats[stats_type][gest_count].

    Return: List of [start, end] frame of gestures that are valid and
    list of [start, end] frames of gestures that are not valid.
    '''
    gest_h5 = h5py.File(self.gest_list_h5, 'r')
    train_group = 'train' if user_file in gest_h5['train'].keys() else 'test'
    gestures = np.array(gest_h5[train_group][user_file][str(gest_type)])
    # We use 1 as a placeholder?
    # Also this is not statistically significant to tell anything
    if gestures.shape[0] <= 1 or len(gestures.shape) == 1:
      return None, None

    gest_list = []
    removed_gest = []
    for i in xrange(gestures.shape[0]):
      curr_stats = {}
      for k in user_stats.keys():
        curr_stats[k] = user_stats[k][i]

      st, end = gestures[i, 0], gestures[i, 1]
      should_filter = False
      if gest_type == 0:
        should_filter = self.filter_sigma_none(
            curr_stats, total_stats, sigma_factor=sigma_factor)
      elif gest_type >= 1 and gest_type <= 5:
        should_filter = self.filter_sigma_nods(
            curr_stats, total_stats, sigma_factor=sigma_factor)
      elif gest_type == 6:
        should_filter = self.filter_sigma_tilts(
            curr_stats, total_stats, sigma_factor=sigma_factor)
      elif gest_type == 7 or gest_type == 8:
        should_filter = self.filter_sigma_shakes(
            curr_stats, total_stats, sigma_factor=sigma_factor)
      elif gest_type == 9 or gest_type == 10:
        should_filter = self.filter_sigma_forward(
            curr_stats, total_stats, sigma_factor=sigma_factor)
      else:
        raise(ValueError, "Invalid gest_type {}".format(gest_type))

      if should_filter:
        gest_list.append([st, end])
      else:
        removed_gest.append([st, end])
    return gest_list, removed_gest

  def filter_global_gest(self, curr_stats, total_stats, threshold=1.):
    return curr_stats['nosetip_x'] >= threshold or \
        curr_stats['nosetip_y'] >= threshold

  def filter_global_gestures(self, user_file, gest_type, total_stats,
      user_stats, threshold=1.0):
    ''' Filter gestures based on a global threshold that applies to all gestures
    together. The difference between thi and `filter_sigma_gestures` is that the
    latter uses per gesture stats to filter gestures while in here we use a
    global value for all gestures.
   '''
    gest_h5 = h5py.File(self.gest_list_h5, 'r')
    train_group = 'train' if user_file in gest_h5['train'].keys() else 'test'
    gestures = np.array(gest_h5[train_group][user_file][str(gest_type)])
    # We use 1 as a placeholder?
    # Also this is not statistically significant to tell anything
    if gestures.shape[0] <= 1 or len(gestures.shape) == 1:
      return None, None

    gest_list = []
    removed_gest = []
    for k, v in user_stats.iteritems():
      if len(v) != gestures.shape[0]:
        print(len(v))
        print(gestures.shape[0])
        print(user_file)

      assert len(v) == gestures.shape[0], "Invalid number of gest stats for user"
    for i in xrange(gestures.shape[0]):
      # Get stats for current gesture
      curr_stats = {}
      for k in user_stats.keys():
        curr_stats[k] = user_stats[k][i]

      st, end = gestures[i, 0], gestures[i, 1]
      should_filter = self.filter_global_gest(
          curr_stats, total_stats, threshold=threshold)
      if should_filter:
        gest_list.append([st, end])
      else:
        removed_gest.append([st, end])

    return gest_list, removed_gest


  def create_global_gest_list(self, threshold=1.0, debug=False):
    ''' Create gesture list based on some global threshold for motion. If the
    motion of any gesture is below this certain threshold then we don't include
    that gesture into our analysis.
    '''
    gest_h5 = h5py.File(self.gest_list_h5, 'r')
    train_users, test_users = gest_h5['train'].keys(), gest_h5['test'].keys()

    # Init values
    new_gest_list_map = {'train': {}, 'test': {}}
    removed_gest_list_map = {'train': {}, 'test': {}}
    for group, users in zip(['train', 'test'], [train_users, test_users]):
      for user in users:
        new_gest_list_map[group][user] = {}
        removed_gest_list_map[group][user] = {}

    all_stats = {}
    stats_per_user_per_gest = {}
    for gest in xrange(11):
      # Get stats for every user
      stats_per_user_per_gest[gest] = {}
      # Add reference for quick access
      stats_per_user = stats_per_user_per_gest[gest]

      for user in train_users+test_users:
        gest_stats = self.get_stats_for_gest(user, gest)
        stats_per_user[user] = gest_stats

      all_stats[gest] = {}
      # Calculate the global stats
      for user in train_users:
        user_stats = stats_per_user[user]
        if user_stats is None: continue

        for k, v in user_stats.iteritems():
          if all_stats[gest].get(k) is None:
            all_stats[gest][k] = []
          all_stats[gest][k] += copy.deepcopy(v)


    # all_stats contains all the `gesture labels` as keys with the value as a
    # dictionary. This dictionary has stat values (mean, etc.) as its keys
    # and all the gesture values for that stat as its value

    # For now let's just get the mean and std values as usual
    all_stats_combined = {}
    for gest in sorted(all_stats.keys()):
      for stat in all_stats[gest].keys():
        val = all_stats[gest][stat]
        if all_stats_combined.get(stat) is None:
          all_stats_combined[stat] = {'mean': [], 'std': []}
        all_stats_combined[stat]['mean'].append(np.mean(val))
        all_stats_combined[stat]['std'].append(np.std(val))

    if debug:
      pprint.pprint(sorted(all_stats.keys()))
      pprint.pprint(all_stats_combined)

    for gest in xrange(11):
      for user in train_users+test_users:
        group = 'train' if user in train_users else 'test'
        gest_list, removed_gest = self.filter_global_gestures(
          user,
          gest,
          all_stats_combined,
          stats_per_user_per_gest[gest][user],
          threshold=threshold)
        new_gest_list_map[group][user][str(gest)] = gest_list
        removed_gest_list_map[group][user][str(gest)] = removed_gest

    return new_gest_list_map, removed_gest_list_map


  def create_sigma_gest_list(self, sigma_factor=1):
    ''' Create a sigma filtered gesture list for all users.
    Return: Tuple of Dictionary with gest_list representation of new gestures
    and another dictionary with removed gest_list representation.
    '''
    gest_h5 = h5py.File(self.gest_list_h5, 'r')
    train_users, test_users = gest_h5['train'].keys(), gest_h5['test'].keys()

    # Init values
    new_gest_list_map = {'train': {}, 'test': {}}
    removed_gest_list_map = {'train': {}, 'test': {}}
    for group, users in zip(['train', 'test'], [train_users, test_users]):
      for user in users:
        new_gest_list_map[group][user] = {}
        removed_gest_list_map[group][user] = {}

    for gest in xrange(11):
      # Get stats for every user
      stats_per_user = {}
      for user in train_users+test_users:
        gest_stats = self.get_stats_for_gest(user, gest)
        stats_per_user[user] = gest_stats

      all_stats = {}
      # Calculate the global stats
      for user in train_users:
        user_stats = stats_per_user[user]
        if user_stats is None: continue

        for k, v in user_stats.iteritems():
          if all_stats.get(k) is None:
            all_stats[k] = []
          all_stats[k] += copy.deepcopy(v)

      # all_stats has keys (nosetip_x, nosetip_y, pose_x, ...) and values is
      # a list of all the values for these gestures for all train users
      # combined.
      total_stats = {}
      for k, v in all_stats.iteritems():
        total_stats[k] = {}
        total_stats[k]['mean'], total_stats[k]['std'] = np.mean(v), np.std(v)
      # print("Gesture: {}".format(gest))
      # print(total_stats)


      for user in train_users+test_users:
        group = 'train' if user in train_users else 'test'
        gest_list, removed_gest = self.filter_sigma_gestures(
            user,
            gest,
            total_stats,
            stats_per_user[user],
            sigma_factor=sigma_factor)
        new_gest_list_map[group][user][str(gest)] = gest_list
        removed_gest_list_map[group][user][str(gest)] = removed_gest

    return new_gest_list_map, removed_gest_list_map,

  def create_gest_list(self):
    gest_h5 = h5py.File(self.gest_list_h5, 'r')
    train_users, test_users = gest_h5['train'].keys(), gest_h5['test'].keys()
    d = {'train': {}, 'test': {}}
    for user in train_users: d['train'][str(user)] = {}
    for user in test_users: d['test'][str(user)] = {}

    removed_gest_map = {}

    for gest in xrange(11):
      for user in train_users + test_users:
        # if '038' not in user: continue
        gest_stats = self.get_stats_for_gest(user, gest)
        gest_list, removed_gest = self.filter_easy_gestures(
          user, gest, gest_stats)
        group = 'train' if user in train_users else 'test'
        if gest_list is not None and len(gest_list) > 0:
          gest_list = np.array(gest_list)
        else:
          gest_list = np.array([0])
        d[group][user][str(gest)] = gest_list

        if removed_gest is not None:
          for s, e in removed_gest:
            item = UpdateGestureCSVItem(user, s, e, gest, '{}:-1'.format(e))
            k = user + '_' + str(s)
            removed_gest_map[k] = item
    EasyGestureGenerator.write_gest_list_h5(self.save_path, d, removed_gest_map)

  def filter_none(self, filter_values):
    def f(mean_vel_x, mean_vel_y):
      return mean_vel_x < filter_values[0] and mean_vel_y < filter_values[1] \
          and np.abs(mean_vel_x - mean_vel_y) < filter_values[2]
    return f

  def filter_nod(self, filter_values):
    def f(mean_vel_x, mean_vel_y):
      return mean_vel_x < filter_values[0] and \
          mean_vel_y > filter_values[1] and \
          (mean_vel_y - mean_vel_x) > filter_values[2]
    return f

  def filter_jerk(self, filter_values):
    def f(mean_vel_x, mean_vel_y):
      return mean_vel_x < filter_values[0] and mean_vel_y > filter_values[1] \
          and (mean_vel_y - mean_vel_x) > filter_values[2]
    return f

  def filter_up(self, filter_values):
    def f(mean_vel_x, mean_vel_y):
      return mean_vel_x < filter_values[0] and mean_vel_y > filter_values[1] \
          and (mean_vel_y - mean_vel_x) > filter_values[2]
    return f

  def filter_down(self, filter_values):
    def f(mean_vel_x, mean_vel_y):
      return mean_vel_x < filter_values[0] and mean_vel_y > filter_values[1] \
          and (mean_vel_y - mean_vel_x) > filter_values[2]
    return f

  def filter_tick(self, filter_values):
    def f(mean_vel_x, mean_vel_y):
      return mean_vel_x < filter_values[0] and mean_vel_y > filter_values[1] \
          and (mean_vel_y - mean_vel_x) > filter_values[2]
    return f

  def filter_tilt(self, filter_values):
    def f(mean_vel_x, mean_vel_y):
      return mean_vel_x >= filter_values[0] and mean_vel_y >= filter_values[1]
    return f

  def filter_shake(self, filter_values):
    def f(mean_vel_x, mean_vel_y):
      return mean_vel_x > filter_values[0] and mean_vel_y < filter_values[1] \
          and mean_vel_x - mean_vel_y > filter_values[2]
    return f

  def filter_turn(self, filter_values):
    def f(mean_vel_x, mean_vel_y):
      return mean_vel_x > filter_values[0] and mean_vel_y < filter_values[1] \
          and mean_vel_x - mean_vel_y > filter_values[2]
    return f

  def filter_forward(self, filter_values):
    """Filter gestures with significant movement else forward gestures are
    very difficult to recognize.
    """
    def f(mean_vel_x, mean_vel_y):
      return mean_vel_x >= filter_values[0] or filter_values[1] >= filter_values[1]
    return f

  def filter_backward(self, filter_values):
    """Filter gestures with significant movement else back gestures are
    very difficult to recognize.
    """
    def f(mean_vel_x, mean_vel_y):
      return mean_vel_x >= filter_values[0] or filter_values[1] >= filter_values[1]
    return f


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
          description='Create easy/medium gestures from FIPCO.')
  parser.add_argument('--h5_dir', nargs='?', type=str, const=1,
      default='../openface_data/face_gestures/dataseto_text',
      help='h5 files directory')
  parser.add_argument('--save', nargs='?', type=int, const=1,
      default=0, help='1 if want to save new gest list as h5 file else 0')
  parser.add_argument('--gest_list_h5', nargs='?', type=str, const=1,
      default='', help='gesture list h5 file to be used as original gestures.')
  parser.add_argument('--cpm_h5_dir', nargs='?', type=str, const=1,
      default='', help='CPM h5 files directory')
  parser.add_argument('--zface_h5_dir', nargs='?', type=str, const=1,
      default='', help='Directory with zface h5 files.')

  args = parser.parse_args()
  print(args)

  FLAGS = args
