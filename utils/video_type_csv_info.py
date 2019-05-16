import numpy as np

import copy
import csv
import os
import pdb

class VideoTypeCSVInfo(object):
  '''Info object to read and query video type csv files.'''
    
  @staticmethod
  def read_video_type_csv(csv_path):
    assert os.path.exists(csv_path), 'File does not exist {}'.format(csv_path)
    data = {}
    with open(csv_path, 'r') as csv_f:
      csv_reader = csv.DictReader(csv_f)
      for row in csv_reader:
        data[row['filename'].strip()] = row['type'].strip()
    return data

  def __init__(self, csv_path):
    self.path = csv_path
    self.data = VideoTypeCSVInfo.read_video_type_csv(csv_path)

  def train_files(self):
    return [k for k,v in self.data.iteritems() if v == 'train']

  def test_files(self):
    return [k for k,v in self.data.iteritems() if v == 'test']

  def incorrect_files(self):
    return [k for k,v in self.data.iteritems() if v == 'incorrect']

  def type_for_file(self, f):
    '''Return the type for file f. Return None if f not present'''
    return self.data.get(f)

