import argparse
import lasagne
import theano
import time
import os
import gc
import h5py
import math
import sys
import pdb
import json
import keras
import itertools

# Add Keras dependencies.
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.engine.topology import Merge
from keras.layers.convolutional import Convolution1D, Convolution2D
from keras.layers.core import Dense, Activation, Dropout, Permute, Reshape
from keras.layers.noise import GaussianNoise
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential, model_from_json
from keras.regularizers import l2, activity_l2

import pandas as pd
from pandas import Series
import numpy as np
import cPickle as cp
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

# Cannot plot requires pyGTK

from sliding_window import sliding_window
from data_loader import DataLoader
from batch_generator import BatchGenerator
from sliding_window_generator import SlidingWindowGenerator

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 148

# Number filters convolutional layers
NUM_FILTERS = 64

# Size filters convolutional layers
FILTER_SIZE = 3

# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 128

# Name of the file used to cache the data created.
CACHE_DATA_FILE_NAME = 'dclstm_ss3_data_148.h5'

# Global config for settings.
config = {}

def get_confusion_matrix_one_hot(model_results, truth):
  '''
  model_results and truth should be for one-hot format, i.e, have >= 2 columns,
  where truth is 0/1, and max along each row of model_results is model result
  '''
  assert model_results.shape == truth.shape
  num_outputs = truth.shape[1]
  confusion_matrix = np.zeros((num_outputs, num_outputs), dtype=np.int32)
  predictions = np.argmax(model_results,axis=1)
  assert len(predictions)==truth.shape[0]

  for actual_class in range(num_outputs):
    idx_examples_this_class = truth[:,actual_class]==1
    prediction_for_this_class = predictions[idx_examples_this_class]
    for predicted_class in range(num_outputs):
      count = np.sum(prediction_for_this_class==predicted_class)
      confusion_matrix[actual_class, predicted_class] = count
  assert np.sum(confusion_matrix)==len(truth)
  assert np.sum(confusion_matrix)==np.sum(truth)
  return confusion_matrix

def get_stats_conf_matrix(conf):
  precision = np.zeros(conf.shape[0])
  recall, f1_score = np.zeros(conf.shape[0]), np.zeros(conf.shape[0])

  for i in xrange(conf.shape[0]):
    # Adding epsilon to avoid diving by 0 issue.
    # correctly predicted / actual_in_class
    recall[i] = float(conf[i, i])/(np.sum(conf[i,:])+0.001)
    # correctlly predicted / total_predicted_in_class
    precision[i] = float(conf[i, i])/(np.sum(conf[:,i])+0.001)
    f1_score[i] = (2.0*recall[i]*precision[i])/(recall[i] + precision[i])

  wt_precision, wt_recall, wt_f1 = 0, 0, 0
  weights, total_weight = np.sum(conf, 0), np.sum(conf)
  for i in xrange(conf.shape[0]):
    wt_precision += float(weights[i]*precision[i])/total_weight
    wt_recall += float(weights[i]*recall[i])/total_weight
    wt_f1 += float(weights[i]*f1_score[i])/total_weight

  return (precision, recall, f1_score), (wt_precision, wt_recall, wt_f1)

def get_file_stats():
  fdir = '../openface_data/face_gestures/dataseto_text'
  d = {}

  for npfile in os.listdir(fdir):
    if npfile.endswith("_static.mp4.txt.h5"):
      hf = h5py.File(fdir + "/" + npfile,'r')
      a = np.array(hf.get('annotations'))
      f = np.array(hf.get('features'))
      X_file, y_file = process_data(f, a)
      print 'File: {}'.format(npfile)
      total_cnt = 0
      d[npfile] = [0]*NUM_CLASSES
      for i in range(NUM_CLASSES):
        d[npfile][i] = np.sum(y_file == i)

  for (file, gest_counts) in d.iteritems():
    sys.stdout.write('File: {}, Gestures: '.format(file))
    for i in xrange(NUM_CLASSES):
      sys.stdout.write('{} '.format(d[file][i]))
    sys.stdout.write('\n')
  sys.stdout.flush()
  return d

def opp_sliding_window(data_x, data_y, ws, ss):
  data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
  data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
  return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)

def reshape_input(X_train, y_train, X_test, y_test):
  """
  Data is reshaped since the input of the network is a 4-D tensor.
  """
  if not config[USE_SIMPLE_LSTM]:
    X_test = X_test.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
    X_train = X_train.reshape((-1, 1, SLIDING_WINDOW_LENGTH,
      NB_SENSOR_CHANNELS))
  # We also need to reformat y to choose the target symbol.
  # DeepConvLSTM chooses the last time step symbol let's do that
  # y_train = y_train.reshape((-1, SLIDING_WINDOW_LENGTH))
  #y_test = y_test.reshape((-1, SLIDING_WINDOW_LENGTH))
  #y_train = y_train[:, -1]
  #y_test = y_test[:, -1]
  return X_train, y_train, X_test, y_test

def get_lstm_model(X_train, y_train, X_test, y_test):
    """
    Creates a simple LSTM model for the training data
    """
    net = {}
    net['input'] = lasagne.layers.InputLayer((BATCH_SIZE,
      SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
    batchsize, seqlen, _ = net['input'].input_var.shape
    net['lstm1'] = lasagne.layers.LSTMLayer(net['input'], NUM_UNITS_LSTM)
    net['dropout'] = lasagne.layers.DropoutLayer(net['lstm1'], 0.3)
    net['shp1'] = lasagne.layers.ReshapeLayer(net['dropout'],
            (-1, NUM_UNITS_LSTM))
    # TODO(Mohit): See what happens when using a softmax layer after dense
    # net['prob'] = lasagne.layers.DenseLayer(net['shp1'], NUM_CLASSES+1,
    # nonlinearity=lasagne.nonlinearities.softmax)

    # lasagne.nonlinearities.linear gives nan
    net['prob'] = lasagne.layers.DenseLayer(net['shp1'], NUM_CLASSES+1,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Tensors reshaped back to the original shape
    net['shp2'] = lasagne.layers.ReshapeLayer(net['prob'],
        (BATCH_SIZE, SLIDING_WINDOW_LENGTH, NUM_CLASSES+1))
    print net['shp2'].output_shape

    # Last sample in the sequence is considered
    net['output'] = lasagne.layers.SliceLayer(net['shp2'], -1, 1)
    print net['output'].output_shape

    return net

def get_model(X_train, y_train, X_test, y_test):
  """
  Creates the network for the model and returns it.
  """
  net = {}
  net['input'] = lasagne.layers.InputLayer((BATCH_SIZE, 1,
      SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
  net['conv1/5x1'] = lasagne.layers.Conv2DLayer(net['input'],
      NUM_FILTERS, (FILTER_SIZE, 1))
  net['conv2/5x1'] = lasagne.layers.Conv2DLayer(net['conv1/5x1'],
      NUM_FILTERS, (FILTER_SIZE, 1))
  net['conv3/5x1'] = lasagne.layers.Conv2DLayer(net['conv2/5x1'],
      NUM_FILTERS, (FILTER_SIZE, 1))
  net['conv4/5x1'] = lasagne.layers.Conv2DLayer(net['conv3/5x1'],
      NUM_FILTERS, (FILTER_SIZE, 1))
  net['shuff'] = lasagne.layers.DimshuffleLayer(net['conv4/5x1'], (0, 2, 1, 3))
  net['lstm1'] = lasagne.layers.LSTMLayer(net['shuff'], NUM_UNITS_LSTM)
  net['lstm2'] = lasagne.layers.LSTMLayer(net['lstm1'], NUM_UNITS_LSTM)
  # In order to connect a recurrent layer to a dense layer, it is necessary to
  # flatten the first two dimensions to cause each time step of each sequence
  # to be processed independently (see Lasagne docs for further information)
  net['shp1'] = lasagne.layers.ReshapeLayer(net['lstm2'], (-1, NUM_UNITS_LSTM))
  net['prob'] = lasagne.layers.DenseLayer(net['shp1'], NUM_CLASSES+1,
      nonlinearity=lasagne.nonlinearities.softmax)

  print("shape before reshaping: " + \
      str(lasagne.layers.get_output_shape(net['prob'])))

  # Tensors reshaped back to the original shape
  net['shp2'] = lasagne.layers.ReshapeLayer(net['prob'],
      (BATCH_SIZE, FINAL_SEQUENCE_LENGTH, NUM_CLASSES+1))

  # Last sample in the sequence is considered
  net['output'] = lasagne.layers.SliceLayer(net['shp2'], -1, 1)
  return net


class PrintCheckPointStats(Callback):

  def __init__(self, X_val, y_val):
    super(PrintCheckPointStats, self).__init__()
    self.X = X_val
    self.y = y_val

  def on_epoch_end(self, epoch, logs={}):
    predicted = self.model.predict(self.X, batch_size=700)
    conf = get_confusion_matrix_one_hot(predicted, self.y)

    # Print to the console just for the sake of it
    opt = np.get_printoptions()
    np.set_printoptions(threshold='nan')
    print('===== Confusion matrix =====')
    print(conf)
    np.set_printoptions(**opt)

def get_16_to_1_model():
  num_features = 52
  model = Sequential()
  model.add(TimeDistributed(Convolution1D(128, 3, border_mode='valid'), 
    input_shape=(config['win_len'], 16, num_features)))
  model.add(TimeDistributed(Activation('relu')))
  model.add(TimeDistributed(Convolution1D(128, 3, border_mode='valid')))
  model.add(TimeDistributed(Activation('relu')))
  model.add(TimeDistributed(Convolution1D(128, 3, border_mode='valid')))
  model.add(TimeDistributed(Activation('relu')))
  model.add(TimeDistributed(MaxPooling1D(2)))
  model.add(TimeDistributed(Convolution1D(128, 3, border_mode='valid')))
  model.add(TimeDistributed(Activation('relu')))
  model.add(TimeDistributed(Convolution1D(64, 3, border_mode='valid')))
  model.add(TimeDistributed(Activation('relu')))
  return model

def get_32_to_1_model():
  num_features = 52
  model = Sequential()
  model.add(TimeDistributed(Convolution1D(128, 3, border_mode='valid'), 
    input_shape=(config['win_len'], 32, num_features)))
  model.add(TimeDistributed(Activation('relu')))
  model.add(TimeDistributed(Convolution1D(128, 3, border_mode='valid')))
  model.add(TimeDistributed(MaxPooling1D(2)))
  model.add(TimeDistributed(Convolution1D(256, 3, border_mode='valid')))
  model.add(TimeDistributed(Activation('relu')))
  model.add(TimeDistributed(Convolution1D(256, 3, border_mode='valid')))
  model.add(TimeDistributed(Activation('relu')))
  model.add(TimeDistributed(MaxPooling1D(2)))
  model.add(TimeDistributed(Convolution1D(128, 3, border_mode='valid')))
  model.add(TimeDistributed(Activation('relu')))
  model.add(TimeDistributed(Convolution1D(64, 3, border_mode='valid')))
  model.add(TimeDistributed(Activation('relu')))
  return model

def get_64_to_1_model():
  num_features = 52
  model = Sequential()
  model.add(TimeDistributed(Convolution1D(128, 3, border_mode='valid'), 
    input_shape=(config['win_len'], 64, num_features))) 
  model.add(TimeDistributed(Activation('relu')))
  model.add(TimeDistributed(Convolution1D(128, 3, border_mode='valid')))
  model.add(TimeDistributed(Activation('relu')))
  model.add(TimeDistributed(MaxPooling1D(2)))
  model.add(TimeDistributed(Convolution1D(256, 3, border_mode='valid')))
  model.add(TimeDistributed(Activation('relu')))
  model.add(TimeDistributed(Convolution1D(256, 3, border_mode='valid')))
  model.add(TimeDistributed(Activation('relu')))
  model.add(TimeDistributed(MaxPooling1D(2)))
  model.add(TimeDistributed(Convolution1D(128, 3, border_mode='valid')))
  model.add(TimeDistributed(Activation('relu')))
  model.add(TimeDistributed(Convolution1D(64, 3, border_mode='valid')))
  model.add(TimeDistributed(Activation('relu')))
  model.add(TimeDistributed(MaxPooling1D(2)))
  model.add(TimeDistributed(Convolution1D(64, 4, border_mode='valid')))
  model.add(TimeDistributed(Activation('relu')))
  return model

def get_multi_conv_model():
  model_16_to_1 = get_16_to_1_model()
  model_32_to_1 = get_32_to_1_model()
  model_64_to_1 = get_64_to_1_model()
  print(model_16_to_1.output_shape)
  print(model_32_to_1.output_shape)
  print(model_64_to_1.output_shape)
  merged = Merge([model_16_to_1, model_32_to_1, model_64_to_1], mode='ave')
  final_model = Sequential()
  final_model.add(merged)
  final_model.add(Reshape((config['win_len'], 64)))
  final_model.add(LSTM(64, return_sequences=True))
  final_model.add(LSTM(64))
  final_model.add(Dense(config['num_classify'], W_regularizer=l2(0.01)))
  return final_model

def get_keras_conv_lstm_model():
  """
  Returns a keras model that can be compiled and used to fit some training
  data while cross-validating against cross-validation data.
  """
  model = Sequential()
  in_neurons = NB_SENSOR_CHANNELS
  middle_neurons = 128
  if config[USE_ONE_VS_ALL]:
    out_neurons = 2
  else:
    out_neurons = NUM_CLASSES
  hidden_neurons = 256

  # The first argument is the sliding window length.
  model.add(Convolution1D(512, 3, border_mode='same',
    input_shape=(SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS)))
  model.add(Activation('relu'))
  model.add(MaxPooling1D(2))
  model.add(Convolution1D(256, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling1D(2))
  model.add(Convolution1D(128, 3, border_mode='same'))
  model.add(Activation('relu'))
  model.add(MaxPooling1D(2))
  model.add(Flatten())
  # model.add(LSTM(hidden_neurons, dropout_W=0.3, return_sequences=False))
  # model.add(Dropout(.5))
  # model.add(LSTM(hidden_neurons, input_shape=(SLIDING_WINDOW_LENGTH,
  #  in_neurons)))
  model.add(Dense(1024, W_regularizer=l2(0.01)))
  model.add(Dropout(0.3))
  model.add(Dense(out_neurons, W_regularizer=l2(0.01)))
  model.add(Activation("softmax"))
  return model


def get_keras_model():
  """
  Returns a keras model that can be compiled and used to fit some training
  data while cross-validating against cross-validation data.
  """
  model = Sequential()
  in_neurons = 52 
  middle_neurons = 128
  if config[USE_ONE_VS_ALL]:
    out_neurons = 5
  else:
    out_neurons = NUM_CLASSES
    # out_neurons = 5
  hidden_neurons = 128

  model.add(Masking(mask_value=0., input_shape=(SLIDING_WINDOW_LENGTH, in_neurons)))
  model.add(Bidirectional(LSTM(hidden_neurons, dropout_W=0.3, dropout_U=0.3, return_sequences=True)))
  model.add(Bidirectional(LSTM(hidden_neurons, dropout_W=0.5, dropout_U=0.3, return_sequences=True)))
  model.add(Bidirectional(LSTM(hidden_neurons, dropout_W=0.5, dropout_U=0.3, return_sequences=False)))
  # model.add(TimeDistributed(AveragePooling1D(hidden_neurons)))
  #model.add(Dropout(.5))
  #model.add(LSTM(hidden_neurons, input_shape=(SLIDING_WINDOW_LENGTH,
  #  in_neurons)))
  model.add(Dense(middle_neurons, W_regularizer=l2(0.01)))
  model.add(Activation('relu'))
  model.add(Dropout(0.3))
  model.add(Dense(out_neurons, W_regularizer=l2(0.01)))
  model.add(Activation("softmax"))
# model.add(Activation("linear"))
  return model


from theano.compile.nanguardmode import NanGuardMode

def compile_model(net):
  """
  Compile the network using appropriate loss functions and target variables.
  Returns the compiled train function and test function.
  """
  # Prepare Theano variables for inputs and targets
  target_var = T.ivector('targets')
  # target_var = T.vector('targets') # Theano requires an ivector

  # Create a loss expression for training, i.e., a scalar objective we want
  # to minimize (for our multi-class problem, it is the cross-entropy loss):
  prediction = lasagne.layers.get_output(net['output'])
  loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
  loss = loss.mean()

  # We could add some weight decay as well here, see lasagne.regularization.
  # TODO(Mohit): Add L2/Dropout regularization on all weights.
  # reg_params = lasagne.layers.get_all_params(net['output'],regularizable=True)


  # Create update expressions for training, i.e., how to modify the
  # parameters at each training step. Here, we'll use Stochastic Gradient
  # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
  params = lasagne.layers.get_all_params(net['output'], trainable=True)

  print("Computing updates ...")
  # TODO(Mohit): Use Adagrad maybe?
  updates = lasagne.updates.nesterov_momentum(loss, params,
      learning_rate=0.01, momentum=0.9)

  # Create a loss expression for validation/testing. The crucial difference
  # here is that we do a deterministic forward pass through the network,
  # disabling dropout layers.
  test_prediction = lasagne.layers.get_output(net['output'],
      deterministic=True)
  test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
      target_var)
  test_loss = test_loss.mean()
  # As a bonus, also create an expression for the classification accuracy:
  test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
      dtype=theano.config.floatX)

  # Compile a function performing a training step on a mini-batch (by giving
  # the updates dictionary) and returning the corresponding training loss:
  print("Compiling functions ...")
  train_fn = theano.function([net['input'].input_var, target_var],
      [loss, prediction], updates=updates,
      #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
      )

  print('===== DEBUG INFO =====')
  #theano.printing.debugprint(train_fn)
  print('===== END =====')

  # Compile a second function computing the validation loss and accuracy:
  val_fn = theano.function([net['input'].input_var, target_var],
      [test_loss, test_acc, test_prediction])

  return train_fn, val_fn

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
  assert len(inputs) == len(targets)
  if shuffle:
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
  for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
    if shuffle:
      excerpt = indices[start_idx:start_idx + batchsize]
    else:
      excerpt = slice(start_idx, start_idx + batchsize)
    yield inputs[excerpt], targets[excerpt]

def count_correct_gestures(pred, target, count_none=False):
  """
  Returns the count of non-zero gestures that were predicted accurately.
  """
  pred_gesture = np.argmax(pred, axis=1)
  # None Gestures which were marked correctly should be removed from calc.
  pred_gesture[pred_gesture==0] = -1
  correct_count = np.sum(pred_gesture == target)
  total_count = np.sum(target != 0)
  return correct_count, total_count

def train_model(X_train, y_train, X_test, y_test, net, train_fn, val_fn):
  use_cached_epoch = config.get(USE_CACHED_EPOCH, False)

  # Finally, launch the training loop.
  train_err_hist, val_err_hist = [], []
  val_acc_hist = []
  loading = 0
  print("Starting training...")
  # We iterate over epochs:
  for epoch in range(NUM_EPOCHS):
    if use_cached_epoch and os.path.isfile('model_ss3.'+str(epoch+1)+'.npz'):
      print("Epoch {} already exists".format(epoch+1))
      f = np.load('model_ss3.'+str(epoch+1)+'.npz')
      param_values = [f['arr_%d' % i] for i in range(len(f.files))]
      lasagne.layers.set_all_param_values(net['output'], param_values)
    else:
      # In each epoch, we do a full pass over the training data:
      train_err = 0
      train_batches = 0
      start_time = time.time()
      for batch in iterate_minibatches(X_train, y_train, BATCH_SIZE,
          shuffle=True):
        inputs, targets = batch
        err, output_pred = train_fn(inputs, targets)
        train_err += err
        train_batches += 1

      train_err_hist.append(train_err)
      print("Epoch {} took {:.3f}s tloss:\t\t{:.6f}".format(
        epoch + 1,
        time.time() - start_time,
        train_err / train_batches))
      # Save training error history to file
      with open('train_hist_convlstm_centered.txt', 'w') as train_hist_file:
        for err in train_err_hist:
          train_hist_file.write('{0:.3f}\n'.format(err))

      #dump the network weights to a file:
      np.savez('model_ss3.'+str(epoch+1)+'.npz',
          *lasagne.layers.get_all_param_values(net['output']))

    # Do a full pass over the validation data:
    # (Mohit) This is not really validation data. We shouldn't be using X_test
    # at all during staging. Implement k-fold validation.
    val_err = 0
    val_acc = 0
    val_batches = 0
    val_correct_gestures = 0
    val_total_gestures = 0
    for batch in iterate_minibatches(X_test, y_test, BATCH_SIZE, shuffle=False):
      inputs, targets = batch
      err, acc, pred = val_fn(inputs, targets)
      correct_gestures, total_gestures = count_correct_gestures(pred, targets)
      val_correct_gestures += correct_gestures
      val_total_gestures += total_gestures
      val_err += err
      val_acc += acc
      val_batches += 1

    # Then we print the results for this epoch:
    print "Epoch {}\tvloss:{:.6f}\tvacc:{:.2f}%\tvgest_acc:{:.2f}%".format(
      epoch + 1, val_err / val_batches, val_acc / val_batches * 100,
      (val_correct_gestures*100.0)/val_total_gestures)
    val_err_hist.append([val_err, val_correct_gestures, val_total_gestures])
    val_acc_hist.append(val_acc)
    # Save training error history to file
    with open('val_hist_convlstm_centered.txt', 'w') as val_hist_file:
      for err in val_err_hist:
        val_hist_file.write('{0:.3f}, {1}, {2}\n'.format(err[0], err[1],
          err[2]))


  return train_err_hist, val_err_hist, val_acc_hist

def test_model(X_test, y_test, net):
  # Compilation of theano functions
  # Obtaining the probability distribution over classes
  test_prediction = lasagne.layers.get_output(net['output'],
      deterministic=True)
  # Returning the predicted output for the given minibatch
  test_fn = theano.function([ net['input'].input_var],
      [T.argmax(test_prediction, axis=1)])

  # Classification of the testing data
  print("Process {0} instances in mini-batches of {1}".format(X_test.shape[0],
    BATCH_SIZE))
  test_pred = np.empty((0))
  test_true = np.empty((0))
  start_time = time.time()
  for batch in iterate_minibatches(X_test, y_test, BATCH_SIZE):
    inputs, targets = batch
    y_pred, = test_fn(inputs)
    test_pred = np.append(test_pred, y_pred, axis=0)
    test_true = np.append(test_true, targets, axis=0)

  pd.DataFrame(test_pred).to_csv("deeplstm_convlstm_centered_ss3.pred.csv")
  pd.DataFrame(test_true).to_csv("deeplstm_convlstm_centered_ss3.tre.csv")

  # Results presentation
  print("||Results||")
  print("\tTook {:.3f}s.".format( time.time() - start_time))
  import sklearn.metrics as metrics
  print("\tTest fscore:\t{:.4f} ".format(metrics.f1_score(test_true, test_pred,
    average='weighted')))

def load_config(args):
  config['num_classify'] = args.num_classify
  config['lr'] = args.learning_rate
  config['fdir'] = args.checkpoints
  config['batch_size'] = args.batch_size
  config['win_len'] = args.win_len
  config['win_step'] = args.win_step
  config['num_epochs'] = args.epochs
  config['use_keras'] = 1
  return config

def plot_graphs(train_err_hist, val_err_hist, val_acc_hist):
  """
  train_err_hist: A list of loss values for the train function.
  val_err_hist: A list of list with loss values, correct gestures and total
  gestures for the validation function.
  val_acc_hist: A list of accuracy on the validation set.
  """
  val_err_arr = np.array(val_err_hist)
  fig, ax = plt.subplots(nrows=2, ncols=1)  # create figure & 1 axis
  plt.subplot(211)
  plt.plot(range(NUM_EPOCHS), train_err_hist, 'b', label='Train loss')
  plt.plot(range(NUM_EPOCHS), val_err_hist[:,0], 'g', label='Test loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(loc='upper left')

  plt.subplot(212)
  plt.plot(range(NUM_EPOCHS), val_err_hist[:, 1], 'r',
      label='Correct gestures')
  plt.xlabel('Epoch')
  plt.ylabel('Correct gestures [not none]')
  plt.legend(log-'upper left')
  fig.savefig('./Plots/lstm_noisy_data.png')
  plt.close(fig)

def evaluate_model(model, X, y, conf_csv=None, pred_csv=None, y_csv=None):
  '''
  Evaluates a given model on the given X and y. Prints the confusion matrix
  for the predicted responses. Optionall store the confusion matrix, predictions
  and actual outputs in csv files.
  '''
  score = model.evaluate(X, y, batch_size=256, verbose=1)
  print(score)
  print('Eval score: ', score[0], 'accuracy: ', score[1], 'frames: ', len(y))
  predicted = model.predict(X, batch_size=700) 
  conf = get_confusion_matrix_one_hot(predicted, y)
  (prec, recall, f1), (wt_prec, wt_recall, wt_f1) = get_stats_conf_matrix(
      conf)
  print('F1 score: {0:.2f}, precision: {0:.2f}, ' \
      'recall: {0:.2f}'.format(wt_f1, wt_prec, wt_recall))
  print('Individual results')
  print('F1 score ', f1)
  print('Precision ', prec)
  print('Recall ', recall)

  if conf_csv:
    pd.DataFrame(conf).to_csv(conf_csv)
  if pred_csv:
    pd.DataFrame(predicted).to_csv(pred_csv)
  if y_csv:
    pd.DataFrame(y).to_csv(y_csv)

  # Print to the console just for the sake of it
  opt = np.get_printoptions()
  np.set_printoptions(threshold='nan')
  print('\n===== Confusion matrix =====')
  print(conf)
  np.set_printoptions(**opt)
  return score 

def weighted_loss_function(weights):
  '''
  Weighted loss function which gives more weightage to certain classes based
  on certain weights.
  '''
  def f(y_true, y_pred):
    output /= output.sum(axis=-1, keepdims=True)
    # avoid numerical instability with _EPSILON clipping
    output = theano.Tensor.clip(output, _EPSILON, 1.0 - _EPSILON)
    return T.nnet.categorical_crossentropy(output, target)
  return f

def main(fdir, args):
  load_config(args)
  print 'Load config, ', config
  data_loader = DataLoader(config, 11, 52, config['win_len'])
  if config.get('save_custom_batch_load_data', False):
    # pass
    data_loader.save_custom_batch_data()

  sliding_window_loader = SlidingWindowGenerator(
    '../openface_data/train_gesture_by_file.h5',
    args.batch_size,
    11,
    args.num_classify,
    args.win_len,
    args.win_step,
    )
  
  if config['use_keras']:

    # X_train, y_train, X_test, y_test = process_keras_data(X_train, y_train,
    #    X_test, y_test)
    # _, _, X_val, y_val, X_test, y_test =  data_loader.load_data_centered_padded()
    gc.collect()
    # data_loader.save_data(X_train, y_train, X_val, y_val, X_test, y_test)

    # X_train, y_train, X_val, y_val, X_test, y_test = \
    #    load_data_centered_not_padded()

    #train_batch_gen = BatchGenerator('../openface_data/train_gesture_by_file.h5',
    #    BATCH_SIZE, 11, SLIDING_WINDOW_LENGTH, 52, num_classify=5)

    optim = keras.optimizers.Adagrad(lr=config['lr'], clipvalue=10.0)
    model = get_multi_conv_model()
    #print(model.summary())
    model.compile(loss="categorical_crossentropy",
        optimizer='adagrad', # optim,
        metrics=['accuracy'],
        #mode=NanGuardMode(nan_is_error=True, inf_is_error=True)
        )

    batch_history = {'loss': [], 'acc': []}
    val_metrics = {'val_loss': [], 'val_acc': []}
    for e in range(config['num_epochs']):
      print('Epoch {}'.format(e))
      for X_train, y_train in sliding_window_loader.next_batch():
        history = model.fit(X_train, y_train, 
            batch_size=config['batch_size'], nb_epoch=1, verbose=0)

        for k, v in history.history.iteritems():
          if batch_history.get(k, None) != None:
            batch_history[k].append(v)
        
      # Evaluate on validation data after every epoch
      val_score = evaluate_model(model, X_val, y_val)
      val_metrics['val_loss'].append(val_score[0])
      val_metrics['val_acc'].append(val_score[1])

    '''
    non_one_hot_y = np.argmax(y_train, 1)
    class_weights = compute_class_weight('balanced', np.unique(non_one_hot_y),
        non_one_hot_y)
    class_weight_dict = {}
    for i in xrange(np.shape(class_weights)[0]):
      class_weight_dict[i] = class_weights[i]
    print('Class weights in Loss {}'.format(class_weight_dict))

    history = model.fit(X_train, y_train, batch_size=256, nb_epoch=50,
        validation_data=(X_val, y_val), verbose=1,
        class_weight=class_weight_dict,
        callbacks=[PrintCheckPointStats(X_val, y_val),
         ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5')
         ])
    '''

    # Write history to json
    with open(fdir + '/' + 'keras_lstm_fc.hist.json', 'w') as fp:
      if not len(batch_history):
        json.dump(batch_history, fp, sort_keys=True, indent=4)
      else:
        hist = dict(batch_history, **val_metrics)
        json.dump(hist, fp, sort_keys=True, indent=4)

    json_string = model.to_json()
    open(fdir + '/' + 'keras_lstm_fc.model.json', 'w').write(json_string)
    model.save_weights(fdir + '/' + 'keras_lstm_fc.h5')

    print('====== Validation Data stats ======')
    evaluate_model(model, X_val, y_val,
        conf_csv=fdir + '/' + 'keras_lstm_fc.conf.val.csv',
        pred_csv=fdir + '/' + 'keras_lstm_fc.pred.val.csv',
        y_csv= fdir + '/' + 'keras_lstm_fc.target.val.csv')
    print('====================================')

    print('====== Test Data stats ======')
    evaluate_model(model, X_test, y_test,
        conf_csv=fdir + '/' + 'keras_lstm_fc.conf.test.csv',
        pred_csv=fdir + '/' + 'keras_lstm_fc.pred.test.csv',
        y_csv=fdir + '/' + 'keras_lstm_fc.target.test.csv')
    print('====================================')


  else:
    X_train, y_train, X_test, y_test = get_data(use_cached=use_cached_data)
    assert NB_SENSOR_CHANNELS == X_train.shape[1]

    # Sensor data is segmented using a sliding window mechanism
    X_train, y_train = opp_sliding_window(X_train, y_train,
        SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    X_test, y_test = opp_sliding_window(X_test, y_test,
        SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
    print(" ..after sliding window (testing): inputs {0}, " \
          "targets {1}".format(X_test.shape, y_test.shape))
    print(" ..after sliding window (training): inputs {0}, " \
          "targets {1}".format(X_train.shape, y_train.shape))

    X_train, y_train, X_test, y_test = reshape_input(X_train, y_train, X_test,
        y_test)
    print "Final datasets with size: | train {0} | test {1} | " \
      .format(X_train.shape,X_test.shape)
    print 'Create model.'
    net = None
    if config.get(USE_SIMPLE_LSTM, False):
      net = get_lstm_model(X_train, y_train, X_test, y_test)
    else:
      net = get_model(X_train, y_train, X_test, y_test)

    print 'Compile model.'
    train_fn, val_fn = compile_model(net)
    print 'Train model.'
    train_err_hist, val_err_hist, val_acc_hist = train_model(X_train, y_train,
        X_test, y_test, net, train_fn, val_fn)
    print 'Test model.'
    test_model(X_test, y_test, net)

if __name__ == '__main__':
  fdir = '.'
  parser = argparse.ArgumentParser(description='Train conv_lstm.')
  parser.add_argument('--learning_rate', nargs='?', type=str, const=1,
      default='0.001', help='Learning rate')
  parser.add_argument('--checkpoints', nargs='?', type=str, const=1,
      default='cv', help='Directory to store results')
  parser.add_argument('--batch_size', nargs='?', type=int, const=1,
      default=50, help='Batch size')
  parser.add_argument('--win_len', nargs='?', type=int, const=1,
      default=10, help='Window length for each input to the model')
  parser.add_argument('--win_step', nargs='?', type=int, const=1,
      default=1,
      help='Window step to take after each window length has been ' \
          'processed.')
  parser.add_argument('--num_classify', nargs='?', type=int, const=1,
    default=11,
    help='Number of classes to use for classification i.e. multi-class ' \
        'binary')
  parser.add_argument('--epochs', nargs='?', type=int, const=1,
      default=50, help='Number of epochs to run')
  args = parser.parse_args()
  print(args)

  fdir = args.checkpoints
  try: 
    os.makedirs(fdir)
  except OSError:
    if not os.path.isdir(fdir):
      raise()
  main(fdir, args)
