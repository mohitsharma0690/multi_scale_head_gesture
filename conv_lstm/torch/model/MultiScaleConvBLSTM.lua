require 'torch'
require 'nn'
require 'rnn'

require 'model/MyLSTM'
require 'model/MS_BLSTM'
require 'model/MS_BRNN'

local utils = require 'util.utils'

local MultiScaleConvBLSTM, parent = torch.class('nn.MultiScaleConvBLSTM', 'nn.Module')

function MultiScaleConvBLSTM:__init(kwargs)
  self.lstm_size = {64, 64}
  self.num_classes = utils.get_kwarg(kwargs, 'num_classes')
  self.num_classify = utils.get_kwarg(kwargs, 'num_classify')
  self.num_features = utils.get_kwarg(kwargs, 'num_features')
  self.h5dir = utils.get_kwarg(kwargs, 'data_dir')
  self.win_len = utils.get_kwarg(kwargs, 'win_len')
  self.win_step = utils.get_kwarg(kwargs, 'win_step')
  self.batch_size = utils.get_kwarg(kwargs, 'batch_size')
  self.num_features = utils.get_kwarg(kwargs, 'num_features')
  self.curr_win_sizes = {16, 32, 64}
  self.start_frame = 36
  self.rnns_16 = {}
  self.rnns_32 = {}
  self.rnns_64 = {}
end

function MultiScaleConvBLSTM:get_16_2d_conv_model()
  local model = nn.Sequential()
  -- kw = 1, kh = 3
  model:add(nn.SpatialConvolution(1, 128, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.ReLU())

  -- Now input is (N, channels, T=10, F=46)
  model:add(nn.Transpose({2, 3}))
  model:add(nn.View(-1, 10, 128*self.num_features))

  -- Input to LSTM should be in 3D format (N, T, D)
  -- Add LSTM
  local lstm1 = nn.MS_BRNN(
    nn.MyLSTM(128*self.num_features, 256),
    nn.MyLSTM(128*self.num_features, 256))
    
  table.insert(self.rnns_16, lstm1)
  model:add(lstm1)

  model:add(nn.Dropout(0.5))
  local lstm2 = nn.MS_BRNN(
    nn.MyLSTM(256, 256),
    nn.MyLSTM(256, 256))
     
  table.insert(self.rnns_16, lstm2)
  model:add(lstm2)

  -- Select the last timestamp state
  model:add(nn.Select(2, 10))
  
  return model
end
 
function MultiScaleConvBLSTM:get_32_2d_conv_model()
  model = nn.Sequential()
  model:add(nn.SpatialConvolution(1, 128, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.ReLU())

  -- Now input is (N, channels, T=10, F=46)
  model:add(nn.Transpose({2, 3}))
  model:add(nn.View(-1, 26, 128*self.num_features))

  -- Input to LSTM should be in 3D format (N, T, D)
  -- Add LSTM
  local lstm1 = nn.MS_BRNN(
    nn.MyLSTM(128*self.num_features, 256),
    nn.MyLSTM(128*self.num_features, 256))

  table.insert(self.rnns_32, lstm1)
  model:add(lstm1)

  model:add(nn.Dropout(0.5))

  local lstm2 = nn.MS_BRNN(
    nn.MyLSTM(256, 256),
    nn.MyLSTM(256, 256))
  table.insert(self.rnns_32, lstm2)
  model:add(lstm2)

  -- Select the last timestamp state
  model:add(nn.Select(2, 26))
  return model
end

function MultiScaleConvBLSTM:get_64_2d_conv_model()
  local model = nn.Sequential()
  model:add(nn.SpatialConvolution(1, 128, 1, 5))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolution(128, 128, 1, 5))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolution(128, 128, 1, 5))
  model:add(nn.ReLU())

  -- Now input is (N, channels, T=10, F=46)
  model:add(nn.Transpose({2, 3}))
  model:add(nn.View(-1, 52, 128*self.num_features))

  -- Input to LSTM should be in 3D format (N, T, D)
  -- Add LSTM
  local lstm1 = nn.MS_BRNN(
    nn.MyLSTM(128*self.num_features, 256),
    nn.MyLSTM(128*self.num_features, 256))
    
  table.insert(self.rnns_64, lstm1)
  model:add(lstm1)

  model:add(nn.Dropout(0.5))

  local lstm2 = nn.MS_BRNN(
    nn.MyLSTM(256, 256),
    nn.MyLSTM(256, 256))
  table.insert(self.rnns_64, lstm2)
  model:add(lstm2)

  -- Select the last timestamp state
  model:add(nn.Select(2, 52))

  return model
end

function MultiScaleConvBLSTM:getConvLSTMModel()
  local final_model = nn.Sequential()
  local model = nn.ParallelTable()
  --[[
  local m16 = nn.Parallel(2, 2)
  local m32 = nn.Parallel(2, 2)
  for i = 1, self.win_len do
    local m16_to_1 = self:get_16_to_1_model()
    m16:add(m16_to_1)
    local m32_to_1 = self:get_32_to_1_model()
    m32:add(m32_to_1)
  end
  ]]
  local m16 = self:get_16_2d_conv_model()
  model:add(m16)
  local m32 = self:get_32_2d_conv_model()
  model:add(m32)
  local m64 = self:get_64_2d_conv_model()
  model:add(m64)
  final_model:add(model)
  -- We now have multiple tables of length (N, 1, F) which we want to join
  -- together
  final_model:add(nn.JoinTable(2))

  -- Finally add a Dense layer
  final_model:add(nn.Linear(256*3, 128))
  final_model:add(nn.ReLU())
  final_model:add(nn.Dropout(0.5))

  final_model:add(nn.Linear(128, self.num_classify))

  self.net = final_model
end

function MultiScaleConvBLSTM:updateType(dtype)
  self.net = self.net:type(dtype)
end

function MultiScaleConvBLSTM:updateOutput(input)
  return self.net:forward(input)
end

function MultiScaleConvBLSTM:backward(input, gradOutput, scale)
  return self.net:backward(input, gradOutput, scale)
end

function MultiScaleConvBLSTM:updateGradInput(input, gradOutput)
  return self.net:updateGradInput(input, gradOutput)
end

function MultiScaleConvBLSTM:parameters()
  return self.net:parameters()
end

function MultiScaleConvBLSTM:training()
  self.net:training()
  parent.training(self)
end

function MultiScaleConvBLSTM:evaluate()
  self.net:evaluate()
  parent.evaluate(self)
end

function MultiScaleConvBLSTM:resetStates()
  for i, rnn in ipairs(self.rnns_16) do 
    rnn:resetStates()
  end
  for i, rnn in ipairs(self.rnns_32) do 
    rnn:resetStates()
  end
  for i, rnn in ipairs(self.rnns_64) do 
    rnn:resetStates()
  end
end

function MultiScaleConvBLSTM:clearState()
  self.net:clearState()
end

