require 'torch'
require 'nn'
require 'rnn'

require 'model/MyLSTM'

local utils = require 'util.utils'

local DenseConvLSTM, parent = torch.class('nn.DenseConvLSTM', 'nn.Module')

function DenseConvLSTM:__init(kwargs)
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
  self.rnns = {}
end
 
-- (16, F) to output (10, F)
function DenseConvLSTM:get_16_to_1_model()
  local model = nn.Sequential()
  model:add(nn.SpatialConvolution(1, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(1, 2))
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  return model
end

-- input (32, F) to output (10, F)
function DenseConvLSTM:get_32_to_1_model()
  local model = nn.Sequential()
  model:add(nn.SpatialConvolution(1, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(1, 2))
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(1, 2))
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  return model
end

function get_64_to_1_model()
  local model = nn.Sequential()
  model:add(nn.SpatialConvolution(1, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(1, 2))
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(1, 2))
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(1, 2))
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  return model
end

function DenseConvLSTM:getDenseConvLSTMModel()
  local final_model = nn.Sequential()
  local model = nn.ParallelTable()
  local m16 = nn.Parallel(2, 2)
  local m32 = nn.Parallel(2, 2)
  local m64 = nn.Parallel(2, 2)
  for i = 1, self.win_len do
    local m16_to_1 = self:get_16_to_1_model()
    m16:add(m16_to_1)
    local m32_to_1 = self:get_32_to_1_model()
    m32:add(m32_to_1)
    local m64_to_1 = get_64_to_1_model()
    m64:add(m64_to_1)
  end
  model:add(m16)
  model:add(m32)
  model:add(m64)
  final_model:add(model)
  -- We now have multiple tables of length (N, 1, F) which we want to join
  -- together
  final_model:add(nn.CMaxTable())

  -- The model here is (N, T*C, 1, F) which is now remapped to (N, T, C, 1, F)
  final_model:add(nn.View(-1, self.win_len, 64, 1, self.num_features))
  -- The model is now remapped to (N, T, C*1*F)
  final_model:add(nn.View(-1, self.win_len, 64*1*self.num_features))

  -- input is (N, channels, T=10, F=46)
  --final_model:add(nn.Transpose({2, 3}))
  --final_model:add(nn.Reshape(-1, 128, 10*self.num_features))
  --final_model:add(nn.View(-1, 10, 64*self.num_features))

  -- Input to LSTM should be in 3D format (N, T, D)
  -- Add LSTM
  local lstm1 = nn.MyLSTM(64*1*self.num_features, 128)
  lstm1.remember_states = false
  table.insert(self.rnns, lstm1)
  final_model:add(lstm1)
  final_model:add(nn.Dropout(0.3))
  local lstm2 = nn.MyLSTM(128, 128)
  lstm2.remember_states = false
  table.insert(self.rnns, lstm2)
  final_model:add(lstm2)
  final_model:add(nn.Dropout(0.3))

  -- After RNN we have tensor of size (N, T, H).
  -- We want to add a couple of dense layers to this so first
  -- remove the time dimension from net
  final_model:add(nn.Select(2, self.win_len))

  -- Finally add a Dense layer
  final_model:add(nn.Linear(128, self.num_classify))

  self.net = final_model
end

function DenseConvLSTM:updateType(dtype)
  self.net = self.net:type(dtype)
end

function DenseConvLSTM:updateOutput(input)
  return self.net:forward(input)
end

function DenseConvLSTM:backward(input, gradOutput, scale)
  return self.net:backward(input, gradOutput, scale)
end

function DenseConvLSTM:parameters()
  return self.net:parameters()
end

function DenseConvLSTM:training()
  self.net:training()
  parent.training(self)
end

function DenseConvLSTM:evaluate()
  self.net:evaluate()
  parent.evaluate(self)
end

function DenseConvLSTM:resetStates()
  for i, rnn in ipairs(self.rnns) do 
    rnn:resetStates()
  end
end

function DenseConvLSTM:clearState()
  self.net:clearState()
end

