require 'torch'
require 'nn'
require 'rnn'

require 'model/MyLSTM'

local utils = require 'util.utils'

local ConvLSTM, parent = torch.class('nn.ConvLSTM', 'nn.Module')

function ConvLSTM:__init(kwargs)
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

function ConvLSTM:get_16_2d_conv_model()
  local model = nn.Sequential()
  -- kw = 1, kh = 3
  model:add(nn.SpatialConvolution(1, 128, 1, 3))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.ReLU())
  return model
end
 
-- (16, F) to output (10, F)
function ConvLSTM:get_16_to_1_model()
  local model = nn.Sequential()
  model:add(nn.TemporalConvolution(self.num_features, 128, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(128, 128, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(128, 128, 3))
  model:add(nn.ReLU())
  --[[
  model:add(nn.TemporalMaxPooling(2, 2))
  model:add(nn.TemporalConvolution(128, 64, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(64, 64, 3))
  model:add(nn.ReLU())
  ]]
  return model
end

function ConvLSTM:get_32_2d_conv_model()
  model = nn.Sequential()
  model:add(nn.SpatialConvolution(1, 128, 1, 3))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(128, 128, 1, 3, 1, 2))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.ReLU())
  --[[
  model:add(nn.SpatialMaxPooling(1, 2))
  model:add(nn.SpatialConvolution(128, 128, 1, 4))
  model:add(nn.ReLU())
  ]]
  return model
end

-- input (32, F) to output (10, F)
function ConvLSTM:get_32_to_1_model()
  model = nn.Sequential()
  model:add(nn.TemporalConvolution(self.num_features, 128, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(128, 128, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(128, 128, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalMaxPooling(2, 2))
  model:add(nn.TemporalConvolution(128, 64, 4))
  model:add(nn.ReLU())
  --[[
  model:add(nn.TemporalConvolution(64, 64, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(64, 64, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalMaxPooling(2, 2))
  model:add(nn.TemporalConvolution(64, 64, 3))
  model:add(nn.ReLU())
  ]]
  return model
end

function ConvLSTM:get_64_2d_conv_model()
  local model = nn.Sequential()
  model:add(nn.SpatialConvolution(1, 128, 1, 3))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.ReLU())
  --model:add(nn.SpatialMaxPooling(1, 2))
  model:add(nn.SpatialConvolution(128, 128, 1, 3, 1, 2))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(128, 128, 1, 3, 1, 2))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.ReLU())
  --model:add(nn.SpatialMaxPooling(1, 2))
  --model:add(nn.SpatialConvolution(64, 64, 1, 2))
  return model
end

function get_64_to_1_model()
  local model = nn.Sequential()
  model:add(nn.TemporalConvolution(self.num_features, 128, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(128, 128, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(128, 128, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalMaxPooling(2, 2))
  model:add(nn.TemporalConvolution(128, 64, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(64, 64, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(64, 64, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalMaxPooling(2, 2))
  model:add(nn.TemporalConvolution(64, 64, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(64, 64, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalMaxPooling(2, 2))
  model:add(nn.TemporalConvolution(64, 64, 3))
  model:add(nn.ReLU())
  return model
end

function ConvLSTM:getConvLSTMModel()
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
  final_model:add(nn.CMulTable())

  -- input is (N, channels, T=10, F=46)
  final_model:add(nn.Transpose({2, 3}))
  --final_model:add(nn.Reshape(-1, 128, 10*self.num_features))
  final_model:add(nn.View(-1, 10, 128*self.num_features))

  -- Input to LSTM should be in 3D format (N, T, D)
  -- Add LSTM
  local lstm1 = nn.MyLSTM(128*self.num_features, 256)
  lstm1.remember_states = false
  table.insert(self.rnns, lstm1)
  final_model:add(lstm1)
  final_model:add(nn.Dropout(0.5))
  local lstm2 = nn.MyLSTM(256, 256)
  lstm2.remember_states = false
  table.insert(self.rnns, lstm2)
  final_model:add(lstm2)
  final_model:add(nn.Dropout(0.5))

  -- After RNN we have tensor of size (N, T, H).
  -- We want to add a couple of dense layers to this so first
  -- remove the time dimension from net
  final_model:add(nn.Select(2, self.win_len))

  -- Finally add a Dense layer
  final_model:add(nn.Linear(256, self.num_classify))

  self.net = final_model
end

function ConvLSTM:updateType(dtype)
  self.net = self.net:type(dtype)
end

function ConvLSTM:updateOutput(input)
  return self.net:forward(input)
end

function ConvLSTM:backward(input, gradOutput, scale)
  return self.net:backward(input, gradOutput, scale)
end

function ConvLSTM:parameters()
  return self.net:parameters()
end

function ConvLSTM:training()
  self.net:training()
  parent.training(self)
end

function ConvLSTM:evaluate()
  self.net:evaluate()
  parent.evaluate(self)
end

function ConvLSTM:resetStates()
  for i, rnn in ipairs(self.rnns) do 
    rnn:resetStates()
  end
end

function ConvLSTM:clearState()
  self.net:clearState()
end

--[[
function ConvLSTM:getModel_2(input_size, fc_size, output_size)
  self.net = nn.Sequential()
  local lstm_input_size = input_size
  for i=1,#self.lstm_size do
    local rnn = nn.SeqLSTM(lstm_input_size, self.lstm_size[i])
    rnn.maskzero = true
    self.net:add(rnn)
    if self.dropout > 0 then
      self.net:add(nn.Dropout(self.dropout))
    end
    lstm_input_size = self.lstm_size[i]
  end
  -- After RNN we have tensor of size (T, N, H).
  -- We want to add a couple of dense layers to this so first
  -- remove the time dimension from net
  -- For now hardcode the sequence length i.e. 240
  self.net:add(nn.Select(1, 240))

  self.net:add(nn.Linear(lstm_input_size, fc_size))
  self.net:add(nn.ReLU())
  self.net:add(nn.Linear(fc_size, output_size))
  return self.net
end

function ConvLSTM:getModel_3(input_size, fc_size, output_size)
  self.net = nn.Sequential()
  local lstm_input_size = input_size
  for i=1,#self.lstm_size do
    local rnn = nn.SeqLSTM(lstm_input_size, self.lstm_size[i])
    rnn.maskzero = true
    --self.net:add(nn.Sequencer(rnn))
    self.net:add(rnn)
    if self.dropout > 0 then
      self.net:add(nn.Dropout(self.dropout))
    end
    lstm_input_size = self.lstm_size[i]
  end
  self.net:add(nn.Sequencer(nn.Linear(lstm_input_size, fc_size)))
  self.net:add(nn.Sequencer(nn.ReLU()))
  self.net:add(nn.Sequencer(nn.Linear(fc_size, output_size)))
  return self.net
end

function ConvLSTM:getLSTMModel(input_size, fc_size, output_size)
  self.net = nn.Sequential()
  local lstm_input_size = input_size
  for i=1,#self.lstm_size do
    local rnn = nn.SeqLSTM(lstm_input_size, self.lstm_size[i])
    rnn.maskzero = true
    --self.net:add(nn.Sequencer(rnn))
    self.net:add(rnn)
    if self.dropout > 0 then
      self.net:add(nn.Dropout(self.dropout))
    end
    lstm_input_size = self.lstm_size[i]
  end
  -- After RNN we have tensor of size (T, N, H).
  -- We want to add a couple of dense layers to this so first
  -- remove the time dimension from net
  -- For now hardcode the sequence length i.e. 240
  self.net:add(nn.Select(1, 240))

  self.net:add(nn.Linear(lstm_input_size, fc_size))
  self.net:add(nn.ReLU())
  self.net:add(nn.Linear(fc_size, output_size))
  return self.net
end

function ConvLSTM:getModel(input_size, fc_size, output_size)
  self.net = nn.Sequential()
  local lstm_input_size = input_size
  for i=1,#self.lstm_size do
    local rnn = nn.SeqLSTM(lstm_input_size, self.lstm_size[i])
    rnn.maskzero = true
    --self.net:add(nn.Sequencer(rnn))
    self.net:add(rnn)
    if self.dropout > 0 then
      self.net:add(nn.Dropout(self.dropout))
    end
    lstm_input_size = self.lstm_size[i]
  end
  -- After RNN we have tensor of size (T, N, H).
  -- We want to add a couple of dense layers to this so first
  -- remove the time dimension from net
  -- For now hardcode the sequence length i.e. 240
  self.net:add(nn.Select(1, 240))

  self.net:add(nn.Linear(lstm_input_size, fc_size))
  self.net:add(nn.ReLU())
  self.net:add(nn.Linear(fc_size, output_size))
  return self.net
end
]]

