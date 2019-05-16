require 'torch'
require 'nn'
require 'rnn'

require 'model/MyLSTM'

local utils = require 'util.utils'

local MultiScaleCNN, parent = torch.class('nn.MultiScaleCNN', 'nn.Module')

function MultiScaleCNN:__init(kwargs)
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

function MultiScaleCNN:get_16_2d_conv_model()
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
 
function MultiScaleCNN:get_32_2d_conv_model()
  model = nn.Sequential()
  model:add(nn.SpatialConvolution(1, 128, 1, 3))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(128, 128, 1, 5))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(128, 128, 1, 3, 1, 2))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.ReLU())
  return model
end

function MultiScaleCNN:get_64_2d_conv_model()
  local model = nn.Sequential()
  model:add(nn.SpatialConvolution(1, 128, 1, 3))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(128, 128, 1, 5))
  model:add(nn.SpatialBatchNormalization(128))
  model:add(nn.ReLU())
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

function MultiScaleCNN:getConvLSTMModel()
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
  -- We now have multiple tables of length (N, T, F) which we want to join
  -- together
  final_model:add(nn.JoinTable(3))

  -- input is (N, channels, T=10, F=46)
  -- final_model:add(nn.Transpose({2, 3}))
  --final_model:add(nn.Reshape(-1, 128, 10*self.num_features))
  -- final_model:add(nn.View(-1, 30, 128*self.num_features))

  -- input is (N, channels, T=30, F=46)
  final_model:add(nn.SpatialConvolution(128, 64, 1, 3, 1, 2))
  -- now we should be at (N, 64, 14, F)
  final_model:add(nn.SpatialConvolution(64, 16, 1, 3, 1, 2))
  final_model:add(nn.View(-1, 16*6*self.num_features))
  final_model:add(nn.Linear(16*6*self.num_features, 128))
  final_model:add(nn.ReLU())
  final_model:add(nn.Dropout(0.5))

  -- Finally add a Dense layer
  final_model:add(nn.Linear(128, self.num_classify))

  self.net = final_model
end

function MultiScaleCNN:updateType(dtype)
  self.net = self.net:type(dtype)
end

function MultiScaleCNN:updateOutput(input)
  return self.net:forward(input)
end

function MultiScaleCNN:backward(input, gradOutput, scale)
  return self.net:backward(input, gradOutput, scale)
end

function MultiScaleCNN:parameters()
  return self.net:parameters()
end

function MultiScaleCNN:training()
  self.net:training()
  parent.training(self)
end

function MultiScaleCNN:evaluate()
  self.net:evaluate()
  parent.evaluate(self)
end

function MultiScaleCNN:resetStates()
  for i, rnn in ipairs(self.rnns) do 
    rnn:resetStates()
  end
end

function MultiScaleCNN:clearState()
  self.net:clearState()
end

