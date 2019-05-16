require 'torch'
require 'nn'
require 'rnn'
require 'loadcaffe'

require 'model/MyLSTM'

local utils = require 'util.utils'

local MultiScaleConvLSTM, parent = torch.class('nn.MultiScaleConvLSTM', 'nn.Module')

function MultiScaleConvLSTM:__init(kwargs)
  self.lstm_size = {64, 64}
  self.num_scales = utils.get_kwarg(kwargs, 'num_scales', 0)
  self.use_48_scale = utils.get_kwarg(kwargs, 'use_48_scale', 0)
  self.num_classes = utils.get_kwarg(kwargs, 'num_classes')
  self.num_classify = utils.get_kwarg(kwargs, 'num_classify')
  self.num_features = utils.get_kwarg(kwargs, 'num_features')
  self.h5dir = utils.get_kwarg(kwargs, 'data_dir')
  self.win_len = utils.get_kwarg(kwargs, 'win_len')
  self.win_step = utils.get_kwarg(kwargs, 'win_step')
  self.batch_size = utils.get_kwarg(kwargs, 'batch_size')
  self.num_features = utils.get_kwarg(kwargs, 'num_features')
  
  -- Optical flow pretrained-model
  self.use_opt_flow = utils.get_kwarg(kwargs, 'use_opt_flow')
  if self.use_opt_flow == 1 then
    assert(self.num_scales == 2)
    self.opt_flow_prototxt = utils.get_kwarg(kwargs, 'opt_flow_prototxt')
    self.opt_flow_caffemodel = utils.get_kwarg(kwargs, 'opt_flow_model')
  end

  self.curr_win_sizes = {16, 32, 64}
  self.start_frame = 36
  self.rnns_16 = {}
  self.rnns_32 = {}
  self.rnns_64 = {}
end

function MultiScaleConvLSTM:getMultiScaleSiameseModel()
  local final_model = nn.Sequential()
  local parallel_net = nn.ParallelTable()

  -- Get the 16 network model
  local model_16 = nn.Sequential()
  model:add(nn.SpatialConvolution(1, 128, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.Transpose({2,3}))

  local model_32 = model_16:clone('weight','bias','gradWeight','gradBias')

  parallel_net:add(model_16)
  parallel_net:add(model_32)

  final_model:add(parallel_net)
  local parallel_view = nn.ParallelTable()
  local lstm_16 = nn.Sequential()
  lstm_16:add(nn.View(-1, 10, 128*self.num_features))
  local lstm_16_1 = nn.MyLSTM(128*self.num_features, 256)
  lstm_16_1.remember_states = false
  table.insert(self.rnns_16, lstm_16_1)
  lstm_16:add(lstm_16_1)
  lstm_16:add(nn.Dropout(0.5))
  local lstm_16_2 = nn.MyLSTM(256, 256)
  lstm_16_2.remember_states = false
  table.insert(self.rnns_16, lstm_16_2)
  lstm_16:add(lstm_16_2)
  lstm_16:add(nn.Select(2, -1))

  local lstm_32 = nn.Sequential()
  lstm_32:add(nn.View(-1, 26, 128*self.num_features))
  local lstm_32_1 = lstm_16_1:clone()
  lstm_32:add(lstm_32_1)
  table.insert(self.rnns_32, lstm_32_1)
  lstm_32:add(nn.Dropout(0.5))
  local lstm_32_2 = lstm_16_2:clone()
  lstm_32:add(lstm_32_2)
  table.insert(self.rnns_32, lstm_32_2)
  lstm_32:add(nn.Select(2, -1))

  parallel_view:add(lstm_16)
  parallel_view:add(lstm_32)

  final_model:add(parallel_view)

  final_model:add(nn.JoinTable(2))

  final_model:add(nn.Linear(256*3, 128))
  final_model:add(nn.ReLU())
  final_model:add(nn.Dropout(0.5))

  final_model:add(nn.Linear(128, self.num_classify))

  self.net = final_model
end

function MultiScaleConvLSTM:getLabelCorrectionModel()
  -- Get the label model
  local label_model = nn.Sequential()
  label_model:add(nn.ReconsLayer(self.num_train_frames, self.batch_size,
    self.num_classify))
  self.labelModel = label_model

  -- Get the prediction model
  self:getOneScaleConvLSTMModel()
end

function MultiScaleConvLSTM:get_16_2d_conv_model()
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
  local lstm1 = nn.MyLSTM(128*self.num_features, 256)
  lstm1.remember_states = false
  table.insert(self.rnns_16, lstm1)
  model:add(lstm1)
  model:add(nn.Dropout(0.5))
  local lstm2 = nn.MyLSTM(256, 256)
  lstm2.remember_states = false
  table.insert(self.rnns_16, lstm2)
  model:add(lstm2)

  -- Select the last timestamp state
  model:add(nn.Select(2, 10))

  return model
end

function MultiScaleConvLSTM:get_32_2d_conv_model()
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
  local lstm1 = nn.MyLSTM(128*self.num_features, 256)
  lstm1.remember_states = false
  table.insert(self.rnns_32, lstm1)
  model:add(lstm1)
  model:add(nn.Dropout(0.5))
  local lstm2 = nn.MyLSTM(256, 256)
  lstm2.remember_states = false
  table.insert(self.rnns_32, lstm2)
  model:add(lstm2)

  -- Select the last timestamp state
  model:add(nn.Select(2, 26))

  return model
end

function MultiScaleConvLSTM:get_48_2d_conv_model()
  local model = nn.Sequential()
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
  model:add(nn.View(-1, 42, 128*self.num_features))

  -- Input to LSTM should be in 3D format (N, T, D)
  -- Add LSTM
  local lstm1 = nn.MyLSTM(128*self.num_features, 256)
  lstm1.remember_states = false
  table.insert(self.rnns_64, lstm1)
  model:add(lstm1)
  model:add(nn.Dropout(0.5))
  local lstm2 = nn.MyLSTM(256, 256)
  lstm2.remember_states = false
  table.insert(self.rnns_64, lstm2)
  model:add(lstm2)

  -- Select the last timestamp state
  model:add(nn.Select(2, 42))

  return model
end

function MultiScaleConvLSTM:get_64_2d_conv_model()
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
  local lstm1 = nn.MyLSTM(128*self.num_features, 256)
  lstm1.remember_states = false
  table.insert(self.rnns_64, lstm1)
  model:add(lstm1)
  model:add(nn.Dropout(0.5))
  local lstm2 = nn.MyLSTM(256, 256)
  lstm2.remember_states = false
  table.insert(self.rnns_64, lstm2)
  model:add(lstm2)

  -- Select the last timestamp state
  model:add(nn.Select(2, 52))

  return model
end

-- Since we implement stacked lstm in convLSTM we assume 4D input to the
-- LSTM model. Hence we need to convert it into 3D appropriately i.e.
-- (N, 1, T, F) to (N, T, F)
function MultiScaleConvLSTM:getStackedLSTMModel()
  local model = nn.Sequential()
  local inp_time = 32
  model:add(nn.View(-1, inp_time, self.num_features)) 
  local lstm_1 = nn.MyLSTM(self.num_features, 256)
  lstm_1.remember_states = false
  table.insert(self.rnns_16, lstm_1)
  model:add(lstm_1)

  model:add(nn.Dropout(0.5))

  local lstm_2 = nn.MyLSTM(256, 256)
  lstm_2.remember_states = false
  table.insert(self.rnns_16, lstm_2)
  model:add(lstm_2)

  model:add(nn.Select(2, -1))

  -- Finally add a Dense layer
  model:add(nn.Linear(256, 128))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))

  model:add(nn.Linear(128, self.num_classify))

  self.net = model
end

function MultiScaleConvLSTM:getOneScaleConvLSTMModel()

  local final_model = self:get_32_2d_conv_model()
  --local final_model = self:get_16_2d_conv_model()
  --local final_model = self:get_64_2d_conv_model()

  local parallel_model_op_size, initial_dense_layer = 256*2
  if self.num_scales == 1 then parallel_model_op_size = 256
  elseif self.num_scales == 2 then parallel_model_op_size = 256 * 2 end

  if self.use_opt_flow == 1 then
    parallel_model_op_size = parallel_model_op_size + 256
  end

  -- Finally add a Dense layer
  final_model:add(nn.Linear(parallel_model_op_size, 128))
  final_model:add(nn.ReLU())
  final_model:add(nn.Dropout(0.5))

  local num_classify = self.num_classify
  if G_global_opts.train_one_vs_all > 0 then num_classify = 2 end

  final_model:add(nn.Linear(128, num_classify))

  self.net = final_model
end

-- This only uses the 16 and 32 length timestamps
function MultiScaleConvLSTM:getTwoScaleConvLSTMModel()
  local final_model = nn.Sequential()
  local model = nn.ParallelTable()
  local m16 = self:get_16_2d_conv_model()
  model:add(m16)
  local m32 = self:get_32_2d_conv_model()
  model:add(m32)
  if self.use_opt_flow == 1 then
    local opt_flow_model = loadcaffe.load(
        self.opt_flow_prototxt, self.opt_flow_caffemodel)
    -- Remove fc8 layer
    for i=39,37,-1 do
      opt_flow_model:remove(i)
    end
    -- Now we have the 4096 vector from optical flow as the output.
    -- Add optical flow model to the Parallel Table model.
    -- Convert the 4096 vector into 256 length so that for FC layer we don't
    -- overweigh it.
    opt_flow_model:add(nn.ReLU())
    opt_flow_model:add(nn.Dropout(0.5))
    opt_flow_model:add(nn.Linear(4096, 256))

    -- We only want to finetune the optical flow network hence set the scale
    -- for it appropriately
    opt_flow_model:set_ms_scale(0.001, 36)
    model:add(opt_flow_model)
  end
  final_model:add(model)
  -- We now have multiple tables of length (N, 1, F) which we want to join
  -- together
  final_model:add(nn.JoinTable(2))

  local parallel_model_op_size, initial_dense_layer = 256*2
  if self.use_opt_flow == 1 then
    parallel_model_op_size = parallel_model_op_size + 256
  end

  -- Finally add a Dense layer
  final_model:add(nn.Linear(parallel_model_op_size, 128))
  final_model:add(nn.ReLU())
  final_model:add(nn.Dropout(0.5))

  final_model:add(nn.Linear(128, self.num_classify))

  self.net = final_model
end

function MultiScaleConvLSTM:getConvLSTMModel()
  if self.num_scales == 1 then
    return self:getOneScaleConvLSTMModel()
  elseif self.num_scales == 2 then
    return self:getTwoScaleConvLSTMModel()
  end

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
  if self.use_48_scale == 1 then
    local m48 = self:get_48_2d_conv_model()
    model:add(m48)
  else
    local m64 = self:get_64_2d_conv_model()
    model:add(m64)
  end
  final_model:add(model)
  -- We now have multiple tables of length (N, 1, F) which we want to join
  -- together
  final_model:add(nn.JoinTable(2))

  -- Finally add a Dense layer
  final_model:add(nn.Linear(256*3, 128))
  final_model:add(nn.ReLU())
  final_model:add(nn.Dropout(0.5))

  local num_classify = self.num_classify
  if G_global_opts.train_one_vs_all > 0 then
    num_classify = 2
  end

  final_model:add(nn.Linear(128, num_classify))

  self.net = final_model
end

function MultiScaleConvLSTM:updateType(dtype)
  self.net = self.net:type(dtype)
  if self.labelModel ~= nil then
    self.labelModel = self.labelModel:type(dtype)
  end
end

function MultiScaleConvLSTM:updateOutput(input)
  return self.net:forward(input)
end

function MultiScaleConvLSTM:backward(input, gradOutput, scale)
  return self.net:backward(input, gradOutput, scale)
end

function MultiScaleConvLSTM:parameters()
  return self.net:parameters()
end

function MultiScaleConvLSTM:training()
  self.net:training()
  parent.training(self)
end

function MultiScaleConvLSTM:evaluate()
  self.net:evaluate()
  parent.evaluate(self)
end

function MultiScaleConvLSTM:resetStates()
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

function MultiScaleConvLSTM:clearState()
  self.net:clearState()
end

