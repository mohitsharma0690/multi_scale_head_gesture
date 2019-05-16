require 'torch'
require 'nn'
require 'rnn'
require 'loadcaffe'

require 'model/MyLSTM'
require 'model/MS_Embedding'
require 'model/ReconsLayer'

local utils = require 'util.utils'

local MultiScaleLatentConvLSTM, parent = torch.class('nn.MultiScaleLatentConvLSTM', 'nn.Module')

function MultiScaleLatentConvLSTM:__init(kwargs)
  self.lstm_size = {64, 64}
  self.num_scales = utils.get_kwarg(kwargs, 'num_scales', 3)
  self.use_48_scale = utils.get_kwarg(kwargs, 'use_48_scale', 0)
  self.num_classes = utils.get_kwarg(kwargs, 'num_classes')
  self.num_classify = utils.get_kwarg(kwargs, 'num_classify')
  self.num_features = utils.get_kwarg(kwargs, 'num_features')
  self.h5dir = utils.get_kwarg(kwargs, 'data_dir')
  self.win_len = utils.get_kwarg(kwargs, 'win_len')
  self.win_step = utils.get_kwarg(kwargs, 'win_step')
  self.batch_size = utils.get_kwarg(kwargs, 'batch_size')
  print(self.num_features)
  self.use_label_correction = utils.get_kwarg(kwargs, 'use_label_correction')

  self.latent_model = utils.get_kwarg(kwargs, 'latent_model')
  -- Latent pretrained model
  if self.latent_model == 'lstm_encoder' then
    self.pretrained_latent_model = utils.get_kwarg(kwargs, 'latent_variable_model')
    self.latent_model = torch.load(self.pretrained_latent_model)
    self.latent_enc = self.latent_model.model.encLSTMs
    self.use_long_term_latent_variable = utils.get_kwarg(kwargs,
      'use_long_term_latent_variable')
  end

  if self.use_label_correction == 1 then
    self.num_train_frames = utils.get_kwarg(kwargs, 'num_train_frames')
  end

  -- Optical flow pretrained-model
  self.use_opt_flow = utils.get_kwarg(kwargs, 'use_opt_flow')
  if self.use_opt_flow == 1 then
    assert(self.num_scales == 1)
    self.opt_flow_prototxt = utils.get_kwarg(kwargs, 'opt_flow_prototxt')
    self.opt_flow_caffemodel = utils.get_kwarg(kwargs, 'opt_flow_model')
  end

  self.curr_win_sizes = {16, 32, 64}
  self.start_frame = 36
  self.rnns_16 = {}
  self.rnns_32 = {}
  self.rnns_64 = {}
end

function MultiScaleLatentConvLSTM:getLabelCorrectionModel()
  -- Get the prediction model
  self:getOneScaleConvLSTMModel()
  self.reconsLayer = nn.ReconsLayer(self.num_train_frames, self.batch_size,
      self.num_classify)
  self.net:add(self.reconsLayer)
end

function MultiScaleLatentConvLSTM:get_16_2d_conv_model()
  local final_model = nn.Sequential()
  local pt = nn.ParallelTable()
  -- kw = 1, kh = 3
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
  model:add(nn.View(-1, 10, 128*self.num_features))

  -- Add convolutions to parallel table
  pt:add(model)

  local enc_16_1 = self.latent_enc[1]:clone()
  local enc_16_2 = self.latent_enc[2]:clone()
  local enc_16 = nn.Sequential()
  enc_16:add(nn.Transpose({1, 2})) -- Transpose to seqlen x batch
  enc_16:add(enc_16_1)
  enc_16:add(enc_16_2)
  enc_16:add(nn.Transpose({1, 2})) -- Transpose back to batch, seqlen
  enc_16:add(nn.Select(2, 10)) -- Get the encoded representation NxH
  enc_16:add(nn.Replicate(10, 1)) -- Replicate across all timesteps TxNxH
  enc_16:add(nn.Transpose({1, 2})) -- NxTxH now
  pt:add(enc_16)

  final_model:add(pt)
  final_model:add(nn.JoinTable(3))

  -- Input to LSTM should be in 3D format (N, T, D)
  -- Add LSTM
  local lstm1 = nn.MyLSTM(128*self.num_features + 128, 256)
  lstm1:set_name('lstm_16_1')
  lstm1.remember_states = false
  table.insert(self.rnns_16, lstm1)
  final_model:add(lstm1)
  final_model:add(nn.Dropout(0.5))
  local lstm2 = nn.MyLSTM(256, 256)
  lstm2:set_name('lstm_16_2')
  lstm2.remember_states = false
  table.insert(self.rnns_16, lstm2)
  final_model:add(lstm2)

  -- Select the last timestamp state
  final_model:add(nn.Select(2, 10))

  return final_model
end

function MultiScaleLatentConvLSTM:get_32_2d_conv_model()
  local final_model = nn.Sequential()

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
  model:add(nn.View(-1, 26, 128*self.num_features))

  -- Add convolutions to parallel table
  final_model:add(model)

  -- Input to LSTM should be in 3D format (N, T, D)
  -- Add LSTM
  local lstm1 = nn.MyLSTM(128*self.num_features, 256)
  lstm1:set_name('lstm_32_1')
  lstm1.remember_states = false
  table.insert(self.rnns_32, lstm1)
  final_model:add(lstm1)
  final_model:add(nn.Dropout(0.5))
  local lstm2 = nn.MyLSTM(256, 256)
  lstm2:set_name('lstm_32_2')
  lstm2.remember_states = false
  table.insert(self.rnns_32, lstm2)
  final_model:add(lstm2)

  -- Select the last timestamp state
  final_model:add(nn.Select(2, 26))

  return final_model
end

function MultiScaleLatentConvLSTM:get_48_2d_conv_model()
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

function MultiScaleLatentConvLSTM:get_64_2d_conv_model()
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

--[[
 We declare three sequential models below. context_model, classif_model and
 final_model. Both context_model and classif_model are ConvLSTM models although
 with different Conv architectures. Their input is concatenated by the
 final_model which contains stacked FC layers.  We have to backpropagate the
 gradients manually in our case. We can train this model by just one final
 classification loss. But, we also use another auxiliarly loss for the context
 layer.
]]
function MultiScaleLatentConvLSTM:getSupervisedContextLatentVariableConvModel(args)
  local latent_inp_size = args.latent_inp_size

  self.init_classif_model = self:get_32_2d_conv_model()

  -- Build the context sequential first 
  local final_time_dim = 26
  local context_seq = nn.Sequential()
  context_seq:add(nn.SpatialConvolution(1, 128, 1, 5, 1, 2))
  context_seq:add(nn.ReLU())
  context_seq:add(nn.Dropout(0.5))
  context_seq:add(nn.SpatialConvolution(128, 128, 1, 5, 1, 2))
  context_seq:add(nn.ReLU())
  context_seq:add(nn.Dropout(0.5))
  context_seq:add(nn.SpatialConvolution(128, 128, 1, 3, 1, 1))
  context_seq:add(nn.ReLU())
  -- Now input is (N, channels, T=10, F=46)
  context_seq:add(nn.Transpose({2, 3}))
  context_seq:add(nn.View(-1, final_time_dim, 128*self.num_features))

  -- Input to LSTM should be in 3D format (N, T, D)
  -- Add LSTM
  local lstm1 = nn.MyLSTM(128*self.num_features, 256)
  lstm1:set_name('context_lstm_32_1')
  lstm1.remember_states = false
  table.insert(self.rnns_32, lstm1)
  context_seq:add(lstm1)
  context_seq:add(nn.Dropout(0.5))
  local lstm2 = nn.MyLSTM(256, 256)
  lstm2:set_name('context_lstm_32_2')
  lstm2.remember_states = false
  table.insert(self.rnns_32, lstm2)
  context_seq:add(lstm2)

  -- Select the last timestamp state
  context_seq:add(nn.Select(2, final_time_dim))
  self.init_context_model = context_seq

  local final_model = nn.Sequential()
  final_model:add(nn.JoinTable(2))
  -- Add the final FC and classification layers.
  final_model:add(nn.Linear(512, 128))
  final_model:add(nn.ReLU())
  final_model:add(nn.Dropout(0.5))

  final_model:add(nn.Linear(128, self.num_classify))
  self.final_classif_model = final_model

  local final_context_model = nn.Sequential()
  final_context_model:add(nn.Linear(256, 128))
  final_context_model:add(nn.Linear(128, 3))
  self.final_context_model = final_context_model

  return final_model
end

function MultiScaleLatentConvLSTM:getUserIdLatentVariableConvModel(args)
  local latent_inp_size = args.latent_inp_size or 14
  local final_model = nn.Sequential()
  local pt = nn.ParallelTable()

  local model = nn.Sequential()
  model:add(nn.SpatialConvolution(1, 128, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.ReLU())

  model:add(nn.View(-1, 26, 128 * self.num_features))  -- Removes channel dimension

  pt:add(model) -- Add convolutions to parallel table

  local latent_seq = nn.Sequential()
  -- latent_seq:add(nn.LookupTable(17, 16))
  --latent_seq:add(nn.View(-1, 16))  -- Since LookupTable creates (N,1,16)
  latent_seq:add(nn.Replicate(26, 1))
  latent_seq:add(nn.Transpose({1, 2}))
  -- latent_seq:add(nn.Transpose({2, 3}))

  -- Add latent sequential to parallel table
  pt:add(latent_seq)

  final_model:add(pt)
  final_model:add(nn.JoinTable(3))

  -- Input to LSTM should be in 3D format (N, T, D)
  -- Add LSTM
  local lstm1 = nn.MyLSTM(128*self.num_features+G_global_opts.latent_num_features, 1024)
  lstm1:set_name('user_id_input_lstm')
  --lstm1:set_name('lstm_32_1')
  lstm1.remember_states = false
  table.insert(self.rnns_32, lstm1)
  final_model:add(lstm1)
  final_model:add(nn.Dropout(0.5))
  local lstm2 = nn.MyLSTM(1024, 512)
  lstm2:set_name('lstm_32_2')
  lstm2.remember_states = false
  table.insert(self.rnns_32, lstm2)
  final_model:add(lstm2)

  -- Select the last timestamp state
  final_model:add(nn.Select(2, -1))

  -- Add the final FC and classification layers.
  final_model:add(nn.Linear(512, 128))
  final_model:add(nn.ReLU())
  final_model:add(nn.Dropout(0.5))

  final_model:add(nn.Linear(128, self.num_classify))

  self.net = final_model
end

function MultiScaleLatentConvLSTM:getLongTermLatentVariableConvModel()
  local final_model = nn.Sequential()
  local pt = nn.ParallelTable()

  local model = nn.Sequential()
  model:add(nn.SpatialConvolution(1, 128, 1, 3))
  model:add(nn.ReLU())
  --model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.ReLU())
  --model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.ReLU())
  -- Now input is (N, channels, T=10, F=46)
  model:add(nn.Transpose({2, 3}))
  model:add(nn.View(-1, 26, 128*self.num_features))

  -- Add convolutions to parallel table
  pt:add(model)

  local latent_seq = nn.Sequential()
  local latent_pt = nn.ParallelTable()

  -- Latent Variable 1
  local enc_32_1 = self.latent_enc[1]:clone()
  local enc_32_2 = self.latent_enc[2]:clone()
  local enc_32 = nn.Sequential()
  enc_32:add(nn.Transpose({1, 2})) -- Transpose to seqlen x batch
  enc_32:add(enc_32_1)
  enc_32:add(nn.Dropout(0.5))
  enc_32:add(enc_32_2)
  enc_32:add(nn.Transpose({1, 2})) -- Transpose back to batch, seqlen
  enc_32:add(nn.Select(2, 32)) -- Get the encoded representation NxH
  latent_pt:add(enc_32)

  -- Latent Variable 2
  local enc_32_2_1 = self.latent_enc[1]:clone()
  local enc_32_2_2 = self.latent_enc[2]:clone()
  local enc_32_2 = nn.Sequential()
  enc_32_2:add(nn.Transpose({1, 2})) -- Transpose to seqlen x batch
  enc_32_2:add(enc_32_2_1)
  enc_32:add(nn.Dropout(0.5))
  enc_32_2:add(enc_32_2_2)
  enc_32_2:add(nn.Transpose({1, 2})) -- Transpose back to batch, seqlen
  enc_32_2:add(nn.Select(2, 32)) -- Get the encoded representation NxH
  latent_pt:add(enc_32_2)

  -- Latent Variable 3
  local enc_32_3_1 = self.latent_enc[1]:clone()
  local enc_32_3_2 = self.latent_enc[2]:clone()
  local enc_32_3 = nn.Sequential()
  enc_32_3:add(nn.Transpose({1, 2})) -- Transpose to seqlen x batch
  enc_32_3:add(enc_32_3_1)
  enc_32:add(nn.Dropout(0.5))
  enc_32_3:add(enc_32_3_2)
  enc_32_3:add(nn.Transpose({1, 2})) -- Transpose back to batch, seqlen
  enc_32_3:add(nn.Select(2, 32)) -- Get the encoded representation NxH
  latent_pt:add(enc_32_3)

  latent_seq:add(latent_pt)
  -- Input is 3 tables of NxH output should be Nx3H
  latent_seq:add(nn.JoinTable(2))
  latent_seq:add(nn.Linear(128*3, 32))
  latent_seq:add(nn.Replicate(26, 1))

  -- Add latent sequential to parallel table
  pt:add(latent_seq)

  final_model:add(pt)
  final_model:add(nn.JoinTable(3))

  -- Input to LSTM should be in 3D format (N, T, D)
  -- Add LSTM
  local lstm1 = nn.MyLSTM(128*self.num_features+32, 256)
  lstm1:set_name('lstm_32_1')
  lstm1.remember_states = false
  table.insert(self.rnns_32, lstm1)
  final_model:add(lstm1)
  final_model:add(nn.Dropout(0.5))
  local lstm2 = nn.MyLSTM(256, 256)
  lstm2:set_name('lstm_32_2')
  lstm2.remember_states = false
  table.insert(self.rnns_32, lstm2)
  final_model:add(lstm2)

  -- Select the last timestamp state
  final_model:add(nn.Select(2, 26))

  -- Add the final FC and classification layers.
  final_model:add(nn.Linear(256, 128))
  final_model:add(nn.ReLU())
  final_model:add(nn.Dropout(0.8))

  final_model:add(nn.Linear(128, self.num_classify))

  self.net = final_model
end

-- Use only 16 or 32 length timesteps
function MultiScaleLatentConvLSTM:getOneScaleConvLSTMModel()

  local final_model = self:get_32_2d_conv_model()

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

  final_model:add(nn.Linear(128, self.num_classify))

  self.net = final_model
end

-- This only uses the 16 and 32 length timestamps
function MultiScaleLatentConvLSTM:getTwoScaleConvLSTMModel()

  local final_model = nn.Sequential()
  local model = nn.ParallelTable()
  local m16 = self:get_16_2d_conv_model()
  model:add(m16)

  local m32 = self:get_32_2d_conv_model()
  model:add(m32)

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

function MultiScaleLatentConvLSTM:getConvLSTMModel()
  if self.use_label_correction == 1 then
    return self:getLabelCorrectionModel()
  end
  local latent_inp_size = G_global_opts.latent_num_features
  if self.latent_model == 'user_id' then
    return self:getUserIdLatentVariableConvModel{latent_inp_size=latent_inp_size}
  elseif self.latent_model == 'pose_threshold_context' then
    return self:getUserIdLatentVariableConvModel{latent_inp_size=latent_inp_size}
  elseif self.latent_model == 'pose_vel_hist_context' then
    --return self:getOneScaleConvLSTMModel()
    return self:getUserIdLatentVariableConvModel{latent_inp_size=latent_inp_size}
  elseif self.latent_model == 'supervised_context' then
    return self:getSupervisedContextLatentVariableConvModel{}
  elseif self.latent_model == 'lstm_encoder' then
    if self.use_long_term_latent_variable == 1 then
      print("Using Long term Latent variable model.")
      return self:getLongTermLatentVariableConvModel()
    end
  else
    assert(false)
  end

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

  final_model:add(nn.Linear(128, self.num_classify))

  self.net = final_model
end

function MultiScaleLatentConvLSTM:updateType(dtype)
  self.net = self.net:type(dtype)
end

function MultiScaleLatentConvLSTM:updateOutput(input)
  return self.net:forward(input)
end

function MultiScaleLatentConvLSTM:backward(input, gradOutput, scale)
  return self.net:backward(input, gradOutput, scale)
end

function MultiScaleLatentConvLSTM:parameters()
  return self.net:parameters()
end

function MultiScaleLatentConvLSTM:training()
  self.net:training()
  parent.training(self)
end

function MultiScaleLatentConvLSTM:evaluate()
  self.net:evaluate()
  parent.evaluate(self)
end

function MultiScaleLatentConvLSTM:resetStates()
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

function MultiScaleLatentConvLSTM:clearState()
  if self.latent_model == 'supervised_context' then
  else
    self.net:clearState()
  end
end

function MultiScaleLatentConvLSTM:getLatentInputSingleScaleGradWeights(loss, x, y)
  local grad_threshold = G_global_opts.grad_threshold
  local w1_16, grad_w1_16;
  if G_global_opts.train_user_classification == 1 then
    w1_16, grad_w1_16= self.net:get(1):get(1):parameters()
  else
    w1_16, grad_w1_16= self.net:get(3):parameters()
  end
  local grad_w1_16_f = grad_w1_16[1]:float()
  grad_w1_16 = torch.max(torch.abs(grad_w1_16_f))
  w16_th = torch.sum(torch.gt(grad_w1_16_f, grad_threshold))

  local max_w1_16, max_w1_32, max_w1_64
  max_w1_16 = torch.max(torch.abs(w1_16[1]:double()))
  local curr_grad_history = nil
  curr_grad_history = {
    max_w_16=max_w1_16, max_grad_w1_16=grad_w1_16,
    grad_w_16_gt_th=w16_th, total_w_16=grad_w1_16_f:nElement(),
  }
  if utils.isNan(max_w_16) or utils.isNan(grad_w1_16) then
    print("======================")
    print("Nan Detected")
    print("Loss "..loss)
    print(curr_grad_history)
    print(x[1])
    assert(false)
    torch.save('model_nan.t7', self:float())
    torch.save('x_1.t7', x[1]:float())
    torch.save('x_2_1.t7', x[2][1]:float())
    torch.save('x_2_2.t7', x[2][2]:float())
    torch.save('x_2_3.t7', x[2][3]:float())
    torch.save('grad_nan.t7', grad_w1_16_f)
    print("======================")
    assert(false)
  end

  return curr_grad_history
end

function MultiScaleLatentConvLSTM:getPoseVelHistContextGradWeights(loss, x, y)
  -- Assuming we have a parallel table that has a sequential (for now just an nn.View)
  -- and an nn.Identity which is then concatenated to input into LSTM
  local grad_threshold = G_global_opts.grad_threshold
  local w1_16, grad_w1_16 = 0, 0, 0, 0
  w1_16, grad_w1_16 = self.net:get(3):parameters()
  local grad_w1_16_f = grad_w1_16[1]:float()
  local w16_th = torch.sum(torch.gt(grad_w1_16_f, grad_threshold))
  grad_w1_16 = torch.max(torch.abs(grad_w1_16_f))
  local max_w1_16 = torch.max(torch.abs(w1_16[1]:double()))
  return {
    max_w_16=max_w1_16, max_grad_w1_16=grad_w1_16,
    grad_w_16_gt_th=w16_th, total_w_16=grad_w1_16_f:nElement(),
  }
end

-- This logs the gradients for the model created in 
-- `getSupervisedContextLatentVariableConvModel`
function MultiScaleLatentConvLSTM:getSupervisedContextGradWeights(loss, x, y)
  local grad_threshold = G_global_opts.grad_threshold
  local w1_16, grad_w1_16, w1_32, grad_w1_32 = 0, 0, 0, 0
  w1_16, grad_w1_16 = self.init_classif_model:get(1):parameters()
  w1_32, grad_w1_32 = self.init_context_model:get(1):parameters()

  local grad_w1_16_f = grad_w1_16[1]:float()
  local grad_w1_32_f = grad_w1_32[1]:float()

  local w16_th = torch.sum(torch.gt(grad_w1_16_f, grad_threshold))
  local w32_th = torch.sum(torch.gt(grad_w1_32_f, grad_threshold))

  grad_w1_16 = torch.max(torch.abs(grad_w1_16_f))
  grad_w1_32 = torch.max(torch.abs(grad_w1_32_f))

  local max_w1_16 = torch.max(torch.abs(w1_16[1]:double()))
  local max_w1_32 = torch.max(torch.abs(w1_32[1]:double()))

  return {
    max_w_classif=max_w1_16, max_grad_w_classif=grad_w1_16,
    grad_classif_gt_th=w16_th, total_w_classif=grad_w1_16_f:nElement(),
    max_w_context=max_w1_32, max_grad_w_context=grad_w1_32,
    grad_context_gt_th=w32_th, total_w_context=grad_w1_32:nElement(),
  }
end

function MultiScaleLatentConvLSTM:getGradWeights(loss, x, y)
  local grad_threshold = G_global_opts.grad_threshold
  if G_global_opts.num_scales == 1 and false then
    return self:getLatentInputSingleScaleGradWeights(loss, x, y)
  end
  if self.latent_model == 'pose_vel_hist_context' then
    return self:getPoseVelHistContextGradWeights(loss, x, y)
  end

  if self.latent_model == 'supervised_context' then
    return self:getSupervisedContextGradWeights(loss, x, y)
  end

  local w1_16, grad_w1_16, w1_32, grad_w1_32 = 0, 0, 0, 0
  if G_global_opts.num_scales == 1 then
    w1_16, grad_w1_16= self.net:get(1):parameters()
  else
    w1_16, grad_w1_16= self.net:get(1):get(1):get(1):parameters()
    w1_32, grad_w1_32 = self.net:get(1):get(2):get(1):parameters()
  end

  local w1_64, grad_w1_64 = 0, 0
  if self.num_scales == 3 then
    w1_64, grad_w1_64 = self.net:get(1):get(3):get(1):parameters()
  end

  local grad_w1_16_f, grad_w1_32_f, grad_w1_64_f = 0, 0, 0
  grad_w1_16_f = grad_w1_16[1]:float()
  if self.num_scales > 1 then grad_w1_32_f = grad_w1_32[1]:float() end
  if self.num_scales == 3 then grad_w1_64_f = grad_w1_64[1]:float() end

  grad_w1_16 = torch.max(torch.abs(grad_w1_16_f))
  if self.num_scales > 1 then grad_w1_32 = torch.max(torch.abs(grad_w1_32_f)) end
  if self.num_scales == 3 then grad_w1_64 = torch.max(torch.abs(grad_w1_64_f)) end

  local w16_th, w32_th, w64_th
  w16_th = torch.sum(torch.gt(grad_w1_16_f, grad_threshold))
  if self.num_scales > 1 then w32_th = torch.sum(torch.gt(grad_w1_32_f, grad_threshold)) end
  if self.num_scales == 3 then w64_th = torch.sum(torch.gt(grad_w1_64_f, grad_threshold)) end

  local max_w1_16, max_w1_32, max_w1_64
  max_w1_16 = torch.max(torch.abs(w1_16[1]:double()))
  if self.num_scales > 1 then max_w1_32 = torch.max(torch.abs(w1_32[1]:double())) end
  if self.num_scales == 3 then max_w1_64 = torch.max(torch.abs(w1_64[1]:double())) end

  local curr_grad_history = nil
  if self.num_scales == 1 then
    curr_grad_history = {
      max_w_16=max_w1_16, max_grad_w1_16=grad_w1_16,
      grad_w_16_gt_th=w16_th, total_w_16=grad_w1_16_f:nElement(),
    }
    if utils.isNan(max_w_16) or utils.isNan(grad_w1_16) then
      print("======================")
      print("Nan Detected")
      print("Loss "..loss)
      print(curr_grad_history)
      print(x[1])
      assert(false)
      torch.save('model_nan.t7', model:float())
      torch.save('x_1.t7', x[1]:float())
      torch.save('x_2_1.t7', x[2][1]:float())
      torch.save('x_2_2.t7', x[2][2]:float())
      torch.save('x_2_3.t7', x[2][3]:float())
      torch.save('grad_nan.t7', grad_w1_16_f)
      print("======================")
      assert(false)
    end
  elseif self.num_scales == 2 then
    curr_grad_history = {
      max_w_16=max_w1_16,
      max_w_32=max_w1_32,
      max_grad_w1_16=grad_w1_16,
      max_grad_w1_32=grad_w1_32,
      grad_w_16_gt_th=w16_th,
      grad_w_32_gt_th=w32_th,
      total_w_16=grad_w1_16_f:nElement(),
      total_w_32=grad_w1_32_f:nElement(),
    }
  else
    curr_grad_history = {
      max_w_16=max_w1_16,
      max_w_32=max_w1_32,
      max_w_64=max_w1_64,
      max_grad_w1_16=grad_w1_16,
      max_grad_w1_32=grad_w1_32,
      max_grad_w1_64=grad_w1_64,
      grad_w_16_gt_th=w16_th,
      grad_w_32_gt_th=w32_th,
      grad_w_64_gt_th=w64_th,
      total_w_16=grad_w1_16_f:nElement(),
      total_w_32=grad_w1_32_f:nElement(),
      total_w_64=grad_w1_64_f:nElement()
    }
  end

  return curr_grad_history
end

