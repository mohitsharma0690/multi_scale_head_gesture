require 'torch'
require 'nn'
require 'rnn'
require 'loadcaffe'

require 'model/MyLSTM'
require 'model/MS_Embedding'

local utils = require 'util.utils'

local MultiScaleSupervisedContextModel, parent = torch.class('nn.MultiScaleSupervisedContextModel', 'nn.Module')

function MultiScaleSupervisedContextModel:__init(kwargs)
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

  self.latent_model = utils.get_kwarg(kwargs, 'latent_model')

  if self.use_label_correction == 1 then
    self.num_train_frames = utils.get_kwarg(kwargs, 'num_train_frames')
  end

  self.curr_win_sizes = {16, 32, 64}
  self.start_frame = 36
  self.rnns_16 = {}
  self.rnns_32 = {}
  self.rnns_64 = {}
  
  self.all_models = {}
end

function MultiScaleSupervisedContextModel:get_16_2d_conv_model()
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

function MultiScaleSupervisedContextModel:get_32_2d_conv_model()
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

function MultiScaleSupervisedContextModel:get_48_2d_conv_model()
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

function MultiScaleSupervisedContextModel:get_64_2d_conv_model()
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
function MultiScaleSupervisedContextModel:getSupervisedContextLatentVariableConvModel()
  self.init_classif_model = self:get_32_2d_conv_model()

  -- Build the context sequential first 
  local final_time_dim = 3
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
  local num_output = G_global_opts.supervised_context_num_classify
  final_context_model:add(nn.Linear(256, 128))
  final_context_model:add(nn.Linear(128, num_output))
  self.final_context_model = final_context_model

  return final_model
end

function MultiScaleSupervisedContextModel:getConvLSTMModel()
  self:getSupervisedContextLatentVariableConvModel()
  table.insert(self.all_models, self.init_classif_model)
  table.insert(self.all_models, self.init_context_model)
  table.insert(self.all_models, self.final_classif_model)
  table.insert(self.all_models, self.final_context_model)
end

function MultiScaleSupervisedContextModel:updateType(dtype)
  for k, model in pairs(self.all_models) do
    model:type(dtype)
  end
end

function MultiScaleSupervisedContextModel:updateOutput(input)
  assert(false)
end

function MultiScaleSupervisedContextModel:backward(input, gradOutput, scale)
  assert(false)
end

function MultiScaleSupervisedContextModel:parameters()
  assert(false)
end

function MultiScaleSupervisedContextModel:training()
  for k, model in pairs(self.all_models) do model:training() end
  parent.training(self)
end

function MultiScaleSupervisedContextModel:evaluate()
  for k, model in pairs(self.all_models) do model:evaluate() end
  parent.evaluate(self)
end

function MultiScaleSupervisedContextModel:resetStates()
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

function MultiScaleSupervisedContextModel:float()
  for k, model in pairs(self.all_models) do model:float() end
end

function MultiScaleSupervisedContextModel:clearState()
  self.init_classif_model:clearState()
  self.init_context_model:clearState()
  self.final_classif_model:clearState()
  self.final_context_model:clearState()
end


-- This logs the gradients for the model created in 
-- `getSupervisedContextLatentVariableConvModel`
function MultiScaleSupervisedContextModel:getSupervisedContextGradWeights(loss, x, y)
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
    grad_context_gt_th=w32_th, total_w_context=grad_w1_32_f:nElement(),
  }
end

function MultiScaleSupervisedContextModel:getGradWeights(loss, x, y)
  return self:getSupervisedContextGradWeights(loss, x, y)
end

