require 'torch'
require 'nn'
require 'rnn';
require 'optim'
require 'xlua'

-- local vanilla_model = require 'model.VanillaLSTM'
require 'model.VanillaLSTM'
require 'util.LatentVariableDataLoader'
require 'model.MultiScaleLatentConvLSTM'
require 'model.MultiScaleSupervisedContextModel'
require 'util.DenseSlidingWindowDataLoader'

local utils = require 'util.utils'
local LSTMAutoEncoder = require 'model.LSTMAutoEncoder'

require('mobdebug').start()

local dtype = 'torch.FloatTensor'

local cmd = torch.CmdLine()

cmd:option('-desc', '')
-- Dataset options
cmd:option('-use_openface_features', 1)
cmd:option('-input_h5', '../../openface_data/mohit_data.h5')
cmd:option('-train_seq_h5', '../../openface_data/main_gest_by_file.h5')
cmd:option('-data_dir', '../../openface_data/face_gestures/dataseto_text')
cmd:option('-aug_gests_h5', '../../openface_data/main_gest_by_file_aug_K_32.h5')
cmd:option('-cpm_h5_dir', '../../openface_data/cpm_output')
cmd:option('-aug_cpm_h5', '')
cmd:option('-use_cpm_features', 1)
cmd:option('-openface_mean_h5', '../../openface_data/mean_std/openface_mean_std_correct.h5')
cmd:option('-cpm_mean_h5', '../../openface_data/mean_std/cpm_mean_std.h5')
cmd:option('-use_zface_features', 1)
cmd:option('-zface_h5_dir', '../../data_zface/filtered_headpose')
cmd:option('-aug_zface_h5', '')
cmd:option('-zface_mean_h5', '../../data_zface/mean_std_cache.h5')
cmd:option('-batch_size', 150)
cmd:option('-num_classes', 11)
cmd:option('-num_classify', 5)
cmd:option('-win_len', 10)
cmd:option('-win_step', 1)
cmd:option('-num_features', 50)
cmd:option('-latent_model', 'pose_vel_hist_context')
cmd:option('-supervised_context_num_classify', 3)
cmd:option('-supervised_context_type', 'user_id')
cmd:option('-latent_num_features', 136)
cmd:option('-latent_variable_aug_h5', 
  '../../openface_data/gest_data_38_2/aug_openface_full_autoencoder_step_3_win_all.h5')
cmd:option('-latent_mean_h5', '../../openface_data/gest_data_38_2/mean_std.h5')
cmd:option('-latent_user_id', 1)
cmd:option('-train_user_classification', 0)
cmd:option('-use_all_face_diff', 0)
cmd:option('-openface_trimmed_aug', 1)
-- If 1 next_val_batch should return the batch data as well to save.
cmd:option('-val_batch_info', 1)

-- Model options
cmd:option('-init_from', '')
cmd:option('-reset_iterations', 1)
cmd:option('-model_type', 'lstm')
cmd:option('-rnn_size', 128)
cmd:option('-num_layers', 2)
cmd:option('-dropout', 0.3)
cmd:option('-batchnorm', 0)
cmd:option('-use_dense_conv_lstm', 0)
cmd:option('-num_scales', 1)
cmd:option('-use_48_scale', 0)
cmd:option('-use_opt_flow', 0)
cmd:option('-finetune', 0)
cmd:option('-finetune_ratio', 0.2)
cmd:option('-use_long_term_latent_variable', 0)
-- The pre-trained model for the latent variable model
cmd:option('-latent_variable_model', '')
cmd:option('-use_label_correction', 0)
cmd:option('-coefL1', 0)
cmd:option('-coefL2', 0)
cmd:option('-use_bootstrap_Beta', 0)
cmd:option('-use_bootstrap_Beta_var', 0)
cmd:option('-bootstrap_Beta', 0.8)
cmd:option('-coef_beta_reg', 0.5)
cmd:option('-bootstrap_Beta_var_opt_together', 0)  -- Optimize y_hat, Beta predictions together
cmd:option('-save_coef_beta', 1)

-- Optimization options
cmd:option('-max_epochs', 200)
cmd:option('-learning_rate', 1e-3)
cmd:option('-grad_clip', 10)
-- For Adam people don't usually decay the learning rate
cmd:option('-lr_decay_every', 20)  -- Decay every n epochs
cmd:option('-lr_decay_factor', 0.5)

-- Output options
cmd:option('-save', 'dense_step_5_cls_5')
cmd:option('-print_every', 75)-- Print every n batches
cmd:option('-checkpoint_every', 1)  -- Checkpoint after every n epochs
cmd:option('-checkpoint_name', 'checkpoint')
cmd:option('-validate_every_batches', 75) -- Run on validation data ever n batches
cmd:option('-train_log', 'train.log')
cmd:option('-test_log', 'test.log')
cmd:option('-debug_weights', 1)
cmd:option('-grad_threshold', 0.00001)
cmd:option('-test_batch', 'test_batch.json')
cmd:option('-test_scores', 'test_scores.json')
cmd:option('-test_preds', 'test_preds.json')
cmd:option('-test_beta', 'test_beta.json')
cmd:option('-save_test_data_stats', 1)

-- Benchmark options
cmd:option('-speed_benchmark', 0)
cmd:option('-memory_benchmark', 0)

-- Backend options
cmd:option('-gpu', 1)
cmd:option('-gpu_backend', 'cuda')

local opt = cmd:parse(arg)
opt.checkpoint_name = opt.save..'/'..opt.checkpoint_name
opt.test_scores = opt.save .. '/' .. opt.test_scores
opt.test_preds = opt.save .. '/' .. opt.test_preds
opt.test_batch = opt.save .. '/' .. opt.test_batch
opt.test_beta = opt.save .. '/' .. opt.test_beta

local trainLogger = optim.Logger(paths.concat(opt.save, opt.train_log))
if opt.use_dense_conv_lstm == 1 then
  trainLogger:setNames{'train_err', 'train_loss', 'l_inf_w1_16',
    'l_inf_w1_32', 'l_inf_w1_64'}
else
  trainLogger:setNames{'train_err', 'train_loss', 'l_inf_w1_16',
    'l_inf_w1_32', 'l_inf_w1_64'}
end

if opt.use_openface_features ~= 1 then
  opt.num_features = 0
end

if opt.use_cpm_features == 1 then
  -- For now we use 6 ConvPoseMachine features
  opt.num_features = opt.num_features + 10
end

if opt.use_zface_features == 1 then
  -- These are the filtered features center (X, y) and pose
  opt.num_features = opt.num_features + 9
end

if opt.use_all_face_diff == 1 then
  -- Outer face(17), eyes(4), 1(above nosetip)
  opt.num_features = opt.num_features + 2*17 + 2*4 + 2
end

assert(opt.num_features > 0)

if opt.gpu == 1 then
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  cutorch.setDevice(1)
  dtype = 'torch.CudaTensor'
  print(string.format('Running with CUDA on GPU 1'))
else
  print('Running in CPU mode')
end

-- Confusion matrix
local classes = {}
for i=1,opt.num_classify do table.insert(classes, i) end
-- Confusion matrix for train data
local confusion = optim.ConfusionMatrix(classes)

local opt_clone = torch.deserialize(torch.serialize(opt))
-- GLobal options variable is a global
G_global_opts = opt_clone
print(G_global_opts)

-- Initialize DataLoader
local num_batch_epochs = 5
local loader = LatentVariableDataLoader(opt_clone)
local model = torch.load(opt_clone.init_from)
model = model.model

-- Remove layers used for loss computation

if opt.latent_model == 'supervised_context' then
  model:updateType('torch.CudaTensor')
else model = model:cuda() end

local start_i = 0
local weights = torch.Tensor({1,1,5,3,5})

local N, T = opt.batch_size, opt.win_len
local train_loss_history, mse_loss_history = {}, {}
local bootstrap_loss_2 = {}
local val_loss_history = {}
local val_loss_history_it = {}
local forward_backward_times = {}
local init_memory_usage, memory_usage = nil, {}
-- Data_co could contain the training or validation data
-- as required. This is initialize below.
local data_co = nil
-- Stores the history (l1,l2, l-inf) norm for gradients of the first layer
-- This serves to check that the models are learning. We also store the
-- number of gradients which are less than a threshold.
local grad_counter = 1
local grads_history = {}
local grad_threshold = 1e-5

-- Train the model

local train_cls
if opt.use_label_correction == 1 then train_cls = require 'layer/train_labelCorrection'
elseif opt.use_bootstrap_Beta == 1 then train_cls = require 'layer/train_bootstrap'
elseif opt.use_bootstrap_Beta_var == 1 then train_cls = require 'layer/train_bootstrap_var'
elseif opt.latent_model == 'supervised_context' then train_cls = require 'layer/train_supervised_context'
else train_cls = require 'layer/train_normal' end
train_cls.setup{
  dtype=dtype,
  model=model,
  val_conf=confusion,
  data_loader=loader,
  grads_history=grads_history
}

data_co = coroutine.create(LatentVariableDataLoader.next_val_batch)
print('Using Sliding window loader')
curr_batches_processed = 0
-- Zero confusion matrix for next loop
confusion:zero()
local test_data_stats = train_cls.validate_save(data_co)

print(confusion)
-- Save the test batch data
paths.mkdir(paths.dirname(opt.test_batch))
utils.write_json(opt.test_batch, test_data_stats.test_data)

-- Save the scores
paths.mkdir(paths.dirname(opt.test_scores))
utils.write_json(opt.test_scores, test_data_stats.test_scores)

-- Save the predictions
paths.mkdir(paths.dirname(opt.test_preds))
utils.write_json(opt.test_preds, test_data_stats.test_preds)

if opt.latent_model == 'supervised_context' then
  utils.write_json(opt.save..'/'..'test_scores_context.json',
      test_data_stats.test_scores_context)
  utils.write_json(opt.save..'/'..'test_preds_context.json',
      test_data_stats.test_preds_context)
else
  -- Save betas
  paths.mkdir(paths.dirname(opt.test_beta))
  utils.write_json(opt.test_beta, test_data_stats.test_beta)
end

