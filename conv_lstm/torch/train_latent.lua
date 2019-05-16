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
cmd:option('-win_step', 3)
cmd:option('-num_features', 50)
cmd:option('-latent_model', 'supervised_context')
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
cmd:option('-coef_beta_reg', 1.0)
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

-- Benchmark options
cmd:option('-speed_benchmark', 0)
cmd:option('-memory_benchmark', 0)

-- Backend options
cmd:option('-gpu', 1)
cmd:option('-gpu_backend', 'cuda')

local opt = cmd:parse(arg)
opt.checkpoint_name = opt.save..'/'..opt.checkpoint_name
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
-- Confusion matrix for validation data
local val_conf = optim.ConfusionMatrix(classes)

local opt_clone = torch.deserialize(torch.serialize(opt))
-- GLobal options variable is a global
G_global_opts = opt_clone
print(G_global_opts)

-- Initialize DataLoader
--local loader = DataLoader(opt)
--local batch_loader = BatchLoader(opt_clone)
--batch_loader:load_data()
local num_batch_epochs = 5
--batch_loader:init_batch(num_batch_epochs)
local loader;
-- Loading data happens implicity in the cons
--loader:load_data()

loader = LatentVariableDataLoader(opt_clone)
local model = nn.MultiScaleLatentConvLSTM(opt_clone)
if opt.latent_model == 'supervised_context' then
  model = nn.MultiScaleSupervisedContextModel(opt_clone)
end
model:getConvLSTMModel()
if opt.latent_model == 'supervised_context' then
  print(model.init_context_model)
  print(model.init_classif_model)
  print(model.final_classif_model)
  print(model.final_context_model)
else 
  print(model.net)
  model = model:cuda()
end

if opt.init_from:len() > 0 then 
  model = torch.load(opt_clone.init_from)
  model = model.model
  if opt.use_label_correction == 1 then
    model.reconsLayer =  nn.ReconsLayer(opt_clone.num_train_frames, 
        opt_clone.batch_size, opt_clone.num_classify)
    model.net:add(model.reconsLayer)
  end
  model = model:cuda()
end

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
local optim_config = {learningRate = opt.learning_rate}
model:training()

-- For display purposes only
local curr_batches_processed = 1
local total_train_batches = loader:getTotalTrainBatches()

local train_cls
if opt.use_label_correction == 1 then train_cls = require 'layer/train_labelCorrection'
elseif opt.use_bootstrap_Beta == 1 then train_cls = require 'layer/train_bootstrap'
elseif opt.use_bootstrap_Beta_var == 1 then train_cls = require 'layer/train_bootstrap_var'
elseif opt.latent_model == 'supervised_context' then train_cls = require 'layer/train_supervised_context'
else train_cls = require 'layer/train_normal' end
train_cls.setup{
  dtype=dtype,
  model=model,
  confusion=confusion,
  val_conf=val_conf,
  data_loader=loader,
  grads_history=grads_history
}
local params, grad_params = train_cls.params, train_cls.grad_params

function run_on_val_data()
  model:evaluate()
  model:resetStates()

  local val_data_co
  if opt.use_dense_conv_lstm == 1 then
    val_data_co = coroutine.create(
      DenseSlidingWindowDataLoader.next_val_batch)
  else
    val_data_co = coroutine.create(
      LatentVariableDataLoader.next_val_batch)
  end
  
  train_cls.validate(val_data_co)
  collectgarbage()
end


for i = 1, opt.max_epochs do
  if opt.use_dense_conv_lstm == 1 then 
    data_co = coroutine.create(DenseSlidingWindowDataLoader.next_train_batch)
    print('Using Dense sliding window loader')
  else
    data_co = coroutine.create(LatentVariableDataLoader.next_train_batch)
    print('Using Sliding window loader')
  end
  curr_batches_processed = 0
  -- Zero confusion matrix for next loop
  confusion:zero()
  
  -- Go through the entire train batch
  while coroutine.status(data_co) ~= 'dead' do
    --xlua.progress(curr_batches_processed, total_train_batches)

    if curr_batches_processed < total_train_batches then
      -- Train
      local loss = train_cls.train(data_co, optim_config,
        {
          curr_epoch=i,
          total_epoch=opt.max_epochs,
          curr_batch=curr_batches_processed,
          total_batch=total_train_batches
        })

      -- Print every few thousand iterations?
      if (opt.print_every > 0 and
          curr_batches_processed > 0 and
          curr_batches_processed % opt.print_every == 0) then
        local float_epoch = i
        local msg = 'Epoch %.2f, total epochs:%d, loss = %f'
        local args = {msg, float_epoch, opt.max_epochs, loss[1]}
        print(string.format(unpack(args)))
        print('Gradient weights for the last batch')
        print(grads_history[#grads_history])
      end

      if (opt.validate_every_batches > 0 and 
        curr_batches_processed > 0 and
        curr_batches_processed % opt.validate_every_batches == 0) then
        run_on_val_data()
      end
      curr_batches_processed = curr_batches_processed + 1
    else
      -- Separate this since we don't want to add 0 loss to the train history.
      local success, x = coroutine.resume(data_co, loader)
      assert(coroutine.status(data_co) == 'dead')
    end

  end

  -- One epoch done
  model:resetStates() -- Reset initial hidden states of LSTM

  -- Print confusion matrix stats
  -- print_conf_matrix_stats(confusion, classes)
  print('Epoch complete, confusion matrix for Train data:')
  print(confusion)

  -- Decay learning rate
  if i % opt.lr_decay_every == 0 then
    local old_lr = optim_config.learningRate
    optim_config = {learningRate = old_lr * opt.lr_decay_factor}
  end


  -- TODO(Mohit): Save a checkpoint
  local check_every = opt.checkpoint_every
  if (check_every > 0 and i % check_every == 0) then
    -- Evaluate loss on the validation set. Note that we reset the state of
    -- the model; this might happen in the middle of an epoch, but that
    -- shouldn't cause too much trouble.
    model:evaluate()
    model:resetStates()
    -- Set the dataloader to load validation data
    local val_data_co
    if opt.use_dense_conv_lstm then
      val_data_co = coroutine.create(
        DenseSlidingWindowDataLoader.next_val_batch)
    else
      val_data_co = coroutine.create(
        LatentVariableDataLoader.next_val_batch)
    end
    train_cls.validate(val_data_co)

    -- First save a JSON checkpoint, excluding the model
    local checkpoint = {
      opt = opt,
      desc = opt.desc,
      train_loss_history = train_loss_history,
      mse_loss_history = mse_loss_history,
      val_loss_history = val_loss_history,
      val_loss_history_it = val_loss_history_it,
      forward_backward_times = forward_backward_times,
      conf = val_conf:__tostring__(),
      train_conf = confusion:__tostring__(),
      epoch = i
    }
    -- Add checkpoint items
    for k,v in pairs(train_cls.getCheckpoint()) do checkpoint[k] = v end

    local filename = string.format('%s_%d.json', opt.checkpoint_name, i)
    -- Make sure the output directory exists before we try to write it
    paths.mkdir(paths.dirname(filename))
    utils.write_json(filename, checkpoint)

    print("DID SAVE ====>  "..filename)

    local grads_filename = string.format('%s/model_grads.json', opt.save)
    utils.write_json(grads_filename, grads_history)


    -- Now save a torch checkpoint with the model
    -- Cast the model to float before saving so it can be used on CPU
    model:clearState()
    model:updateType('torch.FloatTensor')
    checkpoint.model = model
    local filename = string.format('%s_%d.t7', opt.checkpoint_name, i)
    paths.mkdir(paths.dirname(filename))
    torch.save(filename, checkpoint)
    model:updateType(dtype)
    if opt.latent_model == 'supervised_context' then
      train_cls:get_grad_params()
    else
      train_cls.params, train_cls.grad_params = model:getParameters()
    end
    collectgarbage()

  end
end

