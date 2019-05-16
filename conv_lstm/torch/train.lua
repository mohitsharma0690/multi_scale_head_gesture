require 'torch'
require 'nn'
require 'optim'
require 'xlua'

-- local vanilla_model = require 'model.VanillaLSTM'
require 'model.VanillaLSTM'
require 'util.DataLoader'
require 'util.BatchLoader'
require 'util.SlidingWindowDataLoader'
require 'util.KFoldSlidingWindowDataLoader'
require 'util.DenseSlidingWindowDataLoader'
require 'util.MultiScaleSpatialDataLoader'
require 'model.ConvLSTM'
require 'model.DenseConvLSTM'
require 'model.MultiScaleCNN'
require 'model.MultiScaleConvLSTM'
require 'model.MultiScaleConvBLSTM'
require 'model.MultiScaleSpatialConvLSTM'

local utils = require 'util.utils'

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
cmd:option('-num_features', 40)
cmd:option('-use_all_face_diff', 0)
cmd:option('-openface_trimmed_aug', 1)
cmd:option('-use_context', 0)
cmd:option('-test_all', 0, 'Test all frames i.e. no n/5 skipping')
cmd:option('-train_one_vs_all', 0,
    'If non zero the classifier is trained for 1-vs-all classification that class')

-- Model options
cmd:option('-init_from', '')
cmd:option('-reset_iterations', 1)
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
cmd:option('-model_type', '')
cmd:option('-finetune_new_dataset', 0)

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
  opt.num_features = opt.num_features + 6
end

if opt.use_all_face_diff == 1 then
  -- Outer face(17), eyes(4), 1(above nosetip)
  opt.num_features = opt.num_features + 2*17 + 2*4 + 2
end

if opt.model_type == 'multi_scale_spatial_conv_lstm' then
  -- HACK FOR NOW
  opt.num_features = 72
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
if opt.train_one_vs_all > 0 then classes = {1, 2} end
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

local model = nil
if opt.use_dense_conv_lstm == 1 then
  loader = DenseSlidingWindowDataLoader(opt_clone)
  model = nn.DenseConvLSTM(opt_clone)
  model:getDenseConvLSTMModel()
  model:updateType(dtype)
elseif opt.model_type == 'multi_scale_spatial_conv_lstm' then
  loader = MultiScaleSpatialDataLoader(opt_clone)
  model = nn.MultiScaleSpatialConvLSTM(opt_clone)
  model:getConvLSTMModel()
  model:updateType(dtype)
elseif opt.model_type == 'stacked_lstm' then
  loader = KFoldSlidingWindowDataLoader(opt_clone)
  model = nn.MultiScaleConvLSTM(opt_clone)
  model:getStackedLSTMModel()
  model:updateType(dtype)
else
  loader = KFoldSlidingWindowDataLoader(opt_clone)
  --model = nn.ConvLSTM(opt_clone)
  --model = nn.MultiScaleCNN(opt_clone)
  model = nn.MultiScaleConvLSTM(opt_clone)
  --model = nn.MultiScaleConvBLSTM(opt_clone)
  model:getConvLSTMModel()
  model:updateType(dtype)
end

if opt.init_from:len() > 0 then 
  model = torch.load(opt_clone.init_from)
  model = model.model
  model = model:cuda()
end

local start_i = 0
local weights = torch.Tensor({1,1,5,3,5})
local crit = nn.CrossEntropyCriterion():type(dtype)
local params, grad_params = model:getParameters()

local N, T = opt.batch_size, opt.win_len
local train_loss_history = {}
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

-- This function is for the non-dense conv lstm model
local function save_grad_weights_dense(model, loss) 
  local w1_16, grad_w1_16 = model.net:get(1):get(1):get(1):get(1):parameters()
  local w1_32, grad_w1_32 = model.net:get(1):get(1):get(2):get(1):parameters()
  local w1_64, grad_w1_64 = model.net:get(1):get(1):get(3):get(1):parameters()
  local grad_w1_16_f = grad_w1_16[1]:float()
  local grad_w1_32_f = grad_w1_32[1]:float()
  local grad_w1_64_f = grad_w1_64[1]:float()
  grad_w1_16 = torch.max(torch.abs(grad_w1_16_f))
  grad_w1_32 = torch.max(torch.abs(grad_w1_32_f))
  grad_w1_64 = torch.max(torch.abs(grad_w1_64_f))
  local w16_th = torch.sum(torch.gt(grad_w1_16_f, grad_threshold))
  local w32_th = torch.sum(torch.gt(grad_w1_32_f, grad_threshold))
  local w64_th = torch.sum(torch.gt(grad_w1_64_f, grad_threshold))
  local curr_grad_history = {
    max_w1_16=grad_w1_16, 
    max_w1_32=grad_w1_32,
    max_w1_64=grad_w1_64,
    w_16_gt_th=w16_th,
    w_32_gt_th=w32_th,
    w_64_gt_th=w64_th,
    total_w1_16=grad_w1_16_f:nElement(),
    total_w_32=grad_w1_32_f:nElement(),
    total_w_64=grad_w1_64_f:nElement()
  }
  --[[
  print('======')
  print(curr_grad_history)
  print('======')
  ]]
  table.insert(grads_history, curr_grad_history)

  -- Add loss etc. to train logger (note accuracy is 0 for now)
  trainLogger:add{loss, 0.0, grad_w1_16, grad_w1_32, grad_w1_64}
end

-- This function is for the non-dense conv lstm model
local function save_grad_weights(model, loss) 
  --print(model.net:get(1):get(1):get(9):get(1):get(1))
  --local w1_16, grad_w1_16 = model.net:get(1):get(1):get(9):get(1):get(1):get(1):parameters()
  --local w1_32, grad_w1_32 = model.net:get(1):get(1):get(9):get(1):get(1):get(1):parameters()
  local w1_16, grad_w1_16
  local w1_32, grad_w1_32 = 0, 0
  if opt.num_scales == 1 then
    w1_16, grad_w1_16= model.net:get(1):parameters()
    if opt.model_type == 'stacked_lstm' then
      -- First layer is nn.View
      w1_16, grad_w1_16= model.net:get(2):parameters()
    end
  elseif opt.num_scales > 1 then 
    w1_16, grad_w1_16= model.net:get(1):get(1):get(1):parameters()
    w1_32, grad_w1_32 = model.net:get(1):get(2):get(1):parameters()
  end
  local w1_64, grad_w1_64 = 0, 0
  if opt.num_scales > 2 then 
    w1_64, grad_w1_64 = model.net:get(1):get(3):get(1):parameters()
  end

  --print(model.net:get(1):get(1):get(9):get(1):get(1):get(1))
  --print(model.net:get(1):get(1):get(9):get(1):get(1):get(1):parameters())
  local grad_w1_16_f, grad_w1_32_f, grad_w1_64_f = 0, 0, 0
  grad_w1_16_f = grad_w1_16[1]:float()
  if opt.num_scales > 1 then grad_w1_32_f = grad_w1_32[1]:float() end
  if opt.num_scales > 2 then grad_w1_64_f = grad_w1_64[1]:float() end

  grad_w1_16 = torch.max(torch.abs(grad_w1_16_f))
  if opt.num_scales > 1 then grad_w1_32 = torch.max(torch.abs(grad_w1_32_f)) end
  if opt.num_scales > 2 then grad_w1_64 = torch.max(torch.abs(grad_w1_64_f)) end

  local w16_th, w32_th, w64_th
  w16_th = torch.sum(torch.gt(grad_w1_16_f, grad_threshold))
  if opt.num_scales > 1 then w32_th = torch.sum(torch.gt(grad_w1_32_f, grad_threshold)) end
  if opt.num_scales > 2 then w64_th = torch.sum(torch.gt(grad_w1_64_f, grad_threshold)) end

  local max_w1_16, max_w1_32, max_w1_64 = 0, 0, 0
  max_w1_16 = torch.max(torch.abs(w1_16[1]:double()))
  if opt.num_scales > 1 then max_w1_32 = torch.max(torch.abs(w1_32[1]:double())) end
  if opt.num_scales > 2 then max_w1_64 = torch.max(torch.abs(w1_64[1]:double())) end

  local curr_grad_history = nil
  if opt.num_scales == 1 then
    curr_grad_history = {
      max_w_16=max_w1_16, max_grad_w1_16=grad_w1_16, 
      grad_w_16_gt_th=w16_th, total_w_16=grad_w1_16_f:nElement(),
    }
  elseif opt.num_scales == 2 then
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

  --[[
  print('======')
  print(curr_grad_history)
  print('======')
  ]]
  table.insert(grads_history, curr_grad_history)

  -- Add loss etc. to train logger (note accuracy is 0 for now)
  trainLogger:add{loss, 0.0, grad_w1_16, grad_w1_32, grad_w1_64}
end

local function f(w)
  assert(w == params)
  grad_params:zero()

  -- Get a minibach and run the model forward
  local success, x, y = coroutine.resume(data_co, loader)
  if not success then print('Data couroutine returns fail.') end
  --x, y = x:type(dtype), y:type(dtype)
  if x == nil then
    print('x is nil returning 0 loss')
    return 0, grad_params
  elseif success == false then
    -- print crash logs
    if torch.isTensor(x) then 
      print(x:size())
    else
      print(x)
    end
  end

  for i=1,#x do x[i] = x[i]:type(dtype) end
  y = y:type(dtype)
  if opt.num_scales == 1 then x = x[1] end

  local timer
  if opt.speed_benchmark ==1 then
    if cutorch then cutorch.synchronize() end
    timer = torch.Timer()
  end

  local scores = model:forward(x)

  -- update confusion
  confusion:batchAdd(scores, y)

  -- Use the criterion to compute loss
  -- TODO(Mohit): Maybe you need to change this
  local loss = crit:forward(scores, y)

  -- Run the criterion and model backward to compute gradients
  -- TODO(Mohit): This also needs fixing
  local grad_scores = crit:backward(scores, y)
  model:backward(x, grad_scores)


  if opt.speed_benchmark == 1 then
    if cutorch then cutorch.synchronize() end
    local time = timer:time().real
    print('Forward / Backward pass took ', time)
    table.insert(forward_backward_times, time)
  end

  -- TODO(Mohit): Record memory usage

  -- TODO(Mohit): Clip the gradients as required.
  if opt.grad_clip > 0 then
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end

  if opt_clone.debug_weights == 1 then
    if opt_clone.use_dense_conv_lstm == 0 then
      save_grad_weights(model, loss)
    else
      save_grad_weights_dense(model, loss)
    end
  end

  return loss, grad_params

end

-- Train the model
local optim_config = {learningRate = opt.learning_rate}
model:training()

-- For display purposes only
local curr_batches_processed = 1
local total_train_batches = loader:getTotalTrainBatches()

function run_on_val_data()
  model:evaluate()
  model:resetStates()

  local  val_data_co = coroutine.create(
    KFoldSlidingWindowDataLoader.next_val_batch)
  
  val_conf:zero()

  local val_loss = 0
  local num_val = 0

  while coroutine.status(val_data_co) ~= 'dead' do
    local success, xv, yv = coroutine.resume(val_data_co, loader) 
    if success and xv ~= nil then
      for i=1,#xv do xv[i] = xv[i]:type(dtype) end
      yv = yv:type(dtype)
      if opt.num_scales == 1 then xv = xv[1] end
      local scores = model:forward(xv)
      val_loss = val_loss + crit:forward(scores, yv)
      val_conf:batchAdd(scores, yv)
      num_val = num_val + 1
    elseif success ~= true then
      print('Validation data coroutine failed')
      print(xv)
    end
  end

  val_loss = val_loss / num_val
  print('val_loss = ', val_loss)
  table.insert(val_loss_history, val_loss)
  table.insert(val_loss_history_it, i)
  print(val_conf)
  model:resetStates()
  model:training()
  collectgarbage()
end

for i = 1, opt.max_epochs do
  if opt.use_dense_conv_lstm == 1 then 
    data_co = coroutine.create(DenseSlidingWindowDataLoader.next_train_batch)
    print('Using Dense sliding window loader')
  else
    data_co = coroutine.create(KFoldSlidingWindowDataLoader.next_train_batch)
    print('Using Sliding window loader')
  end
  curr_batches_processed = 0
  -- Zero confusion matrix for next loop
  confusion:zero()
  
  -- Go through the entire train batch
  while coroutine.status(data_co) ~= 'dead' do
    xlua.progress(curr_batches_processed, total_train_batches)

    if curr_batches_processed < total_train_batches then
      -- Take a gradient step and print
      local _, loss = optim.adam(f, params, optim_config)
      table.insert(train_loss_history, loss[1])

      -- TODO(Mohit): Print every few thousand iterations?
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

      if (curr_batches_processed > 0 and opt.validate_every_batches > 0 and 
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
    local val_data_co = coroutine.create(
        KFoldSlidingWindowDataLoader.next_val_batch)

    val_conf:zero()
    local val_loss = 0
    local num_val = 0
    while coroutine.status(val_data_co) ~= 'dead' do
      local success, xv, yv = coroutine.resume(val_data_co, loader) 
      if success and xv ~= nil then
        for i=1,#xv do xv[i] = xv[i]:type(dtype) end
        yv = yv:type(dtype)
        if opt.num_scales == 1 then xv = xv[1] end
        local scores = model:forward(xv)
        val_loss = val_loss + crit:forward(scores, yv)
        val_conf:batchAdd(scores, yv)
        num_val = num_val + 1
      elseif success ~= true then
        print('Validation data coroutine failed')
        print(xv)
      end
    end
    val_loss = val_loss / num_val
    print('val_loss = ', val_loss)
    table.insert(val_loss_history, val_loss)
    table.insert(val_loss_history_it, i)
    print(val_conf)
    model:resetStates()
    model:training()

    -- First save a JSON checkpoint, excluding the model
    local checkpoint = {
      opt = opt,
      desc = opt.desc,
      train_loss_history = train_loss_history,
      val_loss_history = val_loss_history,
      val_loss_history_it = val_loss_history_it,
      forward_backward_times = forward_backward_times,
      conf = val_conf:__tostring__(),
      train_conf = confusion:__tostring__(),
      epoch = i
    }
    local filename = string.format('%s_%d.json', opt.checkpoint_name, i)
    -- Make sure the output directory exists before we try to write it
    paths.mkdir(paths.dirname(filename))
    utils.write_json(filename, checkpoint)
    local grads_filename = string.format('%s/model_grads.json', opt.save)
    utils.write_json(grads_filename, grads_history)

    -- Now save a torch checkpoint with the model
    -- Cast the model to float before saving so it can be used on CPU
    model:clearState()
    model:float()
    checkpoint.model = model
    local filename = string.format('%s_%d.t7', opt.checkpoint_name, i)
    paths.mkdir(paths.dirname(filename))
    torch.save(filename, checkpoint)
    model:type(dtype)
    params, grad_params = model:getParameters()
    collectgarbage()
  end
end

