require 'torch'
require 'nn'
require 'rnn'
require 'optim'
require 'xlua'

-- local vanilla_model = require 'model.VanillaLSTM'
require 'model.VanillaLSTM'
require 'util.DataLoader'
require 'util.AirHandwritingDataLoader'
require 'model.AirHandwritingConvLSTM'

local utils = require 'util.utils'

require('mobdebug').start()

local dtype = 'torch.FloatTensor'

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-data_h5', '../../data_airhandwriting/matR_upper_train_test_data.h5')
cmd:option('-aug_gests_h5', '../../openface_data/main_gest_by_file_aug_K_32.h5')
cmd:option('-openface_mean_h5', '../../openface_data/mean_std/openface_mean_std.h5')
cmd:option('-batch_size', 150)
cmd:option('-num_classes', 26)
cmd:option('-num_classify', 26)
cmd:option('-win_len', 200)
cmd:option('-num_features', 13)
cmd:option('-word_data', 1)

-- Model options
cmd:option('-init_from', '')
cmd:option('-reset_iterations', 1)
cmd:option('-model_type', 'lstm')
cmd:option('-rnn_size', 128)
cmd:option('-num_layers', 2)
cmd:option('-use_dense_conv_lstm', 0)
cmd:option('-use_sgd', 1)

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
local val_conf = optim.ConfusionMatrix(classes)
if opt.word_data  == 1 then
  confusion = optim.ConfusionMatrix(28)
  val_conf = optim.ConfusionMatrix(28)
end
-- Confusion matrix for validation data

local opt_clone = torch.deserialize(torch.serialize(opt))

-- Initialize DataLoader
local num_batch_epochs = 5
local loader = AirHandwritingDataLoader(opt_clone)
local vocab_size = loader:get_vocab_size()
opt_clone.vocab_size = vocab_size
opt.vocab_size = vocab_size
local model = nn.AirHandwritingConvLSTM(opt_clone)
model:getConvLSTMModel()
model:updateType(dtype)
print(model.net)

if opt.init_from:len() > 0 then 
  model = torch.load(opt_clone.init_from)
  model = model.model
  model = model:cuda()
end

local start_i = 0
local crit
if opt.word_data == 1 then
  crit = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(),1)):type(dtype)
else
  crit = nn.CrossEntropyCriterion():type(dtype)
end
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
local function save_grad_weights(model, loss) 
  if opt.word_data == 1 then
    local w1_16, grad_w1_16 = model.net:get(1):get(1):parameters()
    local w1_32, grad_w1_32 = model.net:get(1):get(12):parameters()
    local grad_w1_16_f = torch.abs(grad_w1_16[1]:float())
    local grad_w1_32_f = torch.abs(grad_w1_32[1]:float())
    grad_w1_16 = torch.max(grad_w1_16_f)
    grad_w1_32 = torch.max(grad_w1_32_f)
    local w16_th = torch.sum(torch.gt(grad_w1_16_f, grad_threshold))
    local w32_th = torch.sum(torch.gt(grad_w1_32_f, grad_threshold))
    local curr_grad_history = {
      max_w1_conv1=grad_w1_16, 
      max_w1_conv2=grad_w1_32,
      w_conv1_gt_th=w16_th,
      w_conv2_gt_th=w32_th,
      total_w1_conv1=grad_w1_16_f:nElement(),
      total_w_conv2=grad_w1_32_f:nElement(),
      conv1_th = w16_th,
      conv2_th = w32_th,
    }
    table.insert(grads_history, curr_grad_history)

    -- Add loss etc. to train logger (note accuracy is 0 for now)
    trainLogger:add{loss, 0.0, grad_w1_16, grad_w1_32, grad_w1_64}
  end
end

local function update_word_conf_matrix(conf, preds, targets)
  -- preds is in format TxNXF
  local batch_size, num_len = targets:size(1), targets:size(2)
  for b=1,batch_size do
    for i=1,num_len do
      if targets[b][i] > 0 then 
        conf:add(preds[{{b},{i},{}}]:view(28), targets[b][i]) 
      end
    end
  end
  return conf
end

local function f(w)
  assert(w == params)
  grad_params:zero()

  -- Get a minibach and run the model forward
  local success, x, y = coroutine.resume(data_co, loader)
  if not success then print('Data couroutine returns fail.') end
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

  if torch.isTensor(x) then x = x:type(dtype) 
  else for k,v in pairs(x) do x[k] = v:cuda() end end
  y = y:type(dtype)

  local timer
  if opt.speed_benchmark ==1 then
    if cutorch then cutorch.synchronize() end
    timer = torch.Timer()
  end

  local scores = model:forward(x) 
  -- update confusion
  if opt.word_data == 1 then update_word_conf_matrix(confusion, scores, y)
  else confusion:batchAdd(scores, y) end

  -- Use the criterion to compute loss
  -- TODO(Mohit): Maybe you need to change this
  local loss = crit:forward(scores, y)
  if opt.word_data == 1 then
    -- Although the original sequence length will vary for each input we use
    -- an approx value of 5 here.
    -- local avgSeqLen = torch.sum(torch.sign(x[2])) / x[2]:size(2)
    loss = loss / 5 
  end

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
    -- TODO(Mohit): Complete this
    save_grad_weights(model, loss)
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

  local val_data_co = coroutine.create(AirHandwritingDataLoader.next_val_batch)
  val_conf:zero()

  local val_loss = 0
  local num_val = 0

  while coroutine.status(val_data_co) ~= 'dead' do
    local success, xv, yv = coroutine.resume(val_data_co, loader) 
    if success and xv ~= nil then
      if torch.isTensor(xv) then xv = xv:type(dtype)
      else for k,v in pairs(xv) do xv[k] = v:type(dtype) end end
      yv = yv:type(dtype)
      local scores =  model:forward(xv)
      val_loss = val_loss + crit:forward(scores, yv)
      for i=1,#model.last_top5_preds do 
        print(model.last_top5_preds[i])
      end
      print(yv)
      assert(false)
      if opt.word_data == 1 then update_word_conf_matrix(val_conf, scores, yv)
      else val_conf:batchAdd(scores, yv) end
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
  data_co = coroutine.create(AirHandwritingDataLoader.next_train_batch)
  print('Using Sliding window loader')

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

      if (curr_batches_processed >= 0 and opt.validate_every_batches > 0 and 
        curr_batches_processed % opt.validate_every_batches == 0) then
        run_on_val_data()
      end
      curr_batches_processed = curr_batches_processed + 1
    else
      -- Separate this since we don't want to add 0 loss to the train history.
      local success, x = coroutine.resume(data_co, loader)
      --assert(coroutine.status(data_co) == 'dead')
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
        AirHandwritingDataLoader.next_val_batch)
    val_conf:zero()
    local val_loss = 0
    local num_val = 0
    while coroutine.status(val_data_co) ~= 'dead' do
      local success, xv, yv = coroutine.resume(val_data_co, loader) 
      if success and xv ~= nil then
        if torch.isTensor(xv) then xv = xv:type(dtype)
        else for k,v in pairs(xv) do xv[k] = v:type(dtype) end end
        yv = yv:type(dtype)
        local scores = model:forward(xv)
        --print("in train.lua")
        --print(yv)
        val_loss = val_loss + crit:forward(scores, yv)
        if opt.word_data == 1 then 
          update_word_conf_matrix(val_conf, scores, yv)
        else
          val_conf:batchAdd(scores, yv)
        end
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

