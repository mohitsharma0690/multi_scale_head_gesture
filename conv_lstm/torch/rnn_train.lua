require 'torch'
require 'nn'
require 'optim'

-- local vanilla_model = require 'model.VanillaLSTM'
require 'model.VanillaLSTM'
require 'util.DataLoader'
require 'util.BatchLoader'
require 'model.BatchLSTM'

require('mobdebug').start()

local dtype = 'torch.FloatTensor'

local try_gpu = true

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-input_h5', '../openface_data/mohit_data.h5')
cmd:option('-train_seq_h5', '../openface_data/train_gesture_by_file.h5')
cmd:option('-data_dir', '../openface_data/face_gestures/dataseto_text')
cmd:option('-batch_size', 100)
cmd:option('-seq_length', 240)
cmd:option('-use_batch_data', 1)

-- Model options
cmd:option('-use_model', 'rnn')
cmd:option('-init_from', '')
cmd:option('-reset_iterations', 1)
cmd:option('-model_type', 'lstm')
cmd:option('-rnn_size', 128)
cmd:option('-num_layers', 2)
cmd:option('-dropout', 0.3)
cmd:option('-batchnorm', 0)

-- Optimization options
cmd:option('-max_epochs', 50)
cmd:option('-learning_rate', 2e-3)
cmd:option('-grad_clip', 5)
cmd:option('-lr_decay_every', 5)
cmd:option('-lr_decay_factor', 0.5)

-- Output options
cmd:option('-print_every', 1)
cmd:option('-checkpoint_every', 10000)
cmd:option('-checkpoint_name', 'cv/checkpoint')

-- Benchmark options
cmd:option('-speed_benchmark', 0)
cmd:option('-memory_benchmark', 0)

-- Backend options
cmd:option('-gpu', 1)
cmd:option('-gpu_backend', 'cuda')

 local opt = cmd:parse(arg)

if opt.gpu == 1 then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(1)
  dtype = 'torch.CudaTensor'
  print(string.format('Running with CUDA on GPU 1'))
else
  print('Running in CPU mode')
end

-- BEGIN GLOBAL VARIABLES --
-- TODO(Mohit): Remove these global variables.
-- We need these global variables for the below f function. Need to fix.
-- Confusion matrix
local classes = {'1', '2', '3', '4', '5'}
local confusion = optim.ConfusionMatrix(classes)

local opt_clone = torch.deserialize(torch.serialize(opt))
local data_loader

local N, T = opt.batch_size, opt.seq_length
local params, grad_params; 
local model
local crit
-- END GLOBAL VARIABLES --

-- Builds a model based on the options passed in and returns it.
function build_model(opt)
  local net_class
  local net_model
  if opt.use_model == 'lstm' then
    net_model = nn.VanillaLSTM(opt_clone):type(dtype)
  elseif opt.use_model == 'rnn' then
    net_class  = nn.BatchLSTM(opt_clone)
    -- model = net_class:getModel(opt.seq_length, 128, 5) 
    net_model = net_class:getModel(46, 128, 5) 

    -- wrap the model in a Sequencer such that we can forward/backward 
    -- entire sequences of length seqLength at once
    -- model = nn.Sequencer(model)
  else
    assert(false)
  end

  -- TODO(Mohit): Maybe initialize from some checkpoint
  local crossEntropyCrit = nn.CrossEntropyCriterion():type(dtype)
  local mzSeqC = nn.MaskZeroCriterion(crossEntropyCrit, 1)
  -- Declared global (above)
  -- crit = nn.SequencerCriterion(mzSeqC)
  local net_crit = crossEntropyCrit

  --according to http://arxiv.org/abs/1409.2329 this should help model 
  --performance 
  --TODO add in new ref for -0.1 to 0.1 range
  --model:getParameters():uniform(-0.1, 0.1)

  -- Tip as per https://github.com/Element-Research/rnn/issues/125
  -- model:zeroGradParameters()

  if opt.gpu == 1 then 
    net_model = net_model:cuda()
    crossEntropyCrit = crossEntropyCrit:cuda()
    mzSeqC = mzSeqC:cuda()
    net_crit = net_crit:cuda()
  end

  print(net_model)

  -- We should theoretically return model and crit here but we are treating
  -- them as global variables for now.
  return net_model, net_crit
end

local function f(w)
  assert(w == params)
  grad_params:zero()

  -- Get a minibatch and run the model forward
  local x, y = data_loader:next_batch()
  x, y = x:type(dtype), y:type(dtype)
  for i=1, x:size(1) do 
  end
  print(y)
  local timer
  if opt.speed_benchmark ==1 then
    if cutorch then cutorch.synchronize() end
    timer = torch.Timer()
  end

  local scores = model:forward(x)
  print(scores)

  -- update confusion
  --confusion:batchAdd(scores, y)

  -- Use the criterion to compute loss
  -- TODO(Mohit): Maybe you need to change this
  local loss = crit:forward(scores, y)

  -- Run the criterion and model backward to compute gradients
  local grad_scores = crit:backward(scores, y)
  local gradInput = model:backward(x, grad_scores)
  
  if opt.speed_benchmark == 1 then
    if cutorch then cutorch.synchronize() end
    local time = timer:time().real
    print('Forward / Backward pass took ', time)
    table.insert(forward_backward_times, time)
  end

  -- TODO(Mohit): Clip the gradients as required.
  local grad_clip = false
  if grad_clip then
    grad_params:clamp(-10,10)
  end

  return loss, grad_params

end

local function print_conf_matrix_stats(conf, classes)
  local c = torch.Tensor(conf)
  local nclasses = #classes
  local prec = torch.zeros(#classes)
  local recall = torch.zeros(#classes)
  local f1 = torch.zeros(#classes)

  print(type(c))
  local tp  = torch.diag(c):resize(1, nclasses)
  local fn = (torch.sum(c, 2) - torch.diag(c)):t()
  local fp = torch.sum(c, 1) - torch.diag(c)
  local tn = torch.Tensor(1, nclasses):fill(torch.sum(c)):typeAs(tp) - tp - fn - fp

  local acc = tp:sum() / c:sum()
  -- local res = torch.cdiv(tp * 2, tp * 2 + fp + fn)  -- (2*TP)/(TP*2+fp+fn)
  -- local res = remNaN(res,self)

  print('Accuracy '..acc)
  print('Confusion matrix')
  print(conf)
end

function validate(model, data_loader, opt)
  local N, T = opt.batch_size, opt.seq_length
  model:evaluate()
  model:resetStates()
  -- TODO(Mohit): Implement this method.
  local num_val = data_loader:get_validation_size()
  local val_loss = 0
  for j = 1, num_val do
    local xv, yv = data_loader:nextBatch('val')
    xv = xv:type(dtype)
    yv = yv:type(dtype):view(N * T)
    local scores = model:forward(xv):view(N * T, -1)
    val_loss = val_loss + crit:forward(scores, yv)
  end
  val_loss = val_loss / num_val
  print('val_loss = ', val_loss)
  model:resetStates()
  model:training()
  return val_loss
end

function save_checkpoint(train_loss_history, val_loss_history,
  val_loss_history_it, forward_backward_times, val_loss, i, model,
  opt)

  table.insert(val_loss_history, val_loss)
  table.insert(val_loss_history_it, i)

  -- First save a JSON checkpoint, excluding the model
  local checkpoint = {
    opt = opt,
    train_loss_history = train_loss_history,
    val_loss_history = val_loss_history,
    val_loss_history_it = val_loss_history_it,
    forward_backward_times = forward_backward_times,
    i = i
  }
  local filename = string.format('%s_%d.json', opt.checkpoint_name, i)
  -- Make sure the output directory exists before we try to write it
  paths.mkdir(paths.dirname(filename))
  utils.write_json(filename, checkpoint)

  -- Now save a torch checkpoint with the model
  -- Cast the model to float before saving so it can be used on CPU
  model:clearState()
  model:float()
  checkpoint.model = model
  local filename = string.format('%s_%d.t7', opt.checkpoint_name, i)
  paths.mkdir(paths.dirname(filename))
  torch.save(filename, checkpoint)
  model:type(dtype)
  collectgarbage()
end

function train(model, data_loader, opt)
  -- Train the model
  local optim_config = {learningRate = opt.learning_rate}

  -- history keeper
  local train_loss_history = {}
  local val_loss_history = {}
  local val_loss_history_it = {}
  local forward_backward_times = {}

  -- TODO(Mohit): Implement this method
  local num_train = data_loader:get_train_size()
  local batch_size = data_loader.batch_size
  local num_iterations = opt.max_epochs * num_train
  
  params, grad_params = model:getParameters()

  model:training()
  local start_i = 0
  for i = 1, num_iterations, batch_size do
    local epoch = math.floor(i / num_train) + 1
    if i % num_train == 0 then
      -- One epoch done
      model:resetStates() -- Reset hidden states

      -- Print confusion matrix stats
      -- print_conf_matrix_stats(confusion, classes)
      print(confusion)
      -- Zero confusion matrix for next loop
      confusion:zero()
    end

    -- Decay learning rate
    if epoch % opt.lr_decay_every == 0  and false then
      local old_lr = optim_config.learningRate
      optim_config = {learningRate = old_lr * opt.lr_decay_factor}
    end

    -- Take a gradient step and print
    local _, loss = optim.adam(f, params, optim_config)
    table.insert(train_loss_history, loss[1])

    -- TODO(Mohit): Print every few thousand iterations?
    if opt.print_every > 0 and i % opt.print_every == 0 then
      local float_epoch = i / num_train + 1
      local msg = 'Epoch %.2f / %d, i = %d / %d, loss = %f'
      local args = {msg, float_epoch, opt.max_epochs, i, num_iterations, loss[1]}
      print(string.format(unpack(args)))
    end

    -- TODO(Mohit): Save a checkpoint
    local check_every = opt.checkpoint_every
    if (check_every > 0 and i % check_every == 0) or 
      i % num_train == 0 or
      i == num_iterations then
      -- Evaluate loss on the validation set. Note that we reset the state of
      local val_loss = validate(model, data_loader, opt)
      save_checkpoint(train_loss_history, val_loss_history,
        val_loss_history_it, forward_backward_times, val_loss, i,
        model, opt)
      params, grad_params = model:getParameters()
    end
  end
end


function main(opt)
  -- Initialize DataLoader
  -- data_loader is global
  if opt.use_batch_data then
    data_loader = BatchLoader(opt)
    data_loader:load_data()
  else
    data_loader = DataLoader(opt)
  end

 data_loader:init_batch(opt.max_epochs)
  --[[
  for i=1,num_batch_epochs do
    X, y = batch_loader:next_batch()
    assert(X:size(1) == opt.batch_size)
    assert(X:size(2) == opt.seq_length)
    assert(X:size(3) == 46)
    assert(y:size(1) == opt.batch_size)
    print(X:size())
  end
  --]]

  print('Did run through training batch successfully')
  -- model and crit is assigned globally we don't return it here
  model, crit = build_model(opt)
  train(model, data_loader, opt)

end

main(opt_clone)

