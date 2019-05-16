require 'torch'
require 'nn'
require 'optim'

-- local vanilla_model = require 'model.VanillaLSTM'
require 'model.VanillaLSTM'
require 'util.DataLoader'
require 'util.BatchLoader'

require('mobdebug').start()

local dtype = 'torch.FloatTensor'

local try_gpu = false

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-input_h5', '../../openface_data/mohit_data.h5')
cmd:option('-train_seq_h5', '../../openface_data/train_gesture_by_file.h5')
cmd:option('-data_dir', '../../openface_data/face_gestures/dataseto_text')
cmd:option('-batch_size', 100)
cmd:option('-win_len', 10)
cmd:option('-win_step', 1)

-- Model options
cmd:option('-init_from', '')
cmd:option('-reset_iterations', 1)
cmd:option('-model_type', 'lstm')
cmd:option('-rnn_size', 128)
cmd:option('-num_layers', 2)
cmd:option('-dropout', 0.3)
cmd:option('-batchnorm', 0)
cmd:option('-num_classes', 11)
cmd:option('-num_classify', 11)

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
cmd:option('-gpu', 0)
cmd:option('-gpu_backend', 'cuda')

 local opt = cmd:parse(arg)

if try_gpu then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(1)
  dtype = 'torch.CudaTensor'
  print(string.format('Running with CUDA on GPU 1'))
else
  print('Running in CPU mode')
end

-- Confusion matrix
local classes = {}
for i=1,cmd.num_classify do table.insert(classes, i) end
local confusion = optim.ConfusionMatrix(classes)

local opt_clone = torch.deserialize(torch.serialize(opt))

-- Initialize DataLoader
local loader = DataLoader(opt)
local batch_loader = SlidingWindowLoader(opt_clone)
batch_loader:load_data()
local num_batch_epochs = 5
batch_loader:init_batch(num_batch_epochs)

print('Did run through training batch successfully')

local model = nn.VanillaLSTM(opt_clone):type(dtype)
-- TODO(Mohit): Maybe initialize from some checkpoint
local start_i = 0
local crit = nn.CrossEntropyCriterion():type(dtype)
local params, grad_params = model:getParameters()

local N, T = opt.batch_size, opt.seq_length
local train_loss_history = {}
local val_loss_history = {}
local val_loss_history_it = {}
local forward_backward_times = {}
local init_memory_usage, memory_usage = nil, {}

local function f(w)
  assert(w == params)
  grad_params:zero()

  -- Get a minibach and run the model forward
  local x, y = loader:nextBatch('train')
  x, y = x:type(dtype), y:type(dtype)

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
  local grad_clip = false
  if grad_clip then
    grad_params:clamp(-10,10)
  end

  return loss, grad_params

end

-- Train the model
local optim_config = {learningRate = opt.learning_rate}
local num_train = loader.split_sizes['train']
local num_iterations = opt.max_epochs * num_train
model:training()

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

for i = start_i + 1, num_iterations do
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
  if epoch % opt.lr_decay_every == 0 then
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
  if (check_every > 0 and i % check_every == 0) or i == num_iterations then
    -- Evaluate loss on the validation set. Note that we reset the state of
    -- the model; this might happen in the middle of an epoch, but that
    -- shouldn't cause too much trouble.
    model:evaluate()
    model:resetStates()
    local num_val = loader.split_sizes['val']
    local val_loss = 0
    for j = 1, num_val do
      local xv, yv = loader:nextBatch('val')
      xv = xv:type(dtype)
      yv = yv:type(dtype):view(N * T)
      local scores = model:forward(xv):view(N * T, -1)
      val_loss = val_loss + crit:forward(scores, yv)
    end
    val_loss = val_loss / num_val
    print('val_loss = ', val_loss)
    table.insert(val_loss_history, val_loss)
    table.insert(val_loss_history_it, i)
    model:resetStates()
    model:training()

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
    params, grad_params = model:getParameters()
    collectgarbage()
  end
end

