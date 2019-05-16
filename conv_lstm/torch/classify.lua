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
require 'model.ConvLSTM'
require 'model.DenseConvLSTM'
require 'model.MultiScaleConvLSTM'
require 'model.MultiScaleConvBLSTM'
require 'model.MultiScaleCNN'
require 'model.MS_TemperatureSoftMax'

local utils = require 'util.utils'

require('mobdebug').start()

local dtype = 'torch.FloatTensor'

local cmd = torch.CmdLine()

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
cmd:option('-batch_size', 200)
cmd:option('-num_classes', 11)
cmd:option('-num_classify', 5)
cmd:option('-win_len', 10)
cmd:option('-win_step', 3)
cmd:option('-num_features', 50)
cmd:option('-use_all_face_diff', 0)
cmd:option('-openface_trimmed_aug', 1)
cmd:option('-test_all_frames', 1)
cmd:option('-train_one_vs_all', 0,
    'If non zero the classifier is trained for 1-vs-all classification that class')

-- Return the list of which batch inputs are in the current batch for validation
-- data
cmd:option('-val_batch_info', 1)
cmd:option('-finetune_batch_size', 50)

-- Optimization options
cmd:option('-max_epochs', 2)
cmd:option('-learning_rate', 1e-6)
cmd:option('-grad_clip', 10)
-- For Adam people don't usually decay the learning rate
cmd:option('-lr_decay_every', 20)  -- Decay every n epochs
cmd:option('-lr_decay_factor', 0.5)

-- Model options
cmd:option('-init_from', '')
cmd:option('-use_dense_conv_lstm', 0)
cmd:option('-num_scales', 3)
cmd:option('-use_48_scale', 0)
cmd:option('-finetune', 0)
cmd:option('-finetune_ratio', 0.2)
cmd:option('-model_type', '')
cmd:option('-finetune_new_dataset', 0)
cmd:option('-softmax_temperature', 10.0)

-- Output options
cmd:option('-print_every', 100) -- Print every n batches while finetuning
cmd:option('-save', 'dense_step_5_cls_5')
cmd:option('-checkpoint_name', 'checkpoint')
cmd:option('-test_log', 'test.log')
cmd:option('-test_batch', 'test_batch.json')
cmd:option('-test_scores', 'test_scores.json')
cmd:option('-test_preds', 'test_preds.json')
cmd:option('-classification_type', 'none')

-- Backend options
cmd:option('-gpu', 1)
cmd:option('-gpu_backend', 'cuda')

local opt = cmd:parse(arg)
opt.test_log= opt.save..'/'..opt.test_log
opt.test_scores = opt.save .. '/' .. opt.test_scores
opt.test_preds = opt.save .. '/' .. opt.test_preds
opt.test_batch = opt.save .. '/' .. opt.test_batch

--[[
local trainLogger = optim.Logger(paths.concat(opt.save, opt.train_log))
if opt.use_dense_conv_lstm == 1 then
  trainLogger:setNames{'train_err', 'train_loss',}
else
  trainLogger:setNames{'train_err', 'train_loss', 'l_inf_w1_16',
    'l_inf_w1_32', 'l_inf_w1_64'}
end
]]

if opt.use_openface_features ~= 1 then
  opt.num_features = 0
end

-- Include CPM features in input.
if opt.use_cpm_features == 1 then
  -- For now we use 6 ConvPoseMachine features
  opt.num_features = opt.num_features + 10
end

if opt.use_zface_features == 1 then
  -- These are the filtered features center (X, y) and pose
  opt.num_features = opt.num_features + 9
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
if opt.train_one_vs_all > 0 then
  classes = {1, 2}
end
-- Confusion matrix for data
local conf = optim.ConfusionMatrix(classes)

local opt_clone = torch.deserialize(torch.serialize(opt))
-- Global options variable is a global
G_global_opts = opt_clone
print(G_global_opts)

-- Load model
local model = torch.load(opt_clone.init_from)
print(model.conf)
model = model.model
print(model.net)
model = model:cuda()



local loader 
local data_co
local val_data_co
if opt.use_dense_conv_lstm == 1 then
  loader = DenseSlidingWindowDataLoader(opt_clone)
  val_data_co = coroutine.create(DenseSlidingWindowDataLoader.next_val_batch)
else
  loader = KFoldSlidingWindowDataLoader(opt_clone)
  val_data_co = coroutine.create(KFoldSlidingWindowDataLoader.next_val_batch)
end


if opt.finetune == 1 then
  local params, grad_params = model:getParameters()
  local crit = nn.CrossEntropyCriterion():type(dtype)
  local total_train_batches = loader:getTotalTrainBatches()
  local train_loss_history, grads_history = {}, {}

    -- TODO(Mohit): Finetune here
   function f(w)
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

    local scores = model:forward(x)
    local loss = crit:forward(scores, y)

    local grad_scores = crit:backward(scores, y)
    model:backward(x, grad_scores)

    if opt_clone.debug_weights == 1 then
      if opt_clone.use_dense_conv_lstm == 0 then
        save_grad_weights(model, loss)
      else
        save_grad_weights_dense(model, loss)
      end
    end

    -- TODO(Mohit): Clip the gradients as required.
    if opt.grad_clip > 0 then
      grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    end

    return loss, grad_params
  end

  print("Will finetune on "..total_train_batches.." batches")
  print('Using Sliding window loader next_finetune_batch')
  for i=1, opt.max_epochs do
    data_co = coroutine.create(KFoldSlidingWindowDataLoader.next_finetune_batch)
    local curr_batches_processed = 0

    -- Go through the entire train batch
    while coroutine.status(data_co) ~= 'dead' do
      xlua.progress(curr_batches_processed, total_train_batches)

      if curr_batches_processed < total_train_batches then
        -- Take a gradient step and print
        local optim_config = {learningRate = opt.learning_rate}
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
        curr_batches_processed = curr_batches_processed + 1
      else
        -- Separate this since we don't want to add 0 loss to the train history.
        local success, x = coroutine.resume(data_co, loader)
        assert(coroutine.status(data_co) == 'dead')
      end
    end
  end
end

--local softmaxLayer = cudnn.SoftMax():cuda()
local softmaxLayer = nn.MS_TemperatureSoftMax(opt.softmax_temperature):cuda()
model.net:add(softmaxLayer)
--local w1_16, grad_w1_16 = model.net:get(1):get(1):get(1):parameters()
-- Get the weights not the biases
--w1_16 = w1_16[1]

-- Evaluate model
model:evaluate()
model:resetStates()

conf:zero()
local num_test = 0
local test_scores = {}
local test_preds = {}
local test_data = {}

while coroutine.status(val_data_co) ~= 'dead' do
  local success, xv, yv, batch = coroutine.resume(val_data_co, loader) 
  if success and xv ~= nil then
    for i=1,#xv do xv[i] = xv[i]:type(dtype) end
    yv = yv:type(dtype)
    if opt.num_scales == 1 then xv = xv[1] end
    local scores = model:forward(xv)
    local scores_max, scores_max_idx = torch.max(scores, 2)
    for i=1,scores:size(1) do
      if torch.max(scores[{{i},{}}]) > 0.0 then
        local _, idx = torch.max(scores[{{i},{}}], 2)
        --print(yv[i])
        --print(idx[1][1])
        conf:add(idx[1][1], yv[i])
      end
    end
    -- conf:batchAdd(scores, yv)
    num_test = num_test + 1
    -- Add batch data to the global table
    for i=1,#batch do table.insert(test_data, batch[i]) end
    -- Add the predictions to the global table
    scores_max_idx = torch.totable(scores_max_idx)
    for i=1,#scores_max_idx do table.insert(test_preds, scores_max_idx[i]) end
    -- Add the scores to the global table
    scores = torch.totable(scores)
    for i=1, #scores do table.insert(test_scores, scores[i]) end
  elseif success ~= true then
    print('Validation data coroutine failed')
    if torch.isTensor(xv) then 
      print(xv:size())
    else
      print(xv)
    end
  end
end

print('Total frames evaluated '..num_test)
print(conf)
-- Save the test batch data
paths.mkdir(paths.dirname(opt.test_batch))
utils.write_json(opt.test_batch, test_data)

-- Save the scores
paths.mkdir(paths.dirname(opt.test_scores))
utils.write_json(opt.test_scores, test_scores)

-- Save the predictions
paths.mkdir(paths.dirname(opt.test_preds))
utils.write_json(opt.test_preds, test_preds)

