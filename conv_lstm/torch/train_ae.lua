require 'torch'
require 'nn'
require 'optim'
require 'xlua'

require 'util.DataLoader'
require 'util.BatchLoader'
require 'util.AutoEncoderDataLoader'

local Seq2SeqAE = require 'model.Seq2SeqAE'
local LSTMAutoEncoder = require 'model.LSTMAutoEncoder'

local utils = require 'util.utils'

require('mobdebug').start()

local dtype = 'torch.FloatTensor'

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-input_h5', '../../openface_data/mohit_data.h5')
cmd:option('-train_seq_h5', '../../openface_data/main_gest_by_file.h5')
cmd:option('-data_dir', '../../openface_data/face_gestures/dataseto_text')
cmd:option('-aug_gests_h5', '../../openface_data/main_gest_by_file_aug_K_32.h5')
cmd:option('-cpm_h5_dir', '../../openface_data/cpm_output')
cmd:option('-aug_cpm_h5', '')
cmd:option('-use_cpm_features', 0)
cmd:option('-openface_mean_h5', '../../openface_data/mean_std/openface_mean_std.h5')
cmd:option('-cpm_mean_h5', '../../openface_data/mean_std/cpm_mean_std.h5')
cmd:option('-batch_size', 150)
cmd:option('-num_classes', 11)
cmd:option('-num_classify', 5)
cmd:option('-win_len', 16)
cmd:option('-win_step', 8)
cmd:option('-num_features', 136)

-- Model options
cmd:option('-init_from', '')
cmd:option('-reset_iterations', 1)
cmd:option('-model_type', 'lstm_sequence_completor')
cmd:option('-rnn_size', 128)
cmd:option('-num_layers', 2)
cmd:option('-dropout', 0.3)
cmd:option('-batchnorm', 0)
cmd:option('-use_dense_conv_lstm', 0)
cmd:option('-use_two_scale', 0)
cmd:option('-use_48_scale', 0)
cmd:option('-use_opt_flow', 0)
cmd:option('-loss', 'bce')

-- Optimization options
cmd:option('-max_epochs', 200)
cmd:option('-learning_rate', 1e-3)
cmd:option('-grad_clip', 10)
-- For Adam people don't usually decay the learning rate
cmd:option('-lr_decay_every', 20)  -- Decay every n epochs
cmd:option('-lr_decay_factor', 0.5)

-- Output options
cmd:option('-save', 'dense_step_5_cls_5')
cmd:option('-save_val_results_h5', 'val_result')
cmd:option('-print_every', 75)-- Print every n batches
cmd:option('-checkpoint_every', 10)  -- Checkpoint after every n epochs
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
if opt.save_val_results_h5 ~= '' then
  opt.save_val_results_h5 = opt.save..'/'..opt.save_val_results_h5
end

local trainLogger = optim.Logger(paths.concat(opt.save, opt.train_log))
if opt.use_dense_conv_lstm == 1 then
  trainLogger:setNames{'train_err', 'train_loss', 'l_inf_w1_16',
    'l_inf_w1_32', 'l_inf_w1_64'}
else
  trainLogger:setNames{'train_err', 'train_loss', 'l_inf_w1_16',
    'l_inf_w1_32', 'l_inf_w1_64'}
end

if opt.use_cpm_features == 1 then
  -- For now we use 6 ConvPoseMachine features
  opt.num_features = opt.num_features + 6
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

local opt_clone = torch.deserialize(torch.serialize(opt))

-- Initialize DataLoader
local loader = AutoEncoderDataLoader(opt_clone)

-- Need to pass X_Train to creaetAutoencoder, we pass in a fake one.
local model
if opt.model_type == 'seq2seq' then
  Seq2SeqAE:createAutoencoder(
      torch.Tensor(torch.Tensor(opt.batch_size, opt.win_len, opt.num_features)))
  model = Seq2SeqAE.autoencoder
elseif opt.model_type == 'lstm_autoencoder' then
  LSTMAutoEncoder:createAutoencoder(
      torch.Tensor(torch.Tensor(opt.batch_size, 32, opt.num_features)))
  model = LSTMAutoEncoder.autoencoder
elseif opt.model_type == 'lstm_predictor' then
  LSTMAutoEncoder:createPredictor({opt.batch_size,32, opt.num_features})
  model = LSTMAutoEncoder.autoencoder
elseif opt.model_type == 'lstm_sequence_completor' then
  LSTMAutoEncoder:createSequenceCompletor({opt.batch_size, 10, opt.num_features})
  model = LSTMAutoEncoder.autoencoder
elseif opt.model_type == 'lstm_next_step_predictor' then
  LSTMAutoEncoder:createNextStepPredictor({opt.batch_size, 32, opt.num_features})
  model = LSTMAutoEncoder.autoencoder
end

assert(model ~= nil)
model:cuda()

if opt.init_from:len() > 0 then 
  model = torch.load(opt_clone.init_from)
  model = model.model
  model = model:cuda()
end

local start_i = 0
--local crit = nn.BCECriterion():type(dtype)
local crit;
if opt.loss == 'mse' then crit = nn.MSECriterion():type(dtype)
elseif opt.loss == 'l1' then crit = nn.AbsCriterion():type(dtype)
elseif opt.loss == 'bce' then crit = nn.BCECriterion():type(dtype) end
assert(crit ~= nil)

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
  local seq_model = model.parent.dummyContainer
  local enc_wt, enc_grad_wt = seq_model:get(1):get(2):parameters()
  local enc_wt_2, enc_grad_wt_2 = 0,0 -- seq_model:get(1):get(3):parameters()
  local dec_wt, dec_grad_wt = seq_model:get(2):get(2):parameters()
  local enc_grad_wt_f = enc_grad_wt[1]:float()
  local enc2_grad_wt_f = 0 -- enc_grad_wt_2[1]:float()
  local dec_grad_wt_f = dec_grad_wt[1]:float()
  -- L-inf norm
  local max_enc_grad_wt = torch.max(torch.abs(enc_grad_wt_f))
  local max_dec_grad_wt = torch.max(torch.abs(dec_grad_wt_f))
  local max_enc_grad_wt_2 = 0 -- torch.max(torch.abs(enc2_grad_wt_f))
  local enc_th = torch.sum(torch.gt(enc_grad_wt_f, grad_threshold))
  local dec_th = torch.sum(torch.gt(dec_grad_wt_f, grad_threshold))
  local curr_grad_history = {
    enc_max_wt=max_enc_grad_wt, 
    enc_max_wt_2=max_enc_grad_wt_2,
    dec_max_wt=max_dec_grad_wt,
    enc_gt_th=enc_th,
    dec_gt_th=dec_th,
    total_enc=enc_grad_wt_f:nElement(),
    total_dec=dec_grad_wt_f:nElement(),
  }
  --[[
  print('======')
  print(curr_grad_history)
  print('======')
  ]]
  table.insert(grads_history, curr_grad_history)

  -- Add loss etc. to train logger (note accuracy is 0 for now)
  trainLogger:add{loss, 0.0, max_enc_grad_wt, max_dec_grad_wt}
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

  for i=1,#x do x[i] = x[i]:type(dtype) end
  y = y:type(dtype)
  -- Hack since the dictionary only has one item
  x = x[1]
  --[[
  local nanCount = 0
  for i=1,x:size(1) do
    for j=1,x:size(2) do
      for k=1,x:size(3) do
        if x[i][j][k] ~= x[i][j][k] then
          print('i: '..i..' j: '..' k: ')
          print(x[i][j][k])
          nanCount = nanCount + 1
        end
      end
    end
  end
  assert(nanCount == 0)
  ]]
  local timer
  if opt.speed_benchmark ==1 then
    if cutorch then cutorch.synchronize() end
    timer = torch.Timer()
  end

  local scores = model:forward(x)

  -- Use the criterion to compute loss
  if opt.model_type == 'lstm_predictor' then y = x[{{},{25,32},{}}]
  elseif opt.model_type == 'lstm_sequence_completor' then y=x[{{},{21,30},{}}]
  else y = x end

  local loss = crit:forward(scores, y)

  -- Run the criterion and model backward to compute gradients
  local grad_scores = crit:backward(scores, y)
  model:backward(x, grad_scores)

  if opt_clone.debug_weights == 1 then
    save_grad_weights_dense(model, loss)
  end

  if opt.speed_benchmark == 1 then
    if cutorch then cutorch.synchronize() end
    local time = timer:time().real
    print('Forward / Backward pass took ', time)
    table.insert(forward_backward_times, time)
  end

  -- TODO(Mohit): Record memory usage

  if opt.grad_clip > 0 then
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
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
  -- TODO(Mohit): We should reset states here and then again after validation
  -- ends
  --model.:forget()

  local val_data_co = coroutine.create(
      AutoEncoderDataLoader.next_val_batch)

  local val_loss = 0
  local num_val = 0

  while coroutine.status(val_data_co) ~= 'dead' do
    local success, xv, yv, batch = coroutine.resume(val_data_co, loader) 
    if success and xv ~= nil then
      for i=1,#xv do xv[i] = xv[i]:type(dtype) end
      yv = yv:type(dtype)
      -- Hack since the dictionary only has one item
      xv = xv[1]
      local scores = model:forward(xv)
      if opt.model_type == 'lstm_predictor' then yv = xv[{{},{25,32},{}}]
      elseif opt.model_type == 'lstm_sequence_completor' then yv = xv[{{},{21,30},{}}]
      else yv = xv end
      val_loss = val_loss + crit:forward(scores, yv)
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
  model:training()
  collectgarbage()
end

local val_data_to_save, saved_val_data_idx= nil, nil

for i = 1, opt.max_epochs do
  data_co = coroutine.create(AutoEncoderDataLoader.next_train_batch)
  print('Using Sliding window loader')
  curr_batches_processed = 0
  
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

      if (opt.validate_every_batches > 0 and 
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

  -- Decay learning rate
  if i % opt.lr_decay_every == 0 then
    local old_lr = optim_config.learningRate
    optim_config = {learningRate = old_lr * opt.lr_decay_factor}
  end

  local check_every = opt.checkpoint_every
  if (check_every > 0 and i % check_every == 0) then
    -- Evaluate loss on the validation set. Note that we reset the state of
    -- the model; this might happen in the middle of an epoch, but that
    -- shouldn't cause too much trouble.
    model:evaluate()
    -- Set the dataloader to load validation data
    local val_data_co = coroutine.create(
        AutoEncoderDataLoader.next_val_batch)

    local val_save_num = {200,200,200,200,200}
    local val_save_counter = {1,1,1,1,1}

    if val_data_to_save == nil then
      local val_data_stats = loader:val_data_stats()
      for k,v in ipairs(val_data_stats) do
        if k > 1 then 
          val_data_stats[k] = val_data_stats[k-1] + val_data_stats[k] 
        end
      end
      val_data_to_save = {}
      for k, v in ipairs(val_save_num) do 
        local start, e = 1, val_data_stats[k]
        if k > 1 then start = val_data_stats[k-1] + 1 end
        if k == 5 then e = val_data_stats[k] - opt.batch_size end

        for j=1,v do
          table.insert(val_data_to_save, math.floor(torch.uniform(start,e)))
        end
      end
    end

    local val_loss = 0
    local num_val, counter = 0, 0
    -- Create tensors used to save the input sand outputs
    local inp_tensors, op_tensors, inp_info = {}, {}, {}
    for k,v in ipairs(val_save_num) do 
      if v > 0 then
        local inp_size, op_size = 24, 8
        if opt.model_type == 'lstm_sequence_completor' then 
          inp_size, op_size = 30, 10
        elseif opt.model_type == 'seq2seq' then
          inp_size, op_size = 16, 16
        end
        inp_tensors[k] = torch.Tensor(v, inp_size, opt.num_features):zero()
        op_tensors[k] = torch.Tensor(v, op_size, opt.num_features):zero()
        inp_info[k] = {}
      end
    end

    while coroutine.status(val_data_co) ~= 'dead' do
      local success, xv, yv, batch = coroutine.resume(val_data_co, loader) 
      local save_dict = {}
      if success and xv ~= nil then
        for i=1,#xv do xv[i] = xv[i]:type(dtype) end
        yv = yv:type(dtype)
        -- Hack
        xv = xv[1]

        local scores = model:forward(xv)

        -- Save scores and xv both, but which ones to save doesn't
        if opt_clone.save_val_results_h5 ~= '' then 
          for i=counter+1,counter+yv:size(1) do 
            for k, v in ipairs(val_data_to_save) do
              if i == v then 
                local y = math.floor(yv[i-counter])
                local idx = val_save_counter[y]

                if idx <= inp_tensors[y]:size(1) and i-counter <= xv:size(1) then
                  inp_tensors[y][{{idx},{},{}}] = xv[{{i-counter},{1,30},{}}]:float()
                  op_tensors[y][{{idx},{},{}}] = scores[{{i-counter},{},{}}]:float()
                  table.insert(inp_info[y], batch[i-counter])
                  val_save_counter[y] = val_save_counter[y] + 1
                end
              end
            end
          end
        end
        
        if opt.model_type == 'lstm_predictor' then yv = xv[{{},{25,32},{}}]
        elseif opt.model_type == 'lstm_sequence_completor' then yv = xv[{{},{21,30},{}}]
        else yv = xv end

        local curr_val_loss = crit:forward(scores, yv)
        val_loss = val_loss + curr_val_loss
        num_val = num_val + 1
        counter = counter + yv:size(1)
      elseif success ~= true then
        print('Validation data coroutine failed')
        print(xv)
      end
    end
    val_loss = val_loss / num_val
    print('val_loss = ', val_loss)
    table.insert(val_loss_history, val_loss)
    table.insert(val_loss_history_it, i)
    model:training()

    -- Check that we saved the right tensors
    if opt_clone.save_val_results_h5 ~= '' then
      -- print(val_save_num)
      -- print(val_save_counter)
      -- for k,v in pairs(val_save_counter) do assert(v > 5) end
    end

    -- First save a JSON checkpoint, excluding the model
    local checkpoint = {
      opt = opt,
      train_loss_history = train_loss_history,
      val_loss_history = val_loss_history,
      val_loss_history_it = val_loss_history_it,
      epoch = i
    }
    local filename = string.format('%s_%d.json', opt.checkpoint_name, i)
    -- Make sure the output directory exists before we try to write it
    paths.mkdir(paths.dirname(filename))
    utils.write_json(filename, checkpoint)

    if opt.save_val_results_h5 ~= '' then
      local filename = string.format('%s_%d.h5', opt.save_val_results_h5, i)
      local json_filename = string.format('%s_%d.json', opt.save_val_results_h5, i)
      local final_tensors = {}
      for k,v in pairs(inp_tensors) do
        final_tensors['/input/'..k] = v
      end
      for k,v in pairs(op_tensors) do
        final_tensors['/output/'..k] = v
      end
      utils.write_hdf5(filename, nil, final_tensors)
      utils.write_json(json_filename, inp_info)
    end

    -- Now save a torch checkpoint with the model
    -- Cast the model to float before saving so it can be used on CPU
    -- model:clearState()
    model.parent.dummyContainer:float()
    -- Save the Seq
    checkpoint.model = {
      dummyContainer=model.parent.dummyContainer,
      cellSizes=LSTMAutoEncoder.cellSizes,
      encLSTMs=LSTMAutoEncoder.encLSTMs,
      decLSTMs=LSTMAutoEncoder.decLSTMs,
    }
    local filename = string.format('%s_%d.t7', opt.checkpoint_name, i)
    paths.mkdir(paths.dirname(filename))
    torch.save(filename, checkpoint)
    model.parent.dummyContainer:type(dtype)
    params, grad_params = model:getParameters()
    collectgarbage()
  end
end

