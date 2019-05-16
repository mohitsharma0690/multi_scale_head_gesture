local utils = require '../util/utils.lua'

require 'nn'
require 'cunn'
require 'hdf5'
require 'math'
require 'paths'
require 'optim'

local train_cls = {}

local grad_params, params
local train_loss_history, crit2_loss_history, val_loss_history = {}, {}, {}
local dtype

--[[
Use context variable which is trained by some auxiliary supervised representation.
For example the auxiliary loss could be to predict which user is the gesture from.
]]
-- Access opt as G_global_opts
function train_cls.setup(args)
  local self=train_cls
  self.dtype = args.dtype
  self.data_loader = args.data_loader
  self.model = args.model
  self.grads_history = args.grads_history
  self.train_conf = args.confusion
  self.val_conf = args.val_conf
  self.num_context_classes = G_global_opts.supervised_context_num_classify
  self.train_context_conf = optim.ConfusionMatrix(self.num_context_classes)
  self.val_context_conf = optim.ConfusionMatrix(self.num_context_classes)
  self.checkpoint = {
    classif_loss_history={},
    context_loss_history={},
    train_loss_history = {},
    val_loss_history = {},
    val_classif_loss_history={},
    val_context_loss_history={},
  }

  self.classif_crit = nn.CrossEntropyCriterion():type(self.dtype)
  self.context_crit = nn.CrossEntropyCriterion():type(self.dtype)

  -- Convert to dtype
  self.model:updateType(self.dtype)

  self.optim_config_init_context = {}
  self.optim_config_final_classif = {}
  self.optim_config_final_context = {}

  -- Interestingly, this should be done after you've updated the model to 
  -- CUDA or whatever type you're using above otherwise the memory allocated
  -- in getParameters() might be different.
  self:get_grad_params()
end

function train_cls:get_grad_params()
  local self = train_cls
  local model = self.model
  self.init_classif_params, self.init_classif_grad_params = model.init_classif_model:getParameters()
  self.init_context_params, self.init_context_grad_params = model.init_context_model:getParameters()
  self.final_classif_params, self.final_classif_grad_params = model.final_classif_model:getParameters()
  self.final_context_params, self.final_context_grad_params = model.final_context_model:getParameters()
end

function train_cls.read_data_co(data_co, data_loader)
  local success, x, y = coroutine.resume(data_co, data_loader)
  if not success then print('Data couroutine returns fail.') end
  if x == nil then
    print('x is nil returning 0 loss')
  elseif success == false then
    -- print crash logs
    if torch.isTensor(x) then
      print(x:size())
    else
      print(x)
    end
  end
  return success,x,y
end

function train_cls.f_init_context(w)
  local self = train_cls
  assert (w == self.init_context_params)
  return 0, self.init_context_grad_params
end

function train_cls.f_final_classif(w)
  local self = train_cls
  assert(w == self.final_classif_params)
  return 0, self.final_classif_grad_params
end

function train_cls.f_final_context(w)
  local self = train_cls
  assert(w == self.final_context_params)
  return 0, self.final_context_grad_params
end

function train_cls.f(w)
  local self = train_cls
  assert(w == self.init_classif_params)
  -- Zero out all grad_params
  self.init_classif_grad_params:zero()
  self.init_context_grad_params:zero()
  self.final_classif_grad_params:zero()
  self.final_context_grad_params:zero()

  -- Get a minibach and run the model forward
  local success, x, y = self.read_data_co(self.data_co, self.data_loader)
  x = x[1]

  x = utils.convert_to_type(x, self.dtype)
  y = utils.convert_to_type(y, self.dtype)
  -- y = y:type(self.dtype)
  
  local init_classif_score = self.model.init_classif_model:forward(x[1])
  local init_context_score = self.model.init_context_model:forward(x[2])
  local final_classif_input = {
    [1]=init_classif_score:clone(),
    [2]=init_context_score:clone()
  }
  local final_classif_score = self.model.final_classif_model:forward(final_classif_input)

  local final_context_input = init_context_score:clone() 
  local final_context_score = self.model.final_context_model:forward(final_context_input) 

  -- Get the two loss values
  local classif_loss = self.classif_crit:forward(final_classif_score, y[1])
  local context_loss = self.context_crit:forward(final_context_score, y[2])

  -- Backprop through crit
  local final_classif_grad_score = self.classif_crit:backward(
    final_classif_score, y[1])

  local final_context_grad_score = self.context_crit:backward(
    final_context_score, y[2])

  -- First backprop through the final classification layers for each model
  local init_classif_grad_score = self.model.final_classif_model:backward(
      final_classif_input, final_classif_grad_score)

  local init_context_grad_score = self.model.final_context_model:backward(
      final_context_input, final_context_grad_score)

  -- Now backprop through the initial models 
  local classif_grad_score = self.model.init_classif_model:backward(
      x[1], init_classif_grad_score[1])

  -- Need to sum gradients for init_context from two different parts
  local total_context_grad_score = torch.add(
      init_classif_grad_score[2], init_context_grad_score)

  local context_grad_score = self.model.init_context_model:backward(
      x[2], total_context_grad_score)
  
  local timer
  if G_global_opts.speed_benchmark ==1 then
    if cutorch then cutorch.synchronize() end
    timer = torch.Timer()
  end

  self.train_conf:batchAdd(final_classif_score, y[1])
  self.train_context_conf:batchAdd(final_context_score, y[2])

  -- Add losses to logs
  if G_global_opts.debug_weights == 1 then
    local logs = self.checkpoint
    table.insert(logs.classif_loss_history, classif_loss)
    table.insert(logs.context_loss_history, context_loss)
    table.insert(logs.train_loss_history, classif_loss+context_loss)
  end
  

  if G_global_opts.speed_benchmark == 1 then
    if cutorch then cutorch.synchronize() end
    local time = timer:time().real
    print('Forward / Backward pass took ', time)
    table.insert(forward_backward_times, time)
  end

  -- TODO(Mohit): Record memory usage

  -- TODO(Mohit): Clip the gradients as required.
  if G_global_opts.grad_clip > 0 then
    local all_grad_params = {
      self.init_classif_grad_params,
      self.init_context_grad_params,
      self.final_classif_grad_params,
      self.final_context_grad_params,
    }
    for k, grad_params in pairs(all_grad_params) do
      grad_params:clamp(-G_global_opts.grad_clip, G_global_opts.grad_clip)
    end

  end

  if G_global_opts.debug_weights == 1 then
    local curr_grad_history = self.model:getGradWeights(loss, x, y)
    table.insert(self.grads_history, curr_grad_history)
  end

  -- Note the loss returned here is not used anywhere but returned back to user code
  -- at the end so we can have 0 here irrespective of everything else.
  return context_loss + classif_loss, self.init_classif_grad_params
end

function train_cls.validate(val_data_co)
  return train_cls._validate(val_data_co, false)
end

function train_cls.validate_save(val_data_co)
  return train_cls._validate(val_data_co, true)
end

function train_cls._validate(val_data_co, return_data_stats)
  local self = train_cls
  local val_loss, num_val = 0, 0

  self.val_conf:zero()
  self.val_context_conf:zero()
  local test_data, test_scores_classif, test_pred_classif = {}, {}, {}
  local test_scores_context, test_pred_context= {}, {}, {}

  while coroutine.status(val_data_co) ~= 'dead' do
    local success, xv, yv, batch = coroutine.resume(val_data_co, self.data_loader)
    if success and xv ~= nil then

      xv = xv[1]
      xv = utils.convert_to_type(xv, self.dtype)
      yv = utils.convert_to_type(yv, self.dtype)

      local init_classif_score = self.model.init_classif_model:forward(xv[1])
      local init_context_score = self.model.init_context_model:forward(xv[2])
      local final_classif_input = {
        [1]=init_classif_score:clone(),
        [2]=init_context_score:clone()
      }
      local final_classif_score = self.model.final_classif_model:forward(
          final_classif_input)

      local final_context_input = init_context_score:clone() 
      local final_context_score = self.model.final_context_model:forward(
          final_context_input) 

      assert(torch.max(final_classif_score) == torch.max(final_classif_score))
      assert(torch.max(final_context_score) == torch.max(final_context_score))

      local classif_loss = self.classif_crit:forward(final_classif_score, yv[1])
      local context_loss = self.context_crit:forward(final_context_score, yv[2])

      val_loss = val_loss + (classif_loss + context_loss)

      self.val_conf:batchAdd(final_classif_score, yv[1])
      self.val_context_conf:batchAdd(final_context_score, yv[2])
      num_val = num_val + 1

      if return_data_stats then
        for i=1,#batch do table.insert(test_data, batch[i]) end
        -- Add classif score
        for i=1, final_classif_score:size(1) do 
          table.insert(test_scores_classif,
              torch.totable(final_classif_score[{{i},{}}]))
        end
        -- Add context score
        for i=1, final_context_score:size(1) do
          table.insert(test_scores_context,
              torch.totable(final_context_score[{{i},{}}]))
        end
        -- Add classif preds
        _, final_classif_score = torch.max(final_classif_score, 2)
        for i=1,final_classif_score:size(1) do 
          table.insert(test_pred_classif, final_classif_score[i][1])
        end
        -- Add context preds
        _, final_context_score = torch.max(final_context_score, 2)
        for i=1,final_context_score:size(1) do
          table.insert(test_pred_context, final_context_score[i][1])
        end
      end

    elseif success ~= true then
      print('Validation data coroutine failed')
      print(xv)
    end
  end

  val_loss = val_loss / num_val
  print('val_loss = ', val_loss)
  table.insert(self.checkpoint.val_loss_history, val_loss)
  print(self.val_conf)
  self.model:resetStates()
  self.model:training()
  collectgarbage()

  return {
    test_data=test_data,
    test_scores=test_scores_classif,
    test_scores_context=test_scores_context,
    test_preds=test_pred_classif,
    test_preds_context=test_pred_context,
  }
end

function train_cls.train(train_data_co, optim_config, stats)
  local self = train_cls
  self.data_co = train_data_co

  self.optim_config_init_context.learningRate = optim_config.learningRate
  self.optim_config_final_classif.learningRate = optim_config.learningRate
  self.optim_config_final_context.learningRate = optim_config.learningRate

  local _, loss = optim.adam(self.f, self.init_classif_params, optim_config)
  optim.adam(self.f_init_context,
      self.init_context_params, self.optim_config_init_context)
  optim.adam(self.f_final_classif, self.final_classif_params,
      self.optim_config_final_classif)
  optim.adam(self.f_final_context, self.final_context_params, 
      self.optim_config_final_context)

  if (stats.curr_batch > 0 and 
      stats.curr_batch % G_global_opts.print_every == 0) then

    local msg = 'Epoch: [%d/%d]\t Iteration:[%d/%d]\t total loss: %.2f\t'
    msg = msg..'classif loss: %.2f\t context loss: %.2f'
    local logs = self.checkpoint
    local args = {
      msg,
      stats.curr_epoch, stats.total_epoch,
      stats.curr_batch, stats.total_batch,
      logs.train_loss_history[#logs.train_loss_history],
      logs.classif_loss_history[#logs.classif_loss_history],
      logs.context_loss_history[#logs.context_loss_history],
    }
    print(string.format(unpack(args)))
  end
  return loss
end

function train_cls.getCheckpoint()
  local self = train_cls
  local cp = {}
  for k,v in pairs(self.checkpoint) do cp[k] = v end
  cp.train_conf = self.train_conf:__tostring__()
  cp.conf = self.val_conf:__tostring__()
  cp.train_context_conf = self.train_context_conf:__tostring__()
  cp.val_context_conf = self.val_context_conf:__tostring__()
  return cp
end

return train_cls
