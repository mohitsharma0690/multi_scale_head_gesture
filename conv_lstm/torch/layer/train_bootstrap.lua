local utils = require '../util/utils.lua'

require 'nn'
require 'cunn'
require 'hdf5'
require 'math'
require 'paths'

local train_cls = {}

local grad_params, params
local train_loss_history, crit2_loss_history, val_loss_history = {}, {}, {}
local dtype

-- Access opt as G_global_opts
function train_cls.setup(args)
  assert(false)
  local self=train_cls
  self.dtype = args.dtype
  self.params, train_cls.grad_params = args.params, args.grad_params
  self.data_loader = args.data_loader
  self.model = args.model
  self.confusion = args.confusion
  self.val_conf = args.val_conf
  self.train_conf = args.confusion,
  self.grads_history = args.grads_history
  self.checkpoint = {
    train_loss_history={},
    crit1_loss_history={},
    crit2_loss_history={},
    val_loss_history={},
  }

  self.model.net:add(nn.LogSoftMax():type(self.dtype))
  self.crit = nn.ClassNLLCriterion():type(self.dtype)
  self.crit2 = nn.ClassNLLCriterion():type(self.dtype)
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

function train_cls.f(w)
  local _s = train_cls
  assert(w == _s.params)
  _s.grad_params:zero()
  assert(G_global_opts.use_bootstrap_Beta == 1)

  local success, x, y = _s.read_data_co(_s.data_co, _s.data_loader)
  if not x then 
    return 0, _s.grad_params
  end

  if G_global_opts.num_scales == 1 then x = x[1];  end
  x = x[1]
  x = utils.convert_to_type(x, _s.dtype)
  y = utils.convert_to_type(y, _s.dtype)

  local log_scores = _s.model:forward(x)
  -- Use the criterion to compute loss
  local loss_target = _s.crit:forward(log_scores, y)
  local _, scores_max = torch.max(log_scores, 2)
  scores_max = scores_max:view(-1)
  local loss_scores = _s.crit2:forward(log_scores, scores_max)
  -- Run the criterion and model backward to compute gradients
  local grad_target = _s.crit:backward(log_scores, y)
  local grad_scores = _s.crit2:backward(log_scores, scores_max)

  -- Combine the two gradients
  local beta = G_global_opts.bootstrap_Beta
  local total_loss = beta*loss_target + (1.0-beta)*loss_scores
  local grad_scores = torch.add(grad_target:mul(beta), grad_scores:mul(1.0-beta))

  -- Finally backprop the gradients
  _s.model:backward(x, grad_scores)

  _s.confusion:batchAdd(log_scores, y)

  -- TODO(Mohit): Clip the gradients as required.
  if G_global_opts.grad_clip > 0 then
    _s.grad_params:clamp(-G_global_opts.grad_clip, G_global_opts.grad_clip)
  end

  if G_global_opts.debug_weights == 1 then 
    local curr_grad_history = _s.model:getGradWeights(loss, x, y) 
    table.insert(_s.grads_history, curr_grad_history)
  end
  table.insert(_s.checkpoint.crit1_loss_history, loss_target)
  table.insert(_s.checkpoint.crit2_loss_history, loss_scores)

  return loss_target+loss_scores, _s.grad_params
end

function train_cls.validate(val_data_co)
  local self = train_cls
  local val_loss, num_val = 0, 0

  self.val_conf:zero()

  while coroutine.status(val_data_co) ~= 'dead' do
    local success, xv, yv = coroutine.resume(val_data_co, self.data_loader) 
    if success and xv ~= nil then
      if G_global_opts.num_scales == 1 then xv = xv[1]; end
      xv = utils.convert_to_type(xv, self.dtype)
      yv = utils.convert_to_type(yv, self.dtype)
      xv = xv[1]

      local scores = self.model:forward(xv)
      val_loss = val_loss + self.crit:forward(scores, yv)

      self.val_conf:batchAdd(scores, yv)
      num_val = num_val + 1
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
end

function train_cls.train(train_data_co, optim_config)
  local self = train_cls
  self.data_co = train_data_co
  local _, loss = optim.adam(self.f, self.params, optim_config)
  table.insert(self.checkpoint.train_loss_history, loss[1]) 
  return loss
end

function train_cls.getCheckpoint()
  local self = train_cls
  local cp = {}
  for k,v in pairs(self.checkpoint) do cp[k] = v end
  cp.train_conf = self.train_conf:__tostring__()
  cp.conf = self.val_conf:__tostring__()
  return cp
end

return train_cls

