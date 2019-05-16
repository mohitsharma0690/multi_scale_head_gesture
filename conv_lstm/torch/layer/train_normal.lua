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
  local self=train_cls
  self.dtype = args.dtype
  self.data_loader = args.data_loader
  self.model = args.model
  self.grads_history = args.grads_history
  self.train_conf = args.confusion
  self.val_conf = args.val_conf
  self.checkpoint = {
    crit2_loss_history={},
    train_loss_history = {},
    val_loss_history = {},
  }

  self.crit = nn.CrossEntropyCriterion():type(self.dtype)

  self.params, self.grad_params = self.model:getParameters()
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
  local self = train_cls
  assert(w == self.params)
  self.grad_params:zero()

  -- Get a minibach and run the model forward
  local success, x, y = self.read_data_co(self.data_co, self.data_loader)

  --if opt.num_scales == 1 then x = x[1]; x[2] = nil end
  if G_global_opts.num_scales == 1 then x = x[1];  end
  x = utils.convert_to_type(x, self.dtype)

  -- We want a tensor for User Classification
  if G_global_opts.train_user_classification == 1 then x = x[1] end
  y = y:type(self.dtype)
  
  local timer
  if G_global_opts.speed_benchmark ==1 then
    if cutorch then cutorch.synchronize() end
    timer = torch.Timer()
  end

  local scores = self.model:forward(x)

  -- Use the criterion to compute loss
  local loss = self.crit:forward(scores, y)

  -- Run the criterion and model backward to compute gradients
  -- TODO(Mohit): This also needs fixing
  local grad_scores = self.crit:backward(scores, y)

  self.model:backward(x, grad_scores)

  -- penalties (L1 and L2):
  if G_global_opts.coefL1 ~= 0 or G_global_opts.coefL2 ~= 0 then
    -- locals:
    local norm,sign = torch.norm,torch.sign
    local reconsLayerWeight = self.model.reconsLayer:currMaskedWeight()
    -- Loss:
    --
    loss = loss + G_global_opts.coefL1 * norm(reconsLayerWeight,1)
    loss = loss + G_global_opts.coefL2 * norm(reconsLayerWeight,2)^2/2

    -- Gradients:
    grad_params:add(sign(params):mul(G_global_opts.coefL1) + params:clone():mul(G_global_opts.coefL2) )
  end

  self.train_conf:batchAdd(scores, y)

  if G_global_opts.speed_benchmark == 1 then
    if cutorch then cutorch.synchronize() end
    local time = timer:time().real
    print('Forward / Backward pass took ', time)
    table.insert(forward_backward_times, time)
  end

  -- TODO(Mohit): Record memory usage

  -- TODO(Mohit): Clip the gradients as required.
  if G_global_opts.grad_clip > 0 then
    self.grad_params:clamp(-G_global_opts.grad_clip, G_global_opts.grad_clip)
  end

  if G_global_opts.debug_weights == 1 then
    local curr_grad_history = self.model:getGradWeights(loss, x, y)
    table.insert(self.grads_history, curr_grad_history)
  end

  return loss, self.grad_params
end

function train_cls.validate(val_data_co)
  local self = train_cls
  local val_loss, num_val = 0, 0

  self.val_conf:zero()

  while coroutine.status(val_data_co) ~= 'dead' do
    local success, xv, yv = coroutine.resume(val_data_co, self.data_loader)
    if success and xv ~= nil then

      --if opt.num_scales == 1 then x = x[1]; x[2] = nil end
      if G_global_opts.num_scales == 1 then xv = xv[1];  end
      xv = utils.convert_to_type(xv, self.dtype)

      -- We want a tensor for User Classification
      if G_global_opts.train_user_classification == 1 then xv = xv[1] end
      yv = yv:type(self.dtype)

      local scores = self.model:forward(xv)
      assert(torch.max(scores) == torch.max(scores))
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

function train_cls.train(train_data_co, optim_config, stats)
  local self = train_cls
  self.data_co = train_data_co

  local _, loss = optim.adam(self.f, self.params, optim_config)
  table.insert(self.checkpoint.train_loss_history, loss[1])
  local msg = 'Epoch: [%d/%d]\t Iteration:[%d/%d]\t loss: %.2f\t'

  local logs = self.checkpoint
  local args = {
    msg,
    stats.curr_epoch, stats.total_epoch,
    stats.curr_batch, stats.total_batch,
    logs.train_loss_history[#logs.train_loss_history],
  }
  print(string.format(unpack(args)))
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
