local utils = require '../util/utils.lua'

require 'nn'
require 'cunn'
require 'hdf5'
require 'math'
require 'paths'
require 'optim'

require 'model.MS_BootstrapCrossEntropy'

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
  self.val_conf = args.val_conf
  self.train_conf = args.confusion
  self.checkpoint = {
    train_loss_history={},
    crit1_loss_history={},
    crit2_loss_history={},
    pred_loss_history={},  -- This is just beta*crit1 + (1-beta)*crit2
    beta_loss_history={},
    beta_reg_loss_history={},
    val_loss_history={},
    beta_history={}
  }

  if G_global_opts.save_test_data_stats == 1 then
    self.test_data_stats = {
      test_scores = {},
      test_preds = {},
      test_data = {},
      test_beta = {},
    }
  end

  -- The last layer from the model would be 128x5. We remove it to add a
  -- parallel table that predicts y_hat and Beta
  if G_global_opts.save_test_data_stats ~= 1 then
    self.model.net:remove(9)
    self.model.net:remove(8)  -- Remove dropout

    self.model.net:add(nn.Replicate(2)) -- Copy into two tensors
    self.model.net:add(nn.SplitTable(1)) -- Split above tensor into two outputs
    self.final_table = nn.ParallelTable()

    -- Add the prediction model
    self.pred_model = nn.Sequential()
    self.pred_model:add(nn.Linear(128, 5))
    self.pred_model:add(nn.LogSoftMax())
    self.final_table:add(self.pred_model)

    -- Compute Beta
    self.beta_prob = nn.Sequential()
    self.beta_prob:add(nn.Linear(128, 1))
    self.beta_prob:add(nn.Sigmoid())
    self.final_table:add(self.beta_prob)

    self.model.net:add(self.final_table)

    self.model:updateType(self.dtype)
  end

  print(self.model.net)

  self.crit1 = nn.ClassNLLCriterion():type(self.dtype)
  self.crit2 = nn.ClassNLLCriterion():type(self.dtype)

  self.beta_crit = nn.MS_BootstrapCrossEntropy():type(self.dtype)
  self.beta_reg = nn.MSECriterion():type(self.dtype)
  self.coef_beta = G_global_opts['coef_beta_reg']
  self.coef_beta_start = self.coef_beta
  self.coef_beta_end = 0.5
  assert(self.coef_beta ~= nil)

  self.params, self.grad_params = self.model:getParameters()
end

function train_cls.get_current_beta(stats)
  local self = train_cls
  local total_it = stats.total_epoch * stats.total_batch
  local done_it = stats.curr_epoch * stats.total_batch
  local step = (self.coef_beta_start - self.coef_beta_end) / total_it
  local curr_beta = self.coef_beta_start - done_it * step
  if curr_beta < self.coef_beta_end then curr_beta = self.coef_beta_end end
  return curr_beta
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

function train_cls.f_opt_together(w)
  local self = train_cls
  assert(w == self.params)
  self.grad_params:zero()
  assert(G_global_opts.use_bootstrap_Beta_var == 1)
  assert(G_global_opts.bootstrap_Beta_var_opt_together == 1)

  local success, x, y = self.read_data_co(self.data_co, self.data_loader)
  if not x then return 0, self.grad_params end
  
  if G_global_opts.num_scales == 1 then x = x[1];  end
  x = x[1]
  x = utils.convert_to_type(x, self.dtype)
  y = utils.convert_to_type(y, self.dtype)

  local scores = self.model:forward(x)  -- scores is a table
  -- scores[1] is the target scores and scores[2] are the Beta scores (confidence)
  assert(torch.max(scores[1]) == torch.max(scores[1]))
  local _, preds = torch.max(scores[1], 2)
  local beta = scores[2]:clone()

  -- We want Beta to be 1
  local expected_beta = torch.ones(beta:size())
  expected_beta = expected_beta:type(self.dtype)
  local one_minus_beta = torch.add(torch.ones(beta:size()):cuda(), -1, beta)
  local scores = scores[1]:clone()

  local loss

  local loss_target = self.crit1:forward(scores, y)
  local preds_vec = preds:view(-1)
  local loss_pred = self.crit2:forward(scores, preds_vec)
  local loss_beta = self.beta_crit:forward(beta, {y, scores})
  local loss_reg = self.curr_coef_beta * self.beta_reg:forward(beta, expected_beta)

  local grad_target = self.crit1:backward(scores, y)
  local grad_pred = self.crit2:backward(scores, preds_vec)
  local grad_beta = self.beta_crit:backward(x, {y, scores})
  local grad_reg = self.beta_reg:backward(beta, expected_beta)
  grad_reg = grad_reg:mul(self.curr_coef_beta)

  local beta_exp = torch.expand(beta, beta:size(1), grad_target:size(2))
  local one_minus_beta_exp = torch.expand(one_minus_beta, beta:size(1), grad_pred:size(2))
  local grad_scores = torch.add(
      grad_target:cmul(beta_exp), grad_pred:cmul(one_minus_beta_exp))
  grad_beta = grad_beta:add(grad_reg)

  grad_scores = {grad_scores, grad_beta}

  -- Backrop the gradient
  self.model:backward(x, grad_scores)

  -- Update the confusion matrix
  self.train_conf:batchAdd(scores, y)

  -- This by itself is not a correct estimation of the loss but for now its Ok.
  -- Since we should do beta*loss_target + (1-beta)*loss_pred
  local total_pred_loss = torch.add(loss_target * beta, loss_pred * one_minus_beta)
  total_pred_loss = total_pred_loss:sum() / y:size(1)
  
  loss = loss_target + loss_pred + loss_beta + loss_reg

  table.insert(self.checkpoint.crit1_loss_history, loss_target)
  table.insert(self.checkpoint.crit2_loss_history, loss_pred)
  table.insert(self.checkpoint.pred_loss_history, total_pred_loss)
  table.insert(self.checkpoint.beta_loss_history, loss_beta)
  table.insert(self.checkpoint.beta_reg_loss_history, loss_reg)
  table.insert(self.checkpoint.beta_history, self.curr_coef_beta)

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

function train_cls.f(w)
  local _s = train_cls
  assert(w == _s.params)
  _s.grad_params:zero()
  assert(G_global_opts.use_bootstrap_Beta_var == 1)

  local success, x, y
  if _s.train_pred_model then
    success, x, y = _s.read_data_co(_s.data_co, _s.data_loader)
    if x ~= nil then _s.train_x, _s.train_y = torch.deserialize(torch.serialize(x)), y:clone()
    else _s.train_x, _s.train_y = nil, nil end
  else
    success, x, y = true, _s.train_x, _s.train_y
  end

  if not x then 
    return 0, _s.grad_params
  end

  if G_global_opts.num_scales == 1 then x = x[1];  end
  x = x[1]
  x = utils.convert_to_type(x, _s.dtype)
  y = utils.convert_to_type(y, _s.dtype)

  local scores = _s.model:forward(x)  -- scores is a table
  -- scores[1] is the target scores and scores[2] are the Beta scores (confidence)
  assert(torch.max(scores[1]) == torch.max(scores[1]))
  local _, preds = torch.max(scores[1], 2)
  local beta = scores[2]:clone()
  -- We want Beta to be 1
  local expected_beta = torch.ones(beta:size())
  expected_beta = expected_beta:type(_s.dtype)
  local one_minus_beta = torch.add(torch.ones(beta:size()):cuda(), -1, beta)
  local scores = scores[1]:clone()

  local loss
  if _s.train_pred_model then
    -- Train the prediction model with loss beta*y +(1-beta)*pred
    local loss_target = _s.crit1:forward(scores, y)
    local preds_vec = preds:view(-1)
    local loss_pred = _s.crit2:forward(scores, preds_vec)
    local loss_reg = _s.coef_beta * _s.beta_reg:forward(beta, expected_beta)

    local grad_target = _s.crit1:backward(scores, y)
    local grad_pred = _s.crit2:backward(scores, preds_vec)
    local grad_reg = _s.beta_reg:backward(beta, expected_beta)
    grad_reg = grad_reg:mul(_s.coef_beta)

    local total_loss = torch.add(
        torch.mul(beta, loss_target), torch.mul(one_minus_beta, loss_pred))

    local beta_exp = torch.expand(beta, beta:size(1), grad_target:size(2))
    local one_minus_beta_exp = torch.expand(one_minus_beta, beta:size(1), grad_pred:size(2))
    local grad_scores = torch.add(
        grad_target:cmul(beta_exp), grad_pred:cmul(one_minus_beta_exp))

    -- We don't backprop the regualrization loss in the first iteration. We
    -- optimize this only in the second iteration when we are directly 
    -- optimizing Beta.
    grad_scores = {grad_scores, torch.Tensor(beta:size()):zero()}
    grad_scores[2] = grad_scores[2]:type(_s.dtype)

    -- Backrop the gradient
    _s.model:backward(x, grad_scores)

    -- Update the confusion matrix
    _s.train_conf:batchAdd(scores, y)

    loss = loss_target + loss_pred + loss_reg

    table.insert(_s.checkpoint.crit1_loss_history, loss_target)
    table.insert(_s.checkpoint.crit2_loss_history, loss_pred)

  else 
    assert(_s.train_pred_model ~= nil)

    -- Train the bootstrap(confidence)model
    loss = _s.beta_crit:forward(beta, {y, scores})
    local loss_reg = _s.coef_beta * _s.beta_reg:forward(beta, expected_beta)
    local grad_scores = _s.beta_crit:backward(x, {y, scores})
    local grad_reg = _s.beta_reg:backward(beta, expected_beta)
    grad_reg = grad_reg:mul(_s.coef_beta)
    grad_scores = grad_scores:add(grad_reg)
    grad_scores = {torch.Tensor(scores:size()):zero(), grad_scores}
    grad_scores[1] = grad_scores[1]:type(_s.dtype)

    _s.model:backward(x, grad_scores)
    table.insert(_s.checkpoint.beta_loss_history, loss)
  end

  -- TODO(Mohit): Clip the gradients as required.
  if G_global_opts.grad_clip > 0 then
    _s.grad_params:clamp(-G_global_opts.grad_clip, G_global_opts.grad_clip)
  end

  if G_global_opts.debug_weights == 1 then 
    local curr_grad_history = _s.model:getGradWeights(loss, x, y) 
    table.insert(_s.grads_history, curr_grad_history)
  end

  return loss, _s.grad_params
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
      assert(torch.max(scores[1]) == torch.max(scores[1]))
      val_loss = val_loss + self.crit1:forward(scores[1], yv)

      self.val_conf:batchAdd(scores[1], yv)
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

function train_cls.validate_save(val_data_co)
  local self = train_cls
  local val_loss, num_val = 0, 0

  self.val_conf:zero()

  while coroutine.status(val_data_co) ~= 'dead' do
    local success, xv, yv, batch = coroutine.resume(val_data_co, self.data_loader) 
    if success and xv ~= nil then

      if G_global_opts.num_scales == 1 then xv = xv[1]; end
      xv = utils.convert_to_type(xv, self.dtype)
      yv = utils.convert_to_type(yv, self.dtype)
      xv = xv[1]

      local scores = self.model:forward(xv)
      local beta = scores[2]:clone()
      scores = scores[1] 
      assert(torch.max(scores) == torch.max(scores))

      self.val_conf:batchAdd(scores, yv)
      num_val = num_val + 1

      -- Save test data stats 
      scores = torch.exp(scores)
      local scores_max, scores_max_idx = torch.max(scores, 2)
      for i=1,#batch do table.insert(self.test_data_stats.test_data, batch[i]) end
      scores_max_idx = torch.totable(scores_max_idx)
      for i=1,#scores_max_idx do
        table.insert(self.test_data_stats.test_preds, scores_max_idx[i])
      end
      scores = torch.totable(scores)
      for i=1,#scores do 
        table.insert(self.test_data_stats.test_scores, scores[i])
      end
      beta = torch.totable(beta)
      for i=1,#beta do
        table.insert(self.test_data_stats.test_beta, beta[i])
      end

    elseif success ~= true then
      print('Validation data coroutine failed')
      print(xv)
    end
  end

  print('Total frames evaluated: '..num_val)
  return self.test_data_stats
end

function train_cls.train(train_data_co, optim_config, stats)
  local self = train_cls
  self.data_co = train_data_co

  local loss
  if G_global_opts.bootstrap_Beta_var_opt_together == 1 then 
    -- Update beta
    self.curr_coef_beta = self.get_current_beta(stats)

    _, loss = optim.adam(self.f_opt_together, self.params, optim_config)
    table.insert(self.checkpoint.train_loss_history, loss[1]) 
    local msg = 'Epoch: [%d/%d]\t Iteration:[%d/%d]\tTarget(y) loss: %.2f\t'
    msg = msg..'y_hat loss: %.2f\t Actual pred loss: %.2f\t'
    msg = msg..'beta_loss: %.2f\t beta_reg_loss: %.2f\t beta:%.2f'


    local logs = self.checkpoint
    local args = {
      msg,
      stats.curr_epoch, stats.total_epoch,
      stats.curr_batch, stats.total_batch,
      logs.crit1_loss_history[#logs.crit1_loss_history],
      logs.crit2_loss_history[#logs.crit2_loss_history],
      logs.pred_loss_history[#logs.pred_loss_history],
      logs.beta_loss_history[#logs.beta_loss_history],
      logs.beta_reg_loss_history[#logs.beta_reg_loss_history],
      logs.beta_history[#logs.beta_history],
    }
    print(string.format(unpack(args)))
  else

    self.train_pred_model = true
    _, loss = optim.adam(self.f, self.params, optim_config)
    table.insert(self.checkpoint.train_loss_history, loss[1]) 
    print(string.format('Pred model loss: %.2f', loss[1]))
    self.train_pred_model = false
    local _, loss = optim.adam(self.f, self.params, optim_config)

    table.insert(self.checkpoint.beta_loss_history, loss[1]) 
    print(string.format('Beta model loss: %.2f', loss[1]))
  end
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

