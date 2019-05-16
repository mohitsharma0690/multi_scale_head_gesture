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
  local self = train_cls
  self.dtype = args.dtype
  self.params, train_cls.grad_params = args.params, args.grad_params
  self.data_loader = args.data_loader
  self.data_co = args.data_loader
  self.model = args.model
  self.confusion = args.confusion
  self.checkpoint = {
    train_loss_history={},
    crit2_loss_history={},
    train_conf=confusion,
  }

  model.net:add(nn.LogSoftMax())
  self.crit = nn.ClassNLLCriterion():type(dtype)
  self.crit2 = nn.ClassNLLCriterion():type(dtype)
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

function train_cls.f_labelCorrection(w)
  local self = train_cls
  assert(w == params)
  self.grad_params:zero()
  assert(G_global_opts.use_label_correction == 1)

  local success, x, y = self.read_data_co(self.data_co, self.data_loader)
  if x == nil then return 0, self.grad_params end
  local label_corr_inp
  x, label_corr_inp = unpack(x)

  if G_global_opts.num_scales == 1 then x = x[1];  end
  x = utils.convert_to_type(x, dtype)
  y = utils.convert_to_type(y, dtype)

  x = x[1]

  -- First index are the file and timestamp stats
  self.model.reconsLayer:setMaskForInput(label_corr_inp[2])

  local scores = self.model:forward(x)
  local label_softmax = nn.SoftMax():type(dtype)
  local label_y = utils.get_one_hot_tensor(y, opt.num_classify):cuda()

  -- label_scores are softmax_scores
  local label_scores = label_softmax:forward(scores)
  -- We use scores here since CrossEntropy loss assumes LogSoftMax
  local loss = self.crit:forward(scores, y)
  local label_loss = self.label_crit:forward(label_scores, label_y)

  local grad_scores = self.crit:backward(scores, y)
  local label_grad_scores = self.label_crit:backward(label_scores, label_y)
  label_grad_scores = label_softmax:backward(scores, label_grad_scores)

  -- Add the two gradients together to backpropagate
  grad_scores = grad_scores:add(label_grad_scores)

  self.model:backward(x, grad_scores)

  -- penalties (L1 and L2):
  if G_global_opts.coefL1 ~= 0 or G_global_opts.coefL2 ~= 0 then
    -- locals:
    local norm,sign= torch.norm,torch.sign
    local reconsLayerWeight = self.model.reconsLayer:currMaskedWeight()
    -- Loss:
    --
    loss = loss + opt.coefL1 * norm(reconsLayerWeight,1)
    loss = loss + opt.coefL2 * norm(reconsLayerWeight,2)^2/2

    -- Gradients:
    grad_params:add(sign(params):mul(opt.coefL1) + params:clone():mul(opt.coefL2) )
  end

  -- update confusion
  self.confusion:batchAdd(label_scores, y)

  -- TODO(Mohit): Clip the gradients as required.
  if G_global_opts.grad_clip > 0 then
    grad_params:clamp(-G_global_opts.grad_clip, G_global_opts.grad_clip)
  end

  if G_global_opts.debug_weights == 1 then
    self.save_grad_weights_label_corr(model, loss, label_loss, x, y)
  end

  return loss+label_loss, grad_params

end

return train_cls


