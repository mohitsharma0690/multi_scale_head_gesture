require 'nn';
local utils = require '../util/utils.lua'

local MS_BootstrapCrossEntropy, Criterion = torch.class('nn.MS_BootstrapCrossEntropy', 'nn.Criterion')

function MS_BootstrapCrossEntropy:__init()
   Criterion.__init(self)
   self.hard_beta = false
end

function MS_BootstrapCrossEntropy:updateOutput(input, target)
   --input = input:squeeze()
   local beta = input:clone()
   local one_minus_beta = torch.add(torch.ones(beta:size()):cuda(), -1, beta)
   -- NOTE:
   -- target is N length tensor
   -- pred is Nx5 tensor with values in LogSpace (since we do LogSoftMax)
   local target, pred = unpack(target)
   target = target:clone()
   pred = pred:clone()
   
   local dtype = target:type()

   -- Get target in one-hot format
   target = utils.get_one_hot_tensor(target, pred:size(2))
   target = target:type(dtype)
   if self.hard_beta then
     pred = utils.get_one_hot_tensor(torch.max(pred, 2):squeeze(), pred:size())
     pred = pred:type(dtype)
   end

   -- Get pred in normal space from log space
   local pred_normal = torch.exp(pred)
   local beta_exp = torch.expand(beta, beta:size(1), pred:size(2))
   local one_minus_beta_exp = torch.expand(one_minus_beta, beta:size(1), pred:size(2))

   -- sum[k] (beta*y[k] + (1-beta)*y_hat[k])*log(y_hat[k])
   local t = torch.add(target:cmul(beta_exp), pred_normal:cmul(one_minus_beta_exp))
   t = t:cmul(pred)
   self.output = torch.sum(t, 2)
   local N = self.output:size(1)
   self.output = -1*torch.sum(self.output) / N

   return self.output
end

function MS_BootstrapCrossEntropy:updateGradInput(input, target)
   input = input:squeeze()
   local beta = input
   -- NOTE:
   -- target is N length tensor
   -- pred is Nx5 tensor with values in LogSpace (since we do LogSoftMax)
   local target, pred = unpack(target)
   target = target:clone()
   pred = pred:clone()

   local dtype = target:type()

   -- Get target in one-hot format
   target = utils.get_one_hot_tensor(target, pred:size(2))
   target = target:type(dtype)

   -- Get pred in normal space from log space
   local pred_normal = torch.exp(pred)

   -- sum[k] (y[k] - y_hat[k])*log(y_hat[k])
   local t = torch.add(target, -1, pred_normal)
   t = t:cmul(pred)
   self.gradInput = torch.sum(t, 2)

  return self.gradInput
end

return nn.MS_BootstrapCrossEntropy

