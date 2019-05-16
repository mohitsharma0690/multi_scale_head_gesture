require 'model/MS_BRNN'
require 'model/MyLSTM'

local MS_BLSTM, parent = torch.class('nn.MS_BLSTM', 'nn.Module')

function MS_BLSTM:__init(input_dim, hidden_dim)
  parent.__init(self)
  self.fwd = nn.MyLSTM(input_dim, hidden_dim)
  self.fwd.remember_states = false
  self.back = nn.MyLSTM(input_dim, hidden_dim)
  self.back.remember_states = false
  -- BRNN uses the second dimension as T by default
  self.brnn = nn.MS_BRNN(self.fwd, self.back)
end

function MS_BLSTM:resetStates()
  self.fwd:resetStates()
  self.back:resetStates()
end

function MS_BLSTM:updateOutput(input)
  self.output = self.brnn:updateOutput(input)
  return self.output
end

function MS_BLSTM:updateGradInput(input, gradOutput)
  self.gradInput = self.brnn:updateGradInput(input, gradOutput)
  return self.gradInput
end

function MS_BLSTM:updateType(dtype)
  self.brnn:udpateType(dtype)
end

function MS_BLSTM:parameters()
  return self.brnn:parameters()
end

function MS_BLSTM:training()
  self.brnn:training()
  parent.training(self)
end

function MS_BLSTM:evaluate()
  self.brnn:evaluate()
  parent.evaluate(self)
end

function MS_BLSTM:clearState()
  self.brnn:clearState()
end

function MS_BLSTM:accGradParameters(input, gradOutput, scale)
    self.brnn:accGradParameters(input, gradOutput, scale)
end

function MS_BLSTM:accUpdateGradParameters(input, gradOutput, lr)
    self.brnn:accUpdateGradParameters(input, gradOutput, lr)
end

function MS_BLSTM:sharedAccUpdateGradParameters(input, gradOutput, lr)
    self.brnn:sharedAccUpdateGradParameters(input, gradOutput, lr)
end

