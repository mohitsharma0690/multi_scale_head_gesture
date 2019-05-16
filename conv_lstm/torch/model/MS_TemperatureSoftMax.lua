require 'torch'

local MS_TemperatureSoftMax, parent = torch.class('nn.MS_TemperatureSoftMax', 'nn.Module')

function MS_TemperatureSoftMax:__init(temperature)
  self.temperature = temperature
  parent.__init(self)
end

function MS_TemperatureSoftMax:updateOutput(input)
  if self.temperature ~= 1 then
    self.output = torch.csub(input, 
      torch.repeatTensor(torch.max(input, 2), 1, input:size(2)))
    self.output:div(self.temperature)
    self.output:exp()
    self.output:cdiv(torch.repeatTensor(
      torch.sum(self.output, 2), 1, self.output:size(2)))
  else
    input.THNN.SoftMax_updateOutput(
      input:cdata(),
      self.output:cdata()
    )
  end
  return self.output
end

function MS_TemperatureSoftMax:updateGradInput(input, gradOutput)
  -- The gradient isn't impacted by Temperature.
  input.THNN.SoftMax_updateGradInput(
    input:cdata(),
    gradOutput:cdata(),
    self.gradInput:cdata(),
    self.output:cdata()
  )
  return self.gradInput
end

