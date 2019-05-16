
local ReconsLayer, parent = torch.class('nn.ReconsLayer', 'nn.Module')

function ReconsLayer:__init(num_weights, batch_size, num_classes)
  parent.__init(self)

  self.__num_weights = num_weights
  self.__batch_size = batch_size
  self.__num_classes = num_classes
  self.weight = torch.Tensor(self.__num_weights*self.__num_classes, self.__num_classes)
  self.bias = torch.Tensor(self.__num_weights*self.__num_classes):zero()
  -- self.output = torch.Tensor(self.__batch_size)
  self.output = torch.Tensor(self.__batch_size, self.__num_classes):zero()
  self.gradWeight = torch.Tensor(self.__num_weights*self.__num_classes, 5):zero()
  self.gradBias = torch.Tensor(self.__num_weights*self.__num_classes):zero()
  self.gradInput = torch.Tensor(self.__batch_size, self.__num_classes):zero()
  self.__mask = torch.ByteTensor(self.__num_weights*self.__num_classes, 5)
  self.__mask:zero()
  self.mask_set = false
  self:__reset_val()
end

-- m is a Tensor with each element value being the corresponding row index in m
function ReconsLayer:__set_mask(m)
  assert(m ~= nil)
  for k=1,m:size(1) do
    idx = m[k]
    self.__mask[{{idx,idx+self.__num_classes-1},{}}]:fill(1)
  end
  self.mask_idx = m
  self.mask_set = true
end

function ReconsLayer:__reset_mask()
  self.__mask:zero()
  self.mask_set = false
  self.mask_idx = nil
end

-- inp: Table of indexes being used in this minibatch
function ReconsLayer:setMaskForInput(inp)
  local inp_t = torch.Tensor(inp):clone()
  self:__reset_mask()
  -- HACK!! To convert __mask to byte tensor
  self.__mask = self.__mask:eq(0)
  self.__mask:zero()
  self:__set_mask(inp_t)
end

function ReconsLayer:__reset_val()
  self.weight:zero()
  for i=1,self.weight:size(1),self.__num_classes do
    self.weight[{{i,i+self.__num_classes-1},{}}] = 0.01*torch.eye(self.__num_classes)
  end
  self.bias:zero()
end

function ReconsLayer:__get_one_hot_tensor(inp)
  local one_hot_val = torch.Tensor(#inp, self.__num_classes):zero()
  for k, idx in ipairs(inp) do
    one_hot_val[k][idx] = 1
  end
  return one_hot_val
end

function ReconsLayer:currMaskedWeight()
  local maskedWeight = self.weight:maskedSelect(self.__mask)
  return maskedWeight:clone()
end

-- input: Tensor of size (N, 5)
function ReconsLayer:updateOutput(input)
  if self.train == false then
    if self.mask_set then self:__reset_mask() end
    self.output = input:clone()  -- This layer has no effect during testing
    return
  end

  assert(self.mask_set)
  self.recompute_backward = true
  -- We should manually loop through here since we don't know the order in the
  -- input is the usual lexical order for weights
  self.output:zero()
  assert(input:dim() == 2)
  if self.output:dim() ~= 2 then 
    print(input:size())
    print(self.output:size())
    self.output:resize(input:size())
    self.output:zero()
  end
  assert(self.output:dim() == 2)
  local c = self.__num_classes
  for i=1,input:size(1) do
    local inp = input[{{i},{}}]  -- (C,) tensor
    local w_idx = self.mask_idx[i]
    local W = self.weight[{{w_idx,w_idx+c-1},{}}] -- (C,C) tensor
    local b = self.bias[{{w_idx,w_idx+c-1}}] -- (C,) tensor
    -- local output = self.output[{{i},{}}]  -- (C,) tensor
    -- Make sure we initially use Identity matrix
    --[[
    assert(W:sum() == c)
    for j=1,c do assert(W[j][j] == 1) end
    ]]
    assert(self.output:dim() == 2)

    self.output[{{i},{}}] = b + W*inp:t()
  end
  -- self.output should now have the linear configuration
  for i=1,self.output:size(1) do
    for j=1,self.output:size(2) do
      if self.output[i][j] ~= self.output[i][j] then
        print("Nan detected")
        print(self.output)
      end
      assert(self.output[i][j] == self.output[i][j])
    end
  end
  return self.output
end

-- gradOutput will be a tensor that we should convert to 
function ReconsLayer:backward(input, gradOutput, scale)
  self.recompute_backward = false

  assert(gradOutput:size(1) == self.output:size(1))
  assert(gradOutput:size(2) == self.output:size(2))

  for i=1,gradOutput:size(1) do
    for j=1,gradOutput:size(2) do
      if gradOutput[i][j] ~= gradOutput[i][j] then
        print("Nan detected")
        print(self.output)
        print(gradOutput)
      end
      assert(gradOutput[i][j] == gradOutput[i][j])
    end
  end
  if torch.max(gradOutput) ~= torch.max(gradOutput) then
    print("NaN detected")
    print(grad_output)
    assert(false)
  end

  local c = self.__num_classes
  self.gradInput:resizeAs(input)
  self.gradInput:zero()

  local scale = 1.0

  for i=1, input:size(1) do
    local grad_op = gradOutput[{{i},{}}] -- (C,) tensor
    local inp = input[{{i},{}}]  -- (C,) tensor
    local w_idx = self.mask_idx[i]
    local W = self.weight[{{w_idx,w_idx+c-1},{}}] -- (C,C) tensor

    local temp_gradWeight = grad_op:t() * inp
    temp_gradWeight= temp_gradWeight:mul(scale)
    self.gradWeight[{{w_idx,w_idx+c-1},{}}] = temp_gradWeight -- dl/do * x.T
    self.gradBias[{{w_idx,w_idx+c-1}}] = grad_op:clone()
    self.gradInput[{{i},{}}] = W:t() * grad_op:t()
  end

  -- TODO(Mohit): Verify using grad checker.
  return self.gradInput
end

function ReconsLayer:updateGradInput(input, gradOutput)
  if self.recompute_backward then
    self:backward(input, gradOutput, 1.0)
  end
  return self.gradInput
end

function ReconsLayer:accGradParameters(input, gradOutput, scale)
  if self.recompute_backward then
    self:backward(input, gradOutput, scale)
  end
end

--function ReconsLayer:zeroGradParameters()
--end

function ReconsLayer:parameters()
  return {[1]=self.weight, [2]=self.bias}, {[1]=self.gradWeight, [2]=self.gradBias}
  --return {[1]=self.weight, self.bias}, {[1]=self.gradWeight, [2]=self.gradBias}
end

