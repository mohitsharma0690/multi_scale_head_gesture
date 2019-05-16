local nn = require 'nn'
require 'rnn'
require '../modules/Gaussian'

local LSTMVarAutoEncoder = {
  cellSizes = {256, 256}, -- Number of LSTM cells
  encLSTMs = {},
  decLSTMs = {},
  zSize = 256,
  frame_dropout_prob = 0.3,
}

-- Copy encoder cell and output to decoder LSTM
function LSTMVarAutoEncoder:forwardConnect()
  for l = 1, #LSTMVarAutoEncoder.decLSTMs do
    LSTMVarAutoEncoder.decLSTMs[l].userPrevOutput = LSTMVarAutoEncoder.encLSTMs[l].output[self.seqLen]
    LSTMVarAutoEncoder.decLSTMs[l].userPrevCell = LSTMVarAutoEncoder.encLSTMs[l].cell[self.seqLen]
  end
end

-- Copy decoder gradients to encoder LSTM
function LSTMVarAutoEncoder:backwardConnect()
  for l = 1, #LSTMVarAutoEncoder.encLSTMs do
    LSTMVarAutoEncoder.encLSTMs[l].userNextGradCell = LSTMVarAutoEncoder.decLSTMs[l].userGradPrevCell
    LSTMVarAutoEncoder.encLSTMs[l].gradPrevOutput = LSTMVarAutoEncoder.decLSTMs[l].userGradPrevOutput
  end
end

function LSTMVarAutoEncoder:createSequenceCompletor(inp_size)
  local featureSize = inp_size[2] * inp_size[3]
  self.seqLen = inp_size[2] -- Treat rows as a sequence
  
  -- Create encoder
  self.encoder = nn.Sequential()
  self.encoder:add(nn.Transpose({1, 2})) -- Transpose to seqlen x batch
  for l = 1, #LSTMVarAutoEncoder.cellSizes do
    local inputSize = l == 1 and inp_size[3] or LSTMVarAutoEncoder.cellSizes[l - 1]
    self.encLSTMs[l] = nn.SeqLSTM(inputSize, LSTMVarAutoEncoder.cellSizes[l])
    self.encLSTMs[l]:set_name('enc_'..l)
    self.encoder:add(self.encLSTMs[l])
  end
  -- Add mean and log var here
  self.encoder:add(nn.Select(1, self.seqLen))
  local zLayer = nn.ConcatTable()
  -- Mean of z
  zLayer:add(nn.Linear(cellSizes[#cellSizes], LSTMVarAutoEncoder.zSize))
  -- log var^2 of z
  zLayer:add(nn.Linear(cellSizes[#cellSizes], LSTMVarAutoEncoder.zSize))
  self.encoder:add(zLayer)
  
  -- Create sampler
  self.sampler = nn.Sequential() 

  -- Create sigma*eps module
  local noiseModule = nn.Sequential()
  local noiseModuleInternal = nn.ConcatTable()
  local stdModule = nn.Sequential()
  stdModule:add(nn.MulConstant(0.5)) -- This gives us log sigma
  stdModule:add(nn.Exp()) -- Now we have sigma
  noiseModuleInternal:add(stdModule) -- sigma
  noiseModuleInternal:add(nn.Gaussian(0, 1)) -- eps
  noiseModule:add(noiseModuleInternal)
  noiseModule:add(nn.CMulTable()) -- sigma*eps

  local addMeanSigma = nn.ParallelTable()
  addMeanSigma:add(nn.Identity()) -- mean
  addMeanSigma:add(noiseModule) -- sigma*eps
  self.sampler:add(addMeanSigma)
  self.sampler:add(nn.CAddTable()) -- mean + sigma*eps

  -- Create decoder
  self.decoder = nn.Sequential()
  self.decoder:add(nn.Transpose({1, 2})) -- Transpose to seqlen x batch
  for l = 1, #LSTMVarAutoEncoder.cellSizes do
    local inputSize = l == 1 and inp_size[3] or LSTMVarAutoEncoder.cellSizes[l - 1]
    -- Retain hidden state on consecutive calls to forward during evaluation
    -- TODO(Mohit): Should we remember on 'val'
    self.decLSTMs[l] = nn.SeqLSTM(
        inputSize, 
        LSTMVarAutoEncoder.cellSizes[l]) -- :remember('eval')
    self.decLSTMs[l]:set_name('dec_'..l)
    self.decoder:add(self.decLSTMs[l])
  end
  
  -- Reconstruct columns
  self.decoder:add(nn.Sequencer(nn.Linear(
        LSTMVarAutoEncoder.cellSizes[#LSTMVarAutoEncoder.cellSizes], inp_size[3])))
  self.decoder:add(nn.Transpose({1, 2})) -- Transpose back to batch x seqlen
  -- It is not necessary to use sigmoid
  --self.decoder:add(nn.Sigmoid(true))
  
  -- Create dummy container for getParameters (no other way to combine storage
  -- pointers)
  self.dummyContainer = nn.Sequential()
  self.dummyContainer:add(self.encoder)
  self.dummyContainer:add(self.sampler)
  self.dummyContainer:add(self.decoder)

  -- Create autoencoder wrapper
  self.autoencoder = {
    parent = self
  }
  
  -- Create CUDA wrapper
  function self.autoencoder:cuda()
    self.parent.encoder:cuda()
    self.parent.sampler:cuda()
    self.parent.decoder:cuda()
  end

  -- Create replace wrapper
  function self.autoencoder:replace(fn)
    self.parent.dummyContainer:replace(fn)
  end

  -- Create getParameters wrapper
  function self.autoencoder:getParameters()
    return self.parent.dummyContainer:getParameters()
  end

  -- Create training wrapper
  function self.autoencoder:training()
    self.parent.encoder:training()
    self.parent.sampler:training()
    self.parent.decoder:training()
  end

  -- Create evaluate wrapper
  function self.autoencoder:evaluate()
    self.parent.encoder:evaluate()
    self.parent.sampler:evaluate()
    self.parent.decoder:evaluate()
  end

  -- Create forward wrapper
  -- Use 1..10 timesteps to get the encoding vector. We want to predict 11..20
  -- timesteps conditioned on 21..30 timesteps.
  function self.autoencoder:forward(x)
    -- Take the first 30 frames (x should have 32 frames initially)
    local temp_x = x:clone()
    temp_x = temp_x[{{},{1,30},{}}]
    self.encoderOutput = self.parent.encoder:forward(temp_x[{{},{1,10},{}}])

    self.parent.sampler:forward(self.encoderOutput)
    local z = self.parent.sampler.output
    LSTMVarAutoEncoder.decLSTMs[1].userPrevOutput = z

    -- Unconditional decoder
    local decInSeq = temp_x[{{},{21,30},{}}]
    decInSeq = decInSeq:cuda()
    return self.parent.decoder:forward(decInSeq)
  end
  
  -- Create backward wrapper
  -- We don't do the backward pass for KL-loss here but in feval in the training
  -- loop
  function self.autoencoder:backward(x, gradLoss)
    -- Take the first 30 frames (x should have 32 frames initially)
    local temp_x = x:clone()
    temp_x = temp_x[{{},{1,30},{}}]
    local decInSeq = temp_x[{{},{21,30},{}}]
    decInSeq = decInSeq:cuda()

    self.parent.decoder:backward(decInSeq, gradLoss)

    self.parent.sampler:backward(self.encoderOutput,
      LSTMVarAutoEncoder.decLSTMs[1].userGradPrevOutput)

    LSTMVarAutoEncoder.encLSTMs[1].gradPrevOutput = self.parent.sampler.gradInput
     -- seqlen x batch
    local zeroTensor = torch.Tensor(
        10,  -- Since we don't use the entire sequence x
        x:size(1),
        LSTMVarAutoEncoder.cellSizes[#LSTMVarAutoEncoder.cellSizes]):zero():typeAs(x)
    return self.parent.encoder:backward(x, zeroTensor)
  end
end

function LSTMVarAutoEncoder:createPredictor(inp_size)
  local featureSize = inp_size[2] * inp_size[3]
  self.seqLen = inp_size[2] -- Treat rows as a sequence
  
  -- Create encoder
  self.encoder = nn.Sequential()
  self.encoder:add(nn.Transpose({1, 2})) -- Transpose to seqlen x batch
  for l = 1, #LSTMVarAutoEncoder.cellSizes do
    local inputSize = l == 1 and inp_size[3] or LSTMVarAutoEncoder.cellSizes[l - 1]
    self.encLSTMs[l] = nn.SeqLSTM(inputSize, LSTMVarAutoEncoder.cellSizes[l])
    self.encLSTMs[l]:set_name('enc_'..l)
    self.encoder:add(self.encLSTMs[l])
  end
  -- Add mean and log var here
  self.encoder:add(nn.Select(1, self.seqLen))
  local zLayer = nn.ConcatTable()
  -- Mean of z
  zLayer:add(nn.Linear(cellSize, LSTMVarAutoEncoder.zSize))
  -- log var^2 of z
  zLayer:add(nn.Linear(cellSize, LSTMVarAutoEncoder.zSize))
  self.encoder:add(zLayer)
  
  -- Create sampler
  self.sampler = nn.Sequential() 

  -- Create sigma*eps module
  local noiseModule = nn.Sequential()
  local noiseModuleInternal = nn.ConcatTable()
  local stdModule = nn.Sequential()
  stdModule:add(nn.MulConstant(0.5)) -- This gives us log sigma
  stdModule:add(nn.Exp()) -- Now we have sigma
  noiseModuleInternal:add(stdModule) -- sigma
  noiseModuleInternal:add(nn.Gaussian(0, 1)) -- eps
  noiseModule:add(noiseModuleInternal)
  noiseModule:add(nn.CMulTable()) -- sigma*eps

  local addMeanSigma = nn.ParallelTable()
  addMeanSigma:add(nn.Identity()) -- mean
  addMeanSigma:add(noiseModule) -- sigma*eps
  self.sampler:add(addMeanSigma)
  self.sampler:add(nn.CAddTable()) -- mean + sigma*eps

  -- Create decoder
  self.decoder = nn.Sequential()
  self.decoder:add(nn.Transpose({1, 2})) -- Transpose to seqlen x batch
  for l = 1, #LSTMVarAutoEncoder.cellSizes do
    local inputSize = l == 1 and inp_size[3] or LSTMVarAutoEncoder.cellSizes[l - 1]
    -- Retain hidden state on consecutive calls to forward during evaluation
    -- TODO(Mohit): Should we remember on 'val'
    self.decLSTMs[l] = nn.SeqLSTM(
        inputSize, 
        LSTMVarAutoEncoder.cellSizes[l]) -- :remember('eval')
    self.decLSTMs[l]:set_name('dec_'..l)
    self.decoder:add(self.decLSTMs[l])
  end
  
  -- Reconstruct columns
  self.decoder:add(nn.Sequencer(nn.Linear(
        LSTMVarAutoEncoder.cellSizes[#LSTMVarAutoEncoder.cellSizes], inp_size[3])))
  self.decoder:add(nn.Transpose({1, 2})) -- Transpose back to batch x seqlen
  -- It is not necessary to use sigmoid
  --self.decoder:add(nn.Sigmoid(true))
  

  -- Create dummy container for getParameters (no other way to combine storage
  -- pointers)
  self.dummyContainer = nn.Sequential()
  self.dummyContainer:add(self.encoder)
  self.dummyContainer:add(self.sampler)
  self.dummyContainer:add(self.decoder)

  -- Create autoencoder wrapper
  self.autoencoder = {
    parent = self
  }
  
  -- Create CUDA wrapper
  function self.autoencoder:cuda()
    self.parent.encoder:cuda()
    self.parent.sampler:cuda()
    self.parent.decoder:cuda()
  end

  -- Create replace wrapper
  function self.autoencoder:replace(fn)
    self.parent.dummyContainer:replace(fn)
  end

  -- Create getParameters wrapper
  function self.autoencoder:getParameters()
    return self.parent.dummyContainer:getParameters()
  end

  -- Create training wrapper
  function self.autoencoder:training()
    self.parent.encoder:training()
    self.parent.sampler:training()
    self.parent.decoder:training()
  end

  -- Create evaluate wrapper
  function self.autoencoder:evaluate()
    self.parent.encoder:evaluate()
    self.parent.sampler:evaluate()
    self.parent.decoder:evaluate()
  end

  -- Create forward wrapper
  function self.autoencoder:forward(x)
    self.encoderOutput = self.parent.encoder:forward(x)

    self.parent.sampler:forward(self.encoderOutput)
    local z = self.parent.sampler.output
    LSTMVarAutoEncoder.decLSTMs[1].userPrevOutput = z

    -- Unconditional decoder
    local decInSeq = torch.Tensor(x:size()):zero()
    decInSeq = decInSeq:cuda()
    return self.parent.decoder:forward(decInSeq)
  end
  
  -- Create backward wrapper
  function self.autoencoder:backward(x, gradLoss)
    local decInSeq = torch.Tensor(x:size(1),8,x:size(3)):zero()
    decInSeq = decInSeq:cuda()

    self.parent.decoder:backward(decInSeq, gradLoss)
    self.parent:backwardConnect()

     -- seqlen x batch
    local zeroTensor = torch.Tensor(
        x:size(2),
        x:size(1),
        LSTMVarAutoEncoder.cellSizes[#LSTMVarAutoEncoder.cellSizes]):zero():typeAs(x)
    return self.parent.encoder:backward(x, zeroTensor)
  end
end

function LSTMVarAutoEncoder:createNextStepPredictor(X)
  local featureSize = X:size(2) * X:size(3)
  self.seqLen = X:size(2) -- Treat rows as a sequence

  -- Create encoder
  self.encoder = nn.Sequential()
  self.encoder:add(nn.Transpose({1, 2})) -- Transpose to seqlen x batch
  for l = 1, #LSTMVarAutoEncoder.cellSizes do
    local inputSize = l == 1 and X:size(3) or LSTMVarAutoEncoder.cellSizes[l - 1]
    self.encLSTMs[l] = nn.SeqLSTM(inputSize, LSTMVarAutoEncoder.cellSizes[l])
    self.encLSTMs[l]:set_name('enc_'..l)
    self.encoder:add(self.encLSTMs[l])
  end

  -- Create decoder
  self.decoder = nn.Sequential()
  self.decoder:add(nn.Transpose({1, 2})) -- Transpose to seqlen x batch
  for l = 1, #LSTMVarAutoEncoder.cellSizes do
    local inputSize = l == 1 and X:size(3) or LSTMVarAutoEncoder.cellSizes[l - 1]
    -- Retain hidden state on consecutive calls to forward during evaluation
    -- TODO(Mohit): Should we remember on 'val'
    self.decLSTMs[l] = nn.SeqLSTM(
        inputSize, 
        LSTMVarAutoEncoder.cellSizes[l]) -- :remember('eval')
    self.decLSTMs[l]:set_name('dec_'..l)
    self.decoder:add(self.decLSTMs[l])
  end
  self.decoder:add(nn.Sequencer(nn.Linear(
        LSTMVarAutoEncoder.cellSizes[#LSTMVarAutoEncoder.cellSizes], X:size(3)))) -- Reconstruct columns
  self.decoder:add(nn.Transpose({1, 2})) -- Transpose back to batch x seqlen
  -- It is not necessary to use sigmoid
  -- self.decoder:add(nn.Sigmoid(true))

  -- Create dummy container for getParameters (no other way to combine storage pointers)
  self.dummyContainer = nn.Sequential()
  self.dummyContainer:add(self.encoder)
  self.dummyContainer:add(self.decoder)

  -- Create autoencoder wrapper
  self.autoencoder = {
    parent = self
  }
  
  -- Create CUDA wrapper
  function self.autoencoder:cuda()
    self.parent.encoder:cuda()
    self.parent.decoder:cuda()
  end

  -- Create replace wrapper
  function self.autoencoder:replace(fn)
    self.parent.dummyContainer:replace(fn)
  end

  -- Create getParameters wrapper
  function self.autoencoder:getParameters()
    return self.parent.dummyContainer:getParameters()
  end

  -- Create training wrapper
  function self.autoencoder:training()
    self.parent.encoder:training()
    self.parent.decoder:training()
  end

  -- Create evaluate wrapper
  function self.autoencoder:evaluate()
    self.parent.encoder:evaluate()
    self.parent.decoder:evaluate()
  end

  -- Create forward wrapper
  function self.autoencoder:forward(x)
    local encOut = self.parent.encoder:forward(x)
    self.parent:forwardConnect()

    -- Use target vector in training, sample from self in evaluate
    if self.parent.decoder.train then
      -- Shift decoder input sequence by one step forward
      local decInSeq = x:clone()
      decInSeq[{{}, {2, x:size(2)}, {}}] = decInSeq[{{}, {1, x:size(2) - 1}, {}}]
      decInSeq[{{}, {1}, {}}]:zero() -- Start from vector of zeros
      return self.parent.decoder:forward(decInSeq)
    else
      local decOut = torch.zeros(x:size()):typeAs(x)
      local decOutPartial = torch.zeros(
          x:size(1), 1, x:size(3)):typeAs(x) -- Start from vector of zeros

      -- Feed outputs back into self
      self.parent.decoder:forget() -- Clear hidden state
      for t = 1, self.parent.seqLen do
        decOutPartial = self.parent.decoder:forward(decOutPartial)
        decOut[{{}, {t}, {}}] = decOutPartial
      end
      return decOut
    end
  end
  
  -- Create backward wrapper
  function self.autoencoder:backward(x, gradLoss)
    -- Shift decoder input sequence by one step forward
    local decInSeq = x:clone()
    decInSeq[{{}, {2, x:size(2)}, {}}] = decInSeq[{{}, {1, x:size(2) - 1}, {}}]
    decInSeq[{{}, {1}, {}}]:zero()
    self.parent.decoder:backward(decInSeq, gradLoss)
    self.parent:backwardConnect()

    -- seqlen x batch
    local zeroTensor = torch.Tensor(
        x:size(2),
        x:size(1),
        LSTMVarAutoEncoder.cellSizes[#LSTMVarAutoEncoder.cellSizes]):zero():typeAs(x) 
    return self.parent.encoder:backward(x, zeroTensor)
  end
end

function LSTMVarAutoEncoder:createAutoencoder(X)
  local featureSize = X:size(2) * X:size(3)
  self.seqLen = X:size(2) -- Treat rows as a sequence
  local cellSize = LSTMVarAutoEncoder.cellSizes[#LSTMVarAutoEncoder.cellSizes]

  -- Create encoder
  self.encoder = nn.Sequential()
  self.encoder:add(nn.Transpose({1, 2})) -- Transpose to seqlen x batch
  for l = 1, #LSTMVarAutoEncoder.cellSizes do
    local inputSize = l == 1 and X:size(3) or LSTMVarAutoEncoder.cellSizes[l - 1]
    self.encLSTMs[l] = nn.SeqLSTM(inputSize, LSTMVarAutoEncoder.cellSizes[l])
    self.encLSTMs[l]:set_name('enc_'..l)
    self.encoder:add(self.encLSTMs[l])
  end
  -- Add mean and log var here
  self.encoder:add(nn.Select(1, self.seqLen))
  local zLayer = nn.ConcatTable()
  -- Mean of z
  zLayer:add(nn.Linear(cellSize, LSTMVarAutoEncoder.zSize))
  -- log var^2 of z
  zLayer:add(nn.Linear(cellSize, LSTMVarAutoEncoder.zSize))
  self.encoder:add(zLayer)
  
  -- Create sampler
  self.sampler = nn.Sequential() 

  -- Create sigma*eps module
  local noiseModule = nn.Sequential()
  local noiseModuleInternal = nn.ConcatTable()
  local stdModule = nn.Sequential()
  stdModule:add(nn.MulConstant(0.5)) -- This gives us log sigma
  stdModule:add(nn.Exp()) -- Now we have sigma
  noiseModuleInternal:add(stdModule) -- sigma
  noiseModuleInternal:add(nn.Gaussian(0, 1)) -- eps
  noiseModule:add(noiseModuleInternal)
  noiseModule:add(nn.CMulTable()) -- sigma*eps

  local addMeanSigma = nn.ParallelTable()
  addMeanSigma:add(nn.Identity()) -- mean
  addMeanSigma:add(noiseModule) -- sigma*eps
  self.sampler:add(addMeanSigma)
  self.sampler:add(nn.CAddTable()) -- mean + sigma*eps

  -- Create decoder
  self.decoder = nn.Sequential()
  self.decoder:add(nn.Transpose({1, 2})) -- Transpose to seqlen x batch
  for l = 1, #LSTMVarAutoEncoder.cellSizes do
    local inputSize = l == 1 and X:size(3) or LSTMVarAutoEncoder.cellSizes[l - 1]
    -- Retain hidden state on consecutive calls to forward during evaluation
    -- TODO(Mohit): Should we remember on 'val'
    self.decLSTMs[l] = nn.SeqLSTM(
        inputSize, 
        LSTMVarAutoEncoder.cellSizes[l]) -- :remember('eval')
    self.decLSTMs[l]:set_name('dec_'..l)
    self.decoder:add(self.decLSTMs[l])
  end
  
  -- Reconstruct columns
  self.decoder:add(nn.Sequencer(nn.Linear(
        LSTMVarAutoEncoder.cellSizes[#LSTMVarAutoEncoder.cellSizes], X:size(3))))
  self.decoder:add(nn.Transpose({1, 2})) -- Transpose back to batch x seqlen
  -- It is not necessary to use sigmoid
  -- self.decoder:add(nn.Sigmoid(true))

  -- Create dummy container for getParameters (no other way to combine storage
  -- pointers)
  self.dummyContainer = nn.Sequential()
  self.dummyContainer:add(self.encoder)
  self.dummyContainer:add(self.sampler)
  self.dummyContainer:add(self.decoder)

  -- Create autoencoder wrapper
  self.autoencoder = {
    parent = self
  }
  
  -- Create CUDA wrapper
  function self.autoencoder:cuda()
    self.parent.encoder:cuda()
    self.parent.sampler:cuda()
    self.parent.decoder:cuda()
  end

  -- Create replace wrapper
  function self.autoencoder:replace(fn)
    self.parent.dummyContainer:replace(fn)
  end

  -- Create getParameters wrapper
  function self.autoencoder:getParameters()
    return self.parent.dummyContainer:getParameters()
  end

  -- Create training wrapper
  function self.autoencoder:training()
    self.parent.encoder:training()
    self.parent.sampler:training()
    self.parent.decoder:training()
  end

  -- Create evaluate wrapper
  function self.autoencoder:evaluate()
    self.parent.encoder:evaluate()
    self.parent.sampler:evaluate()
    self.parent.decoder:evaluate()
  end

  -- Create forward wrapper
  function self.autoencoder:forward(x)
    self.encoderOutput = self.parent.encoder:forward(x)

    self.parent.sampler:forward(self.encoderOutput)
    local z = self.parent.sampler.output
    LSTMVarAutoEncoder.decLSTMs[1].userPrevOutput = z:clone()

    -- Use target vector in training, sample from self in evaluate
    -- Shift decoder input sequence by one step forward
    local decInSeq = x:clone()
    -- Reverse x to get decInSeq
    -- decInSeq = decInSeq:index(2, torch.linspace(x:size(2),1,x:size(2)):long())
    -- Shift decInSeq by 1 to the right
    decInSeq[{{}, {2, x:size(2)}, {}}] = decInSeq[{{}, {1, x:size(2) - 1}, {}}]
    decInSeq[{{}, {1}, {}}]:zero() -- Start from vector of zeros
    for i=2,decInSeq:size(2) do
      if torch.uniform() < LSTMVarAutoEncoder.frame_dropout_prob then
        decInSeq[{{},{i},{}}]:zero()
      end
    end
    self.decoderInput = decInSeq
    return self.parent.decoder:forward(decInSeq)
  end

  -- Create backward wrapper
  function self.autoencoder:backward(x, gradLoss)
    --[[
    -- Shift decoder input sequence by one step forward
    local decInSeq = x:clone()
    -- Reverse x to get decInSeq
    decInSeq = decInSeq:index(2, torch.linspace(x:size(2),1,x:size(2)):long())
    decInSeq[{{}, {2, x:size(2)}, {}}] = decInSeq[{{}, {1, x:size(2) - 1}, {}}]
    decInSeq[{{}, {1}, {}}]:zero()
    ]]
    self.parent.decoder:backward(self.decoderInput, gradLoss)

    -- backward through sampler
    self.parent.sampler:backward(self.encoderOutput,
      LSTMVarAutoEncoder.decLSTMs[1].userGradPrevOutput)

    -- set backward for encoder
    --LSTMVarAutoEncoder.encLSTMs[1].gradPrevOutput = self.parent.sampler.gradInput
    --print('did backward through sampler')

     -- seqlen x batch
     --[[
    local zeroTensor = torch.Tensor(
        x:size(2),
        x:size(1),
        LSTMVarAutoEncoder.cellSizes[#LSTMVarAutoEncoder.cellSizes]):zero():typeAs(x)
    ]]
    return self.parent.encoder:backward(x, self.parent.sampler.gradInput)
  end
end

return LSTMVarAutoEncoder

