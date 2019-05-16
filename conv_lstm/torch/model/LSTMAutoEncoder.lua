local nn = require 'nn'
require 'rnn'

local LSTMAutoEncoder = {
  cellSizes = {128,128}, -- Number of LSTM cells
  encLSTMs = {},
  decLSTMs = {}
}

-- Copy encoder cell and output to decoder LSTM
function LSTMAutoEncoder:forwardConnect()
  for l = 1, #LSTMAutoEncoder.decLSTMs do
    LSTMAutoEncoder.decLSTMs[l].userPrevOutput = LSTMAutoEncoder.encLSTMs[l].output[self.seqLen]
    LSTMAutoEncoder.decLSTMs[l].userPrevCell = LSTMAutoEncoder.encLSTMs[l].cell[self.seqLen]
  end
end

-- Copy decoder gradients to encoder LSTM
function LSTMAutoEncoder:backwardConnect()
  for l = 1, #LSTMAutoEncoder.encLSTMs do
    LSTMAutoEncoder.encLSTMs[l].userNextGradCell = LSTMAutoEncoder.decLSTMs[l].userGradPrevCell
    LSTMAutoEncoder.encLSTMs[l].gradPrevOutput = LSTMAutoEncoder.decLSTMs[l].userGradPrevOutput
  end
end

function LSTMAutoEncoder:createSequenceCompletor(inp_size)
  local featureSize = inp_size[2] * inp_size[3]
  self.seqLen = inp_size[2] -- Treat rows as a sequence
  
  -- Create encoder
  self.encoder = nn.Sequential()
  self.encoder:add(nn.Transpose({1, 2})) -- Transpose to seqlen x batch
  for l = 1, #LSTMAutoEncoder.cellSizes do
    local inputSize = l == 1 and inp_size[3] or LSTMAutoEncoder.cellSizes[l - 1]
    self.encLSTMs[l] = nn.SeqLSTM(inputSize, LSTMAutoEncoder.cellSizes[l])
    self.encLSTMs[l]:set_name('enc_'..l)
    self.encoder:add(self.encLSTMs[l])
  end

  -- Create decoder
  self.decoder = nn.Sequential()
  self.decoder:add(nn.Transpose({1, 2})) -- Transpose to seqlen x batch
  for l = 1, #LSTMAutoEncoder.cellSizes do
    local inputSize = l == 1 and inp_size[3] or LSTMAutoEncoder.cellSizes[l - 1]
    -- Retain hidden state on consecutive calls to forward during evaluation
    -- TODO(Mohit): Should we remember on 'val'
    self.decLSTMs[l] = nn.SeqLSTM(
        inputSize, 
        LSTMAutoEncoder.cellSizes[l]) -- :remember('eval')
    self.decLSTMs[l]:set_name('dec_'..l)
    self.decoder:add(self.decLSTMs[l])
  end
  
  -- Reconstruct columns
  self.decoder:add(nn.Sequencer(nn.Linear(
        LSTMAutoEncoder.cellSizes[#LSTMAutoEncoder.cellSizes], inp_size[3])))
  self.decoder:add(nn.Transpose({1, 2})) -- Transpose back to batch x seqlen
  -- It is not necessary to use sigmoid
  --self.decoder:add(nn.Sigmoid(true))
  
  -- Create dummy container for getParameters (no other way to combine storage
  -- pointers)
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
  -- Use 1..10 timesteps to get the encoding vector. We want to predict 11..20
  -- timesteps conditioned on 21..30 timesteps.
  function self.autoencoder:forward(x)
    -- Take the first 30 frames (x should have 32 frames initially)
    local temp_x = x:clone()
    temp_x = temp_x[{{},{1,30},{}}]
    local encOut = self.parent.encoder:forward(temp_x[{{},{1,10},{}}])
    self.parent:forwardConnect()

    -- Unconditional decoder
    local decInSeq = temp_x[{{},{21,30},{}}]
    decInSeq = decInSeq:cuda()
    return self.parent.decoder:forward(decInSeq)
  end
  
  -- Create backward wrapper
  function self.autoencoder:backward(x, gradLoss)
    -- Take the first 30 frames (x should have 32 frames initially)
    local temp_x = x:clone()
    temp_x = temp_x[{{},{1,30},{}}]
    local decInSeq = temp_x[{{},{21,30},{}}]
    decInSeq = decInSeq:cuda()

    self.parent.decoder:backward(decInSeq, gradLoss)
    self.parent:backwardConnect()

     -- seqlen x batch
    local zeroTensor = torch.Tensor(
        10,  -- Since we don't use the entire sequence x
        x:size(1),
        LSTMAutoEncoder.cellSizes[#LSTMAutoEncoder.cellSizes]):zero():typeAs(x)
    return self.parent.encoder:backward(x, zeroTensor)
  end
end

function LSTMAutoEncoder:createPredictor(inp_size)
  local featureSize = inp_size[2] * inp_size[3]
  self.seqLen = inp_size[2] -- Treat rows as a sequence
  
  -- Create encoder
  self.encoder = nn.Sequential()
  self.encoder:add(nn.Transpose({1, 2})) -- Transpose to seqlen x batch
  for l = 1, #LSTMAutoEncoder.cellSizes do
    local inputSize = l == 1 and inp_size[3] or LSTMAutoEncoder.cellSizes[l - 1]
    self.encLSTMs[l] = nn.SeqLSTM(inputSize, LSTMAutoEncoder.cellSizes[l])
    self.encLSTMs[l]:set_name('enc_'..l)
    self.encoder:add(self.encLSTMs[l])
  end

  -- Create decoder
  self.decoder = nn.Sequential()
  self.decoder:add(nn.Transpose({1, 2})) -- Transpose to seqlen x batch
  for l = 1, #LSTMAutoEncoder.cellSizes do
    local inputSize = l == 1 and inp_size[3] or LSTMAutoEncoder.cellSizes[l - 1]
    -- Retain hidden state on consecutive calls to forward during evaluation
    -- TODO(Mohit): Should we remember on 'val'
    self.decLSTMs[l] = nn.SeqLSTM(
        inputSize, 
        LSTMAutoEncoder.cellSizes[l]) -- :remember('eval')
    self.decLSTMs[l]:set_name('dec_'..l)
    self.decoder:add(self.decLSTMs[l])
  end
  
  -- Reconstruct columns
  self.decoder:add(nn.Sequencer(nn.Linear(
        LSTMAutoEncoder.cellSizes[#LSTMAutoEncoder.cellSizes], inp_size[3])))
  self.decoder:add(nn.Transpose({1, 2})) -- Transpose back to batch x seqlen
  -- It is not necessary to use sigmoid
  --self.decoder:add(nn.Sigmoid(true))
  

  -- Create dummy container for getParameters (no other way to combine storage
  -- pointers)
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

    -- Unconditional decoder
    local decInSeq = torch.Tensor(x:size(1),8,x:size(3)):zero()
    decInSeq = decInSeq:cuda()
    --[[
    -- Use target vector in training, sample from self in evaluate
    -- Shift decoder input sequence by one step forward
    -- Reverse x to get decInSeq
    decInSeq = decInSeq:index(2, torch.linspace(x:size(2),1,x:size(2)):long())
    -- Shift decInSeq by 1 to the right
    decInSeq[{{}, {2, x:size(2)}, {}}] = decInSeq[{{}, {1, x:size(2) - 1}, {}}]
    decInSeq[{{}, {1}, {}}]:zero() -- Start from vector of zeros
    ]]
    return self.parent.decoder:forward(decInSeq)
  end
  
  -- Create backward wrapper
  function self.autoencoder:backward(x, gradLoss)
    local decInSeq = torch.Tensor(x:size(1),8,x:size(3)):zero()
    decInSeq = decInSeq:cuda()
    --[[
    -- Shift decoder input sequence by one step forward
    local decInSeq = x:clone()
    -- Reverse x to get decInSeq
    decInSeq = decInSeq:index(2, torch.linspace(x:size(2),1,x:size(2)):long())
    decInSeq[{{}, {2, x:size(2)}, {}}] = decInSeq[{{}, {1, x:size(2) - 1}, {}}]
    decInSeq[{{}, {1}, {}}]:zero()
    ]]
    self.parent.decoder:backward(decInSeq, gradLoss)
    self.parent:backwardConnect()

     -- seqlen x batch
    local zeroTensor = torch.Tensor(
        x:size(2),
        x:size(1),
        LSTMAutoEncoder.cellSizes[#LSTMAutoEncoder.cellSizes]):zero():typeAs(x)
    return self.parent.encoder:backward(x, zeroTensor)
  end
end

function LSTMAutoEncoder:createNextStepPredictor(X)
  local featureSize = X:size(2) * X:size(3)
  self.seqLen = X:size(2) -- Treat rows as a sequence

  -- Create encoder
  self.encoder = nn.Sequential()
  self.encoder:add(nn.Transpose({1, 2})) -- Transpose to seqlen x batch
  for l = 1, #LSTMAutoEncoder.cellSizes do
    local inputSize = l == 1 and X:size(3) or LSTMAutoEncoder.cellSizes[l - 1]
    self.encLSTMs[l] = nn.SeqLSTM(inputSize, LSTMAutoEncoder.cellSizes[l])
    self.encLSTMs[l]:set_name('enc_'..l)
    self.encoder:add(self.encLSTMs[l])
  end

  -- Create decoder
  self.decoder = nn.Sequential()
  self.decoder:add(nn.Transpose({1, 2})) -- Transpose to seqlen x batch
  for l = 1, #LSTMAutoEncoder.cellSizes do
    local inputSize = l == 1 and X:size(3) or LSTMAutoEncoder.cellSizes[l - 1]
    -- Retain hidden state on consecutive calls to forward during evaluation
    -- TODO(Mohit): Should we remember on 'val'
    self.decLSTMs[l] = nn.SeqLSTM(
        inputSize, 
        LSTMAutoEncoder.cellSizes[l]) -- :remember('eval')
    self.decLSTMs[l]:set_name('dec_'..l)
    self.decoder:add(self.decLSTMs[l])
  end
  self.decoder:add(nn.Sequencer(nn.Linear(
        LSTMAutoEncoder.cellSizes[#LSTMAutoEncoder.cellSizes], X:size(3)))) -- Reconstruct columns
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
        LSTMAutoEncoder.cellSizes[#LSTMAutoEncoder.cellSizes]):zero():typeAs(x) 
    return self.parent.encoder:backward(x, zeroTensor)
  end
end

function LSTMAutoEncoder:createAutoencoder(X)
  local featureSize = X:size(2) * X:size(3)
  self.seqLen = X:size(2) -- Treat rows as a sequence


  -- Create encoder
  self.encoder = nn.Sequential()
  self.encoder:add(nn.Transpose({1, 2})) -- Transpose to seqlen x batch
  for l = 1, #LSTMAutoEncoder.cellSizes do
    local inputSize = l == 1 and X:size(3) or LSTMAutoEncoder.cellSizes[l - 1]
    self.encLSTMs[l] = nn.SeqLSTM(inputSize, LSTMAutoEncoder.cellSizes[l])
    self.encLSTMs[l]:set_name('enc_'..l)
    self.encoder:add(self.encLSTMs[l])
  end

  -- Create decoder
  self.decoder = nn.Sequential()
  self.decoder:add(nn.Transpose({1, 2})) -- Transpose to seqlen x batch
  for l = 1, #LSTMAutoEncoder.cellSizes do
    local inputSize = l == 1 and X:size(3) or LSTMAutoEncoder.cellSizes[l - 1]
    -- Retain hidden state on consecutive calls to forward during evaluation
    -- TODO(Mohit): Should we remember on 'val'
    self.decLSTMs[l] = nn.SeqLSTM(
        inputSize, 
        LSTMAutoEncoder.cellSizes[l]) -- :remember('eval')
    self.decLSTMs[l]:set_name('dec_'..l)
    self.decoder:add(self.decLSTMs[l])
  end
  
  -- Reconstruct columns
  self.decoder:add(nn.Sequencer(nn.Linear(
        LSTMAutoEncoder.cellSizes[#LSTMAutoEncoder.cellSizes], X:size(3))))
  self.decoder:add(nn.Transpose({1, 2})) -- Transpose back to batch x seqlen
  -- It is not necessary to use sigmoid
  -- self.decoder:add(nn.Sigmoid(true))

  -- Create dummy container for getParameters (no other way to combine storage
  -- pointers)
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
    -- Shift decoder input sequence by one step forward
    local decInSeq = x:clone()
    -- Reverse x to get decInSeq
    decInSeq = decInSeq:index(2, torch.linspace(x:size(2),1,x:size(2)):long())
    -- Shift decInSeq by 1 to the right
    decInSeq[{{}, {2, x:size(2)}, {}}] = decInSeq[{{}, {1, x:size(2) - 1}, {}}]
    decInSeq[{{}, {1}, {}}]:zero() -- Start from vector of zeros
    return self.parent.decoder:forward(decInSeq)
  end
  
  -- Create backward wrapper
  function self.autoencoder:backward(x, gradLoss)
    -- Shift decoder input sequence by one step forward
    local decInSeq = x:clone()
    -- Reverse x to get decInSeq
    decInSeq = decInSeq:index(2, torch.linspace(x:size(2),1,x:size(2)):long())
    decInSeq[{{}, {2, x:size(2)}, {}}] = decInSeq[{{}, {1, x:size(2) - 1}, {}}]
    decInSeq[{{}, {1}, {}}]:zero()
    self.parent.decoder:backward(decInSeq, gradLoss)
    self.parent:backwardConnect()

     -- seqlen x batch
    local zeroTensor = torch.Tensor(
        x:size(2),
        x:size(1),
        LSTMAutoEncoder.cellSizes[#LSTMAutoEncoder.cellSizes]):zero():typeAs(x)
    return self.parent.encoder:backward(x, zeroTensor)
  end
end

return LSTMAutoEncoder

