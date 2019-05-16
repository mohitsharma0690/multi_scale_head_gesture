require 'torch'
require 'nn'
require 'rnn'
require 'loadcaffe'

require 'model/MyLSTM'

local utils = require 'util.utils'

local AirHandwritingConvLSTM, parent = torch.class('nn.AirHandwritingConvLSTM', 'nn.Module')

function AirHandwritingConvLSTM:__init(kwargs)
  self.lstm_size = {64, 64}
  self.use_two_scale = utils.get_kwarg(kwargs, 'use_two_scale', 0)
  self.use_48_scale = utils.get_kwarg(kwargs, 'use_48_scale', 0)
  self.num_classes = utils.get_kwarg(kwargs, 'num_classes')
  self.num_classify = utils.get_kwarg(kwargs, 'num_classify')
  self.num_features = utils.get_kwarg(kwargs, 'num_features')
  self.win_len = utils.get_kwarg(kwargs, 'win_len')
  self.word_data = utils.get_kwarg(kwargs, 'word_data')
  self.vocab_size = utils.get_kwarg(kwargs, 'vocab_size')
  self.use_sgd = utils.get_kwarg(kwargs, 'use_sgd')
  self.last_top5_preds, self.last_top5_prob = {}, {}

  self.curr_win_sizes = {200}
  self.rnns = {}
end

-- Use latent variable inferred from the speed of the gesture?
function AirHandwritingConvLSTM:get_2d_latent_conv_model()
  local final_model = nn.Sequential()
  -- kw = 1, kh = 3
  local p = nn.ParallelTable()
  local model = nn.Sequential()

  model:add(nn.SpatialConvolution(1, 128, 1, 5, 1, 2))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolution(128, 128, 1, 5, 1, 2))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.ReLU())

  -- Now input is (N, channels, T=43, F=13). So convert T to second dimension
  model:add(nn.Transpose({2, 3}))
  -- (N, T, C, F)
  model:add(nn.View(-1, 43, 128*self.num_features))
  -- Make T dimension as the first dimension
  model:add(nn.Transpose({1, 2}))

  p:add(model)

  local latent_model = nn.Sequential()
  -- Make T dimension as the first dimension
  latent_model:add(nn.Transpose({1, 2}))
  latent_model:add(nn.Identity())
  p:add(latent_model)

  final_model:add(p)
  final_model:add(nn.JoinTable(2))

  -- Input to LSTM should be in 3D format (N, T, D)
  -- Add LSTM
  local lstm1 = nn.SeqLSTM(128*self.num_features, 256)
  table.insert(self.rnns, lstm1)
  final_model:add(lstm1)
  final_model:add(nn.Dropout(0.5))
  local lstm2 = nn.SeqLSTM(256, 256)
  table.insert(self.rnns, lstm2)
  final_model:add(lstm2)

  -- Select the last timestamp state
  final_model:add(nn.Select(1, 43))
  
  print(model)
  return model
end

function AirHandwritingConvLSTM:get_word_2d_conv_model()
  local model = nn.Sequential()
  model:add(nn.SpatialConvolution(1, 128, 1, 5, 1, 2))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(128, 128, 1, 5, 1, 2))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(128, 128, 1, 5, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(128, 128, 1, 5, 1, 1))
  model:add(nn.ReLU())

  -- Now input is (N, channels, T=43, F=13). So convert T to second dimension
  model:add(nn.Transpose({2, 3}))
  -- (N, T, C, F)
  model:add(nn.View(-1, 239, 128*self.num_features))
  -- Make T dimension as the first dimension
  model:add(nn.Transpose({1, 2}))

  -- Input to LSTM should be in 3D format (N, T, D)
  -- Add LSTM
  local lstm1 = nn.SeqLSTM(128*self.num_features, 1024)
  table.insert(self.rnns, lstm1)
  model:add(lstm1)
  model:add(nn.Dropout(0.5))
  self.encoderLSTM = nn.SeqLSTM(1024, 1024)
  table.insert(self.rnns, self.encoderLSTM)
  model:add(self.encoderLSTM)

  -- Select the last timestamp state
  -- model:add(nn.Select(1, 43))
  
  return model
end

function AirHandwritingConvLSTM:get_2d_conv_model()
  local model = nn.Sequential()
  -- kw = 1, kh = 3
  model:add(nn.SpatialConvolution(1, 128, 1, 5, 1, 2))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolution(128, 128, 1, 5, 1, 2))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolution(128, 128, 1, 3))
  model:add(nn.ReLU())

  -- Now input is (N, channels, T=43, F=13). So convert T to second dimension
  model:add(nn.Transpose({2, 3}))
  -- (N, T, C, F)
  model:add(nn.View(-1, 43, 128*self.num_features))
  -- Make T dimension as the first dimension
  model:add(nn.Transpose({1, 2}))

  -- Input to LSTM should be in 3D format (N, T, D)
  -- Add LSTM
  local lstm1 = nn.SeqLSTM(128*self.num_features, 1024)
  table.insert(self.rnns, lstm1)
  model:add(lstm1)
  model:add(nn.Dropout(0.5))
  local lstm2 = nn.SeqLSTM(1024, 1024)
  table.insert(self.rnns, lstm2)
  model:add(lstm2)

  -- Select the last timestamp state
  model:add(nn.Select(1, 43))
  
  return model
end
 

--[[ Forward coupling: Copy encoder cell and output to decoder LSTM ]]--
function AirHandwritingConvLSTM:forwardConnect()
  local inputSeqLen = self.encoderLSTM.output:size(1)
  --local inputSeqLen = 239 
  self.decoderLSTM.userPrevOutput = self.encoderLSTM.output[inputSeqLen]
  self.decoderLSTM.userPrevCell = self.encoderLSTM.cell[inputSeqLen]
end

--[[ Backward coupling: Copy decoder gradients to encoder LSTM ]]--
function AirHandwritingConvLSTM:backwardConnect()
  local inputSeqLen = 43
  self.encoderLSTM.userNextGradCell = self.decoderLSTM.userGradPrevCell
  self.encoderLSTM.gradPrevOutput = self.decoderLSTM.userGradPrevOutput
end

function AirHandwritingConvLSTM:getWordConvLSTMModel_nopad()
  -- Encoder
  local model = nn.Sequential()
  model:add(nn.SpatialConvolution(1, 128, 1, 5, 1, 2))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(128, 128, 1, 5, 1, 2))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(128, 128, 1, 5, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(128, 128, 1, 5, 1, 1))
  model:add(nn.ReLU())

  -- Now input is (N=1, channels, T=43, F=13). So convert T to second dimension
  model:add(nn.Transpose({2, 3}))
  -- batch size should always be 1 in this case
  model:add(nn.View(1, -1, 128*self.num_features))
  -- Convert T to first dimension
  model:add(nn.Transpose({1, 2}))

  --self.encoderLSTM = nn.FastLSTM(128*self.num_features, 1024)
  --self.encoderSequencer = nn.Sequencer(self.encoderLSTM)
  self.encoderLSTM = nn.SeqLSTM(128*self.num_features, 1024)
  self.encoderLSTM:set_name('encoder LSTM')
  model:add(self.encoderLSTM)
  model:add(nn.Select(1, -1))

  self.encoder = model

  -- Decoder
  self.decoder = nn.Sequential()
  self.decoder:add(nn.Transpose({1, 2})) -- Sequencer requires input in TxNxF format
  self.decoder:add(nn.LookupTableMaskZero(self.vocab_size, 1024))
  -- self.decoderLSTM = nn.FastLSTM(1024, 1024):maskZero(1)
  self.decoderLSTM = nn.SeqLSTM(1024, 1024)
  self.decoderLSTM:maskZero()
  self.decoderLSTM:set_name('Decoder LSTM')
  --self.decoder:add(nn.Sequencer(self.decoderLSTM))
  self.decoder:add(self.decoderLSTM)
  self.decoder:add(nn.Sequencer(nn.MaskZero(nn.Linear(1024, self.vocab_size),1)))
  self.decoder:add(nn.Sequencer(nn.MaskZero(nn.LogSoftMax(),1)))
  self.decoder:add(nn.Transpose({1, 2})) -- Convert back

  -- dummy container
  self.net = nn.Sequential()
  self.net:add(self.encoder)
  self.net:add(self.decoder)
  
  return self.net
end

function AirHandwritingConvLSTM:forward_nopad(input)
  local encInp =  input.enc_inp
  local decInp = input.dec_inp
  self.lastEncoderOutput = {}
  if self.train then
    self.output = torch.Tensor(encInp:size(1), decInp:size(2), 28):zero()
    self.output = self.output:cuda()
    for i=1, encInp:size(1) do
      local currEncInp = encInp[{{i},{}}]
      local currDecInp = decInp[{{i},{}}]
      local encoderOutput = self.encoder:forward(currEncInp)
      -- Save for backprop
      local encoderOutput_clone = encoderOutput:clone()
      table.insert(self.lastEncoderOutput, encoderOutput_clone)
      self:forwardConnect()

      local decoderOutput = self.decoder:forward(currDecInp)
      self.output[{{i},{}}] = decoderOutput
    end
    return self.output
  else
    -- TODO(Mohit): For validation we should not sample from training data
    -- For validation we need to sample from our output only
    -- Remember during validation
    self.decoderLSTM:remember('eval')

    local MAX_OUTPUT_SIZE = 7
    local goToken, eosToken = 1, 2
    local decInput = torch.zeros(encInp:size(1), MAX_OUTPUT_SIZE)
    decInput[{{1},{1}}] = goToken
    local preds, top5_prob, top5_preds = {1}, {}, {}

    self.output = torch.Tensor(encInp:size(1), MAX_OUTPUT_SIZE, 28):zero()
    self.output = self.output:cuda()

    -- Forward through the encoder initially
    local currEncInp = encInp[{{1},{}}]
    local encoderOutput = self.encoder:forward(currEncInp)
    self:forwardConnect()

    -- Forward through the decoder step by step
    for i=1, MAX_OUTPUT_SIZE do
      --prediction will be 1xNx28 in size
      local decInput = preds[#preds]
      local decInput_cuda = torch.Tensor({{decInput}}):cuda()
      local prediction = self.decoder:forward(decInput_cuda)
      self.output[{{1},{i},{}}] = prediction:clone()

      local prob, wordIds = prediction:view(28):topk(5, 1, true, true)
      table.insert(preds, wordIds[1])
      table.insert(top5_preds, wordIds:float())
      table.insert(top5_prob, prob:float())

      -- We end the generation process
      if preds[#preds] == eosToken then
        break
      end
    end
    self:forget()
    self.last_top5_preds, self.last_top5_prob = top5_preds, top5_prob
    return self.output
  end
end

function AirHandwritingConvLSTM:backward_nopad(input, gradOutput, scale)
  assert(self.word_data == 1)
  local decInp = input.dec_inp
  local encInp = input.enc_inp
  -- We should return gradInput from backward although it's not super 
  -- neceessary here but still good to have.
  --print('gradoutput max '..torch.max(torch.abs(gradOutput)))
  for i=1,decInp:size(1) do
    local currDecInp = decInp[{{i},{}}]
    --local currGradOutput = gradOutput[{{i},{}}]:view(-1, 28)
    local currGradOutput = gradOutput[{{i},{}}]

    self.decoder:backward(currDecInp, currGradOutput)
    
    self:backwardConnect()
    --local encoderOutput = self.lastEncoderOutput[i]:view(-1, 1024)
    local encoderOutput = self.lastEncoderOutput[i]
    local inp = encInp[{{i},{}}]
    assert(decInp:size(1) == 1)
    return self.encoder:backward(inp, encoderOutput:zero())
  end
end

function AirHandwritingConvLSTM:getWordConvLSTMModel()
  -- Encoder
  -- Use conv LSTM as the encoder and get a single vector as the video 
  -- representation
  self.encoder = self:get_word_2d_conv_model()

  -- Decoder
  self.decoder = nn.Sequential()
  self.decoder:add(nn.LookupTableMaskZero(self.vocab_size, 1024))
  self.decoder:add(nn.Transpose({1, 2})) -- Sequencer requires input in TxNxF format
  self.decoderLSTM = nn.FastLSTM(1024, 1024):maskZero(1)
  -- We don't need nn.Sequencer() above since we use SeqLSTM above which already
  -- handles sequences but apparently FastLSTM doesn't
  self.decoder:add(nn.Sequencer(self.decoderLSTM))
  self.decoder:add(nn.Sequencer(nn.MaskZero(nn.Linear(1024, self.vocab_size),1)))
  self.decoder:add(nn.Sequencer(nn.MaskZero(nn.LogSoftMax(),1)))
  self.decoder:add(nn.Transpose({1, 2}))

  -- dummy container
  self.net = nn.Sequential()
  self.net:add(self.encoder)
  self.net:add(self.decoder)
  print(self.net)
  return self.net
end

-- This only uses the 16 and 32 length timestamps 
function AirHandwritingConvLSTM:getConvLSTMModel()
  if self.word_data == 1 then
    if self.use_sgd == 1 then
      return self:getWordConvLSTMModel_nopad()
    else
      self:getWordConvLSTMModel()
      return self.net
    end
  end
  local final_model = nn.Sequential()
  local m16 = self:get_2d_conv_model()
  final_model:add(m16)

  local parallel_model_op_size = 256

  -- Finally add a Dense layer
  final_model:add(nn.Linear(parallel_model_op_size, 128))
  final_model:add(nn.ReLU())
  final_model:add(nn.Dropout(0.5))

  final_model:add(nn.Linear(128, self.num_classify))

  self.net = final_model
end

function AirHandwritingConvLSTM:updateType(dtype)
  self.net = self.net:type(dtype)
end

function AirHandwritingConvLSTM:forward(input)
  if self.use_sgd == 1 then
    return self:forward_nopad(input)
  else
    local encInp =  input.enc_inp
    local encoderOutput = self.encoder:forward(encInp)
    self:forwardConnect()
    if self.train then
      local decInp = input.dec_inp
      local decoderOutput = self.decoder:forward(decInp)
      self.output = decoderOutput
      -- save for backprop
      self.curr_encoderOutput = encoderOutput:clone()
      return self.output
    else
      -- For validation we need to sample from our output only
      local MAX_OUTPUT_SIZE = 7
      local decInput = torch.zeros(encInp:size(1), MAX_OUTPUT_SIZE)
      local decInputPartial = torch.zeros(encInp:size(1), 1)
      local goToken, eosToken = 1, 2
      local output_tensor = torch.Tensor(7, encInp:size(1), 28):zero()
      decInputPartial = decInputPartial:cuda()
      for i=1,decInputPartial:size(1) do decInputPartial[{{i}, {}}] = goToken end
      for i=1, MAX_OUTPUT_SIZE do
        --prediction will be 1xNx28 in size
        local prediction = self.decoder:forward(decInputPartial)
        output_tensor[{{i},{},{}}] = prediction:clone():float()

        -- prepare decInputPartial for next input
        decInputPartial = decInputPartial:zero()

        for j=1, prediction:size(2) do
          local prob, wordIds = prediction[{{},{j},{}}]:view(28):topk(5, 1, true, true)
          local next_output = wordIds[1]
          -- If next output is eosToken we will input 0 to the system since we
          -- no longer care about the output
          if next_output ~= eosToken then
            decInputPartial[{{j},{}}] = next_output
          end
        end
      end
      -- This works 
      -- self.output = torch.Tensor(7, encInp:size(1), 28):zero():cuda()
      -- but this doesn't
      --output_tensor = output_tensor:zero()
      self.output = output_tensor:clone():cuda()
      return self.output
    end
  end
end

function AirHandwritingConvLSTM:getParameters()
  return self.net:getParameters()
end
function AirHandwritingConvLSTM:updateOutput(input)
  if self.word_data == 1 then assert(false) end
  return self.net:forward(input)
end

function AirHandwritingConvLSTM:backward(input, gradOutput, scale)
  if self.word_data == 1 then
    if self.use_sgd == 1 then
      return self:backward_nopad(input, gradOutput, scale)
    else
      local decInp = input.dec_inp
      self.decoder:backward(decInp, gradOutput)
      self:backwardConnect()
      return self.encoder:backward(
          input.enc_inp, self.curr_encoderOutput:zero():cuda())
    end
  else
    return self.net:backward(input, gradOutput, scale)
  end
end

function AirHandwritingConvLSTM:parameters()
  return self.net:parameters()
end

function AirHandwritingConvLSTM:training()
  self.net:training()
  parent.training(self)
end

function AirHandwritingConvLSTM:evaluate()
  self.net:evaluate()
  parent.evaluate(self)
end

function AirHandwritingConvLSTM:resetStates()
  for i, rnn in ipairs(self.rnns) do 
    rnn:forget()
  end
end

function AirHandwritingConvLSTM:clearState()
  self.net:clearState()
end

