require 'torch'
require 'nn'
require 'model/MyLSTM'

local VanillaLSTM, parent = torch.class('nn.VanillaLSTM', 'nn.Module')


function VanillaLSTM:__init(kwargs)
  local lstm_size, fc_size, op_size = 128, 64, 2
  local input_dim = 36
  local rnn_layers = 1

  self.rnns = {}
  self.dropout = 0.3
  self.fc_size = fc_size
  self.op_size = op_size
  self.hidden_dim = lstm_size
  
  self.net = nn.Sequential()
  -- self.net:add(nn.LookupTable(batch_size, input_dim))
  
  for i = 1, rnn_layers do
    local prev_dim = lstm_size
    if i == 1 then prev_dim = input_dim end
    local rnn = nn.MyLSTM(prev_dim, lstm_size)
    rnn.remember_states = true

    table.insert(self.rnns, rnn)
    self.net:add(rnn)

    if self.dropout > 0 then 
      self.net:add(nn.Dropout(self.dropout))     
    end
  end
  
  -- After RNN we have tensor of size (N, T, H).
  -- We want to add a couple of dense layers to this so first
  -- remove the time dimension from net
  self.net:add(nn.Narrow(2, lstm_size, 1))

  -- Reshape tensor from (N, 1, H) to (N, H)
  self.net:add(nn.Reshape(self.hidden_dim, true))

  -- Add dense layer
  self.net:add(nn.Linear(self.hidden_dim, self.fc_size))
  self.net:add(nn.ReLU())

  -- Add final classification layer
  self.net:add(nn.Linear(self.fc_size, self.op_size))

end


function VanillaLSTM:updateOutput(input)
  return self.net:forward(input)
end

function VanillaLSTM:backward(input, gradOutput, scale)
  return self.net:backward(input, gradOutput, scale)
end

function VanillaLSTM:parameters()
  return self.net:parameters()
end

function VanillaLSTM:training()
  self.net:training()
  parent.training(self)
end

function VanillaLSTM:evaluate()
  self.net:evaluate()
  parent.evaluate(self)
end

function VanillaLSTM:resetStates()
  for i, rnn in ipairs(self.rnn) do 
    rnn:resetStates()
  end
end

function VanillaLSTM:clearState()
  self.net:clearState()
end

