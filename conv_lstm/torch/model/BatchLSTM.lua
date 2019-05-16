require 'torch'
require 'nn'
require 'rnn'

local utils = require 'util.utils'

local BatchLSTM, parent = torch.class('nn.BatchLSTM', 'nn.Module')

function BatchLSTM:__init(kwargs)
  self.lstm_size = self:getLSTMSize()
  self.num_classes = utils.get_kwarg(kwargs, 'num_classes', 11)
  self.num_classify = utils.get_kwarg(kwargs, 'num_classify', 5)
  self.timestep = utils.get_kwarg(kwargs, 'seq_length', 180)
  self.num_features = utils.get_kwarg(kwargs, 'num_features', 40)
  self.dropout = utils.get_kwarg(kwargs, 'dropout', 0.3) 
end

function BatchLSTM:getLSTMSize()
  return {256}
end

function BatchLSTM:getModel_2(input_size, fc_size, output_size)
  self.net = nn.Sequential()
  local lstm_input_size = input_size
  for i=1,#self.lstm_size do
    local rnn = nn.SeqLSTM(lstm_input_size, self.lstm_size[i])
    rnn.maskzero = true
    self.net:add(rnn)
    if self.dropout > 0 then 
      self.net:add(nn.Dropout(self.dropout))
    end
    lstm_input_size = self.lstm_size[i]
  end
  -- After RNN we have tensor of size (T, N, H).
  -- We want to add a couple of dense layers to this so first
  -- remove the time dimension from net
  -- For now hardcode the sequence length i.e. 240
  self.net:add(nn.Select(1, 240))

  self.net:add(nn.Linear(lstm_input_size, fc_size))
  self.net:add(nn.ReLU())
  self.net:add(nn.Linear(fc_size, output_size))
  return self.net
end

function BatchLSTM:getModel_3(input_size, fc_size, output_size)
  self.net = nn.Sequential()
  local lstm_input_size = input_size
  for i=1,#self.lstm_size do
    local rnn = nn.SeqLSTM(lstm_input_size, self.lstm_size[i])
    rnn.maskzero = true
    --self.net:add(nn.Sequencer(rnn))
    self.net:add(rnn)
    if self.dropout > 0 then 
      self.net:add(nn.Dropout(self.dropout))
    end
    lstm_input_size = self.lstm_size[i]
  end
  self.net:add(nn.Sequencer(nn.Linear(lstm_input_size, fc_size)))
  self.net:add(nn.Sequencer(nn.ReLU()))
  self.net:add(nn.Sequencer(nn.Linear(fc_size, output_size)))
  return self.net
end

function BatchLSTM:getVanillaLSTMModel(input_size, fc_size, output_size)
  self.net = nn.Sequential()
  local lstm_input_size = input_size
  for i=1,#self.lstm_size do
    local rnn = nn.SeqLSTM(lstm_input_size, self.lstm_size[i])
    rnn.maskzero = true
    --self.net:add(nn.Sequencer(rnn))
    self.net:add(rnn)
    if self.dropout > 0 then 
      self.net:add(nn.Dropout(self.dropout))
    end
    lstm_input_size = self.lstm_size[i]
  end
  -- After RNN we have tensor of size (T, N, H).
  -- We want to add a couple of dense layers to this so first
  -- remove the time dimension from net
  -- For now hardcode the sequence length i.e. 240
  self.net:add(nn.Select(1, 240))

  self.net:add(nn.Linear(lstm_input_size, fc_size))
  self.net:add(nn.ReLU())
  self.net:add(nn.Linear(fc_size, output_size))
  return self.net
end

function BatchLSTM:getModel(input_size, fc_size, output_size)
  self.net = nn.Sequential()
  local lstm_input_size = input_size
  for i=1,#self.lstm_size do
    local rnn = nn.SeqLSTM(lstm_input_size, self.lstm_size[i])
    rnn.maskzero = true
    --self.net:add(nn.Sequencer(rnn))
    self.net:add(rnn)
    if self.dropout > 0 then 
      self.net:add(nn.Dropout(self.dropout))
    end
    lstm_input_size = self.lstm_size[i]
  end
  -- After RNN we have tensor of size (T, N, H).
  -- We want to add a couple of dense layers to this so first
  -- remove the time dimension from net
  -- For now hardcode the sequence length i.e. 240
  self.net:add(nn.Select(1, 240))

  self.net:add(nn.Linear(lstm_input_size, fc_size))
  self.net:add(nn.ReLU())
  self.net:add(nn.Linear(fc_size, output_size))
  return self.net
end

