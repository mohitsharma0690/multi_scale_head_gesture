require 'torch'
require 'hdf5'

local utils = require 'util.utils'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(kwargs)
  local h5file = utils.get_kwarg(kwargs, 'input_h5', '../../openface_data/mohit_data.h5')
  self.batch_size = utils.get_kwarg(kwargs, 'batch_size')
  self.seq_length = utils.get_kwarg(kwargs, 'seq_length')

  local N, T = self.batch_size, self.seq_length

  local splits = {}
  local f = hdf5.open(h5file, 'r')
  splits.train = f:read('/train'):all()
  splits.val = f:read('/val'):all()
  splits.test = f:read('/test'):all()

  self.x_splits = {}
  self.y_splits = {}
  self.split_sizes = {}

  for split, v in pairs(splits) do 
    -- local num = v:nElement()
    local num = v.X:size(1)
    local extra = num % N 
    local x_time_dim, x_input_dim = v.X:size(2), v.X:size(3)

    if extra == 0 then
      -- extra = N * T
    end

    -- local vx = v[{{1, num - extra}}]:view(N, -1, T):transpose(1, 2):clone()
    -- local vy = v[{{2, num - extra + 1}}]:view(N, -1, T):transpose(1, 2):clone()
    local vx = v.X:narrow(1, 1, num-extra):view((num-extra)/N, N, x_time_dim, x_input_dim):clone()
    local vy_val, vy_idx = torch.max(v.y, 2)
    local vy = vy_idx:narrow(1, 1, num-extra):view((num-extra)/N, N, -1):clone()

    self.x_splits[split] = vx
    self.y_splits[split] = vy
    self.split_sizes[split] = vx:size(1)
  end

  self.split_idxs = { train=1, val=1, test=1 }
end

function DataLoader:nextBatch(split)
  local idx = self.split_idxs[split]
  assert(idx, 'Invalid split ' .. idx)

  local x = self.x_splits[split][idx]
  local y = self.y_splits[split][idx]
  
  if idx == self.split_sizes[split] then
    self.split_idxs[split] = 1
  else
    self.split_idxs[split] = idx + 1
  end
  return x, y
end


