require 'nn'
require 'cunn'
require 'cudnn'
require 'rnn'

function get_16_to_1_model()
  local model = nn.Sequential()
  model:add(nn.SpatialConvolution(1, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(1, 2))
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  return model
end

function get_32_to_1_model()
  local model = nn.Sequential()
  model:add(nn.SpatialConvolution(1, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(1, 2))
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(1, 2))
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  return model
end

function get_64_to_1_model()
  local model = nn.Sequential()
  model:add(nn.SpatialConvolution(1, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(1, 2))
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(1, 2))
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(1, 2))
  model:add(nn.SpatialConvolution(64, 64, 1, 3))
  model:add(nn.ReLU())
  return model
end

function f()
  final_model = nn.Sequential()
  model = nn.ParallelTable()
  m16 = nn.Parallel(2, 2)
  m32 = nn.Parallel(2, 2)
  m64 = nn.Parallel(2, 2)
  for i = 1, 4 do
    m16_to_1 = get_16_to_1_model()
    m16:add(m16_to_1)
    m32_to_1 = get_32_to_1_model()
    m32:add(m32_to_1)
    m64_to_1 = get_64_to_1_model()
    m64:add(m64_to_1)
  end
  model:add(m16)
  model:add(m32)
  model:add(m64)
  final_model:add(model)
  -- We now have multiple tables of length (N, T, 1, F) which we want to join
  -- together
  final_model:add(nn.CMaxTable())
  final_model:add(nn.View(-1, 4, 64, 1, 5))
  final_model:add(nn.View(-1, 4, 64*1*5))


  -- (N, T, C, W, F) (batch_size, time_len, channels, window_len, num_features)
  X = {torch.rand(3, 4, 1, 16, 5), torch.rand(3, 4, 1, 32, 5),
  torch.rand(3, 4, 1, 64, 5)}
  scores = final_model:forward(X)
  print(scores)
  print(scores:size())
end

function f_mask_zero()
  local model = nn.Sequential()
  model:add(nn.SpatialConvolution(1, 1, 3, 1))
  model:add(nn.Tanh())
  model:add(nn.SpatialConvolution(1, 1, 3, 1))
  model:add(nn.Tanh())
  -- model:add(nn.Transpose({3, 4}))
  local final_model = nn.Sequential()
  final_model:add(nn.Transpose({3, 4}))
  final_model:add(nn.MaskZero(model, 3))

  local inp = torch.Tensor(2, 1, 6, 4):zero()
  inp[{{},{}, {},{3,4}}] = torch.rand(2,1,6,2)
  local op = final_model:forward(inp)
  print(inp)
  print(op)
end

f_mask_zero()

