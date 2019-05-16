require 'nn'
require 'cunn'
require 'cudnn'

function get_16_to_1_model()
  local model = nn.Sequential()
  model:add(nn.TemporalConvolution(5, 128, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(128, 128, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(128, 128, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalMaxPooling(2, 2))
  model:add(nn.TemporalConvolution(128, 64, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(64, 64, 3))
  model:add(nn.ReLU())
  return model
end

function get_32_to_1_model()
  local model = nn.Sequential()
  model:add(nn.TemporalConvolution(5, 128, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(128, 128, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(128, 128, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalMaxPooling(2, 2))
  model:add(nn.TemporalConvolution(128, 64, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(64, 64, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(64, 64, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalMaxPooling(2, 2))
  model:add(nn.TemporalConvolution(64, 64, 3))
  model:add(nn.ReLU())
  return model
end

function get_64_to_1_model()
  local model = nn.Sequential()
  model:add(nn.TemporalConvolution(5, 128, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(128, 128, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(128, 128, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalMaxPooling(2, 2))
  model:add(nn.TemporalConvolution(128, 64, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(64, 64, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(64, 64, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalMaxPooling(2, 2))
  model:add(nn.TemporalConvolution(64, 64, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalConvolution(64, 64, 3))
  model:add(nn.ReLU())
  model:add(nn.TemporalMaxPooling(2, 2))
  model:add(nn.TemporalConvolution(64, 64, 3))
  model:add(nn.ReLU())
  return model
end

final_model = nn.Sequential()
model = nn.ParallelTable()
m16 = nn.Parallel(2, 2)
m32 = nn.Parallel(2, 2)
m64 = nn.Parallel(2, 2)
--[[
for i = 1, 1 do
  m16_to_1 = get_16_to_1_model()
  m16:add(m16_to_1)
  m32_to_1 = get_32_to_1_model()
  m32:add(m32_to_1)
  m64_to_1 = get_64_to_1_model()
  m64:add(m64_to_1)
end
]]
model:add(get_16_to_1_model())
model:add(get_32_to_1_model())
model:add(get_64_to_1_model())
final_model:add(model)
  -- We now have multiple tables of length (N, T, 1, F) which we want to join
  -- together
final_model:add(nn.CMaxTable())


X = {torch.rand(2, 16, 5), torch.rand(2, 32, 5), torch.rand(2, 64, 5)}
scores = final_model:forward(X)
print(scores)
