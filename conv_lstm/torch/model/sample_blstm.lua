require 'nn'
--require 'cunn'
--require 'cudnn'

MS_BLSTM = require 'model.MS_BLSTM'

local model = nn.MS_BLSTM(4, 5)
local x = torch.rand(2, 3, 4)
local output = model:forward(x)
print(output)
