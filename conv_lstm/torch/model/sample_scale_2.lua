require 'nn';
require 'optim';

local model = nn.Sequential()
model:add(nn.Linear(3,3,false))
model:add(nn.Linear(3,2,false))
model:add(nn.Linear(2,2,false))
model:set_ms_scale(0.0001, 2)

local crit = nn.CrossEntropyCriterion()

local inp = torch.rand(3)
local y = torch.Tensor(2):zero()
y[1] = 1

local params,grad_params = model:getParameters()
local initial_params = params:clone()

function f(w)
  assert(w == params)
  grad_params:zero()

  local scores = model:forward(inp)
  local loss = crit:forward(scores, y)
  local grad_scores = crit:backward(scores, y)
  model:backward(inp, y)
  -- returns f(x) and df/dx
  return loss, grad_params
end

local config = {learningRate=1}
local gp, loss = optim.sgd(f, params, config)

print(gp)
print(loss)
initial_params:csub(params)
print('Params diff ...')
print(initial_params)
