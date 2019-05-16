require 'torch'
require 'hdf5'
require 'math'
require 'paths'

local utils = require 'util.utils'

local BatchLoader = torch.class('BatchLoader')

function BatchLoader:__init(kwargs)
  self.train_h5file = utils.get_kwarg(kwargs, 'train_seq_h5',
    '../../openface_data/train_gesture_by_file.h5')
  self.h5dir = utils.get_kwarg(kwargs, 'data_dir',
    '../../openface_data/face_gestures/dataseto_text')

  self.batch_size = utils.get_kwarg(kwargs, 'batch_size', 256)
  self.num_classes = utils.get_kwarg(kwargs, 'num_classes', 11)
  self.num_classify = utils.get_kwarg(kwargs, 'num_classify', 5)
  self.timestep = utils.get_kwarg(kwargs, 'seq_length', 180)
  self.num_features = utils.get_kwarg(kwargs, 'num_features', 40)
end

-- We return a table with keys from 1 to num_classes. Each key is mapped to a
-- table where each element of the table is another table of type (file_name,
-- gest_begin_time, gest_end_time)
function BatchLoader:get_all_gest_by_type()
  local gest_by_type = {}
  for i=1,self.num_classes do table.insert(gest_by_type, {}) end
  local h5_file = hdf5.open(self.train_h5file, 'r')

  local f_contents = h5_file:read("/"):all()
  for k, v in pairs(f_contents) do
    file_gest_by_type = v
    for i=1, self.num_classes do
      -- gest_t is a tensor Nx2 i.e. all gestures of type i in h5 file
      gest_t = file_gest_by_type[tostring(i-1)]
      if gest_t:nElement() > 0 then
        for j=1, gest_t:size(1) do
          -- File name as first argument
          table.insert(gest_by_type[i], {k, gest_t[j][1], gest_t[j][2]})
        end
      end
    end
  end
  h5_file:close()
  return gest_by_type
end

function BatchLoader:check_if_usable_h5(f)
  local ends_with = "static.mp4.txt.h5"
  return string.sub(f,-string.len(ends_with)) == ends_with
end

function BatchLoader:normalize_data(X,y)
  local mean = {}
  local std = {}
  for i = 1,X:size(2) do
    -- normalize each channel globally
    mean[i] = X[{{}, i}]:mean()
    std[i] = X[{{}, i}]:std()
    X[{{}, i}]:add(-mean[i])
    X[{{}, i}]:div(std[i])
  end
  return X, y
end

function BatchLoader:process_data(X, y)
  assert(X:size(2) > self.num_features)
  X = X[{{}, {1, 148}}]
  -- Should smooth data
  -- In numpy :12 will not get us the 12'th row but in Lua it will. Although it
  -- is compensated by the fact that in Lua we start from 1 while in python from
  -- 0.
  local X_pose = X[{{}, {1, 12}}]
  local X_size = X:size()
  local X_pose_diff = torch.Tensor(X_pose:size(1), 2):zero()
  X_pose_diff[{{2,-1},{}}] = X[{{1,-2}, {6,7}}] - X[{{2,-1},{6,7}}]

  local landmarks = {
    28, 28 + 68, -- forehead
    34, 34 + 68, -- nosetip
    2,   2 + 68, -- left side of face
    4,   4 + 68,
    8,   8 + 68, -- bottom (right)
    10, 10 + 68,
    14, 14 + 68, -- top
    16, 16 + 68
  }
  local X_landmarks = torch.Tensor(X:size(1), #landmarks):zero()
  for i=1,#landmarks do
    X_landmarks[{{},{i}}] = X[{{},landmarks[i]}]
  end
  local X_landmarks_diff = torch.Tensor(X_landmarks:size()):zero()
  X_landmarks_diff[{{2,-1},{}}] = X_landmarks[{{1,-2},{}}] - X_landmarks[{{2,-1},{}}]

  -- TODO(Mohit): Add the 3 direction vectors on our face which can change
  -- as we move

  X = torch.cat({X_pose, X_pose_diff, X_landmarks, X_landmarks_diff})
  X, y = self:normalize_data(X, y)
  return X, y
end

function BatchLoader:load_all_features()
  local X_by_file = {}
  local y_by_file = {}

  for f in paths.files(self.h5dir) do
    if self:check_if_usable_h5(f) then
      local f_path = self.h5dir .. '/' .. f
      local hdf5_file = hdf5.open(f_path, 'r')
      local annots = hdf5_file:read('/annotations'):all()
      local feats = hdf5_file:read('/features'):all()
      local X1 = torch.Tensor(feats:size()):copy(feats)
      local y1 = torch.Tensor(annots:size()):copy(annots)
      local X, y = self:process_data(X1, y1)
      X_by_file[f] = X
      y_by_file[f] = y
    end
  end
  return X_by_file, y_by_file
end

function BatchLoader:final_class_for_gest(gest_type)
  if gest_type == 1 then return 1
  elseif gest_type >= 2 and gest_type <= 6 then return 2
  elseif gest_type == 7 then return 3
  elseif gest_type == 8 or gest_type == 9 then return 4
  elseif gest_type == 10 or gest_type == 11 then return 5
  else assert(false) end
end

function BatchLoader:group_gest_into_classes(gest_by_type)
  local gest_by_classify = {}
  for i=1, self.num_classify do table.insert(gest_by_classify, {}) end
  for i=1, #gest_by_type do
    local new_class = self:final_class_for_gest(i)
    for j=1, #gest_by_type[i] do
      table.insert(gest_by_classify[new_class], gest_by_type[i][j])
    end
  end
  return gest_by_classify
end

function BatchLoader:load_data()
  local gest_by_type = self:get_all_gest_by_type()
  self.gest_by_type = self:group_gest_into_classes(gest_by_type)

  X_by_file, y_by_file = self:load_all_features()
  self.X_by_file = X_by_file
  self.y_by_file = y_by_file
end

function BatchLoader:get_features_for_batch(batch, noise_mask)
  -- Retun the actual features for the batch.
  local X = torch.Tensor(self.batch_size, self.timestep, 46):zero()
  local y = torch.Tensor(self.batch_size):zero()
  for i, v in ipairs(batch) do
    local X_gest = self.X_by_file[v[1]]
    local y_gest = self.y_by_file[v[1]]
    y[i] = self:final_class_for_gest(y_gest[v[3]-1] + 1)
    assert(y[i] >= 1 and y[i] <= self.num_classify)

    -- Add gaussian noise based on noise mask
    if noise_mask[i] then
      local rand_mean, rand_var = 5, 10
      local rand_noise = math.floor(torch.normal(rand_mean, rand_var))
      while (v[2] - rand_noise < 1 or
        (v[3] - v[2] + rand_noise) < 10 or
        (v[3] - v[2] + rand_noise) > self.timestep) do
        rand_noise = math.floor(torch.normal(rand_mean, rand_var))
      end
      v[2] = v[2] - rand_noise
    end
    local move_above = math.floor((v[3]-v[2])/2)
    local start_idx = math.floor(self.timestep/2) - move_above
    start_idx = math.max(start_idx, 1)
    local x = X_gest[{{v[2],v[3]-1},{}}]
    X[{i,{start_idx, start_idx+v[3]-v[2]-1},{}}] = x
  end
  y = y:type('torch.IntTensor')
  return X, y
end

function BatchLoader:get_train_size()
  assert(#(self.gest_by_type) == self.num_classify)
  return #self.gest_by_type[2] * self.num_classify
end

function BatchLoader:get_validation_size()
  return 0
end

function BatchLoader:init_batch(num_epochs)
  -- Note: num_classify is the number of classes we want our network to classify
  -- in the end after softmax.  num_classes is the total number of classes in
  -- our data. Here we have to sample from num_classify

  total_samples_in_epoch = #self.gest_by_type[2] * self.num_classify
  self.num_batches = math.floor(total_samples_in_epoch / self.batch_size)
  print('Total samples in epoch ' .. total_samples_in_epoch .. ' num_batches ' .. self.num_batches)
  print('Wasting '.. total_samples_in_epoch % self.batch_size ..' samples in epoch')

  self.idx_by_type = {}
  for i=1,self.num_classify do table.insert(self.idx_by_type, 1) end
  self.curr_epoch = 1
  self.num_epoch = num_epoch
  self.curr_batch = 1
end

function BatchLoader:reset_batch()
  self.curr_batch = 1
  self.curr_epoch = self.curr_epoch + 1
  for i=1,self.num_classify do self.idx_by_type[i] = 1 end
end

function BatchLoader:next_batch()
  -- Return current batch and increment batch number
  local class_samples_in_batch = self.batch_size / self.num_classify
  local batch = {}
  local noise_mask = torch.Tensor(self.num_classify*class_samples_in_batch):zero()

  for cl=1, self.num_classify do
    for s=1, class_samples_in_batch do
      if #self.gest_by_type[cl] > self.idx_by_type[cl] then
        local gest_type = self.gest_by_type[cl][self.idx_by_type[cl] ]
        if gest_type == nil then
          print(string.format('cl %d self.idx_by_type[cl] %d',
            cl, self.idx_by_type[cl]))
        end
        assert(gest_type ~= None)
        table.insert(batch, self.gest_by_type[cl][self.idx_by_type[cl] ])
        self.idx_by_type[cl] = self.idx_by_type[cl] + 1
      else
        local rand_idx = math.random(1, #self.gest_by_type[cl])
        table.insert(batch, self.gest_by_type[cl][rand_idx])
        noise_mask[(cl-1)*class_samples_in_batch + s] = 1
      end
    end
  end

  X_batch, y_batch = self:get_features_for_batch(batch, noise_mask)
  self.curr_batch = self.curr_batch + 1
  if self.curr_batch > self.num_batches then self:reset_batch() end
  -- rnn library in torch requires data to be in
  -- (timestep, batch_size, num_features) format.
  X_batch = X_batch:transpose(1, 2)

  return X_batch, y_batch
end

