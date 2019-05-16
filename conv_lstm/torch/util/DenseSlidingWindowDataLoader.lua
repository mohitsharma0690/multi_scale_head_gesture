require 'torch'
require 'hdf5'
require 'math'
require 'paths'

local utils = require 'util.utils'

local DenseSlidingWindowDataLoader = torch.class('DenseSlidingWindowDataLoader')

function DenseSlidingWindowDataLoader:__init(kwargs)
  self.train_h5file = utils.get_kwarg(kwargs, 'train_seq_h5',
    '../../openface_data/main_gest_by_file.h5')
  self.h5dir = utils.get_kwarg(kwargs, 'data_dir',
    '../../openface_data/face_gestures/dataseto_text')
  self.aug_gest_h5file = utils.get_kwarg(kwargs, 'aug_gests_h5',
    '../../openface_data/main_gest_by_file_aug_32.h5')
  self.val_batch_info = utils.get_kwarg(kwargs, 'val_batch_info', 0)
  if self.val_batch_info ~= 1 then
    self.aug_h5 = hdf5.open(self.aug_gest_h5file, 'r')
  end

  self.batch_size = utils.get_kwarg(kwargs, 'batch_size')
  self.num_classes = utils.get_kwarg(kwargs, 'num_classes')
  self.num_classify = utils.get_kwarg(kwargs, 'num_classify')
  self.win_len = utils.get_kwarg(kwargs, 'win_len')
  self.win_step = utils.get_kwarg(kwargs, 'win_step')
  self.num_features = utils.get_kwarg(kwargs, 'num_features', 52)
  self.curr_win_sizes = {16, 32, 64}
  --self.curr_win_sizes = {16, 32}
  self.start_frame = 36

  self.h5_by_file = self:load_data()
  self.X_by_file = self.h5_by_file.X
  self.y_by_file = self.h5_by_file.y

  -- This is time consuming hence do this initially in the beginning once.
  if self.val_batch_info == 0 then
    self.gest_by_type = self:get_all_gest_by_type('train')
    self.gest_by_type = self:unroll_gest_sequences(self.gest_by_type, 'train')
    self.gest_by_type = self:group_gest_into_classes(self.gest_by_type)
  end

  self.test_gest_by_type = self:get_all_gest_by_type('test')
  self.test_gest_by_type = self:unroll_gest_sequences(self.test_gest_by_type,
    'test')
  self.test_gest_by_type = self:group_gest_into_classes(self.test_gest_by_type)

  self.shuffle_order = nil
  -- Load augmented gestures (Too much data to load into memory)
  -- self.aug_gests = self:load_augmented_gestures(self.aug_gest_h5file
end

function DenseSlidingWindowDataLoader:isTrainFile(f)
  local fname = paths.basename(f)
  local fnum = string.match(fname, '%d+')
  fnum = tonumber(fnum)
  if fnum > 11 then return true
  else return false end
end

function DenseSlidingWindowDataLoader:load_data()
  -- create two tables per training set (i.e. 2 each for train, val)
  -- 1) list of all h5 files in each set
  -- 2) the h5 files for each dataset loaded into memory
  local h5_by_file = {}
  for fp in paths.files(self.h5dir) do
    if self:check_if_usable_h5(fp) then
      local f_path = self.h5dir .. '/' .. fp
      local hdf5_file = hdf5.open(f_path, 'r')
      local annots = hdf5_file:read('/annotations'):all()
      local feats = hdf5_file:read('/features'):all()
      local X1 = torch.Tensor(feats:size()):copy(feats)
      local y1 = torch.Tensor(annots:size()):copy(annots)
      local X, y = utils.process_data(X1, y1)
      h5_by_file[fp] = {X=X, y=y}
    end
  end
  return h5_by_file
end

function DenseSlidingWindowDataLoader:get_aug_curr_win_inputs(curr_file,
  curr_t, win_sizes, org_gest)
  local inp = {}
  local start_t = math.floor(curr_t - self.win_len/2)
  local end_t = start_t + self.win_len
  for t=start_t, end_t do
    table.insert(inp, {})
    for i=1, #win_sizes do
      local w_size = win_sizes[i]
      -- Get the (Num_Augmentations, Num_features, Win_size) values
      local init_idx = '/'..curr_file..tostring(org_gest)
      local all_aug = self.aug_h5:read(init_idx..'/'..t..'/'..w_size):all()
      local aug_idx = torch.random(1, all_aug:size(1))
      local gest = all_aug[aug_idx]:t()
      table.insert(inp[t-start_t+1], all_aug[aug_idx])
    end
  end
  return inp
end

-- Returns a table of table. The outer table consists of self.win_len elements
-- where each element is a table. This table consists of #self.win_sizes
-- elements where each element is a Tensor.
function DenseSlidingWindowDataLoader:get_curr_win_inputs(X, y, curr_t,
  win_sizes, noise)
  local inp = {}
  local start_t = math.floor(curr_t - self.win_len/2)
  local end_t = start_t + self.win_len
  for t=start_t, end_t do
    table.insert(inp, {})
    for i=1, #win_sizes do
      local w_size = win_sizes[i]
      table.insert(inp[t-start_t+1], X[{{t-w_size/2, t-1+w_size/2},{}}])
    end
  end
  return inp
end

-- batch_inp: 3D table (batch_size, T, win_sizes)
function DenseSlidingWindowDataLoader:convert_to_inp_format(batch_inp, batch_op)
  local final_X = {}
  for i=1,#self.curr_win_sizes do
    table.insert(final_X, torch.zeros(
      #batch_inp, self.win_len, 1, self.curr_win_sizes[i], self.num_features))
  end

  for i=1, #self.curr_win_sizes do
    for b=1, #batch_inp do
      for t=1, self.win_len do
        final_X[i][b][t][1] = batch_inp[b][t][i]
      end
    end
  end
  y = torch.Tensor(batch_op)
  return final_X, y
end

function DenseSlidingWindowDataLoader:get_batch_sample_ratio()
  num_samples = {}
  for i=1, self.num_classify do
    table.insert(num_samples, math.floor(self.batch_size/self.num_classify))
  end
  return num_samples
end

function DenseSlidingWindowDataLoader:get_noise_mask_for_samples(samples)
  noise_mask = {}
  for i=1,#samples do
    for j=1,samples[i] do table.insert(noise_mask, 0) end
  end
  return noise_mask
end

function DenseSlidingWindowDataLoader:get_shuffle_order(gest_by_type)
  local shuffle_order = {}
  for i=1,#self.gest_by_type do
    table.insert(shuffle_order, torch.randperm(#self.gest_by_type[i]))
  end
  return shuffle_order
end

function DenseSlidingWindowDataLoader:getTotalTrainBatches()
  local num_train_nods = #self.gest_by_type[2]
  local total_samples_in_epoch = num_train_nods * self.num_classify
  local num_batches = math.floor(total_samples_in_epoch / self.batch_size)
  return num_batches
end

function DenseSlidingWindowDataLoader:next_train_batch()
  --self:next_batch(self.train_files, self.train_h5_by_file)
  local num_train_nods = #self.gest_by_type[2]
  local total_samples_in_epoch = num_train_nods * self.num_classify
  local num_batches = math.floor(total_samples_in_epoch / self.batch_size)
  -- TODO(Mohit): Complete this
  self.shuffle_order = self:get_shuffle_order(self.gest_by_type)

  idx_by_type = {}
  for i=1, self.num_classify do table.insert(idx_by_type, 1) end
  print('Total batches in 1 epoch '..num_batches)

  for i=1,num_batches do
    local class_samples_in_batch = self:get_batch_sample_ratio()
    local noise_mask = self:get_noise_mask_for_samples(class_samples_in_batch)

    --print(class_samples_in_batch)
    local batch = {}
    -- Reset noise mask
    local batch_idx = 1

    for c=1,self.num_classify do
      local num_gests = #self.gest_by_type[c]
      for s=1,class_samples_in_batch[c] do
        if num_gests >= idx_by_type[c] then
          local shuffled_idx = self.shuffle_order[c][idx_by_type[c]]
          --if c==1 and s < 5 then print(shuffled_idx) end
          table.insert(batch, self.gest_by_type[c][shuffled_idx])
          idx_by_type[c] = idx_by_type[c] + 1
        else
          local rand_idx = torch.random(1, num_gests)
          local shuffled_idx = self.shuffle_order[c][rand_idx]
          table.insert(batch, self.gest_by_type[c][shuffled_idx])
          noise_mask[batch_idx] = 1
        end
      end
    end
    local X_batch, y_batch = self:get_features_for_batch(batch, noise_mask)
    coroutine.yield(X_batch, y_batch)
  end
end

function DenseSlidingWindowDataLoader:next_val_batch()
  local batch_size = self.batch_size
  local num_test = 1
  for c=1, self.num_classify do
    num_test = num_test + #self.test_gest_by_type[c]
  end
  batch = {}
  for c=1, self.num_classify do
    for i=1, #self.test_gest_by_type[c] do
      table.insert(batch, self.test_gest_by_type[c][i])

      if #batch == batch_size then
        local X_batch, y_batch = self:get_features_for_batch(batch, nil)
        coroutine.yield(X_batch, y_batch)
        batch = {}
        collectgarbage()
      end
    end
  end
end

function DenseSlidingWindowDataLoader:next_batch(files, h5_by_file)
  local shuffle = torch.randperm(#files)
  local max_win_size = math.max(unpack(self.curr_win_sizes))

  local batch_inp, batch_op = {}, {}
  local inp = {}

  for i=1, shuffle:size(1) do
    local curr_file = files[shuffle[i]]
    local video_len = h5_by_file[curr_file].y:size(1)
    local t = self.start_frame
    local y = h5_by_file[curr_file].y

    while t <= video_len - self.start_frame do
      local curr_t = t
      if y[t] == 0 and torch.uniform() > 0.2 then
        t = t + self.win_step
      else
        while curr_t < t + self.win_len and curr_t + max_win_size/2 < video_len do
          local inp_win = self:get_curr_win_inputs(
            h5_by_file[curr_file].X,
            h5_by_file[curr_file].y,
            curr_t,
            self.curr_win_sizes)
          table.insert(inp, inp_win)
          curr_t = curr_t + 1
        end

        -- Move to next window
        if #inp == self.win_len then
          table.insert(batch_inp, inp)
          table.insert(batch_op, y[t])
          inp = {}
        end

        -- Check if we are done with current batch
        if #batch_inp == self.batch_size then
          local X_temp, y_temp = self:convert_to_inp_format(batch_inp, batch_op)
          coroutine.yield(X_temp, y_temp)
          batch_inp = {}
          batch_op = {}
          collectgarbage()
        end

        t = t + self.win_step
      end
    end
  end

end

-- We return a table with keys from 1 to num_classes. Each key is mapped to a
-- table where each element of the table is another table of type (file_name,
-- gest_begin_time, gest_end_time)
function DenseSlidingWindowDataLoader:get_all_gest_by_type(group_name)
  local gest_by_type = {}
  for i=1,self.num_classes do table.insert(gest_by_type, {}) end
  local h5_file = hdf5.open(self.train_h5file, 'r')

  local f_contents = h5_file:read("/"):all()
  f_contents = f_contents[group_name]
  for k, v in pairs(f_contents) do
    local file_gest_by_type = v
    local num_frames = self.h5_by_file[k].y:size(1)
    for i=1, self.num_classes do
      -- gest_t is a tensor Nx2 i.e. all gestures of type i in h5 file
      gest_t = file_gest_by_type[tostring(i-1)]
      if gest_t:nElement() > 0 then
        for j=1, gest_t:size(1) do
          -- Only insert if we have sufficient frames at the end for the window
          -- Compare gest[j][2] == 0 since we need a hack for cases where hdf5
          -- isn't able to read empty tensors stored in h5 files. For those cases
          -- we add [0, 0] as the indexes to save and remove them here.

          if (torch.isTensor(gest_t[j]) and
              num_frames - gest_t[j][2] > 100 and
              gest_t[j][1] > 100) then
            -- File name as first argument
            table.insert(gest_by_type[i], {k, gest_t[j][1], gest_t[j][2]})
          end
        end
      end
    end
  end
  h5_file:close()
  return gest_by_type
end

function DenseSlidingWindowDataLoader:load_augmented_gestures(aug_h5)
  local h5_file = hdf5.open(aug_h5, 'r')

  local f_contents = h5_file:read("/"):all()
  local aug_gests = {}
  for file_name, v in pairs(f_contents) do
    aug_gests[file_name] = {}
    local count = 0
    for l, v_l in pairs(v) do
      aug_gests[file_name][l] = {}
      for t, v_t in pairs(v_l) do
        aug_gests[file_name][l][t] = {}
        for w, v_w in pairs(v_t) do
          aug_gests[file_name][l][t][w] = v_w
          count = count + v_w:size(1)
        end
      end
    end
    print('Did load ' .. count .. ' augmented gestures for ' .. file_name)
  end
  return aug_gests
end

function DenseSlidingWindowDataLoader:check_if_usable_h5(f)
  local ends_with = "static.mp4.txt.h5"
  return string.sub(f,-string.len(ends_with)) == ends_with
end

function DenseSlidingWindowDataLoader:final_class_for_gest(gest_type)
  if self.num_classes == self.num_classify then return gest_type end

  if gest_type == 1 then return 1
  elseif gest_type >= 2 and gest_type <= 6 then return 2
  elseif gest_type == 7 then return 3
  elseif gest_type == 8 or gest_type == 9 then return 4
  elseif gest_type == 10 or gest_type == 11 then return 5
  else assert(false) end
end

function DenseSlidingWindowDataLoader:group_gest_into_classes(gest_by_type)
  if self.num_classes == self.num_classify then return gest_by_type end

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

function DenseSlidingWindowDataLoader:unroll_gest_sequences(gest_by_type,
  data_type)
  --[[
  gest_by_type: Table of the sequences of each gesture type. We unroll each
  gesture sequence to multiple values in the list where each value
  represents one frame of the sequence and we use that value as the center
  frame for one data point.
  ]]
  local unroll_gest_by_type = {}
  local canSkip = true
  local win_step = self.win_step
  -- Dense labeling for test sequences
  if data_type == 'test' then
    canSkip = false
    win_step = 1
  end
  for c=1,#gest_by_type do
    table.insert(unroll_gest_by_type, {})
    for i=1,#gest_by_type[c] do
      local is_valid = true
      if canSkip and c == 1 and torch.uniform() > 0.2 then is_valid = false end

      if is_valid then
        local curr_gest = gest_by_type[c][i]
        local gest_len = curr_gest[3] - curr_gest[2]
        local gest_start = curr_gest[2] + math.floor(gest_len/5)
        local gest_end = curr_gest[3] - math.floor(gest_len/5)
        for j=gest_start, gest_end, self.win_step do
          table.insert(unroll_gest_by_type[c], {curr_gest[1], j, j})
        end
      end
    end
  end
  local unroll_stats = 'Unrolled gests len: ['
  for c=1,#unroll_gest_by_type do
    unroll_stats = unroll_stats .. #unroll_gest_by_type[c] .. ', '
  end
  unroll_stats = unroll_stats .. ']'
  print(unroll_stats)
  return unroll_gest_by_type
end

function DenseSlidingWindowDataLoader:get_features_for_batch(batch, noise_mask)
  -- Retun the actual features for the batch.
  local max_win_size = math.max(unpack(self.curr_win_sizes))
  local batch_inp, batch_op = {}, {}
  for i, v in ipairs(batch) do
    local y_gest = self.h5_by_file[v[1]].y
    local org_gest = y_gest[v[3]]
    local op_win = self:final_class_for_gest(y_gest[v[3]]+1)
    assert(op_win >= 1 and op_win <= self.num_classify)

    local noise = 0
    if noise_mask ~= nil then noise = noise_mask[i] end
    local curr_t = v[2]
    local curr_file = v[1]
    local inp_win = nil
    -- For now let's ignore all noise in None class
    if (noise == 1 and org_gest == 0) then noise = 0 end
    if noise == 0 then
      inp_win = self:get_curr_win_inputs(
        self.h5_by_file[curr_file].X,
        self.h5_by_file[curr_file].y,
        curr_t,
        self.curr_win_sizes,
        noise)
    else
      inp_win = self:get_aug_curr_win_inputs(
        curr_file,
        curr_t,
        self.curr_win_sizes,
        org_gest)
    end
    table.insert(batch_inp, inp_win)
    table.insert(batch_op, op_win)
  end
  --[[ Shuffles elements within batch. Not required.
  local new_shuffle_order = torch.randperm(#batch_inp)
  local new_batch_inp, new_batch_op = {}, {}
  for i=1, #batch_inp do
    table.insert(new_batch_inp, batch_inp[new_shuffle_order[i] ])
    table.insert(new_batch_op, batch_op[new_shuffle_order[i] ])
  end
  batch_inp = new_batch_inp
  batch_op = new_batch_op
  ]]
  --print(batch_op)

  -- X_batch is a table of Tensors, y_batch is a Tensor
  local X_batch, y_batch = self:convert_to_inp_format(batch_inp, batch_op)
  y_batch = y_batch:clone()
  y_batch = y:type('torch.IntTensor')
  return X_batch, y_batch
end

function DenseSlidingWindowDataLoader:get_train_size()
  assert(#(self.gest_by_type) == self.num_classify)
  return #self.gest_by_type[2] * self.num_classify
end

function DenseSlidingWindowDataLoader:get_validation_size()
  return 0
end

function DenseSlidingWindowDataLoader:init_batch(num_epochs)
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

function DenseSlidingWindowDataLoader:reset_batch()
  self.curr_batch = 1
  self.curr_epoch = self.curr_epoch + 1
  for i=1,self.num_classify do self.idx_by_type[i] = 1 end
end

