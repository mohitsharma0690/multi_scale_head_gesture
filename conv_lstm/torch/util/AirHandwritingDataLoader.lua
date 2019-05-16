require 'torch'
require 'hdf5'
require 'math'
require 'paths'
require 'image'

local utils = require 'util.utils'

local AirHandwritingDataLoader = torch.class('AirHandwritingDataLoader')

function AirHandwritingDataLoader:__init(kwargs)
  self.data_h5 = utils.get_kwarg(kwargs, 'data_h5')
  self.aug_gest_h5file = utils.get_kwarg(kwargs, 'aug_gests_h5',
    '../../openface_data/main_gest_by_file_aug_K_32.h5')

  self.val_batch_info = utils.get_kwarg(kwargs, 'val_batch_info', 0)
  if self.val_batch_info ~= 1 then
    -- self.aug_h5 = hdf5.open(self.aug_gest_h5file, 'r')
  end

  self.batch_size = utils.get_kwarg(kwargs, 'batch_size')
  self.num_classes = utils.get_kwarg(kwargs, 'num_classes')
  self.num_classify = utils.get_kwarg(kwargs, 'num_classify')
  self.win_len = utils.get_kwarg(kwargs, 'win_len')
  self.num_features = utils.get_kwarg(kwargs, 'num_features')
  self.word_data = utils.get_kwarg(kwargs, 'word_data')
  self.use_sgd = utils.get_kwarg(kwargs, 'use_sgd')

  if self.use_two_scale == 1 then
    self.curr_win_sizes = {16, 32}
  else
    if self.word_data == 1 then self.curr_win_sizes = {1000}
    elseif self.word_data == 1 and self.use_sgd == 1 then self.curr_win_sizes = {-1} 
    else self.curr_win_sizes = {200} end
  end
  print(self.curr_win_sizes)
  self.start_frame = 36

  --[[
  local openface_mean_h5 = utils.get_kwarg(kwargs, 'openface_mean_h5')
  self.openface_mean_std = self:load_mean_h5(openface_mean_h5)
  ]]

  self.vocab_size = self:get_vocab_size()
  self.valid_words = self:load_valid_words()
  self.data = self:load_data()
  self.train_data = self.data['train']
  self.test_data = self.data['test']
  
  self.shuffle_order = nil

  print('Did read and load gestures')
end

function AirHandwritingDataLoader:isTrainFile(f) 
  local fname = paths.basename(f)
  local fnum = string.match(fname, '%d+')
  fnum = tonumber(fnum)
  if fnum > 11 then return true
  else return false end
end

function AirHandwritingDataLoader:load_mean_h5(h5_file_path)
  local f = hdf5.open(h5_file_path, 'r')
  local mean = f:read('/mean'):all()
  local std = f:read('/std'):all()
  f:close()
  return {mean=mean, std=std}
end

function AirHandwritingDataLoader:filter_word_classes_from_dict(d, words)
  local filter_d = {}
  for valid_word_idx, valid_word in ipairs(words) do 
    -- We can add this word to our training/testing example
    if d[valid_word] ~= nil then
      local total_samples = 1
      table.insert(filter_d, {})
      for k1, v1 in pairs(d[valid_word]) do 
        table.insert(filter_d[valid_word_idx], {})
      end
      for k1, v1 in pairs(d[valid_word]) do 
        local x = torch.Tensor(v1:size()):copy(v1)
        -- Convert features x time to time x features
        x = x:t()
        filter_d[valid_word_idx][k1+1] = x 
      end
    end
  end
  assert(filter_d[1] ~= nil)
  assert(#filter_d[1] ~= 0)
  return filter_d
end

function AirHandwritingDataLoader:filter_classes_from_dict(d, low, high)
  local filter_d = {}
  for k, v in pairs(d) do
    k_int = string.byte(k)
    if k_int >= low and k_int <= high then 
      local total_samples = 1
      filter_d[k_int-low+1] = {}
      for k1, v1 in pairs(v) do table.insert(filter_d[k_int-low+1], {}) end
      for k1, v1 in pairs(v) do
        local x = torch.Tensor(v1:size()):copy(v1)
        -- Convert features x time to time x features
        x = x:t()
        filter_d[k_int-low+1][k1+1] = x 
      end
    end
  end
  assert(filter_d[1] ~= nil)
  assert(#filter_d[1] ~= 0)
  return filter_d
end

function AirHandwritingDataLoader:process_data(args)
  local X = args.X
  local mean, std = args.mean, args.std
  local norm_type = args.norm_type or "z-norm"
  -- TODO(Mohit): Add normalization
  
  -- Remove time from feature set
  for l_k, l_v in pairs(X) do
    for k, v in pairs(l_v) do
      l_v[k] = v[{{},{2,v:size(2)}}]
    end
  end
  return X
end

function AirHandwritingDataLoader:load_data()
  -- create two tables per training set (i.e. 2 each for train, val)
  -- 1) list of all h5 files in each set
  -- 2) the h5 files for each dataset loaded into memory
  local h5_by_file = {}
  local hdf5_file = hdf5.open(self.data_h5, 'r')
  local train_data = hdf5_file:read("/train"):all()
  local test_data = hdf5_file:read("/test"):all()

  if self.word_data == 1 then 
    train_data = self:filter_word_classes_from_dict(
      train_data, self.valid_words)
    test_data = self:filter_word_classes_from_dict(
      test_data, self.valid_words)
  else
    train_data = self:filter_classes_from_dict(
      train_data, string.byte("A"), string.byte("Z"))
    test_data = self:filter_classes_from_dict(test_data,
      string.byte("A"), string.byte("Z"))
  end
  
  local train_data = self:process_data {
    X=train_data, 
    mean=0,
    std=1,
  }
  local test_data = self:process_data {
    X=test_data,
    mean=0,
    std=1,
  }
  -- Non-Maximal suppression of velocity vectors
  -- X = utils.do_non_maximal_suppression(X)
  return {train=train_data, test=test_data} 
end

function AirHandwritingDataLoader:get_aug_curr_win_inputs(curr_file, 
  curr_t, win_sizes, org_gest)
  local inp = {}
  for i=1, #win_sizes do
    local w_size = win_sizes[i]
    -- Get the (Num_Augmentations, Num_features, Win_size) values
    local temp_idx = curr_file .. '/' .. tostring(org_gest)
    local all_aug = self.aug_h5:read('/'..temp_idx..'/'..curr_t..'/'..w_size):all()
    local aug_idx = torch.random(1, all_aug:size(1))
    local gest = all_aug[aug_idx]:t()
    gest = utils.get_augmented_data_from_trimmed{
      X=gest,
      mean=self.openface_mean_std.mean[curr_file],
      std=self.openface_mean_std.std[curr_file]
    }
    gest = utils.do_non_maximal_suppression(gest)

    if self.cpm_aug_h5 ~= nil then
      local all_aug = self.cpm_aug_h5:read(
          '/'..temp_idx..'/'..curr_t..'/'..w_size):all()
      local cpm_gest = all_aug[aug_idx]:t()
      cpm_gest = utils.get_augmented_cpm_data_from_trimmed{X=cpm_gest}
      cpm_gest = utils.do_non_maximal_suppression_cpm_data(cpm_gest)
      gest = torch.cat(gest, cpm_gest)
    end
    table.insert(inp, gest)
  end
  return inp
end

function AirHandwritingDataLoader:get_opt_flow_file_name(t)
  -- If we have skipped frames in opt flow then we should do the corresponding
  -- math here.
  return string.format('flow_x_%05d.jpg', t) , string.format('flow_y_%05d.jpg', t)
end

function AirHandwritingDataLoader:get_opt_flow_input(curr_file, curr_t)
  local s, e = string.find(curr_file, '%d%d%d')
  local file_num = string.sub(curr_file, s, e)
  local dir = paths.concat(self.opt_flow_dir, file_num)
  dir = paths.concat(dir, 'cropped_224_224')
  -- We need to convert the curr_t into a file number. It is usually prefixex
  -- as flow_x_000ab.jpg flow_y_000ab.jpg
  -- The pretrained network uses 10 optical flow timestamps.
  local opt_flow_inp = torch.Tensor(20, 224, 224)
  local i = 1
  for t=curr_t-5,curr_t+4 do
    local flow_x_img, flow_y_img = self:get_opt_flow_file_name(t)
    flow_x_img = paths.concat(dir, flow_x_img)
    flow_y_img = paths.concat(dir, flow_y_img)
    local x_img = image.load(flow_x_img, 1)
    local y_img = image.load(flow_y_img, 1)
    --[[
    local c = utils.OPT_FLOW_IMAGE_CROP[curr_file]
    x_img = image.crop(x_img, c[2],c[1],c[4],c[3])
    y_img = image.crop(y_img, c[2],c[1],c[4],c[3])
    ]]

    x_img = image.scale(x_img, 224, 224)
    y_img = image.scale(x_img, 224, 224)
    
    opt_flow_inp[{i,{},{}}] = x_img
    opt_flow_inp[{i+1,{},{}}] = y_img
    i = i + 2
  end
  return opt_flow_inp
end

function AirHandwritingDataLoader:get_curr_win_inputs(args)
  local X, y, curr_t, win_sizes = args.X, args.y, args.curr_t, args.win_sizes
  local noise = args.noise or nil
  local cpm_X = args.cpm_X or nil
  local inp = {}
  for i=1, #win_sizes do
    local w_size = win_sizes[i]
    if noise == nil or noise == 0 then
      if cpm_X == nil then
        table.insert(inp, X[{{curr_t-w_size/2, curr_t-1+w_size/2},{}}])
      else
        local X_openface = X[{{curr_t-w_size/2, curr_t-1+w_size/2},{}}]
        local X_cpm = cpm_X[{{curr_t-w_size/2, curr_t-1+w_size/2},{}}]
        local X_final = torch.cat(X_openface, X_cpm)
        table.insert(inp, X_final)
      end
    else
      if cpm_X == nil then
        local x = X[{{curr_t-w_size/2, curr_t-1+w_size/2},{}}]
        -- Flip the tensor
        x = x:index(1, torch.linspace(x:size(1),1,x:size(1)):long())
        table.insert(inp, x)
      else
        local X_openface = X[{{curr_t-w_size/2, curr_t-1+w_size/2},{}}]
        local X_cpm = cpm_X[{{curr_t-w_size/2, curr_t-1+w_size/2},{}}]
        local X_final = torch.cat(X_openface, X_cpm)
        X_final = X_final:index(
            1, torch.linspace(X_final:size(1),1,X_final:size(1)):long())
        table.insert(inp, X_final)
      end
    end
  end
  return inp
end

function AirHandwritingDataLoader:convert_to_sgd_inp_format(batch_inp, batch_op, is_val_batch)
  -- Pass in the batch inp and output without any padding since we are
  -- assuming pure SGD
  -- Get the decoder input and output
  -- each output is of maximum length 6
  assert(#batch_inp == 1)
  local final_X = torch.Tensor(1, batch_inp[1]:size(1), batch_inp[1]:size(2)):zero()
  final_X[{{1},{},{}}] = batch_inp[1]

  is_val_batch = is_val_batch or false

  local max_decoder_len = string.len(self.valid_words[batch_op[1]]) + 1
  if is_val_batch  then max_decoder_len = 7 end
  local decoder_output = torch.Tensor(#batch_op, max_decoder_len):zero()
  local decoder_input = torch.Tensor(#batch_op, max_decoder_len):zero()
  if is_val_batch then decoder_output[{}] = self:get_char_embedding("<end>") end
  for i=1, #batch_op do
    local word_idx = batch_op[i] 
    local word = self.valid_words[word_idx]
    assert(string.len(word) <= 6 and string.len(word) > 0)
    for j=1,string.len(word) do
      decoder_output[{{i},{max_decoder_len-string.len(word)+j}}] = self:get_char_embedding(string.byte(word, j))
      decoder_input[{{i},{max_decoder_len-string.len(word)+j}}] = self:get_char_embedding(string.byte(word, j))
    end

    -- Shift decoder to left and add <EOS> at the end
    decoder_output[{{i}, {1,-2}}] = decoder_output[{{i},{2,-1}}]
    decoder_output[{{i}, {-1}}] = self:get_char_embedding("<end>")

    -- Add start to the left of ABC -> <pad><pad><start>ABC
    decoder_input[{{i}, {max_decoder_len-string.len(word)}}] = self:get_char_embedding("<start>")

  end

  return {enc_inp=final_X, dec_inp=decoder_input}, decoder_output
end

function AirHandwritingDataLoader:convert_to_inp_format(batch_inp, batch_op, is_val_batch) 
  if self.use_sgd == 1 then
    return self:convert_to_sgd_inp_format(batch_inp, batch_op, is_val_batch) 
  end

  local final_X = {}
  for i=1,#self.curr_win_sizes do
    table.insert(final_X, torch.zeros(
      #batch_inp, 1, self.curr_win_sizes[i], self.num_features))
  end

  for i=1,#self.curr_win_sizes do
    for b=1,#batch_inp do
      local inp = batch_inp[b] 
      if inp:size(1) <= self.curr_win_sizes[1] then
        local pad_start = self.curr_win_sizes[1] - inp:size(1) + 1
        final_X[i][b][1][{{pad_start,self.curr_win_sizes[i]},{}}] = inp
      else
        -- Gesture is greater than window length so for now just choose the
        -- mid part of the gesture
        local mid = math.floor(inp:size(1) / 2)
        local left = math.floor(mid - self.curr_win_sizes[i]/2)
        if left == 0 then left = left + 1 end
        final_X[i][b][1] = inp[{{left, left+self.curr_win_sizes[i]-1},{}}]
      end
    end
  end

  if #self.curr_win_sizes == 1 then
    -- We don't need a table here
    final_X = final_X[1]
  end

  if self.word_data == 1 then
    -- Get the decoder input and output
    -- each output is of maximum length 6
    local max_decoder_len = 7
    local decoder_output = torch.Tensor(#batch_op, max_decoder_len):zero()
    local decoder_input = torch.Tensor(#batch_op, max_decoder_len):zero()
    for i=1, #batch_op do
      local word_idx = batch_op[i] 
      local word = self.valid_words[word_idx]
      assert(string.len(word) <= 6 and string.len(word) > 0)
      for j=1,string.len(word) do
        decoder_output[{{i},{max_decoder_len-string.len(word)+j}}] = self:get_char_embedding(string.byte(word, j))
        decoder_input[{{i},{max_decoder_len-string.len(word)+j}}] = self:get_char_embedding(string.byte(word, j))
      end

      -- Shift decoder to left and add <EOS> at the end
      decoder_output[{{i}, {1,-2}}] = decoder_output[{{i},{2,-1}}]
      decoder_output[{{i}, {-1}}] = self:get_char_embedding("<end>")

      -- Add start to the left of ABC -> <pad><pad><start>ABC
      decoder_input[{{i}, {max_decoder_len-string.len(word)}}] = self:get_char_embedding("<start>")

    end

    return {enc_inp=final_X, dec_inp=decoder_input}, decoder_output
  else
    y = torch.Tensor(batch_op)
    return final_X, y
  end
end

function AirHandwritingDataLoader:get_batch_sample_ratio()
  num_samples = {}
  for i=1, self.num_classify do 
    table.insert(num_samples, math.floor(self.batch_size/self.num_classify))
    if num_samples[#num_samples] == 0 then num_samples[#num_samples] = 1 end
  end
  return num_samples
end

function AirHandwritingDataLoader:get_noise_mask_for_samples(samples)
  noise_mask = {}
  for i=1,#samples do
    for j=1,samples[i] do table.insert(noise_mask, 0) end
  end
  return noise_mask
end

function AirHandwritingDataLoader:get_shuffle_order(gest_by_type)
  local shuffle_order = {}
  for i=1, self.num_classify do
    table.insert(shuffle_order, torch.randperm(#self.train_data[i]))
  end
  return shuffle_order
end

function AirHandwritingDataLoader:getTotalTrainBatches()
  local total_train = 0
  for i, v in ipairs(self.train_data) do
    for i1, v1 in pairs(v) do total_train = total_train + 1 end
  end
  return math.floor(total_train / self.batch_size)
end

function AirHandwritingDataLoader:next_train_batch()
  local num_batches = self:getTotalTrainBatches()
  self.shuffle_order = self:get_shuffle_order(self.train_data)
  
  local idx_by_type = {}
  for i=1, self.num_classify do table.insert(idx_by_type, 1) end
  print('Total batches in 1 epoch '..num_batches)
  
  for i=1,num_batches do
    local class_samples_in_batch = self:get_batch_sample_ratio()  
    --print(class_samples_in_batch)
    local batch = {}
    local noise_mask = self:get_noise_mask_for_samples(class_samples_in_batch)
    
    local batch_idx = 1
    for c=1,self.num_classify do
      local num_gests = #self.train_data[c]
      for s=1,class_samples_in_batch[c] do
        if num_gests >= idx_by_type[c] then
          local shuffled_idx = self.shuffle_order[c][idx_by_type[c]]
          -- add {<class name>, <table key>, <table index>} into the batch
          table.insert(batch, {c, 'train', shuffled_idx})
          idx_by_type[c] = idx_by_type[c] + 1
        else
          local shuffled_idx = self.shuffle_order[c][torch.random(num_gests)]
          -- add {<class name>, <table key>, <table index>} into the batch
          table.insert(batch, {c, 'train', shuffled_idx})
          idx_by_type[c] = idx_by_type[c] + 1
          --assert(false, "Invalid Index")
        end
        if #batch == self.batch_size then 
          local X_batch, y_batch = self:get_features_for_batch(batch, noise_mask) 
          coroutine.yield(X_batch, y_batch)
          batch = {}
        end
        batch_idx = batch_idx + 1
      end
    end
    if #batch > 0 then
      local X_batch, y_batch = self:get_features_for_batch(batch, noise_mask) 
      coroutine.yield(X_batch, y_batch)
    end
  end
end

function AirHandwritingDataLoader:getTotalValBatches()
  local total_test = 0
  for i, v in ipairs(self.test_data) do
    for i1, v1 in pairs(v) do total_test = total_test + 1 end
  end
  return math.floor(total_test / self.batch_size)
end

function AirHandwritingDataLoader:next_val_batch()
  local batch_size = self.batch_size
  local num_test = self:getTotalValBatches()
  batch = {}
  for c=1, self.num_classify do
    for i=1, #self.test_data[c] do
      table.insert(batch, {c, 'test', i})

      if #batch == batch_size then
        local X_batch, y_batch = self:get_features_for_batch(batch, nil, true)
        if self.val_batch_info == 1 then
          coroutine.yield(X_batch, y_batch, batch)
        else
          coroutine.yield(X_batch, y_batch)
        end
        batch = {}
        collectgarbage()
      end
    end
  end
end

-- We return a table with keys from 1 to num_classes. Each key is mapped to a
-- table where each element of the table is another table of type (file_name,
-- gest_begin_time, gest_end_time)
function AirHandwritingDataLoader:get_all_gest_by_type(group_name)
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
      if gest_t:nElement() > 0 and torch.isTensor(gest_t) then
        for j=1, gest_t:size(1) do
          -- Only insert if we have sufficient frames at the end for the window
          -- Compare gest[j][2] == 0 since we need a hack for cases where hdf5
          -- isn't able to read empty tensors stored in h5 files. For those cases
          -- we add [0, 0] as the indexes to save and remove them here.
          
          if (torch.isTensor(gest_t[j]) and
              num_frames - gest_t[j][2] > 60 and
              gest_t[j][2] > 0) then
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

function AirHandwritingDataLoader:load_augmented_gestures(aug_h5)
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

function AirHandwritingDataLoader:check_if_usable_h5(f)
  local ends_with = "static.mp4.txt.h5"
  return string.sub(f,-string.len(ends_with)) == ends_with
end

function AirHandwritingDataLoader:final_class_for_gest(gest_type)
  if self.num_classes == self.num_classify then return gest_type end

  if gest_type == 1 then return 1
  elseif gest_type >= 2 and gest_type <= 6 then return 2
  elseif gest_type == 7 then return 3
  elseif gest_type == 8 or gest_type == 9 then return 4
  elseif gest_type == 10 or gest_type == 11 then return 5
  else assert(false) end
end

function AirHandwritingDataLoader:group_gest_into_classes(gest_by_type)
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

function AirHandwritingDataLoader:unroll_gest(
  gest_by_type, curr_gest, classification_type, win_step)
  local gest_len = curr_gest[3] - curr_gest[2]
  local gest_start, gest_end
  if data_type == 'test' then
    gest_start = curr_gest[2]
    gest_end = curr_gest[3]
  else
    gest_start = curr_gest[2] + math.floor(gest_len/5)
    gest_end = curr_gest[3] - math.floor(gest_len/5)
  end
  local step_size = win_step
  if classification_type == 'none' then
    for i=gest_start, gest_end, step_size do
      table.insert(gest_by_type, {curr_gest[1], i, i})
    end
  elseif classification_type == 'middle' then
    local gest_mid = math.floor((gest_start + gest_end) / 2)
    gest_start = math.max(gest_start, gest_mid - 3)
    gest_end = math.min(gest_end, gest_mid + 3)
    for i=gest_start, gest_end do 
      table.insert(gest_by_type, {curr_gest[1], i, i})
    end
  elseif classification_type == 'dense' then
    gest_start = curr_gest[2]
    gest_end = curr_gest[3]
    for i=gest_start, gest_end do
      table.insert(gest_by_type, {curr_gest[1], i, i})
    end
  elseif classification_type == 'average' then
    for i=gest_start, gest_end, step_size do
      table.insert(gest_by_type, {curr_gest[1], i, i})
    end
  end
end

function AirHandwritingDataLoader:unroll_gest_sequences(gest_by_type, 
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
  local classification_type = 'none'
  -- Dense labeling for test sequences
  if data_type == 'test' then 
    canSkip = false 
    classification_type = self.classification_type
    win_step = 1
  end
  for c=1,#gest_by_type do
    table.insert(unroll_gest_by_type, {})
    for i=1,#gest_by_type[c] do
      local is_valid = true
      if canSkip and c == 1 and torch.uniform() > 0.3 then is_valid = false end

      if is_valid then
        local curr_gest = gest_by_type[c][i]
        self:unroll_gest(unroll_gest_by_type[c], curr_gest,
            classification_type, win_step)
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

function AirHandwritingDataLoader:get_features_for_batch(batch, noise_mask, is_val_batch)
  -- Retun the actual features for the batch.
  local batch_inp, batch_op = {}, {}
  for i, v in ipairs(batch) do
    local y_gest = v[1]
    local op_win = y_gest
    assert(y_gest >= 1 and y_gest <= self.num_classify)

    -- Add gaussian noise based on noise mask.
    local noise = 0
    if noise_mask ~= nil then noise = noise_mask[i] end

    local inp_win
    if noise == 0 then 
      inp_win = self.data[v[2]][v[1]][v[3]]
    else 
      assert(false, "Noise not implemented yet.")
    end
    table.insert(batch_inp, inp_win)
    table.insert(batch_op, op_win)
  end

  -- X_batch is a table of Tensors, y_batch is a Tensor
  local X_batch, y_batch = self:convert_to_inp_format(batch_inp, batch_op, is_val_batch)
  y_batch = y_batch:clone()
  y_batch = y_batch:type('torch.IntTensor')
  return X_batch, y_batch
end

function AirHandwritingDataLoader:get_vocab_size()
  -- 1 for start and 1 for end
  -- Our vocab is <st>, <EOS>, <A>, ... , <Z>
  return 26 + 1 + 1
end

function AirHandwritingDataLoader:get_char_embedding(char)
  if char == '<start>' then return 1
  elseif char == '<end>' then return 2
  else return char - string.byte('A') + 3 end
end

function AirHandwritingDataLoader:load_valid_words()
  local words = {
    [1]="ABC",
    [2]="CBS",
    [3]="CNN",
    --DISCOVERY
    [4]="DISNEY",
    [5]="ESPN",
    [6]="FOX",
    [7]="HBO",
    [8]="NBC",
    [9]="TBS",
    [10]="BBC",
    [11]="FX",
    [12]="HULU",
    [13]="TNT",
    [14]="MUSIC",
    [15]="JAZZ",
    [16]="ROCK",
    [17]="DRAMA",
    [18]="MOVIE",
    [19]="SPORT",
    --WEATHER
    [20]="NEWS",
    [21]="MLB",
    [22]="NFL",
    [23]="TRAVEL",
    [24]="POKER",
    [25]="FOOD",
    [26]="KID",
    [27]="MAP",
    [28]="TV",
    [29]="GAME",
    [30]="VOICE",
    [31]="CALL",
    [32]="MAIL",
    [33]="MSG",
    [34]="FB",
    [35]="YOU",
    [36]="GOOGLE",
    [37]="SKYPE",
    [38]="QUIZ",
  }
  return words
end
