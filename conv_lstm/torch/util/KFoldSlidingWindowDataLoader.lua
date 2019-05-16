require 'torch'
require 'hdf5'
require 'math'
require 'paths'
require 'image'

local utils = require 'util.utils'

local KFoldSlidingWindowDataLoader = torch.class('KFoldSlidingWindowDataLoader')

function KFoldSlidingWindowDataLoader:__init(kwargs)
  self.train_h5file = utils.get_kwarg(kwargs, 'train_seq_h5',
    '../../openface_data/main_gest_by_file.h5')
  self.h5dir = utils.get_kwarg(kwargs, 'data_dir',
    '../../openface_data/face_gestures/dataseto_text')
  self.aug_gest_h5file = utils.get_kwarg(kwargs, 'aug_gests_h5',
    '../../openface_data/main_gest_by_file_aug_K_32.h5')

  self.model_type = utils.get_kwarg(kwargs, 'model_type', nil)

  self.use_openface_features = utils.get_kwarg(kwargs, 'use_openface_features', 1)
  self.num_scales = utils.get_kwarg(kwargs, 'num_scales')
  self.use_opt_flow = utils.get_kwarg(kwargs, 'use_opt_flow', 0)
  if self.use_opt_flow == 1 then
    self.opt_flow_dir = utils.get_kwarg(kwargs, 'opt_flow_dir')
    self.opt_flow_cache = {}
    self.cache_opt_flow_images = utils.get_kwarg(kwargs, 'cache_opt_flow_images')
  end
  self.finetune = utils.get_kwarg(kwargs, 'finetune')
  self.finetune_batch_size = utils.get_kwarg(kwargs, 'finetune_batch_size', 20)
  self.finetune_new_dataset = utils.get_kwarg(kwargs, 'finetune_new_dataset')
  assert(self.finetune ~= 1 or self.finetune_new_dataset ~= 1)
  self.use_all_face_diff = utils.get_kwarg(kwargs, 'use_all_face_diff')
  -- self.test_all_frames = utils.get_kwarg(kwargs, 'test_all_frames')

  self.val_batch_info = utils.get_kwarg(kwargs, 'val_batch_info', 0)
  if self.val_batch_info ~= 1 then
    self.aug_h5 = hdf5.open(self.aug_gest_h5file, 'r')
  end

  self.classification_type = utils.get_kwarg(
      kwargs, 'classification_type', 'none')
  -- CPM features
  self.use_cpm_features = utils.get_kwarg(kwargs, 'use_cpm_features', 1)
  self.cpm_h5_dir = utils.get_kwarg(kwargs, 'cpm_h5_dir', '')
  self.cpm_aug_h5file = utils.get_kwarg(kwargs, 'aug_cpm_h5', '')
  if self.val_batch_info ~= 1 and self.use_cpm_features  == 1 then
    self.cpm_aug_h5 = hdf5.open(self.cpm_aug_h5file, 'r')
  end

  -- Zface features
  self.use_zface_features = utils.get_kwarg(kwargs, 'use_zface_features', 1)
  self.zface_h5_dir = utils.get_kwarg(kwargs, 'zface_h5_dir', '')
  self.zface_aug_h5file = utils.get_kwarg(kwargs, 'aug_zface_h5', '')
  if self.val_batch_info ~= 1 and self.use_zface_features == 1 then
    self.zface_aug_h5 = hdf5.open(self.zface_aug_h5file, 'r')
  end

  self.batch_size = utils.get_kwarg(kwargs, 'batch_size')
  self.num_classes = utils.get_kwarg(kwargs, 'num_classes')
  self.num_classify = utils.get_kwarg(kwargs, 'num_classify')
  self.win_len = utils.get_kwarg(kwargs, 'win_len')
  self.win_step = utils.get_kwarg(kwargs, 'win_step')
  self.num_features = utils.get_kwarg(kwargs, 'num_features', 46)
  if self.num_scales == 1 then
    self.curr_win_sizes = {32}
  elseif self.num_scales == 2 then
    self.curr_win_sizes = {16, 32}
  else
    self.curr_win_sizes = {16, 32, 64}
  end
  print(self.curr_win_sizes)
  self.start_frame = 36

  local openface_mean_h5 = utils.get_kwarg(kwargs, 'openface_mean_h5')
  local cpm_mean_h5 = utils.get_kwarg(kwargs, 'cpm_mean_h5')
  local zface_mean_h5 = utils.get_kwarg(kwargs, 'zface_mean_h5')

  self.openface_mean_std = self:load_mean_h5(openface_mean_h5)
  if self.use_cpm_features == 1 then
    self.cpm_mean_std = self:load_mean_h5(cpm_mean_h5)
  end
  if self.use_zface_features == 1 then
    self.zface_mean_std = self:load_mean_h5(zface_mean_h5)
  end

  self.h5_by_file = self:load_data()
  self.X_by_file = self.h5_by_file.X
  self.y_by_file = self.h5_by_file.y
  if self.use_cpm_features == 1 then
    self.cpm_h5_by_file = self:load_cpm_data()
  end
  if self.use_zface_features == 1 then
    self.zface_h5_by_file = self:load_zface_data()
  end

  -- This is time consuming hence do this initially in the beginning once.
  if self.val_batch_info == 0 then
    self.gest_by_type = self:get_all_gest_by_type('train')
    self.gest_by_type = self:unroll_gest_sequences(self.gest_by_type, 'train')
    self.gest_by_type = self:group_gest_into_classes(self.gest_by_type)
  elseif self.finetune == 1 then
    self.gest_by_type = self:get_all_gest_by_type('test', true)
    -- Use train here since we don't really want to do a dense classification
    self.gest_by_type = self:unroll_gest_sequences(self.gest_by_type, 'train')
    self.gest_by_type = self:group_gest_into_classes(self.gest_by_type)
  end
  
  self.test_gest_by_type = self:get_all_gest_by_type('test')
  self.test_gest_by_type = self:unroll_gest_sequences(self.test_gest_by_type,
    'test')
  self.test_gest_by_type = self:group_gest_into_classes(self.test_gest_by_type)

  -- Load augmented gestures (Too much data to load into memory)
  -- self.aug_gests = self:load_augmented_gestures(self.aug_gest_h5file)

  self.shuffle_order = nil
  self.train_one_vs_all = utils.get_kwarg(kwargs, 'train_one_vs_all')

  print('Did read and load gestures')
end

function KFoldSlidingWindowDataLoader:isTrainFile(f) 
  local fname = paths.basename(f)
  local fnum = string.match(fname, '%d+')
  fnum = tonumber(fnum)
  if fnum > 11 then return true
  else return false end
end

function KFoldSlidingWindowDataLoader:load_mean_h5(h5_file_path)
  local f = hdf5.open(h5_file_path, 'r')
  local mean = f:read('/mean'):all()
  local std = f:read('/std'):all()
  f:close()
  return {mean=mean, std=std}
end


function KFoldSlidingWindowDataLoader:load_zface_data()
  local h5_by_file = {}
  for fp in paths.files(self.zface_h5_dir) do
    if self:check_if_usable_h5(fp, zelf.zface_h5_dir) then
      local f_path = self.zface_h5_dir .. '/' .. fp
      local hdf5_file = hdf5.open(f_path, 'r')
      local feats = hdf5_file:read('/features'):all()
      local X = torch.Tensor(feats:size()):copy(feats)
      X = utils.process_zface_data{
        X=X,
        mean=self.zface_mean_std['mean'][fp],
        std=self.zface_mean_std['std'][fp],
      }
      -- Non-Maximal suppression of velocity vectors
      --X = utils.do_non_maximal_suppression_zface_data(X)
      h5_by_file[fp] = {X=X}
      hdf5_file:close()
    end
  end
  return h5_by_file
end

function KFoldSlidingWindowDataLoader:load_cpm_data()
  local h5_by_file = {}
  for fp in paths.files(self.cpm_h5_dir) do
    print('process cpm')
    if self:check_if_usable_h5(fp, cpm_h5_dir) then
      local f_path = self.cpm_h5_dir.. '/' .. fp
      local hdf5_file = hdf5.open(f_path, 'r')
      local joints = hdf5_file:read('/joints'):all()
      local X = torch.Tensor(joints:size()):copy(joints)
      X = utils.process_cpm_data{X=X}
      -- Non-Maximal suppression of velocity vectors
      --X = utils.do_non_maximal_suppression_cpm_data(X)
      h5_by_file[fp] = {X=X}
      hdf5_file:close()
    end
  end
  return h5_by_file
end

function KFoldSlidingWindowDataLoader:load_data()
  -- create two tables per training set (i.e. 2 each for train, val)
  -- 1) list of all h5 files in each set
  -- 2) the h5 files for each dataset loaded into memory
  local h5_by_file = {}
  for fp in paths.files(self.h5dir) do
    if self:check_if_usable_h5(fp, self.h5dir) then
      print('Process file '..fp)
      local f_path = self.h5dir .. '/' .. fp
      local hdf5_file = hdf5.open(f_path, 'r')
      local annots = hdf5_file:read('/annotations'):all()
      local feats = hdf5_file:read('/features'):all()
      local X1 = torch.Tensor(feats:size()):copy(feats)
      local y1 = torch.Tensor(annots:size()):copy(annots)
      local X, y = utils.process_data{
        X=X1, 
        y=y1,
        mean=self.openface_mean_std.mean[fp],
        std=self.openface_mean_std.std[fp],
      }
      -- Non-Maximal suppression of velocity vectors
      --X = utils.do_non_maximal_suppression(X)
      h5_by_file[fp] = {X=X, y=y}
    end
  end
  return h5_by_file
end

function KFoldSlidingWindowDataLoader:get_aug_curr_win_inputs(curr_file, 
  curr_t, win_sizes, org_gest)
  local debug = false
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
    --gest = utils.do_non_maximal_suppression(gest)
    if self.use_openface_features ~= 1 then gest = nil end
    if debug == true then print("Got openface aug gesture.") end

    if self.cpm_aug_h5 ~= nil then
      local all_aug = self.cpm_aug_h5:read(
          '/'..temp_idx..'/'..curr_t..'/'..w_size):all()
      local cpm_gest = all_aug[aug_idx]:t()
      cpm_gest = utils.get_augmented_cpm_data_from_trimmed{X=cpm_gest}
      --cpm_gest = utils.do_non_maximal_suppression_cpm_data(cpm_gest)
      if self.use_openface_features == 1 then
        gest = torch.cat({gest, cpm_gest})
      else
        gest = cpm_gest
      end
      if debug == true then print("Got CPM aug gesture.") end
    end

    if self.zface_aug_h5 ~= nil then
      local all_aug = self.zface_aug_h5:read(
        '/'..temp_idx..'/'..curr_t..'/'..w_size):all()
      local zface_gest = all_aug[aug_idx]:t()
      zface_gest = utils.get_augmented_zface_data_from_trimmed{
        X=zface_gest,
        mean=self.zface_mean_std['mean'][curr_file],
        std=self.zface_mean_std['std'][curr_file],
      }
      --zface_gest = utils.do_non_maximal_suppression_zface_data(zface_gest)
      if self.use_openface_features == 1 or gest ~= nil then
        gest = torch.cat({gest, zface_gest})
      else 
        gest = zface_gest
      end
      if debug == true then print("Got z-face aug gesture.") end
    end
    table.insert(inp, gest)
  end
  return inp
end

function KFoldSlidingWindowDataLoader:get_opt_flow_file_name(t)
  -- If we have skipped frames in opt flow then we should do the corresponding
  -- math here.
  return string.format('flow_x_%05d.jpg', t) , string.format('flow_y_%05d.jpg', t)
end

function KFoldSlidingWindowDataLoader:get_opt_flow_images(dir, t)
  local x_img_name, y_img_name = self:get_opt_flow_file_name(t)
  local flow_x_img_path = paths.concat(dir, x_img_name)
  local flow_y_img_path = paths.concat(dir, y_img_name)
  if self.cache_opt_flow_images == 1 and self.opt_flow_cache[flow_x_img_path] ~= nil then
    return self.opt_flow_cache[flow_x_img_path], self.opt_flow_cache[flow_y_img_path]
  end
  local x_img = image.load(flow_x_img_path, 1)
  local y_img = image.load(flow_y_img_path, 1)

  x_img = image.scale(x_img, 224, 224)
  y_img = image.scale(y_img, 224, 224)
  if self.cache_opt_flow_images == 1 then
    self.opt_flow_cache[flow_x_img_path] = x_img
    self.opt_flow_cache[flow_y_img_path] = y_img
  end
  return x_img, y_img
end

function KFoldSlidingWindowDataLoader:get_opt_flow_input(curr_file, curr_t)
  local st_time = os.clock()
  local s, e = string.find(curr_file, '%d%d%d')
  local file_num = string.sub(curr_file, s, e)
  local dir = paths.concat(self.opt_flow_dir, file_num..'_skip_4')
  dir = paths.concat(dir, 'cropped')
  -- We need to convert the curr_t into a file number. It is usually prefixex
  -- as flow_x_000ab.jpg flow_y_000ab.jpg
  -- The pretrained network uses 10 optical flow timestamps.
  local opt_flow_inp = torch.Tensor(20, 224, 224)
  local i = 1
  local start = math.floor(curr_t / 4) - 4
  local _end = start + 9
  for t=start,_end do
    local x_img, y_img = self:get_opt_flow_images(dir, t)
    opt_flow_inp[{i,{},{}}] = x_img:clone()
    opt_flow_inp[{i+1,{},{}}] = y_img:clone()
    i = i + 2
  end
  local end_time = os.clock()
  --print(string.format("Time elapsed loading %.3f ", end_time - st_time)) 
  collectgarbage()
  return opt_flow_inp
end

function KFoldSlidingWindowDataLoader:get_curr_win_inputs(args)
  local X, y, curr_t, win_sizes = args.X, args.y, args.curr_t, args.win_sizes
  local noise = args.noise or nil
  local cpm_X = args.cpm_X or nil
  local zface_X = args.zface_X or nil
  local inp = {}
  for i=1, #win_sizes do
    local w_size = win_sizes[i]
    local X_openface, X_cpm, X_zface, X_final
    if X~= nil and self.use_openface_features == 1 then
      X_openface = X[{{curr_t-w_size/2, curr_t-1+w_size/2},{}}]
      X_final = X_openface:clone()
    end
    if cpm_X ~= nil then
      X_cpm = cpm_X[{{curr_t-w_size/2, curr_t-1+w_size/2},{}}]
      if X_final == nil then X_final = X_cpm:clone()
      else X_final = torch.cat(X_final, X_cpm) end
    end
    if zface_X ~= nil then 
      local X_zface = zface_X[{{curr_t-w_size/2, curr_t-1+w_size/2},{}}]
      if X_final == nil then X_final = X_zface:clone()
      else X_final = torch.cat(X_final, X_zface) end
    end
    if noise ~= nil and noise == 1 then
      X_final = X_final:index(
          1, torch.linspace(X_final:size(1),1,X_final:size(1)):long())
    end
    table.insert(inp, X_final)
  end
  return inp
end

function KFoldSlidingWindowDataLoader:convert_to_binary_classif(X, y, main_target_label)
  if main_target_label == nil then
    error('Trying to convert to binary classif without target')
  end

  local final_X, final_y

  local main_target_idx = {}
  for i=1,y:size(1) do
    if y[i] == main_target_label then
      table.insert(main_target_idx, i)
    end
  end
  local zero_label_idx = {}
  local rand_y_idx = torch.randperm(y:size(1))
  for i=1,rand_y_idx:size(1) do
    if y[rand_y_idx[i]] ~= main_target_label then
      table.insert(zero_label_idx, rand_y_idx[i])
    end
    if #zero_label_idx == #main_target_idx then
      break
    end
  end
  --assert(#zero_label_idx == #main_target_idx)
  
  -- We have to choose these zero label idx and main target idx that
  -- were created when choosing our inputs
  if torch.isTensor(X) then
    error('Not implemented!!')
  else
    -- X is a table need to select values at every X
    final_X = {}
    for scale=1,#X do
      local org_size = X[scale]:size()
      org_size[1] = 2 * #main_target_idx
      table.insert(final_X, torch.Tensor(org_size):zero())
      for i=1,#main_target_idx do
        local idx = main_target_idx[i]
        final_X[scale][{{i},{}}] = X[scale][{{idx},{}}]
      end
      for i=1,#zero_label_idx do
        local idx = zero_label_idx[i]
        final_X[scale][{{#main_target_idx+i},{}}] = X[scale][{{idx},{}}]
      end
    end
    final_y = torch.Tensor(2 * #main_target_idx):zero()
    final_y[{{1,#main_target_idx}}] = 2
    final_y[{{#main_target_idx+1, #main_target_idx+#zero_label_idx}}] = 1
  end
  return final_X, final_y
end

function KFoldSlidingWindowDataLoader:convert_to_inp_format(
  batch_inp, batch_op, batch_opt_flow_inp) 

  local debug = false

  local final_X = {}
  for i=1,#self.curr_win_sizes do
    table.insert(final_X, torch.zeros(
      #batch_inp, 1, self.curr_win_sizes[i], self.num_features))
  end

  for i=1,#self.curr_win_sizes do
    for b=1,#batch_inp do
      if debug then 
        print(final_X[i][b][1]:size())
        print(batch_inp[b][i]:size())
      end
      assert(final_X[i][b][1]:size(1) == batch_inp[b][i]:size(1))
      final_X[i][b][1] = batch_inp[b][i] 
    end
  end

  if batch_opt_flow_inp ~= nil and #batch_opt_flow_inp == #batch_inp then
    table.insert(final_X, torch.zeros(#batch_inp, 20, 224, 224))
    for i=1,#batch_opt_flow_inp do
      final_X[#final_X][i] = batch_opt_flow_inp[i]
    end
  end

  y = torch.Tensor(batch_op)
  return final_X, y
end

function KFoldSlidingWindowDataLoader:get_batch_sample_ratio()
  num_samples = {}
  for i=1, self.num_classify do 
    table.insert(num_samples, math.floor(self.batch_size/self.num_classify))
  end
  return num_samples
end

function KFoldSlidingWindowDataLoader:get_noise_mask_for_samples(samples)
  noise_mask = {}
  for i=1,#samples do
    for j=1,samples[i] do table.insert(noise_mask, 0) end
  end
  return noise_mask
end

function KFoldSlidingWindowDataLoader:get_shuffle_order(gest_by_type)
  local shuffle_order = {}
  for i=1,#self.gest_by_type do
    if #self.gest_by_type[i] > 0 then
      table.insert(shuffle_order, torch.randperm(#self.gest_by_type[i]))
    else table.insert(shuffle_order, 0) end
  end
  return shuffle_order
end

function KFoldSlidingWindowDataLoader:getTotalTrainBatches()
  if self.finetune_new_dataset == 1 then
    local min_gest_num, min_gest_type = 10000, -1
    for i=1,#self.gest_by_type do
      if #self.gest_by_type[i] > 0 then
        min_gest_num = math.min(min_gest_num, #self.gest_by_type[i])
        if min_gest_num == #self.gest_by_type[i] then min_gest_type = i end
      end
    end
    return min_gest_num
  end
  if self.finetune == 1 then
    local num_train = 0
    for i=1, self.num_classify do num_train = num_train + #self.gest_by_type[i] end 
    return math.floor(num_train / self.finetune_batch_size)
  end

  local num_train_nods = #self.gest_by_type[2]
  local total_samples_in_epoch = num_train_nods * self.num_classify
  local num_batches = math.floor(total_samples_in_epoch / self.batch_size)
  return num_batches
end

function KFoldSlidingWindowDataLoader:next_ratio_batch()
  -- sample in ratio terms such that each gesture is present a certain times
  -- in each batch
  local min_gest_num, min_gest_type = 10000, -1
  for i=1,#self.gest_by_type do
    if #self.gest_by_type[i] > 0 then
      min_gest_num = math.min(min_gest_num, #self.gest_by_type[i])
      if min_gest_num == #self.gest_by_type[i] then min_gest_type = i end
    end
  end
  local batch_ratio = {}
  for i=1,#self.gest_by_type do 
    table.insert(batch_ratio, 
        math.min(5, math.floor(#self.gest_by_type[i]/min_gest_num)+1)) 
    if #self.gest_by_type[i] == 0 then batch_ratio[#batch_ratio] = 0 end
  end

  local num_batches = min_gest_num

  -- TODO(Mohit): Complete this
  self.shuffle_order = self:get_shuffle_order(self.gest_by_type)

  local idx_by_type = {}
  -- All examples in batch are real examples only, no augmented gestures.
  local num_real_in_batch = batch_ratio
  for i=1, self.num_classify do table.insert(idx_by_type, 1) end
  print('Total batches in 1 epoch '..num_batches)

  for i=1,num_batches do
    -- In this loop we don't do a fixed batch representation based on things
    -- but rather arbitrarily decide the ratio to be similar in representation
    -- to the original one
    local class_samples_in_batch = batch_ratio
    --print(class_samples_in_batch)
    local batch = {}
    
    local batch_idx = 1
    for c=1,self.num_classify do
      local num_gests = #self.gest_by_type[c]
      for s=1,class_samples_in_batch[c] do
        if num_gests >= idx_by_type[c] and num_real_in_batch[c] >= s then
          local shuffled_idx = self.shuffle_order[c][idx_by_type[c]]
          table.insert(batch, self.gest_by_type[c][shuffled_idx])
          idx_by_type[c] = idx_by_type[c] + 1
        else
          local rand_idx = torch.random(1, num_gests)
          local shuffled_idx = self.shuffle_order[c][rand_idx]
          table.insert(batch, self.gest_by_type[c][shuffled_idx])
        end
        batch_idx = batch_idx + 1
      end
    end
    local X_batch, y_batch = self:get_features_for_batch(batch, noise_mask) 
    coroutine.yield(X_batch, y_batch)
    collectgarbage()
  end
end

function KFoldSlidingWindowDataLoader:next_train_batch()
  --self:next_batch(self.train_files, self.train_h5_by_file)
  if self.finetune_new_dataset == 1 then
    return self:next_ratio_batch()
  end

  local num_train_nods = #self.gest_by_type[2]
  local total_samples_in_epoch = num_train_nods * self.num_classify
  local num_batches = math.floor(total_samples_in_epoch / self.batch_size)
  -- TODO(Mohit): Complete this
  self.shuffle_order = self:get_shuffle_order(self.gest_by_type)
  
  local idx_by_type = {}
  local num_real_in_batch = {}
  for i=1, self.num_classify do table.insert(idx_by_type, 1) end
  for i=1, self.num_classify do
    table.insert(num_real_in_batch,
        math.ceil(#self.gest_by_type[i]/num_batches))
  end
  print('Total batches in 1 epoch '..num_batches)
  
  for i=1,num_batches do
    local class_samples_in_batch = self:get_batch_sample_ratio()  
    --print(class_samples_in_batch)
    local batch = {}
    local noise_mask = self:get_noise_mask_for_samples(class_samples_in_batch)
    
    local batch_idx = 1
    for c=1,self.num_classify do
      local num_gests = #self.gest_by_type[c]
      for s=1,class_samples_in_batch[c] do
        if num_gests >= idx_by_type[c] and num_real_in_batch[c] >= s then
          local shuffled_idx = self.shuffle_order[c][idx_by_type[c]]
          table.insert(batch, self.gest_by_type[c][shuffled_idx])
          idx_by_type[c] = idx_by_type[c] + 1
        else
          local rand_idx = torch.random(1, num_gests)
          local shuffled_idx = self.shuffle_order[c][rand_idx]
          table.insert(batch, self.gest_by_type[c][shuffled_idx])
          noise_mask[batch_idx] = 1
        end
        batch_idx = batch_idx + 1
      end
    end
    local X_batch, y_batch = self:get_features_for_batch(
        batch, noise_mask, 'train') 
    coroutine.yield(X_batch, y_batch)
    collectgarbage()
  end
end

-- While we unroll the gesutres we should save some of them into the dictionary
-- with 'finetune' key and use them to finetune on. Or we could simply use the
-- train key for them and not have the train dataset.
-- We also need to save some gestures from every user into this finetune 
-- dataset above.
function KFoldSlidingWindowDataLoader:next_finetune_batch()
  local batch_size = self.finetune_batch_size
  local num_test = 1
  for c=1, self.num_classify do 
    num_test = num_test + #self.gest_by_type[c]
  end
  local batch = {}
  for c=1, self.num_classify do
    for i=1, #self.gest_by_type[c] do
      table.insert(batch, self.gest_by_type[c][i])

      if #batch == batch_size then
        local X_batch, y_batch = self:get_features_for_batch(batch, nil)
        coroutine.yield(X_batch, y_batch)
        batch = {}
        collectgarbage()
      end
    end
  end
end

function KFoldSlidingWindowDataLoader:next_val_batch()
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
        local X_batch, y_batch = self:get_features_for_batch(
            batch, nil, 'test')
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
function KFoldSlidingWindowDataLoader:get_all_gest_by_type(group_name, finetune)
  finetune = finetune or false

  local gest_by_type = {}
  for i=1,self.num_classes do table.insert(gest_by_type, {}) end
  local h5_file = hdf5.open(self.train_h5file, 'r')

  local f_contents = h5_file:read("/"):all()
  f_contents = f_contents[group_name]
  for k, v in pairs(f_contents) do
    local file_gest_by_type = v
    local num_frames = self.h5_by_file[k].y:size(1)
    local finetune_frames = utils.FINETUNE_FRAMES_BY_FILE[k]

    local did_add_gests = false
    if group_name == 'test' and self.test_all == 1 then
      -- We test every frame irrespective of everything i.e irrespective
      -- of what frames are correct etc. We want this to know how the
      -- algorithm behaves in general
      for j=200, num_frames-100 do
        local gest = self.h5_by_file[k].y[j] + 1
        -- local gest = self.h5_by_file[k].y
        table.insert(gest_by_type[gest], {k, j, j})
      end
      did_add_gests = true
    end

    if not did_add_gests then
      for i=1, self.num_classes do
        -- gest_t is a tensor Nx2 i.e. all gestures of type i in h5 file
        gest_t = file_gest_by_type[tostring(i-1)]
        if gest_t:nElement() > 0 and torch.isTensor(gest_t) then

          for j=1, gest_t:size(1) do
            -- Only insert if we have sufficient frames at the end for the window
            -- Compare gest[j][2] == 0 since we need a hack for cases where hdf5
            -- isn't able to read empty tensors stored in h5 files. For those cases
            -- we add [0, 0] as the indexes to save and remove them here.

            if (self.finetune == 1 and group_name == 'test' and
              j < finetune_frames[i]) then
              -- Since we are finetuning on these gestures we don't want them to
              -- be included in our validation dataset

            else
              if torch.isTensor(gest_t[j]) and 
                finetune and
                j < finetune_frames[i] then
                table.insert(gest_by_type[i], {k, gest_t[j][1], gest_t[j][2]})
              elseif (torch.isTensor(gest_t[j]) and
                num_frames - gest_t[j][2] > 60 and
                gest_t[j][2] > 0) then
                -- File name as first argument
                table.insert(gest_by_type[i], {k, gest_t[j][1], gest_t[j][2]})
              end
            end
          end
        end
      end 
    end
  end
  h5_file:close()
  return gest_by_type
end

function KFoldSlidingWindowDataLoader:load_augmented_gestures(aug_h5)
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

function KFoldSlidingWindowDataLoader:check_if_usable_h5(f, f_dir)
  if string.find(f_dir, 'cardiff') ~= nil then
    local ends_with = '.h5'
    return string.sub(f,-string.len(ends_with)) == ends_with
  else
    local ends_with = "mp4.txt.h5"
    --local ends_with = ".h5"
    return string.sub(f,-string.len(ends_with)) == ends_with
  end
end

function KFoldSlidingWindowDataLoader:final_class_for_gest_7(gest_type)
  if gest_type == 1 then return 1
  elseif gest_type == 2 or gest_type == 3 or gest_type == 6 then return 2
  elseif gest_type == 4 or gest_type == 5 then return 3
  elseif gest_type == 7 then return 4
  elseif gest_type == 8 then return 5
  elseif gest_type == 9 then return 6
  elseif gest_type == 10 or gest_type == 11 then return 7
  else assert(false) end
end

function KFoldSlidingWindowDataLoader:final_class_for_gest(gest_type)
  if self.num_classes == self.num_classify then return gest_type end
  if self.num_classify == 7 then
    return self:final_class_for_gest_7(gest_type)
  end

  if gest_type == 1 then return 1
  elseif gest_type >= 2 and gest_type <= 6 then return 2
  elseif gest_type == 7 then return 3
  elseif gest_type == 8 or gest_type == 9 then return 4
  elseif gest_type == 10 or gest_type == 11 then return 5
  else assert(false) end
end

function KFoldSlidingWindowDataLoader:group_gest_into_classes(gest_by_type)
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

function KFoldSlidingWindowDataLoader:unroll_gest(
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

function KFoldSlidingWindowDataLoader:unroll_gest_sequences(gest_by_type, 
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

function KFoldSlidingWindowDataLoader:get_features_for_batch(
      batch, noise_mask, data_type)

  -- Retun the actual features for the batch.
  local debug = false
  local max_win_size = math.max(unpack(self.curr_win_sizes))
  local batch_inp, batch_op, batch_opt_flow_inp = {}, {}, {}
  for i, v in ipairs(batch) do
    local y_gest = self.h5_by_file[v[1]].y
    local org_gest = y_gest[v[3]]
    local op_win = self:final_class_for_gest(y_gest[v[3]]+1)
    assert(op_win >= 1 and op_win <= self.num_classify)

    -- Add gaussian noise based on noise mask.
    local noise = 0
    if noise_mask ~= nil then noise = noise_mask[i] end

    local curr_t = v[2]
    local curr_file = v[1]
    local inp_win = nil
    local cpm_X, zface_X = nil, nil
    if self.cpm_h5_by_file ~= nil then 
      cpm_X = self.cpm_h5_by_file[curr_file].X 
    end
    if self.zface_h5_by_file ~= nil then 
      zface_X = self.zface_h5_by_file[curr_file].X
    end

    if noise == 0 or (noise==1 and org_gest <= 1) then 
      if debug then print('get real data') end
      inp_win = self:get_curr_win_inputs{
        X=self.h5_by_file[curr_file].X,
        y=self.h5_by_file[curr_file].y,
        curr_t=curr_t,
        win_sizes=self.curr_win_sizes,
        noise=noise,
        cpm_X=cpm_X,
        zface_X=zface_X,
      }
    else 
      if debug then print('get aug data') end
      inp_win = self:get_aug_curr_win_inputs(
        curr_file,
        curr_t,
        self.curr_win_sizes,
        org_gest)
    end
    if self.use_opt_flow == 1 then
      local opt_flow_inp = self:get_opt_flow_input(curr_file, curr_t) 
      table.insert(batch_opt_flow_inp, opt_flow_inp)
    end
    table.insert(batch_inp, inp_win)
    table.insert(batch_op, op_win)
  end
  if debug then print('convert to inp format') end
  -- X_batch is a table of Tensors, y_batch is a Tensor
  local X_batch, y_batch = self:convert_to_inp_format(
      batch_inp, batch_op, batch_opt_flow_inp)

  if self.train_one_vs_all > 0 then
    if data_type == 'train' then
      X_batch, y_batch = self:convert_to_binary_classif(
         X_batch, y_batch, self.train_one_vs_all+1, data_type)
    else
      y_batch = y_batch:clone()
      for i=1,y_batch:size(1) do 
        if y_batch[i] == self.train_one_vs_all+1 then
          y_batch[i] = 2
        else
          y_batch[i] = 1
        end
      end
    end
  end

  y_batch = y_batch:clone()
  y_batch = y_batch:type('torch.IntTensor')
  --print(y_batch)
  --assert(false)
  return X_batch, y_batch
end

function KFoldSlidingWindowDataLoader:get_train_size()
  assert(#(self.gest_by_type) == self.num_classify)
  return #self.gest_by_type[2] * self.num_classify
end

function KFoldSlidingWindowDataLoader:get_validation_size()
  return 0
end

function KFoldSlidingWindowDataLoader:init_batch(num_epochs)
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

function KFoldSlidingWindowDataLoader:reset_batch()
  self.curr_batch = 1
  self.curr_epoch = self.curr_epoch + 1
  for i=1,self.num_classify do self.idx_by_type[i] = 1 end
end


