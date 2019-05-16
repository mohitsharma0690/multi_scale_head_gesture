require 'torch'
require 'hdf5'
require 'math'
require 'paths'
require 'image'

local utils = require 'util.utils'

local MultiScaleSpatialDataLoader = torch.class('MultiScaleSpatialDataLoader')

function MultiScaleSpatialDataLoader:__init(kwargs)
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

  print('Did read and load gestures')
end

function MultiScaleSpatialDataLoader:isTrainFile(f) 
  local fname = paths.basename(f)
  local fnum = string.match(fname, '%d+')
  fnum = tonumber(fnum)
  if fnum > 11 then return true
  else return false end
end

function MultiScaleSpatialDataLoader:load_mean_h5(h5_file_path)
  local f = hdf5.open(h5_file_path, 'r')
  local mean = f:read('/mean'):all()
  local std = f:read('/std'):all()
  f:close()
  return {mean=mean, std=std}
end

function MultiScaleSpatialDataLoader:load_zface_data()
  local h5_by_file = {}
  for fp in paths.files(self.zface_h5_dir) do
    if self:check_if_usable_h5(fp) then
      local f_path = self.zface_h5_dir .. '/' .. fp
      local hdf5_file = hdf5.open(f_path, 'r')
      local feats = hdf5_file:read('/features'):all()
      local X = torch.Tensor(feats:size()):copy(feats)
      X = utils.process_zface_data{X=X}
      -- Non-Maximal suppression of velocity vectors
      X = utils.do_non_maximal_suppression_zface_data(X)
      h5_by_file[fp] = {X=X}
      hdf5_file:close()
    end
  end
  return h5_by_file
end

function MultiScaleSpatialDataLoader:load_cpm_data()
  local h5_by_file = {}
  for fp in paths.files(self.cpm_h5_dir) do
    if self:check_if_usable_h5(fp) then
      local f_path = self.cpm_h5_dir.. '/' .. fp
      local hdf5_file = hdf5.open(f_path, 'r')
      local joints = hdf5_file:read('/joints'):all()
      local X = torch.Tensor(joints:size()):copy(joints)
      X = utils.process_cpm_data{X=X}
      -- Non-Maximal suppression of velocity vectors
      X = utils.do_non_maximal_suppression_cpm_data(X)
      h5_by_file[fp] = {X=X}
      hdf5_file:close()
    end
  end
  return h5_by_file
end

function MultiScaleSpatialDataLoader:process_data(args)
  local X, y = args.X, args.y
  local mean, std = args.mean, args.std
  local norm_type = args.norm_type
  local normalize_signals = args.signals

  assert(X:size(2) > 53)
  X = X[{{}, {1, 148}}]
  -- Should smooth data
  -- In numpy :12 will not get us the 12'th row but in Lua it will. Although it
  -- is compensated by the fact that in Lua we start from 1 while in python from
  -- 0.
  local X_pose = X[{{}, {1, 12}}]
  local X_size = X:size()
  local X_pose_vel= torch.Tensor(X_pose:size(1), 6):zero()
  X_pose_vel[{{2,-1},{}}] = X[{{2,-1}, {7,12}}] - X[{{1,-2},{7,12}}]
  -- Convert into degrees since the difference in radians could be very small. 
  X_pose_vel[{{},{4,6}}] = X_pose_vel[{{},{4,6}}] * 180.0 / math.pi

  local X_landmarks_vel = torch.Tensor(X:size(1),36*2):zero()
  X_landmarks_vel[{{2,-1},{1,36}}] = X[{{2,-1},{13,13+35}}] - X[{{1,-2},{13,13+35}}]
  --- 12 + 68 + 1 = 81
  X_landmarks_vel[{{2,-1},{37,72}}] = X[{{2,-1},{81,81+35}}] - X[{{1,-2},{81,81+35}}]

  -- X = torch.cat({X_pose_vel, X_landmarks_vel})
  X = X_landmarks_vel:clone()
  -- Do not normalize velocity
  -- X = utils.normalize_data_with{ X=X, mean=mean, std=std }
  
  assert(normalize_signals == nil)

  return X, y
end
function MultiScaleSpatialDataLoader:load_data()
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
      local X, y = self:process_data{
        X=X1, 
        y=y1,
        mean=self.openface_mean_std.mean[fp],
        std=self.openface_mean_std.std[fp],
      }
      -- Non-Maximal suppression of velocity vectors
      X = utils.do_non_maximal_suppression(X)
      h5_by_file[fp] = {X=X, y=y}
    end
  end
  return h5_by_file
end

function MultiScaleSpatialDataLoader:get_aug_curr_win_inputs(curr_file, 
  curr_t, win_sizes, org_gest)
  local inp = {}
  for i=1, #win_sizes do
    local w_size = win_sizes[i]
    -- Get the (Num_Augmentations, Num_features, Win_size) values
    local temp_idx = curr_file .. '/' .. tostring(org_gest)
    local all_aug = self.aug_h5:read('/'..temp_idx..'/'..curr_t..'/'..w_size):all()
    local aug_idx = torch.random(1, all_aug:size(1))
    local gest = all_aug[aug_idx]:t()
    local new_gest = torch.cat(gest[{{},{13,13+35}}], gest[{{},{81,81+35}}])
    new_gest[{{},{2,-1}}] = new_gest[{{},{2,-1}}] - new_gest[{{},{1,-2}}]
    for j=1,new_gest:size(2) do new_gest[{{1},{j}}] = 0 end
    gest = utils.do_non_maximal_suppression(new_gest)

    if self.cpm_aug_h5 ~= nil then
      local all_aug = self.cpm_aug_h5:read(
          '/'..temp_idx..'/'..curr_t..'/'..w_size):all()
      local cpm_gest = all_aug[aug_idx]:t()
      cpm_gest = utils.get_augmented_cpm_data_from_trimmed{X=cpm_gest}
      cpm_gest = utils.do_non_maximal_suppression_cpm_data(cpm_gest)
      if self.use_openface_features == 1 then
        gest = torch.cat({gest, cpm_gest})
      else
        gest = cpm_gest
      end
    end
    if self.zface_aug_h5 ~= nil then
      local all_aug = self.zface_aug_h5:read(
        '/'..temp_idx..'/'..curr_t..'/'..w_size):all()
      local zface_gest = all_aug[aug_idx]:t()
      zface_gest = utils.get_augmented_zface_data_from_trimmed{X=zface_gest}
      zface_gest = utils.do_non_maximal_suppression_zface_data(zface_gest)
      gest = torch.cat({gest, zface_gest})
    end
    table.insert(inp, gest)
  end
  return inp
end

function MultiScaleSpatialDataLoader:get_opt_flow_file_name(t)
  -- If we have skipped frames in opt flow then we should do the corresponding
  -- math here.
  return string.format('flow_x_%05d.jpg', t) , string.format('flow_y_%05d.jpg', t)
end

function MultiScaleSpatialDataLoader:get_opt_flow_images(dir, t)
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

function MultiScaleSpatialDataLoader:get_opt_flow_input(curr_file, curr_t)
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

function MultiScaleSpatialDataLoader:get_curr_win_inputs(args)
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

function MultiScaleSpatialDataLoader:convert_to_inp_format(
  batch_inp, batch_op, batch_opt_flow_inp) 

  local final_X = {}
  for i=1,#self.curr_win_sizes do
    table.insert(final_X, torch.zeros(
      #batch_inp, 1, self.curr_win_sizes[i], self.num_features))
  end

  for i=1,#self.curr_win_sizes do
    for b=1,#batch_inp do
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

function MultiScaleSpatialDataLoader:get_batch_sample_ratio()
  num_samples = {}
  for i=1, self.num_classify do 
    table.insert(num_samples, math.floor(self.batch_size/self.num_classify))
  end
  return num_samples
end

function MultiScaleSpatialDataLoader:get_noise_mask_for_samples(samples)
  noise_mask = {}
  for i=1,#samples do
    for j=1,samples[i] do table.insert(noise_mask, 0) end
  end
  return noise_mask
end

function MultiScaleSpatialDataLoader:get_shuffle_order(gest_by_type)
  local shuffle_order = {}
  for i=1,#self.gest_by_type do
    table.insert(shuffle_order, torch.randperm(#self.gest_by_type[i]))
  end
  return shuffle_order
end

function MultiScaleSpatialDataLoader:getTotalTrainBatches()
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

function MultiScaleSpatialDataLoader:next_train_batch()
  --self:next_batch(self.train_files, self.train_h5_by_file)
  local num_train_nods = #self.gest_by_type[2]
  local total_samples_in_epoch = num_train_nods * self.num_classify
  local num_batches = math.floor(total_samples_in_epoch / self.batch_size)
  -- TODO(Mohit): Complete this
  self.shuffle_order = self:get_shuffle_order(self.gest_by_type)
  
  idx_by_type = {}
  num_real_in_batch = {}
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
    local X_batch, y_batch = self:get_features_for_batch(batch, noise_mask) 
    coroutine.yield(X_batch, y_batch)
    collectgarbage()
  end
end

-- While we unroll the gesutres we should save some of them into the dictionary
-- with 'finetune' key and use them to finetune on. Or we could simply use the
-- train key for them and not have the train dataset.
-- We also need to save some gestures from every user into this finetune 
-- dataset above.
function MultiScaleSpatialDataLoader:next_finetune_batch()
  local batch_size = self.finetune_batch_size
  local num_test = 1
  for c=1, self.num_classify do 
    num_test = num_test + #self.gest_by_type[c]
  end
  batch = {}
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

function MultiScaleSpatialDataLoader:next_val_batch()
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
function MultiScaleSpatialDataLoader:get_all_gest_by_type(group_name, finetune)
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
  h5_file:close()
  return gest_by_type
end

function MultiScaleSpatialDataLoader:load_augmented_gestures(aug_h5)
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

function MultiScaleSpatialDataLoader:check_if_usable_h5(f)
  local ends_with = "static.mp4.txt.h5"
  return string.sub(f,-string.len(ends_with)) == ends_with
end

function MultiScaleSpatialDataLoader:final_class_for_gest(gest_type)
  if self.num_classes == self.num_classify then return gest_type end

  if gest_type == 1 then return 1
  elseif gest_type >= 2 and gest_type <= 6 then return 2
  elseif gest_type == 7 then return 3
  elseif gest_type == 8 or gest_type == 9 then return 4
  elseif gest_type == 10 or gest_type == 11 then return 5
  else assert(false) end
end

function MultiScaleSpatialDataLoader:group_gest_into_classes(gest_by_type)
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

function MultiScaleSpatialDataLoader:unroll_gest(
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

function MultiScaleSpatialDataLoader:unroll_gest_sequences(gest_by_type, 
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

function MultiScaleSpatialDataLoader:get_features_for_batch(batch, noise_mask)
  -- Retun the actual features for the batch.
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

  -- X_batch is a table of Tensors, y_batch is a Tensor
  local X_batch, y_batch = self:convert_to_inp_format(
      batch_inp, batch_op, batch_opt_flow_inp)
  y_batch = y_batch:clone()
  y_batch = y:type('torch.IntTensor')
  --print(y_batch)
  --assert(false)
  return X_batch, y_batch
end

function MultiScaleSpatialDataLoader:get_train_size()
  assert(#(self.gest_by_type) == self.num_classify)
  return #self.gest_by_type[2] * self.num_classify
end

function MultiScaleSpatialDataLoader:get_validation_size()
  return 0
end

function MultiScaleSpatialDataLoader:init_batch(num_epochs)
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

function MultiScaleSpatialDataLoader:reset_batch()
  self.curr_batch = 1
  self.curr_epoch = self.curr_epoch + 1
  for i=1,self.num_classify do self.idx_by_type[i] = 1 end
end

