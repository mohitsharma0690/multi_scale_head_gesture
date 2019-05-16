require 'torch'
require 'hdf5'
require 'math'
require 'paths'
require 'image'

local utils = require 'util.utils'

local LatentVariableDataLoader = torch.class('LatentVariableDataLoader')

function LatentVariableDataLoader:__init(kwargs)
  self.train_h5file = utils.get_kwarg(kwargs, 'train_seq_h5',
    '../../openface_data/main_gest_by_file.h5')
  self.h5dir = utils.get_kwarg(kwargs, 'data_dir',
    '../../openface_data/face_gestures/dataseto_text')
  self.aug_gest_h5file = utils.get_kwarg(kwargs, 'aug_gests_h5',
    '../../openface_data/main_gest_by_file_aug_K_32.h5')

  self.use_openface_features = utils.get_kwarg(kwargs, 'use_openface_features', 1)
  self.num_scales = utils.get_kwarg(kwargs, 'num_scales')
  self.use_opt_flow = utils.get_kwarg(kwargs, 'use_opt_flow', 0)
  if self.use_opt_flow == 1 then
    self.opt_flow_dir = utils.get_kwarg(kwargs, 'opt_flow_dir')
  end
  self.finetune = utils.get_kwarg(kwargs, 'finetune')
  self.finetune_batch_size = utils.get_kwarg(kwargs, 'finetune_batch_size', 20)
  self.use_long_term_latent_variable = utils.get_kwarg(kwargs,
      'use_long_term_latent_variable')
  self.use_all_face_diff = utils.get_kwarg(kwargs, 'use_all_face_diff')

  self.user_classification = utils.get_kwarg(kwargs, 'train_user_classification')
  if self.user_classification == 1 then
    self.train_user_classify, self.test_user_classify = self:get_classes_for_user_classification()
  end

  self.use_label_correction = utils.get_kwarg(kwargs, 'use_label_correction')

  -- Latent variable dataset
  self.latent_aug_h5file = utils.get_kwarg(kwargs, 'latent_variable_aug_h5')
  self.latent_aug_h5 = hdf5.open(self.latent_aug_h5file, 'r')
  self.latent_num_features = utils.get_kwarg(kwargs, 'latent_num_features')
  self.latent_user_id = utils.get_kwarg(kwargs, 'latent_user_id')
  self.latent_model = utils.get_kwarg(kwargs, 'latent_model')
  if self.latent_model == 'supervised_context' then
    self.supervised_context_type = utils.get_kwarg(kwargs, 'supervised_context_type')
  end

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
  self.num_features = utils.get_kwarg(kwargs, 'num_features')
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
  self.latent_h5_by_file = self:load_latent_data()
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
    assert(self.gest_by_type ~= nil)
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
  if self.use_label_correction == 1 then
    self.train_files, self.train_file_size = self:load_correction_stats()
    local total_size = 0
    for k, v in pairs(self.train_file_size) do
      total_size = total_size + v[2]
    end
    self.num_train_frames = total_size
    kwargs.num_train_frames = total_size
    --print("Total frames per file:")
    --print(self.train_file_size)
  end
  

  self.shuffle_order = nil
  self.last_batch = {}

  print('Did read and load gestures')
end

function LatentVariableDataLoader:isTrainFile(f) 
  local fname = paths.basename(f)
  local fnum = string.match(fname, '%d+')
  fnum = tonumber(fnum)
  if fnum > 11 then return true
  else return false end
end

function LatentVariableDataLoader:load_mean_h5(h5_file_path)
  local f = hdf5.open(h5_file_path, 'r')
  local mean = f:read('/mean'):all()
  local std = f:read('/std'):all()
  f:close()
  return {mean=mean, std=std}
end

function LatentVariableDataLoader:get_label_corr_idx(f, t)
  if self.train_file_size[f] == nil then return 0 end
  local start_size, f_size = unpack(self.train_file_size[f])
  -- This needs to be corrected since even train_size * 5 should be done
  return 5*start_size + 5*(t - 1) + 1
end

function LatentVariableDataLoader:load_correction_stats()
  local train_files, train_file_size = {}, {}
  local h5_file = hdf5.open(self.train_h5file, 'r')
  local f_contents = h5_file:read("/"):all()
  f_contents = f_contents['train']

  local total_size = 0

  for f, v in pairs(f_contents) do
    local X, y = self.h5_by_file[f].X, self.h5_by_file[f].y
    table.insert(train_files, f)
    assert(X:size(1) == y:size(1))
    -- Store the length of each person's gesture. Each timestep is
    -- associated with a separate weight array for correction. The order
    -- of the weight array used in ReconsLayer will be based on the order
    -- of files in train_files array with its corresponding size in
    -- train_file_size
    train_file_size[f] = {total_size, X:size(1)}
    total_size = total_size + X:size(1)
  end
  h5_file:close()
  print("Total number of frames in all train files: "..total_size)
  return train_files, train_file_size
end

function LatentVariableDataLoader:load_zface_data()
  local h5_by_file = {}
  for fp in paths.files(self.zface_h5_dir) do
    if self:check_if_usable_h5(fp) then
      local f_path = self.zface_h5_dir .. '/' .. fp
      local hdf5_file = hdf5.open(f_path, 'r')
      local feats = hdf5_file:read('/features'):all()
      local X = torch.Tensor(feats:size()):copy(feats)
      X = utils.process_zface_data{X=X}
      -- Non-Maximal suppression of velocity vectors
      --X = utils.do_non_maximal_suppression_zface_data(X)
      h5_by_file[fp] = {X=X}
      hdf5_file:close()
    end
  end
  return h5_by_file
end

function LatentVariableDataLoader:load_cpm_data()
  local h5_by_file = {}
  for fp in paths.files(self.cpm_h5_dir) do
    if self:check_if_usable_h5(fp) then
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

function LatentVariableDataLoader:load_data()
  -- create two tables per training set (i.e. 2 each for train, val)
  -- 1) list of all h5 files in each set
  -- 2) the h5 files for each dataset loaded into memory
  local h5_by_file = {}
  for fp in paths.files(self.h5dir) do
    if self:check_if_usable_h5(fp) then
      print('Process file '..fp)
      local f_path = self.h5dir .. '/' .. fp
      local hdf5_file = hdf5.open(f_path, 'r')
      local annots = hdf5_file:read('/annotations'):all()
      local feats = hdf5_file:read('/features'):all()
      local X1 = torch.Tensor(feats:size()):copy(feats)
      local y1 = torch.Tensor(annots:size()):copy(annots)
      X1[torch.gt(X1, 10000)] = 10000
      X1[torch.lt(X1, -10000)] = -10000
      local X, y = utils.process_data{
        X=X1, 
        y=y1,
        mean=self.openface_mean_std.mean[fp],
        std=self.openface_mean_std.std[fp],
      }
      -- Non-Maximal suppression of velocity vectors
      --X = utils.do_non_maximal_suppression(X)
      h5_by_file[fp] = {X=X, y=y}
      hdf5_file:close()
    end
  end
  return h5_by_file
end

function LatentVariableDataLoader:load_latent_data()
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
      X1[torch.gt(X1, 10000)] = 10000
      local X, y = utils.process_latent_data{
        X=X1, 
        y=y1,
        mean=self.openface_mean_std.mean[fp][{{13, 13+2*68}}],
        std=self.openface_mean_std.std[fp][{{13, 13+2*68}}],
      }
      -- Non-Maximal suppression of velocity vectors
      --X = utils.do_non_maximal_suppression(X)
      h5_by_file[fp] = {X=X}
      hdf5_file:close()
    end
  end
  return h5_by_file
end

function LatentVariableDataLoader:get_aug_curr_win_inputs(args)
  local curr_file, curr_t = args.curr_file, args.curr_t
  local win_sizes, org_gest = args.win_sizes, args.org_gest
  local X_latent = args.X_latent

  local debug = false
  local inp, latent_inp = {}, {}
  local latent_inp_len = {32, 32, 32}
  for i=1, #win_sizes do
    local w_size = win_sizes[i]
    -- Get the (Num_Augmentations, Num_features, Win_size) values
    local temp_idx = curr_file .. '/' .. tostring(org_gest)
    local all_aug = self.aug_h5:read('/'..temp_idx..'/'..curr_t..'/'..w_size):all()
    local aug_idx = torch.random(1, all_aug:size(1))
    local gest = all_aug[aug_idx]:t()

    -- Add latent variable input
    local latent_gest = nil
    assert(self.latent_aug_h5 ~= nil)
    if self.latent_aug_h5 ~= nil then
      gest = utils.get_augmented_data_from_trimmed {
        X=gest,
        mean=self.openface_mean_std.mean[curr_file],
        std=self.openface_mean_std.std[curr_file]
      }
      --gest = utils.do_non_maximal_suppression(gest)
      if self.use_openface_features ~= 1 then gest = nil end
      if debug then print("Got openface aug gesture.") end

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
        if debug then print("Got CPM aug gesture.") end
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
        if debug then print("Got z-face aug gesture.") end
      end

      if self.latent_model == 'user_id' then
        latent_gest = self:get_one_hot_user_id_for_file(curr_file)
      elseif self.latent_model == 'pose_threshold_context' then
        local openface_X = self.h5_by_file[curr_file].X
        latent_gest = utils.get_pose_vel_category_context{
          X=openface_X,
          curr_t=curr_t,
          th_pose={0.3,0.4,0.6}
        }
      elseif self.latent_model == 'pose_vel_hist_context' then
        local openface_X = self.h5_by_file[curr_file].X
        -- latent_gest = utils.get_pose_vel_hist_context{
        latent_gest = utils.get_nosetip_vel_hist_context{
          X=openface_X,
          curr_t=curr_t,
        }
      elseif self.latent_model == 'supervised_context' then
        latent_gest = gest:clone()
      elseif self.latent_model == 'lstm_encoder' then
        if self.use_long_term_latent_variable == 1 then
          -- Use original input here directly
          latent_gest = {}
          assert(X_latent ~= nil)

          local w = latent_inp_len[i]
          table.insert(latent_inp, X_latent[{{curr_t-3*w+1,curr_t-2*w},{}}])
          table.insert(latent_inp, X_latent[{{curr_t-2*w+1,curr_t-w},{}}])
          table.insert(latent_inp, X_latent[{{curr_t-w+1,curr_t},{}}])
        else
          local all_aug = self.latent_aug_h5:read(
          '/'..temp_idx..'/'..curr_t..'/'..w_size):all()
          latent_gest = all_aug[aug_idx]:t()
          latent_gest = utils.get_latent_augmented_data_from_trimmed{
            X=latent_gest,
            mean=self.openface_mean_std.mean[curr_file][{{13, 13+2*68-1}}],
            std=self.openface_mean_std.std[curr_file][{{13, 13+2*68-1}}],
          }
          --latent_gest = latent_gest[{{w_size/2-4,w_size/2+5},{}}]
        end
      else
        assert(false)
      end
      if debug then print("Got latent inp aug gesture.") end
    end
    table.insert(inp, gest)
    if latent_gest ~= nil then table.insert(latent_inp, latent_gest) end
  end
  return inp, latent_inp
end

function LatentVariableDataLoader:get_opt_flow_file_name(t)
  -- If we have skipped frames in opt flow then we should do the corresponding
  -- math here.
  return string.format('flow_x_%05d.jpg', t) , string.format('flow_y_%05d.jpg', t)
end

function LatentVariableDataLoader:get_opt_flow_input(curr_file, curr_t)
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

--[[ 
Return the features for the current window. Only retuns the input features. The
output features for the window are obtained by the caller directly.  Gets
features for each particular latent model type. To add a new latent model you
should add the feature that you need for your latent model in the below code.
--]]
function LatentVariableDataLoader:get_curr_win_inputs(args)
  local X, y, curr_t, win_sizes = args.X, args.y, args.curr_t, args.win_sizes
  local curr_file = args.curr_file or nil
  local latent_X = args.X_latent
  local noise = args.noise or nil
  local cpm_X = args.cpm_X or nil
  local zface_X = args.zface_X or nil
  local inp, latent_inp = {}, {}
  local latent_inp_len = {32, 32}
  local debug = false

  for i=1, #win_sizes do
    local w_size = win_sizes[i]
    local X_openface, X_cpm, X_zface, X_final, X_latent
    if X~= nil and self.use_openface_features == 1 then
      X_openface = X[{{curr_t-w_size/2, curr_t-1+w_size/2},{}}]
      X_final = X_openface:clone()
      if debug then print("Did get openface features") end
    end
    if cpm_X ~= nil then
      X_cpm = cpm_X[{{curr_t-w_size/2, curr_t-1+w_size/2},{}}]
      if X_final == nil then X_final = X_cpm:clone()
      else X_final = torch.cat(X_final, X_cpm) end
      if debug then print("Did get CPM features") end
    end
    if zface_X ~= nil then 
      local X_zface = zface_X[{{curr_t-w_size/2, curr_t-1+w_size/2},{}}]
      if X_final == nil then X_final = X_zface:clone()
      else X_final = torch.cat(X_final, X_zface) end
      if debug then print("Did get zface features") end
    end
    if noise ~= nil and noise == 1 then
      X_final = X_final:index(
          1, torch.linspace(X_final:size(1),1,X_final:size(1)):long())
    end
    table.insert(inp, X_final)

    -- Add latent input which is a 10 frame window centered at t for now
    if self.latent_model == 'user_id' then
      assert(curr_file ~= nil)
      table.insert(latent_inp, self:get_one_hot_user_id_for_file(curr_file))
    elseif self.latent_model == 'pose_threshold_context' then
      local latent_gest = utils.get_pose_vel_category_context {
        X=args.X, curr_t=curr_t, th_pose={0.3,0.4,0.6} }

      table.insert(latent_inp, latent_gest)
    elseif self.latent_model == 'pose_vel_hist_context' then
      --local latent_gest = utils.get_pose_vel_category_context {
      local latent_gest = utils.get_nosetip_vel_hist_context {
        X=args.X, curr_t=curr_t}

      table.insert(latent_inp, latent_gest)

    elseif self.latent_model == 'lstm_encoder' then
      if self.use_long_term_latent_variable == 1 then
        -- We need to take samples of [curr_t-3*win_len,curr_t-2*win_len], 
        -- [curr_t-2*win_len,curr_t-win_len] and [curr_t-win_len, curr_t]
        local w = latent_inp_len[i]
        table.insert(latent_inp, latent_X[{{curr_t-3*w+1,curr_t-2*w},{}}])
        table.insert(latent_inp, latent_X[{{curr_t-2*w+1,curr_t-w},{}}])
        table.insert(latent_inp, latent_X[{{curr_t-w+1,curr_t},{}}])
      else
        X_latent = latent_X[{{curr_t-latent_inp_len[i]+1,curr_t},{}}]
        table.insert(latent_inp, latent_x)
      end
    elseif self.latent_model == 'supervised_context' then
      --TODO(Mohit): Take a longer window for supervised context
      table.insert(latent_inp, X_final:clone()) 
    else
      assert(false)
    end

    if debug then print("Did get latent input features") end
  end
  return inp, latent_inp
end

function LatentVariableDataLoader:convert_batch_stats_to_inp_format(batch_stats)
  local final_batch_stats = {{}, {}}
  for k,v in ipairs(batch_stats) do
    table.insert(final_batch_stats[1], {v[1], v[2]})
    table.insert(final_batch_stats[2], v[3])
  end
  assert(#final_batch_stats[1] == #batch_stats)
  return final_batch_stats
end

--[[
Convert the Tensors in the tables into one Tensor which contains all the batch
members.
batch_inp: Table with input tensors
batch_op: Table with output tensors or values
batch_opt_flow_inp: Table with optical flow input

Return: Input and output tensor obtained by concatenating all batch tensors.
]]
function LatentVariableDataLoader:convert_to_inp_format(
  batch_inp, batch_op, batch_opt_flow_inp) 

  local debug = false

  local final_X = {}
  for i=1,#self.curr_win_sizes do
    local x = {}
    table.insert(x, torch.zeros(
      #batch_inp, 1, self.curr_win_sizes[i], self.num_features))
    if self.latent_model == 'user_id' then
      table.insert(x, torch.zeros(#batch_inp, 14)) 
    elseif (self.latent_model == 'pose_threshold_context' or
            self.latent_model == 'pose_vel_hist_context') then
      table.insert(x, torch.zeros(#batch_inp, self.latent_num_features)) 
    elseif self.latent_model == 'supervised_context' then
      table.insert(x, torch.zeros(#batch_inp, 1, 32, self.num_features))
    elseif self.latent_model == 'lstm_encoder' then
      if self.use_long_term_latent_variable == 1 then
        table.insert(x, {})
        for j=1,3 do 
          table.insert(x[2], torch.zeros(#batch_inp, 32, self.latent_num_features))
        end
      else
        table.insert(x, torch.zeros(
          #batch_inp, 32, self.latent_num_features))
      end
    else
      assert(false)
    end
    table.insert(final_X, x)
  end

  if debug then print('Did initialize data for input data') end

  for i=1,#self.curr_win_sizes do
    for b=1,#batch_inp do
      if self.user_classification == 1 then
        assert(final_X[i][1][b][1]:size(1) == batch_inp[b][1]:size(1))
        assert(final_X[i][1][b][1]:size(2) == batch_inp[b][1]:size(2))
        final_X[i][1][b][1] = batch_inp[b][1] 
      else
        assert(final_X[i][1][b][1]:size(1) == batch_inp[b][1][i]:size(1))
        assert(final_X[i][1][b][1]:size(2) == batch_inp[b][1][i]:size(2))
        final_X[i][1][b][1] = batch_inp[b][1][i] 
        if self.latent_model == 'user_id' then
          final_X[i][2][b] = batch_inp[b][2][i]
        elseif self.latent_model == 'pose_threshold_context' or
              self.latent_model == 'pose_vel_hist_context' then
          final_X[i][2][b] = batch_inp[b][2][i]
        elseif self.latent_model == 'supervised_context' then
          assert(batch_inp[b][2][i] ~= nil)
          final_X[i][2][b][1] = batch_inp[b][2][i]
        elseif self.latent_model == 'lstm_encoder' then
          if self.use_long_term_latent_variable == 1 then
            for j=1,3 do
              final_X[i][2][j][b] = batch_inp[b][2][j]
            end
          else
            final_X[i][2][b] = batch_inp[b][2][i]
          end
        else
          assert(false)
        end
      end
    end
  end

  if debug then print('Did convert input data') end

  -- This wouldn't work with above logic but it's not being used right now.
  if batch_opt_flow_inp ~= nil and #batch_opt_flow_inp == #batch_inp then
    table.insert(final_X, torch.zeros(#batch_inp, 20, 224, 224))
    for i=1,#batch_opt_flow_inp do
      final_X[#final_X][i] = batch_opt_flow_inp[i]
    end
    if debug then print('Did convert optical flow input data') end
  end

  -- Get the output batch. If the elements of batch_op are also tables we create
  -- a table with that many tensors.
  local final_y
  if type(batch_op[1]) == type({}) then
    local num_op = #batch_op
    local num_op_tensors = #batch_op[1] 
    final_y = {}
    for i=1,num_op_tensors do 
      table.insert(final_y, torch.Tensor(num_op):zero())
      for j=1,num_op do
        final_y[i][j] = batch_op[j][i]
      end
    end
  else 
    final_y = torch.Tensor(batch_op)
  end
  return final_X, final_y
end

function LatentVariableDataLoader:get_batch_sample_ratio()
  num_samples = {}
  for i=1, self.num_classify do 
    table.insert(num_samples, math.floor(self.batch_size/self.num_classify))
  end
  return num_samples
end

function LatentVariableDataLoader:get_noise_mask_for_samples(samples)
  noise_mask = {}
  for i=1,#samples do
    for j=1,samples[i] do table.insert(noise_mask, 0) end
  end
  return noise_mask
end

function LatentVariableDataLoader:get_shuffle_order(gest_by_type)
  local shuffle_order = {}
  for i=1,#self.gest_by_type do
    table.insert(shuffle_order, torch.randperm(#self.gest_by_type[i]))
  end
  return shuffle_order
end

function LatentVariableDataLoader:getTotalTrainBatches()
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

function LatentVariableDataLoader:next_train_batch()
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
    self.last_batch = batch
    local X_batch, y_batch = self:get_features_for_batch(batch, noise_mask) 
    coroutine.yield(X_batch, y_batch)
  end
end

-- While we unroll the gesutres we should save some of them into the dictionary
-- with 'finetune' key and use them to finetune on. Or we could simply use the
-- train key for them and not have the train dataset.
-- We also need to save some gestures from every user into this finetune 
-- dataset above.
function LatentVariableDataLoader:next_finetune_batch()
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

function LatentVariableDataLoader:next_unsorted_train_batch()
  local batch_size = self.batch_size
  local num_train = 1
  for c=1, self.num_classify do 
    num_train = num_train + #self.gest_by_type[c]
  end
  batch = {}
  for c=1, self.num_classify do
    for i=1, #self.gest_by_type[c] do
      table.insert(batch, self.gest_by_type[c][i])
      if #batch == batch_size then
        local X_batch, y_batch = self:get_features_for_batch(batch, nil)
        coroutine.yield(X_batch, y_batch, batch)
        batch = {}
        collectgarbage()
      end
    end
  end
end

function LatentVariableDataLoader:next_val_batch()
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
-- gest_begin_time, gest_end_time). 
-- NOTE: This is different from the function below in the sense that we use
-- self.train_user_classify and self.test_user_classify to get the train and
-- test class names instead of picking them from the h5 file itself.
function LatentVariableDataLoader:get_per_user_gest_by_type(group_name)
  local gest_by_type = {}
  for i=1,self.num_classes do table.insert(gest_by_type, {}) end

  local file_names = self.train_user_classify
  if group_name == 'test' then file_names = self.test_user_classify end

  for f_name, f_id in pairs(file_names) do
    local num_frames = self.h5_by_file[f_name].y:size(1)
    local start_frame, end_frame = 200, num_frames - 200
    local num_frames_to_choose = 3000 
    local skip_frames = math.floor((end_frame - start_frame) / num_frames_to_choose)
    gest_by_type[f_id] = {}
    for i=1,num_frames_to_choose do
      local frame = start_frame+(i-1)*skip_frames
      table.insert(gest_by_type[f_id], {f_name, frame, frame})
    end
  end
  return gest_by_type
end

-- We return a table with keys from 1 to num_classes. Each key is mapped to a
-- table where each element of the table is another table of type (file_name,
-- gest_begin_time, gest_end_time)
function LatentVariableDataLoader:get_all_gest_by_type(group_name, finetune)
  if self.user_classification == 1 then
    return self:get_per_user_gest_by_type(group_name)
  end

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

function LatentVariableDataLoader:load_augmented_gestures(aug_h5)
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

function LatentVariableDataLoader:check_if_usable_h5(f)
  local ends_with = "static.mp4.txt.h5"
  return string.sub(f,-string.len(ends_with)) == ends_with
end

function LatentVariableDataLoader:final_class_for_gest(gest_type)
  if self.num_classes == self.num_classify then return gest_type end

  if gest_type == 1 then return 1
  elseif gest_type >= 2 and gest_type <= 6 then return 2
  elseif gest_type == 7 then return 3
  elseif gest_type == 8 or gest_type == 9 then return 4
  elseif gest_type == 10 or gest_type == 11 then return 5
  else assert(false) end
end

function LatentVariableDataLoader:group_gest_into_classes(gest_by_type)
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

function LatentVariableDataLoader:unroll_gest(
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

function LatentVariableDataLoader:unroll_gest_sequences(gest_by_type, 
  data_type)
  --[[ 
  gest_by_type: Table of the sequences of each gesture type. We unroll each
  gesture sequence to multiple values in the list where each value
  represents one frame of the sequence and we use that value as the center
  frame for one data point.
  ]]
  if self.user_classification == 1 then return gest_by_type end
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

function LatentVariableDataLoader:get_user_id_features_for_batch(batch, noise_mask)
  local max_win_size = math.max(unpack(self.curr_win_sizes))
  local batch_inp, batch_op = {}, {}
  for i, v in ipairs(batch) do

    local curr_file = v[1]
    local curr_t = v[2]
    local inp_win, latent_inp_win = nil, nil
    local cpm_X, zface_X = nil, nil
    if self.cpm_h5_by_file ~= nil then 
      cpm_X = self.cpm_h5_by_file[curr_file].X 
    end
    if self.zface_h5_by_file ~= nil then 
      zface_X = self.zface_h5_by_file[curr_file].X
    end
    assert(noise_mask == nil or noise_mask[i] == 0)
    inp_win, latent_inp_win = self:get_curr_win_inputs{
      X=self.h5_by_file[curr_file].X,
      y=self.h5_by_file[curr_file].y,
      X_latent=self.latent_h5_by_file[curr_file].X,
      curr_t=curr_t,
      win_sizes=self.curr_win_sizes,
      noise=noise,
      cpm_X=cpm_X,
      zface_X=zface_X,
      curr_file=curr_file,
    }
    assert(latent_inp_win == nil or #latent_inp_win == 0)

    local y = self.train_user_classify[curr_file] or self.test_user_classify[curr_file]
    assert(y~= nil and y >= 1 and y <= self.num_classify)

    table.insert(batch_inp, inp_win)
    table.insert(batch_op, y)
  end
  -- X_batch is a table of Tensors, y_batch is a Tensor
  local X_batch, y_batch = self:convert_to_inp_format(batch_inp, batch_op, nil)
  if torch.isTensor(y_batch) then
    y_batch = y_batch:clone()
    y_batch = y:type('torch.IntTensor')
  else
    y_batch = utils.convert_to_type(y_batch, 'torch.IntTensor')
  end
  return X_batch, y_batch
end

function LatentVariableDataLoader:get_features_for_batch(batch, noise_mask)
  local debug = false
  if self.user_classification == 1 then 
    return self:get_user_id_features_for_batch(batch, noise_mask)
  end
  -- Retun the actual features for the batch.
  local max_win_size = math.max(unpack(self.curr_win_sizes))
  local batch_inp, batch_op, batch_opt_flow_inp = {}, {}, {}
  local batch_stats = {}
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
    local inp_win, latent_inp_win = nil, nil
    local cpm_X, zface_X = nil, nil
    if self.cpm_h5_by_file ~= nil then 
      cpm_X = self.cpm_h5_by_file[curr_file].X 
    end
    if self.zface_h5_by_file ~= nil then 
      zface_X = self.zface_h5_by_file[curr_file].X
    end
    if self.use_label_correction == 1 then
      local label_corr_idx = self:get_label_corr_idx(curr_file, curr_t)
      table.insert(batch_stats, {curr_file, curr_t, label_corr_idx})
    end

    if noise == 0 or (noise==1 and org_gest <= 1) then 
      if debug then print("Will get real input") end
      inp_win, latent_inp_win = self:get_curr_win_inputs{
        X=self.h5_by_file[curr_file].X,
        y=self.h5_by_file[curr_file].y,
        X_latent=self.latent_h5_by_file[curr_file].X,
        curr_t=curr_t,
        win_sizes=self.curr_win_sizes,
        noise=noise,
        cpm_X=cpm_X,
        zface_X=zface_X,
        curr_file=curr_file,
      }
    else 
      if debug then print("Will get aug input") end
      inp_win, latent_inp_win = self:get_aug_curr_win_inputs{
        curr_file=curr_file,
        curr_t=curr_t,
        win_sizes=self.curr_win_sizes,
        org_gest=org_gest,
        X_latent=self.latent_h5_by_file[curr_file].X,
      }
    end
    if self.use_opt_flow == 1 then
      local opt_flow_inp = self:get_opt_flow_input(curr_file, curr_t) 
      table.insert(batch_opt_flow_inp, opt_flow_inp)
    end
    -- Note both inp_win, latent_inp_win are tables of length 16, 32 respectively
    table.insert(batch_inp, {inp_win, latent_inp_win})
    if self.latent_model == 'supervised_context' then
      local supervised_context_op = self:get_supervised_context_output(
          curr_file, curr_t, org_gest)
      table.insert(batch_op, {op_win, supervised_context_op})
    else
      table.insert(batch_op, op_win)
    end
  end

  -- X_batch is a table of Tensors, y_batch is a Tensor
  if debug then print("Convert to inp format") end
  local X_batch, y_batch = self:convert_to_inp_format(
      batch_inp, batch_op, batch_opt_flow_inp)
  if torch.isTensor(y_batch) then
    y_batch = y_batch:clone()
    y_batch = y_batch:type('torch.IntTensor')
  else
    for k,v in pairs(y_batch) do y_batch[k] = v:clone() end
    y_batch = utils.convert_to_type(y_batch, 'torch.IntTensor')
  end

  if self.use_label_correction == 1 then
    local final_batch_stats = self:convert_batch_stats_to_inp_format(batch_stats)
    return {X_batch, final_batch_stats}, y_batch
  else 
    return X_batch, y_batch
  end
end

function LatentVariableDataLoader:get_train_size()
  assert(#(self.gest_by_type) == self.num_classify)
  return #self.gest_by_type[2] * self.num_classify
end

function LatentVariableDataLoader:get_validation_size()
  return 0
end

function LatentVariableDataLoader:init_batch(num_epochs)
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

function LatentVariableDataLoader:reset_batch()
  self.curr_batch = 1
  self.curr_epoch = self.curr_epoch + 1
  for i=1,self.num_classify do self.idx_by_type[i] = 1 end
end

function LatentVariableDataLoader:get_classes_for_user_classification()
  local train = {
    ["007_static.mp4.txt.h5"]= 1, 
    ["008_static.mp4.txt.h5"]= 2,
    ["009_static.mp4.txt.h5"]= 3,
    ["010_static.mp4.txt.h5"]= 4,
    ["017_static.mp4.txt.h5"]= 5,
    ["019_static.mp4.txt.h5"]= 6,
    ["022_static.mp4.txt.h5"]= 7,
    ["025_static.mp4.txt.h5"]= 8,
    ["029_static.mp4.txt.h5"]= 9,
    ["030_static.mp4.txt.h5"]=10,
  } 
  local test = {
    ["041_static.mp4.txt.h5"]=1, 
    ["040_static.mp4.txt.h5"]=2,
    ["034_static.mp4.txt.h5"]=3,
    ["023_static.mp4.txt.h5"]=4,
    ["036_static.mp4.txt.h5"]=5, 
    ["038_static.mp4.txt.h5"]=6,
    ["031_static.mp4.txt.h5"]=7,
    ["032_static.mp4.txt.h5"]=8,
    ["042_static.mp4.txt.h5"]=9,
    ["043_static.mp4.txt.h5"]=10,
  }
  return train, test
end

function LatentVariableDataLoader:get_user_id_for_file(f)
  assert(f ~= nil or f ~= "")
  if string.find(f, "007") or string.find(f, "041") then return 1 end
  if string.find(f, "008") or string.find(f, "040") or string.find(f, "013") or string.find(f, "030") then 
    return 2
  end
  if string.find(f, "009") or string.find(f, "014") or string.find(f, "034") then
    return 3
  end
  if string.find(f, "010") or string.find(f, "012") or string.find(f, "023") then
    return 4
  end
  if string.find(f, "011") then return 5 end
  if string.find(f, "015") then return 6 end
  if string.find(f, "017") or string.find(f, "036") then return 7 end
  if string.find(f, "019") or string.find(f, "038") or string.find(f, "026") or string.find(f, "037") then 
    return 8
  end
  if string.find(f, "022") or string.find(f, "031") then return 9 end
  if string.find(f, "024") then return 10 end
  if string.find(f, "025") or string.find(f, "032") then return 11 end
  if string.find(f, "029") or string.find(f, "042") then return 12 end
  if string.find(f, "035") or string.find(f, "043") then return 13 end
  if string.find(f, "044") then return 14 end

  assert(false)
  return 0
end

function LatentVariableDataLoader:get_supervised_context_output(f, curr_t, gest)
  if self.supervised_context_type == 'user_id' then
    if string.find(f, "007") or string.find(f, "041") then return 1 end 
    if string.find(f, "008") or string.find(f, "040") then return 2 end 
    if string.find(f, "009") or string.find(f, "034") then return 3 end 
    if string.find(f, "017") or string.find(f, "036") then return 4 end 
    if string.find(f, "019") or string.find(f, "038") then return 5 end 
    if string.find(f, "022") or string.find(f, "031") then return 6 end 
    if string.find(f, "025") or string.find(f, "032") then return 7 end 
    if string.find(f, "035") or string.find(f, "043") then return 8 end 
  elseif self.supervised_context_type == 'user_id_original' then
    
  end


  assert(false)
  return 0
end

function LatentVariableDataLoader:get_one_hot_user_id_for_file(f)
  local user_id = self:get_user_id_for_file(f)
  local one_hot_user_id = torch.Tensor(14):zero()  
  one_hot_user_id[user_id] = 1
  return one_hot_user_id
end

