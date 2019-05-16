local cjson = require 'cjson'

require 'hdf5' require 'math'
require 'paths'

local utils = {}

local CPM_INDEX_TO_JOINT_TYPE = {
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
   --18: "Bkg",  # CPM output doesn't seem to emit this value
}

local OPENFACE_LANDMARKS_TO_USE = {
    28, 28 + 68, -- forehead
    34, 34 + 68, -- nosetip
    2,   2 + 68, -- left side of face
    4,   4 + 68,
    8,   8 + 68, -- bottom (right)
    10, 10 + 68,
    14, 14 + 68, -- top
    16, 16 + 68
}
-- In Lua index starts from 1 therefore the first landmark index will be 13.
-- 13 = 12 + 1 and thus for the above landmarks we need to add 12 as below.
for i=1,#OPENFACE_LANDMARKS_TO_USE do
  OPENFACE_LANDMARKS_TO_USE[i] = OPENFACE_LANDMARKS_TO_USE[i] + 12
end

-- Landmarks to use for face diff vectors
local OPENFACE_ALL_LANDMARKS = {}
-- 1 to 17 are all the facial landmark on the outer side of face
for i=1,17 do table.insert(OPENFACE_ALL_LANDMARKS, i+12); table.insert(OPENFACE_ALL_LANDMARKS, i+12+68) end
table.insert(OPENFACE_ALL_LANDMARKS, 28+12); table.insert(OPENFACE_ALL_LANDMARKS, 28+12+68)
local eye_landmarks = {37, 40, 43, 46}
for i=1,#eye_landmarks do
  table.insert(OPENFACE_ALL_LANDMARKS, eye_landmarks[i]+12)
  table.insert(OPENFACE_ALL_LANDMARKS, eye_landmarks[i]+12+68)
end

-- TODO(Mohit): Change these appropriately
local OPT_FLOW_CROP = {}
OPT_FLOW_CROP["007_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["008_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["009_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["010_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["011_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["012_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["013_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["014_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["015_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["017_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["019_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["022_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["023_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["024_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["025_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["026_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["029_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["030_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["031_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["032_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["034_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["035_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["036_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["037_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["038_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["040_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["041_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["042_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["043_static.mp4.txt.h5"]={350,650,900,1200}
OPT_FLOW_CROP["044_static.mp4.txt.h5"]={350,650,900,1200}

local FINETUNE_FRAMES = {}
FINETUNE_FRAMES["037_static.mp4.txt.h5"]={5,2,2,2,2,2,2,2,2,2,2}
FINETUNE_FRAMES["038_static.mp4.txt.h5"]={5,2,2,2,2,2,2,2,2,2,2}
FINETUNE_FRAMES["040_static.mp4.txt.h5"]={5,2,2,2,2,2,2,2,2,2,2}
FINETUNE_FRAMES["041_static.mp4.txt.h5"]={5,2,2,2,2,2,2,2,2,2,2}
FINETUNE_FRAMES["042_static.mp4.txt.h5"]={5,2,2,2,2,2,2,2,2,2,2}
FINETUNE_FRAMES["043_static.mp4.txt.h5"]={5,2,2,2,2,2,2,2,2,2,2}
FINETUNE_FRAMES["044_static.mp4.txt.h5"]={5,2,2,2,2,2,2,2,2,2,2}


utils.CPM_INDEX_TO_JOINT_TYPE = CPM_INDEX_TO_JOINT_TYPE
utils.OPENFACE_LANDMARKS_TO_USE = OPENFACE_LANDMARKS_TO_USE
utils.OPENFACE_ALL_LANDMARKS = OPENFACE_ALL_LANDMARKS
utils.OPT_FLOW_IMAGE_CROP = OPT_FLOW_CROP
utils.FINETUNE_FRAMES_BY_FILE = FINETUNE_FRAMES

POSE_VEL_DIST={}

function utils.post_process_data(args)
  local X, y = args.X, args.y
  local curr_file = args.file

  local X_pose = X[{{}, {7, 12}}]
  local X_size = X:size()
  local X_pose_diff = torch.Tensor(X_pose:size(1), 3):zero()
  X_pose_diff[{{2,-1},{}}] = X[{{2,-1}, {10,12}}] - X[{{1,-2},{10,12}}]

  local x_dist,y_dist,z_dist = {0,0,0,0,0}, {0,0,0,0,0}, {0,0,0,0,0}
  local th_x, th_y, th_z = 0.4, 0.3, 0.5
  for i=200,X_pose_diff:size(1)-100 do
    if X_pose_diff[i][1] < 0.005 then x_dist[1] = x_dist[1] + 1
    elseif X_pose_diff[i][1] < 0.01 then x_dist[2] = x_dist[2] + 1
    elseif X_pose_diff[i][1] < 0.04 then x_dist[3] = x_dist[3] + 1
    elseif X_pose_diff[i][1] < 0.07 then x_dist[4] = x_dist[4] + 1
    else x_dist[5] = x_dist[5] + 1 end

    if X_pose_diff[i][2] < 0.005 then y_dist[1] = y_dist[1] + 1
    elseif X_pose_diff[i][2] < 0.01 then y_dist[2] = y_dist[2] + 1
    elseif X_pose_diff[i][2] < 0.04 then y_dist[3] = y_dist[3] + 1
    elseif X_pose_diff[i][2] < 0.07 then y_dist[4] = y_dist[4] + 1
    else y_dist[5] = y_dist[5] + 1 end

    if X_pose_diff[i][3] < 0.005 then z_dist[1] = z_dist[1] + 1
    elseif X_pose_diff[i][3] < 0.01 then z_dist[2] = z_dist[2] + 1
    elseif X_pose_diff[i][3] < 0.04 then z_dist[3] = z_dist[3] + 1
    elseif X_pose_diff[i][3] < 0.07 then z_dist[4] = z_dist[4] + 1
    else z_dist[5] = z_dist[5] + 1 end
  end
  print("============================")
  print("============================")
  print("FILE: "..curr_file)
  print("pose_rot stats: ")
  print(x_dist)
  print(y_dist)
  print(z_dist)
  print("============================")
  print("============================")
end

function utils.get_kwarg(kwargs, name, default_val)
  if kwargs == nil then return default_val end
  if kwargs[name] == nil and default_val == nil then
    assert(false, string.format('"%s" expected and not given', name))
  elseif kwargs[name] == nil then
    return default_val
  else
      return kwargs[name]
  end
end

function utils.read_json(path)
  local f = io.open(path, 'r')
  local s = f:read('*all')
  f:close()
  return cjson.decode(s)
end


function utils.write_json(path, obj)
  local s = cjson.encode(obj)
  local f = io.open(path, 'w')
  f:write(s)
  f:close()
end

function utils.get_cpm_joint_type_to_idx(joint)
  for k, v in ipairs(utils.CPM_INDEX_TO_JOINT_TYPE) do
    if v == joint then return k end
  end
  assert(false)
  return -1
end

function utils.get_cpm_joint_type_to_h5_idx(joint)
  -- Since the first index in lua starts with 1
  return 3 * (utils.get_cpm_joint_type_to_idx(joint) - 1) + 1
end

function utils.get_shoulder_neck_nose_angle(args)
  local X = args.X
  local nose_idx,neck_idx = args.nose_idx, args.neck_idx
  local i1, i2 = args.left_shoulder_idx, args.right_shoulder_idx

  local dist_left_shoulder_neck = torch.csub(
      X[{{},{neck_idx,neck_idx+1}}], X[{{},{i1,i1+1}}])
  dist_left_shoulder_neck:pow(2)
  dist_left_shoulder_neck = torch.add(
      dist_left_shoulder_neck[{{},{1}}], dist_left_shoulder_neck[{{},{2}}])
  dist_left_shoulder_neck:pow(0.5)

  local dist_right_shoulder_neck = torch.csub(
      X[{{},{neck_idx,neck_idx+1}}], X[{{},{i2,i2+1}}])
  dist_right_shoulder_neck:pow(2)
  dist_right_shoulder_neck = torch.add(
      dist_right_shoulder_neck[{{},{1}}], dist_right_shoulder_neck[{{},{2}}])
  dist_right_shoulder_neck:pow(0.5)

  local dist_nose_neck = torch.csub(
      X[{{},{neck_idx,neck_idx+1}}], X[{{},{nose_idx,nose_idx+1}}])
  dist_nose_neck:pow(2)
  dist_nose_neck = torch.add(
      dist_nose_neck[{{},{1}}], dist_nose_neck[{{},{2}}])
  dist_nose_neck:pow(0.5)

  local dist_left_shoulder_nose = torch.csub(
      X[{{},{nose_idx,nose_idx+1}}], X[{{},{i1,i1+1}}])
  dist_left_shoulder_nose:pow(2)
  dist_left_shoulder_nose= torch.add(
      dist_left_shoulder_nose[{{},{1}}], dist_left_shoulder_nose[{{},{2}}])
  dist_left_shoulder_nose:pow(0.5)

  local dist_right_shoulder_nose = torch.csub(
      X[{{},{nose_idx,nose_idx+1}}], X[{{},{i2,i2+1}}])
  dist_right_shoulder_nose:pow(2)
  dist_right_shoulder_nose = torch.add(
      dist_right_shoulder_nose[{{},{1}}], dist_right_shoulder_nose[{{},{2}}])
  dist_right_shoulder_nose:pow(0.5)

  -- Use the cosine formula
  local left_angle = torch.pow(dist_left_shoulder_neck, 2)
  left_angle:add(torch.pow(dist_nose_neck, 2))
  left_angle:csub(torch.pow(dist_left_shoulder_nose, 2))
  local den = torch.cmul(dist_left_shoulder_neck, dist_nose_neck)
  den:mul(2)
  left_angle:cdiv(den)
  -- Note the result is in radians we should convert it into degrees and use
  -- angular velocity.
  left_angle:acos()
  left_angle:mul(180/math.pi)
  left_angle[{{2,-1},{}}] = left_angle[{{2,-1},{}}] - left_angle[{{1,-2},{}}]
  left_angle[{{1},{1}}] = 0

  local right_angle = torch.pow(dist_right_shoulder_neck, 2)
  right_angle:add(torch.pow(dist_nose_neck, 2))
  right_angle:csub(torch.pow(dist_right_shoulder_nose, 2))
  den = torch.cmul(dist_right_shoulder_neck, dist_nose_neck)
  den:mul(2)
  right_angle:cdiv(den)
  right_angle:acos()
  right_angle:mul(180/math.pi)
  right_angle[{{2,-1},{}}] = right_angle[{{2,-1},{}}] - right_angle[{{1,-2},{}}]
  right_angle[{{1},{1}}] = 0

  left_angle[left_angle:ne(left_angle)] = 0
  right_angle[right_angle:ne(right_angle)] = 0

  return left_angle, right_angle
end

function utils.get_augmented_cpm_data_from_trimmed(args)
  args.aug = true
  return utils.__process_cpm_data(args)
end

function utils.process_cpm_data(args)
  return utils.__process_cpm_data(args)
end

-- Process data from convolution pose machines.
function utils.__process_cpm_data(args)

  local X = args.X
  local mean, std = args.mean, args.std
  local aug = args.aug or false

  -- We should calculate distances etc. here
  local nose_idx = utils.get_cpm_joint_type_to_h5_idx('Nose')
  local neck_idx = utils.get_cpm_joint_type_to_h5_idx('Neck')
  local l_shldr_idx = utils.get_cpm_joint_type_to_h5_idx('LShoulder')
  local r_shldr_idx = utils.get_cpm_joint_type_to_h5_idx('RShoulder')
  if aug then
    nose_idx, neck_idx, l_shldr_idx, r_shldr_idx = 1, 3, 7, 5
  end
  -- Use the nose velocity, neck velocity and nose-neck velocity as input
  -- features for now.
  local nose_neck_diff = X[{{},{nose_idx,nose_idx+1}}] - X[{{},{neck_idx,neck_idx+1}}]
  -- TODO(Mohit): We need to precalculate the mean/std for these distance
  -- vectors or maybe we can just use the mean values directly.
  if mean ~= nil then
    for i=1,nose_neck_diff:size(2) do
      nose_neck_diff[{{},{i}}] = (nose_neck_diff[{{},{i}}] - mean[i]) / std[i]
    end
  end

  local nose_vel = torch.Tensor(X:size(1), 2):zero()
  nose_vel[{{2,-1},{}}] = X[{{2,-1},{nose_idx,nose_idx+1}}] - X[{{1,-2},{nose_idx,nose_idx+1}}]
  local neck_vel = torch.Tensor(X:size(1), 2):zero()
  neck_vel[{{2,-1},{}}] = X[{{2,-1},{neck_idx,neck_idx+1}}] - X[{{1,-2},{neck_idx,neck_idx+1}}]

  local l_shldr_vel = torch.Tensor(X:size(1), 2):zero()
  local i1, i2 = l_shldr_idx, r_shldr_idx
  l_shldr_vel[{{2,-1},{}}] = X[{{2,-1},{i1,i1+1}}] - X[{{1,-2},{i1,i1+1}}]

  local r_shldr_vel = torch.Tensor(X:size(1), 2):zero()
  r_shldr_vel[{{2,-1},{}}] = X[{{2,-1},{i2,i2+1}}] - X[{{1,-2},{i2,i2+1}}]

  -- Use cosine formula to calculate angle between shoulder, nose, neck joints
  local left_angle_vel, right_angle_vel = utils.get_shoulder_neck_nose_angle{
    X=X,
    nose_idx=nose_idx,
    neck_idx=neck_idx,
    left_shoulder_idx=l_shldr_idx,
    right_shoulder_idx=r_shldr_idx,
  }

  --local X_new = torch.cat({nose_vel, neck_vel, nose_neck_diff, l_shldr_vel,
  --  r_shldr_vel, left_angle_vel, right_angle_vel})
  local X_new = torch.cat({nose_vel, neck_vel, l_shldr_vel, r_shldr_vel,
    left_angle_vel, right_angle_vel})

  local nan_mask = X_new:ne(X_new)
  assert(torch.sum(nan_mask) == 0)

  -- local X_new = torch.cat({nose_vel, neck_vel, nose_neck_diff})
  return X_new
end

-- Do non maximal suppression for Z-face data.
function utils.do_non_maximal_suppression_zface_data(X)
  -- TODO(Mohit): Maybe add non-maximal suppression for zface data
  return X
end

-- Do non maximal suppression for CPM data.
function utils.do_non_maximal_suppression_cpm_data(X)
  -- We use 0.2 for landmark velocity as well
  X = utils.non_maximal_suppression(X, 1, 4, 0.2)
  -- suppress noise in difference between neck and nose
  X = utils.non_maximal_suppression(X, 5, 6, 0.1)
  return X
end

function utils.get_augmented_zface_data_from_trimmed(args)
  args.aug = true
  return utils.__process_zface_data(args)
end

-- Process data from Z-face
function utils.process_zface_data(args)
  return utils.__process_zface_data(args)
end

function utils.__process_zface_data(args)
  local X = args.X
  local mean, std = args.mean, args.std

  local transition_vel = torch.Tensor(X:size(1), 2):zero()
  transition_vel[{{2,-1},{}}] = X[{{2,-1},{2,3}}] - X[{{1,-2},{2,3}}]
  local angular_vel = torch.Tensor(X:size(1), 3):zero()
  angular_vel[{{2,-1},{}}] = X[{{2,-1},{4,6}}] - X[{{1,-2},{4,6}}]

  --[[
  local X_new = torch.cat({X[{{},{1}}], transition_vel, angular_vel})
  local X_new = torch.cat({X[{{},{1}}], X[{{},{4,6}}],
      transition_vel, angular_vel})
  ]]

  -- Pure 6DOF z-normalized
  local X_new = X:clone()
  X_new = utils.normalize_data_with{X=X_new, mean=mean, std=std}

  -- Check for Nan's
  if torch.sum(X_new:ne(X_new)) ~= 0 then
    print(X_new)
    assert(false)
  end

  return X_new
end


function utils.zero_one_normalize(X, max, min)
  assert(X:dim() == 1, "Incorrect number of dimensions")
  return (X[{}] - min) / max - min
end

function utils.z_normalize(X, mean, std)
  assert(X:dim() == 1, "Incorrect number of dimensions")
  assert(std >= 0.0001, "0 std will lead to NaN")
  return (X[{}] - mean) / std
end

function utils.normalize_data_with(args)
  local mean, std = args.mean, args.std
  local max, min = args.max, args.min
  local X = args.X
  local signals, end_signals = args.signals, X:size(2)
  if signals ~= nil then end_signals = #signals end
  for i=1,end_signals do
    local j = i
    if signals ~= nil then j = signals[i] end
    if mean ~= nil and std ~= nil then
      X[{{},j}] = utils.z_normalize(X[{{},j}], mean[j], std[j])
    elseif max ~= nil and min ~= nil then
      X[{{},j}] = utils.z_normalize(X[{{},j}], max[j], min[j])
    end
  end
  return X
end

function utils.normalize_data(X, y, signals, zero_to_one_normalization)
  local mean = {}
  local std = {}
  local num_signals = X:size(2)
  local norm_type = 'z-norm'
  if zero_to_one_normalization ~= nil and zero_to_one_normalization then
    norm_type = '0-1'
  end
  if signals == nil then
    for i = 1,X:size(2) do
      -- normalize each channel globally
      if norm_type == 'z-norm' then
        mean[i] = X[{{}, i}]:mean()
        std[i] = X[{{}, i}]:std()
        X[{{}, i}]:add(-mean[i])
        X[{{}, i}]:div(std[i])
      else
        local min = X[{{},i}]:min()
        local max = X[{{},i}]:max()
        X[{{},i}]:add(-min)
        if max - min ~= 0 then
          X[{{}, i}]:div(max - min)
        end
        assert(X[{{}, i}]:max() < 1.1)
      end
    end
  else
    local T = X:size(1)
    for i=1,#signals do
      local idx = signals[i]
      local max_signal = X[{{}, idx}]:max()
      local min_signal = X[{{}, idx}]:min()

      -- Assume initial 200 frames are just always wrong
      -- (camera setup etc. in videos)
      local j = 200
      while torch.abs(X[j][idx]) < 0.0001 do j = j + 1 end

      if norm_type == '0-1' then
        local min_signal = X[{{j, T}, idx}]:min()
        X[{{j, T}, idx}] = (X[{{j, T}, idx}] - min_signal) / (max_signal - min_signal)
      else
        mean[i] = X[{{j,T}, idx}]:mean()
        std[i] = X[{{j,T}, idx}]:std()
        X[{{}, idx}]:add(-mean[i])
        X[{{}, idx}]:div(std[i])
      end
    end
  end
  return X, y
end

function utils.do_non_maximal_suppression(X)
  -- Non-Maximal suppression of velocity vectors
  assert(false)
  X = utils.non_maximal_suppression(X, 35, 50, 0.5)
  X = utils.non_maximal_suppression(X, 16, 18, (0.01*180)/math.pi)
  -- X = utils.non_maximal_suppression(X, 57, 57, 0.8)  -- 0.8 degree
  return X
end

function utils.non_maximal_suppression(X, supp_idx_st, supp_idx_end, supp_th)
  for i=supp_idx_st,supp_idx_end do
    local val = X[{{},{i}}]
    local idx = torch.cmul(val:lt(supp_th), val:gt(-supp_th))
    val[idx] = 0
    X[{{},{i}}] = val
  end
  return X
end

-- This function is used to normalize values explicitly. Hence whenever
-- changing the type of features being used for classification we should change
-- this method to make sure that we aren't calculating wrong mean or variance
-- for the input.
function utils.normalize_openface_feats(X, mean, std, landmarks)
  -- Normalize pose values
  for i=1,3 do
    assert(std[i+6] >= 0.0001)
    X[{{},{i}}] = (X[{{},{i}}] - mean[i+6]) / std[i+6]
  end

  -- Normalize landmarks
  --for i=13,28 do
  --  assert(std[landmarks[i-12]] >= 0.0001)
  --  X[{{},{i}}] = (X[{{},{i}}] - mean[landmarks[i-12]]) / std[landmarks[i-12]]
  --end
  return X
end

function utils.get_face_difference(a, b)
  local face_vec = a - b
  face_vec[{{2,-1},{}}] = face_vec[{{2,-1},{}}] - face_vec[{{1,-2},{}}]
  for i=1,face_vec:size(2) do face_vec[{{1},{i}}] = 0 end

  return face_vec
end

function utils.get_latent_augmented_data_from_trimmed(args)
  local X = args.X
  local mean, std = args.mean, args.std
  local norm_type = args.norm_type
  local normalize_signals = args.normalize_signals

  assert(norm_type == nil)
  assert(normalize_signals == nil)

  local X_new = X[{{},{13, 13+135}}]
  X_new = utils.normalize_data_with{
    X=X_new,
    mean=mean,
    std=std,
  }
  return X_new
end

-- Return the facial differene vectors based on landmark_idx. The nose_idx is
-- the index of nosetip and is taken as the reference point.
function utils.get_all_face_diff(X, landmark_x_idx, landmark_y_idx, nose_x_idx, nose_y_idx)
  local face_diff = torch.Tensor(X:size(1), 2*#landmark_x_idx):zero()
  for i=1,#landmark_x_idx do
    local x_idx, y_idx = landmark_x_idx[i], landmark_y_idx[i]
    face_diff[{{},{2*i-1}}] = utils.get_face_difference(
        X[{{},{x_idx}}], X[{{},{nose_x_idx}}])
    face_diff[{{},{2*i}}] = utils.get_face_difference(
        X[{{},{y_idx}}], X[{{},{nose_y_idx}}])
  end
  return face_diff
end

function utils.get_augmented_data_from_trimmed(args)
  -- The below landmark indexes only work for trimmed augmentations.
  args.trimmed_aug = G_global_opts['openface_trimmed_aug'] == 1
  if args.trimmed_aug then
    args.face_landmarks = {
      1+12, 3+12,
      1+1+12, 3+1+12,
      5+12, 3+12,
      5+1+12, 3+1+12,
      13+12, 3+12,
      13+1+12, 3+1+12,
    }
  else
    args.face_landmarks = {
      28+12, 34+12,
      28+12+68, 34+12+68,
      2+12, 34+12,
      2+12+68, 34+12+68,
      14+12, 34+12,
      14+12+68, 34+12+68,
    }
  end
  args.aug = true
  return utils.__process_data(args)
end

function utils.process_data(args)
  args.face_landmarks = {
    28+12, 34+12,
    28+12+68, 34+12+68,
    2+12, 34+12,
    2+12+68, 34+12+68,
    14+12, 34+12,
    14+12+68, 34+12+68,
  }
  return utils.__process_data(args)
end

function utils.__process_data(args)
  local X, y = args.X, args.y
  local mean, std = args.mean, args.std
  local norm_type = args.norm_type
  local normalize_signals, fl = args.signals, args.face_landmarks
  local aug, trimmed_aug = args.aug or false, args.trimmed_aug or false

  -- For augmented gestures in old cases we only augment gaze(6), pose(6) and
  -- facial landmarks(16)
  assert(X:size(2) > 53 or X:size(2) == 28)
  if X:size(2) > 53 then X = X[{{}, {1, 148}}] end

  -- Should smooth data
  -- In numpy :12 will not get us the 12'th row but in Lua it will. Although it
  -- is compensated by the fact that in Lua we start from 1 while in python from
  -- 0.
  -- Not including gaze
  local X_pose = X[{{}, {7, 12}}]
  local X_size = X:size()
  local X_pose_diff = torch.Tensor(X_pose:size(1), 6):zero()
  X_pose_diff[{{2,-1},{}}] = X[{{2,-1}, {7,12}}] - X[{{1,-2},{7,12}}]
  -- Convert into degrees since the difference in radians could be very small.
  X_pose_diff[{{},{4,6}}] = X_pose_diff[{{},{4,6}}] * 180.0 / math.pi

  local landmarks = utils.OPENFACE_LANDMARKS_TO_USE
  -- Trimmed augmentations have gaze(6), pose(6), landmarks(12)
  if trimmed_aug then
    landmarks = {}
    for i=13,28 do table.insert(landmarks, i) end
  end
  local X_landmarks = torch.Tensor(X:size(1), #landmarks):zero()

  for i=1,#landmarks do
    X_landmarks[{{},i}] = X[{{},landmarks[i]}]
  end
  local X_landmarks_diff = torch.Tensor(X_landmarks:size()):zero()
  X_landmarks_diff[{{2,-1},{}}] = X_landmarks[{{2,-1},{}}] - X_landmarks[{{1,-2},{}}]
  -- Reject outliers from data
  --[[
  for i=1,X_landmarks_diff:size(2) do 
    X_landmarks_diff[{{},{i}}] = utils.reject_outliers{
      data=X_landmarks_diff[{{},{i}}],
      max_median_scale=20.0
    }
  end
  ]]
  
  --[[
  local X_landmarks_acc = torch.Tensor(X_landmarks:size()):zero()
  X_landmarks_acc[{{2,-1},{}}] = X_landmarks_diff[{{2,-1},{}}] - X_landmarks_diff[{{1,-2},{}}]
  ]]

  local face_diff_vec
  if G_global_opts['use_all_face_diff'] == 1 then
    local face_diff_landmarks = utils.OPENFACE_ALL_LANDMARKS
    local face_diff_landmarks_x, face_diff_landmarks_y = {}, {}
    for i=1,#face_diff_landmarks,2 do
      table.insert(face_diff_landmarks_x, i)
      table.insert(face_diff_landmarks_y, i+1)
    end
    face_diff_vec = utils.get_all_face_diff(
        X, face_diff_landmarks_x, face_diff_landmarks_y, 34, 34+68)
  else
    face_diff_vec = utils.get_all_face_diff(
        X, {fl[1],fl[5],fl[9]}, {fl[3],fl[7],fl[11]}, fl[2], fl[4])

  end

  -- difference between left,right side of face
  --[[
  local X_face_vec_left =  X_landmarks[{{},{7,8}}] - X_landmarks[{{},{5,6}}]
  local X_face_vec_right =  X_landmarks[{{},{13,14}}] - X_landmarks[{{},{15,16}}]

  local face_vec_1_angle = torch.abs(X_face_vec_1_x)
  face_vec_1_angle[face_vec_1_angle:eq(0)] = 0.00001  -- Avoid 0 division.
  face_vec_1_angle = torch.atan(
      torch.cdiv(torch.abs(X_face_vec_1_y), face_vec_1_angle))
  face_vec_1_angle = face_vec_1_angle * 180.0 / math.pi
  face_vec_1_angle[{{2,-1},{}}] = face_vec_1_angle[{{2,-1},{}}] - face_vec_1_angle[{{1,-2},{}}]
  ]]

  -- X = torch.cat({X_pose, X_pose_diff, X_landmarks, X_landmarks_diff,
  --    face_diff_vec})
  X = torch.cat({X_pose, X_pose_diff, X_landmarks_diff})

  if mean == nil then
    if norm_type == '0-1' then
      X, y = utils.normalize_data(X, y, normalize_signals, true)
    else
      X, y = utils.normalize_data(X, y, normalize_signals)
    end
  else
    X = utils.normalize_openface_feats(
      X, mean, std, utils.OPENFACE_LANDMARKS_TO_USE)
  end
  return X, y
end

-- Create a binary vector from curr_t to 128 frames in the past based on
-- threshold
function utils.get_binary_context_vec(args)
  local X, curr_t = args.X, args.curr_t
  local past_frames = args.past_frames or 100
  local th_pose_x, th_pose_y, th_pose_z = unpack(args.th_pose)

  local X_pose = X[{{}, {7, 12}}]
  local X_size = X:size()
  local X_pose_diff = torch.Tensor(X_pose:size(1), 3):zero()
  X_pose_diff[{{2,-1},{}}] = X[{{2,-1}, {10,12}}] - X[{{1,-2},{10,12}}]

  local context_vec = torch.Tensor(past_frames, 3):zero()
  local j = 1
  for i=curr_t-1,curr_t-past_frames, -1 do
    context_vec[j][1] = math.abs(X_pose_diff[i][1]) > th_pose_x and 1 or 0
    context_vec[j][2] = math.abs(X_pose_diff[i][2]) > th_pose_y and 1 or 0
    context_vec[j][3] = math.abs(X_pose_diff[i][3]) > th_pose_z and 1 or 0
    j = j + 1
  end
  -- TODO(Mohit): Reverse context_vec?
  assert(j == context_vec:size(1)+1)
  return context_vec:view(-1)
end

-- Returns the index where val fits in the table `t`. If val < t[1] return 0
function utils.get_index_in_sorted_array(t, val)
  for i=1,#t do
    if val <= t[i] then return i end
  end
  return #t + 1
end

function utils.get_nosetip_vel_hist_context(args)
  local X, curr_t = args.X, args.curr_t
  local window_len, num_windows = 16, 8  -- finetune this (visualize)??
  assert(X:size(2) == 28)
  -- This is assuming we have landmark velocity.
  local X_nosetip = X[{{}, {15, 16}}]

  local bins_x = torch.totable(torch.linspace(0, 2, 5))
  local bins_y = torch.totable(torch.linspace(0, 2, 5))
  local context_x, context_y = {}, {}
  for i=1,#bins_x+1 do table.insert(context_x, 0); table.insert(context_y, 0) end

  for i=1, num_windows do
    local win_st = curr_t - i * window_len
    local win_end = win_st + window_len

    if (win_st > 0 and win_end <= X_nosetip:size(1)) then

      local mean_x = torch.mean(torch.abs(X_nosetip[{{win_st, win_end},{1}}]))
      local mean_y = torch.mean(torch.abs(X_nosetip[{{win_st, win_end},{2}}]))

      local function _get_context(context, bins, val)
        for i=1,#context do 
          if i < #bins and val >= bins[i] and val < bins[i+1] then
            context[i] = context[i] + 1
            break
          elseif i == #bins then
            context[#context] = context[#context] + 1
          end
        end
        return context
      end

      context_x = _get_context(context_x, bins_x, mean_x)
      context_y = _get_context(context_y, bins_y, mean_y)
    end
  end

  context_x = torch.Tensor(context_x)
  context_x = context_x:div(torch.sum(context_x))
  context_y = torch.Tensor(context_y)
  context_y = context_y:div(torch.sum(context_y))
  local final_hist = torch.cat({context_x, context_y})
  return final_hist
end

function utils.get_pose_vel_hist_context(args)
  local X, curr_t = args.X, args.curr_t
  local past_frames = 128

  local X_pose = X[{{}, {7, 12}}]
  local X_size = X:size()
  local X_pose_diff = torch.Tensor(X_pose:size(1), 3):zero()
  local bins_x = {0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0}
  local bins_y = {0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0}
  local bins_z = {0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0}

  X_pose_diff[{{2,-1},{}}] = X[{{2,-1}, {7,9}}] - X[{{1,-2},{7,9}}]

  local idx = 1
  local hist_x, hist_y, hist_z = {}, {}, {}
  for i=1,#bins_x+1 do 
    table.insert(hist_x, 0)
    table.insert(hist_y, 0)
    table.insert(hist_z, 0)
  end

  for i=curr_t-1,math.max(curr_t-past_frames, 1),-1 do
    local x_idx = utils.get_index_in_sorted_array(bins_x, X_pose_diff[i][1])
    hist_x[x_idx] = hist_x[x_idx] + 1
    local y_idx = utils.get_index_in_sorted_array(bins_y, X_pose_diff[i][2])
    hist_y[y_idx] = hist_y[y_idx] + 1
    local z_idx = utils.get_index_in_sorted_array(bins_z, X_pose_diff[i][3])
    hist_z[z_idx] = hist_z[z_idx] + 1
  end
  -- Convert this histogram into some other form of representation ??
  -- e.g. probability and input the probability values into the classification
  -- network?
  local prob_hist_x = torch.Tensor(hist_x) 
  prob_hist_x = prob_hist_x:div(prob_hist_x:sum())
  local prob_hist_y = torch.Tensor(hist_y)
  prob_hist_y = prob_hist_y:div(prob_hist_y:sum())
  local prob_hist_z = torch.Tensor(hist_z)
  prob_hist_z = prob_hist_z:div(prob_hist_z:sum())

  local final_hist = torch.cat({prob_hist_x, prob_hist_y, prob_hist_z})
  return final_hist
end

function utils.get_pose_vel_category_context(args)
  local X, curr_t = args.X, args.curr_t
  local past_frames = 128

  local X_pose = X[{{}, {7, 12}}]
  local X_size = X:size()
  local X_pose_diff = torch.Tensor(X_pose:size(1), 3):zero()
  local th_x, th_y, th_z = 0.01, 0.01, 0.01
  local bin_size_x, bin_size_y, bin_size_z = 0.1, 0.1, 0.1

  X_pose_diff[{{2,-1},{}}] = X[{{2,-1}, {10,12}}] - X[{{1,-2},{10,12}}]

  local context_vec = torch.Tensor(1):zero()
  local idx = 1
  for i=curr_t-1,curr_t-past_frames, -8 do
    local num_gt_th = 0
    for j=i,i-7,-1 do
      if j < 1 then
      elseif (math.abs(X_pose_diff[j][1]) > th_x or
          math.abs(X_pose_diff[j][2]) > th_y or
          math.abs(X_pose_diff[j][3]) > th_z) then
          num_gt_th = num_gt_th + 1
        end
    end
    if num_gt_th >= 3 then context_vec[1] = context_vec[1] + 1 end
  end
  if context_vec[1] == 0 then context_vec[1] = 1 end
  return context_vec
end

-- Create a binary vector from curr_t to 128 frames in the past based on
-- threshold
function utils.get_binary_context_vec(args)
  local X, curr_t = args.X, args.curr_t
  local past_frames = args.past_frames or 100
  local th_pose_x, th_pose_y, th_pose_z = unpack(args.th_pose)

  local X_pose = X[{{}, {7, 12}}]
  local X_size = X:size()
  local X_pose_diff = torch.Tensor(X_pose:size(1), 3):zero()
  X_pose_diff[{{2,-1},{}}] = X[{{2,-1}, {10,12}}] - X[{{1,-2},{10,12}}]

  local context_vec = torch.Tensor(past_frames, 3):zero()
  local j = 1
  for i=curr_t-1,curr_t-past_frames, -1 do
    context_vec[j][1] = math.abs(X_pose_diff[i][1]) > th_pose_x and 1 or 0
    context_vec[j][2] = math.abs(X_pose_diff[i][2]) > th_pose_y and 1 or 0
    context_vec[j][3] = math.abs(X_pose_diff[i][3]) > th_pose_z and 1 or 0
    j = j + 1
  end
  -- TODO(Mohit): Reverse context_vec?
  assert(j == context_vec:size(1)+1)
  return context_vec:view(-1)
end

-- We return a table with keys from 1 to num_classes. Each key is mapped to a
-- table where each element of the table is another table of type (file_name,
-- gest_begin_time, gest_end_time)
function utils.get_all_gest_by_type(gest_h5file, group_name, num_classes)
  local gest_by_type = {}
  for i=1, num_classes do table.insert(gest_by_type, {}) end
  local h5_file = hdf5.open(gest_h5file, 'r')

  local f_contents = h5_file:read("/"):all()
  f_contents = f_contents[group_name]
  for k, v in pairs(f_contents) do
    local file_gest_by_type = v
    for i=1, num_classes do
      -- gest_t is a tensor Nx2 i.e. all gestures of type i in h5 file
      gest_t = file_gest_by_type[tostring(i-1)]
      if gest_t:nElement() > 0 and torch.isTensor(gest_t) then
        for j=1, gest_t:size(1) do
          -- Only insert if we have sufficient frames at the end for the window
          -- Compare gest[j][2] == 0 since we need a hack for cases where hdf5
          -- isn't able to read empty tensors stored in h5 files. For those cases
          -- we add [0, 0] as the indexes to save and remove them here.

          if (torch.isTensor(gest_t[j]) and gest_t[j][2] > 0) then
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

function utils.process_latent_data(args)
  local X, y = args.X, args.y
  local mean, std = args.mean, args.std
  local norm_type = args.norm_type
  local normalize_signals = args.signals

  assert(X:size(2) > 53)
  X = X[{{},{13,148}}]
  X = utils.normalize_data_with{ X=X, mean=mean, std=std }

  assert(normalize_signals == nil)

  return X, y
end

function utils.get_inputs(X, y, curr_t, win_sizes)
  local noise = nil

  local inp = {}
  for i=1, #win_sizes do
    local w_size = win_sizes[i]
    if noise == nil or noise == 0 then
      table.insert(inp, X[{{curr_t-w_size/2, curr_t-1+w_size/2},{}}])
    else
      local x = X[{{curr_t-w_size/2, curr_t-1+w_size/2},{}}]
      -- Flip the tensor
      x = x:index(1, torch.linspace(x:size(1),1,x:size(1)):long())
      table.insert(inp, x)
    end
  end
  return inp
end

-- This implementation is a bit slow but I guess for now it suffices.
function utils.get_inputs_from_file(f_path, curr_t, win_sizes)
  local h5_by_file = {}
  local hdf5_file = hdf5.open(f_path, 'r')
  local annots = hdf5_file:read('/annotations'):all()
  local feats = hdf5_file:read('/features'):all()
  local X1 = torch.Tensor(feats:size()):copy(feats)
  local y1 = torch.Tensor(annots:size()):copy(annots)
  local X, y = utils.process_data(X1, y1)
  local inp = utils.get_inputs(X, y, curr_t, win_sizes)
  return inp
end

-- Choose first 5 inputs from each gesture class and insert it into a table
-- Thus we return a table of tables where each inner table contains 5 signals
-- for that gesture type.
function utils.get_input_from_gest_type(gest_by_type, num_classes, win_size)
   local fdir = '../../openface_data/face_gestures/dataseto_text'
   local input_by_gest = {}
   for i=1, num_classes do table.insert(input_by_gest, {}) end
   local NUM_INPUTS = 5
   for i=1, num_classes do
       for j=1, NUM_INPUTS do
            local seq = gest_by_type[i][j]
            local curr_file = seq[1]
            local curr_t = seq[2]
            local f_path = paths.concat(fdir, curr_file)
            local inp = utils.get_inputs_from_file(f_path, curr_t, {win_size})
            table.insert(input_by_gest[i], inp[1])
       end
   end
   return input_by_gest
end

function utils.computeKappa(mat)
  local N = mat:size(1)
  local tmp = torch.range(1, N):view(1, N)
  local tmp1 = torch.range(1, N):view(N, 1)
  local W= tmp:expandAs(mat)-tmp1:expandAs(mat)
  W:cmul(W)
  W:div((N-1)*(N-1))
  local total = mat:sum()
  local row_sum = mat:sum(1)/total
  local col_sum = mat:sum(2)
  local E = torch.cmul(row_sum:expandAs(mat), col_sum:expandAs(mat))
  mat = mat:double()
  E = E:double()
  local den = torch.cmul(W, E):sum()
  if den == 0  then
    den = 0.01
  end

  local kappa = 1 - torch.cmul(W, mat):sum()/den
  return kappa
end

function utils.write_hdf5(file_path, path_prefix, tensor_dict)
  print(file_path)
  local f = hdf5.open(file_path, 'w')
  for k,v in pairs(tensor_dict) do
    local p = k
    if path_prefix ~= nil then p = path_prefix..'/'..k end
    f:write(p, v)
  end
  f:close()
end

function utils.isNan(num)
  return num ~= num
end

function utils.reject_outliers(args)
  local data, m = args.data, max_median_scale or 10.0
  local data_c = data:clone():view(-1)
  local d = torch.abs(data_c - torch.median(data_c)[1])
  local mdev = torch.median(d)[1]
  local s = d:clone()
  s = s:div(mdev)
  if mdev <= 0.00001 and mdev >= -0.00001 then
    -- print("WARNING: Error in median calculation during rejecting outliers")
    return data_c
  end
  data_c[s:gt(m)] = 0
  return data_c
end

function utils.get_one_hot_tensor(inp, num_classes)
  local one_hot_val = torch.Tensor(inp:size(1), num_classes):zero()
  for i=1,inp:size(1) do
    one_hot_val[i][inp[i]] = 1
  end
  return one_hot_val
end

function utils.convert_to_type(x, dtype)
  if torch.isTensor(x) then 
    x = x:type(dtype)
    return x
  end
  for i=1,#x do 
    if torch.isTensor(x[i]) then x[i] = x[i]:type(dtype)
    else for j=1,#x[i] do x[i][j] = x[i][j]:type(dtype) end end
  end
  return x
end

return utils

