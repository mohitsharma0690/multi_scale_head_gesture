import cv2
import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import sys

from scipy.signal import savgol_filter

import data_utils
import convert_cpm_json_to_h5

# Sample Usage:
#
# python video_frames.py <video-file> \
#   <open-face annotations for vid> \
#   <predictions file for given video>  \
#   <num classes used in prediction>
#
# python video_frames.py \
#   ../videos/042/042_static.mp4 \
#   ../videos/042/042_static.mp4.txt.h5 \
#   pred.h5
#   5

DISPLAY_ANGULAR_VELOCITY_TEXT = 0
DISPLAY_FRAME_NUMBER = 1

FLAGS = None

def smooth_data(X):
  window_len, poly_order = 11, 2
  for i in xrange(X.shape[1]):
    X_data = X[:,i]
    X[:, i] = savgol_filter(X_data, window_len, poly_order)
  return X

def normalize_data_each_sensor_signal(X, y):
  '''
  Normalize data X and y.
  '''
  mean = np.mean(X, 0)
  std = np.std(X, 0)
  norm_X = (X - mean) / std
  # norm_X = X - mean
  return norm_X, y

def non_maximal_suppression(X, suppress_idx, supp_th):
  for col_idx in suppress_idx:
    val = X[:, col_idx]
    b = (val<supp_th) * (val>-supp_th)
    val[b] = 0
    X[:, col_idx] = val

  return X

def read_h5_file(h5_path):
  h5_file = h5py.File(h5_path)
  y = np.array(h5_file['annotations'])
  X = np.array(h5_file['features'])
  X, y = data_utils.process_data(X, y)
  h5_file.close()
  '''
  X = non_maximal_suppression(X, range(34, 50), supp_th=0.5)
  X = non_maximal_suppression(X, range(15, 17), supp_th=(0.01*180)/np.pi)
  X = non_maximal_suppression(X, range(56, 57), supp_th=1)  # 1 degree
  '''

  return X, y

def read_pred_file(h5_pred_file, vid_file):
  h5_file = h5py.File(h5_pred_file)
  pred = np.array(h5_file[str(vid_file)]['pred']).astype(int)
  prob = np.array(h5_file[str(vid_file)]['prob'])
  beta = None
  if "beta" in h5_file[str(vid_file)].keys():
      beta = np.array(h5_file[str(vid_file)]['beta'])

  h5_file.close()
  return pred, prob, beta

def get_class_text_for_gest_5(gest_type, prob):
  if gest_type < 0:
    return 'N/A', ''
  gest_type_text = ['0-None', '1-Nod', '2-Tilt', '3-Shake/Rotate', '4-Fwd/Back']
  prob_str = '( '
  for i in range(len(prob)):
    p = prob[i]
    prob_str += '{:.1f} '.format(p)
  prob_str += ')'

  return gest_type_text[int(gest_type)], prob_str

def get_class_text_for_gest(gest_type, pred_prob=None):
  gest_type_text = ['None', '1-Nod', '1-Jerk', '1-Up', '1-Down', '1-Tick',
      '2-Tilt', '3-Shake', '3-Rotate', '4-Forward', '4-Back']
  prob_str = '('
  if pred_prob is not None:
    for i in range(len(pred_prob)):
      p = pred_prob[i]
      prob_str += '{:.1f} '.format(p)
    prob_str += ')'

  prob_str = prob_str if len(prob_str) > 1 else None
  return gest_type_text[gest_type], prob_str

#####
# 0-5:      (gaze_0_x gaze_0_y gaze_0_z) (gaze_1_xyz)
# 6-11:     (pose_xyz, pose_rotation_angle_xyz)
# 12-17:    (pose velocity and angular velocity xyz)
# 18-33:    (forehead, nosetip, left face side (4), bottom (4), right (4)
# 34-49:    (velocity for each of the above ones)
# 50-55:    (face difference vectors xyz)
#####

def idx_to_plot(plots):
  plots_by_name = [
      'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
      'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
      'pose_x', 'pose_y', 'pose_z', 'pose_rot_x', 'pose_rot_y', 'pose_rot_z',
      'vel_pose_x', 'vel_pose_y', 'vel_pose_z',
      'ang_vel_pose_x', 'ang_vel_pose_y', 'ang_vel_pose_z',
      'forehead_x', 'forehead_y', 'nosetip_x', 'nosetip_y', 'left_x', 'left_y',
      '', '', 'bottom_x', 'bottom_y', '', '', 'right_x', 'right_y', '', '',
      'vel_forehead_x', 'vel_forehead_y', 'vel_nosetip_x', 'vel_nosetip_y',
      'vel_left_x', 'vel_left_y', '', '', 'vel_bottom_x', 'vel_bottom_y', '',
      '', 'vel_right_x', 'vel_right_y', '', '',
      'forehead_nosetip_x', 'forehead_nosetip_y',
      'left_nosetip_x', 'left_nosetip_y',
      'right_nosetip_x', 'right_nosetip_y', 'nose_forehead_angle',
  ]
  assert(len(plots_by_name) == 57)

  text, idxs = [], []
  for p in plots:
    for i in range(len(plots_by_name)):
      if p == plots_by_name[i]:
        idxs.append(i)
        text.append(p)
  return text, idxs

def get_polyline_points(X, y, all_polyline_pts, poly_pos,
    x_idxs, curr_frame_idx, st_frame, end_frame, shift_amt=5):

  for i in xrange(len(all_polyline_pts)):
    polyline_pts, idx, pos = all_polyline_pts[i], x_idxs[i], poly_pos[i]
    shifted_X = X[st_frame:end_frame + 1, idx].copy()
    shifted_X = shifted_X * (2**shift_amt)
    shifted_X = shifted_X.astype('int')
    # Use - to flip the values
    val = pos[1] - shifted_X
    val[val > 50] = 50
    val[val < -50] = -50
    polyline_pts[0:shifted_X.shape[0], 1] = pos[1] - shifted_X

  return all_polyline_pts

def get_colors_for_plots():
  colors = [
      (255, 102, 255), # Blue
      (178, 255, 102), # Green
      (0, 0, 255), # Red
      (255, 255, 0), # Yellow
      (255, 0, 255),
      (0, 255, 255),
      (102, 127, 255),
      (255, 0, 127),
  ]
  return colors

def get_trimmed_rect_coords(filename, width, height):
  '''
  Returns (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
  '''
  if "041" in filename:
    return (400, 400, 1200, 1200)
  elif "040" in filename:
    return (300, 300, 1200, 1200)
  elif "043" in filename:
    return (300, 300, 1200, 1200)
  elif "044" in filename:
    return (300, 300, 1200, 1200)
  elif "038" in filename:
    return (300, 500, 1100, 1300)
  elif "024" in filename:
    return (300, 500, 1100, 1300)
  else:
    return (0, 0, width-1, height-1)

def get_prob_rect_positions(filename):
  ht = 340
  if "041" in filename:
    prob_rect_pos = [
        (100, ht), (130, ht), (160, ht), (190, ht), (220, ht)
    ]
  else:
    prob_rect_pos = [
        (100, ht), (130, ht), (160, ht), (190, ht), (220, ht)
    ]
  return prob_rect_pos

def get_poly_positions(num_poly):
  l = 50  # Left space
  ht, ht_diff = 50, 100

  if num_poly == 5:
    poly_pos = [
        (l, ht), (l, ht+1*ht_diff), (l, ht+2*ht_diff),
        (l, ht+3*ht_diff), (l, ht+4*ht_diff)
        ]
  elif num_poly == 8:
    poly_pos = [
        (l, 200), (l, 300), (l, 400),
        (l, 500), (l, 600), (l, 700),
        (l, 800), (l, 900)
        ]
  else:
    poly_pos = []
  return poly_pos

def read_cpm_h5(cpm_h5_filepath):
    if cpm_h5_filepath is None or len(cpm_h5_filepath) == 0:
        return None
    cpm_fp = h5py.File(cpm_h5_filepath, 'r')
    joints = np.array(cpm_fp['/joints'])
    cpm_fp.close()
    return joints

def show_video_with_annotations(file_path, X, y, pred=None, pred_probs=None,
    beta=None, cpm_joints=None, pred_type=11):

  NEXT_FRAME_IN_MS = 1

  cap = cv2.VideoCapture(file_path)

  frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  trimmed_img_rect = get_trimmed_rect_coords(
      os.path.basename(file_path), width, height)

  joint_to_display = ['Nose', 'Neck']
  joint_idx = [convert_cpm_json_to_h5.get_joint_type_to_h5_idx(j) \
          for j in joint_to_display]

  while not cap.isOpened():
    cap = cv2.VideoCapture(file_path)
    cv2.waitKey(1000)
    print "Wait for the header"

  pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

  plots_name = [
    'ang_vel_pose_x', 'ang_vel_pose_y', 'ang_vel_pose_z',
    'vel_nosetip_x', 'vel_nosetip_y'
  ]
  # Get probability rectangle positions
  prob_rect_pos = get_prob_rect_positions(os.path.basename(file_path))

  # Get plot names and indexes to plot
  polyline_text, polyline_idxs = idx_to_plot(plots_name)

  # Get plot positions
  poly_pos = get_poly_positions(len(plots_name))

  # Get colors for plots
  poly_colors = get_colors_for_plots()

  # Initialize polynomials
  all_polyline_pts = []
  shift_amt, disp_frame_len = 5, 20
  for i in range(len(polyline_idxs)):
    polylines_pts = np.zeros((2*disp_frame_len+1, 2), dtype='int')
    # 100 is the position where polylines should be on the graph i.e. x-axis
    polylines_pts[:, 0] = poly_pos[i][0] + 2*np.arange(1,
        polylines_pts.shape[0]+1).astype('int')
    all_polyline_pts.append(polylines_pts)

  fig = None
  '''
  # Python plots for convenience
  fig, ax = plt.subplots(4, 2, sharex=True)
  #fig, ax = plt.subplots(1, sharex=True)
  # Interactive mode is required to keep processing openCV and matplotlib's
  # visual backends together.
  ax[0,0].plot(range(0,60), range(0,60))
  ax[0,1].plot(range(0,60), range(0,60))
  ax[1,0].plot(range(0,60), range(0,60))
  ax[1,1].plot(range(0,60), range(0,60))
  ax[2,0].plot(range(0,60), range(0,60))
  plt.ion()
  plt.show()
  '''

  while True:
    flag, frame = cap.read()

    if fig is not None:
      fig.clf()
    if flag:
      # The frame is ready and already captured
      pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
      frame_idx = int(pos_frame)
      start_frame = max(0, frame_idx - disp_frame_len)
      end_frame = min(X.shape[0], frame_idx + disp_frame_len)
      class_txt, _ = get_class_text_for_gest(y[int(pos_frame)])
      pred_txt, beta_txt = None, None
      if pred is not None:
        pred_prob = pred_probs[int(pos_frame)]
        if pred_type == 11:
          pred_txt, prob_txt = get_class_text_for_gest(
                  pred[int(pos_frame)],
                  pred_prob=pred_prob)
        else:
          pred_txt, prob_txt = get_class_text_for_gest_5(
              pred[int(pos_frame)], pred_prob)

      if beta is not None:
        beta_txt = '{:.2f}'.format(beta[int(pos_frame)])

      if pred_txt is not None:
        b = beta_txt if beta_txt is not None else 'N/A'
        print('Frame no. {}, gest: {}, pred: {}, beta: {}'.format(
          int(pos_frame), class_txt, pred_txt, beta_txt))
      else:
        print('Frame no. {}, gest: {}'.format(int(pos_frame), class_txt))



      # Adding the overlay
      # overlay_frame = frame.copy()
      overlay_frame = np.ones((600, 300, 3), np.uint8)

      points = np.arange(1, 101, dtype='int').reshape(50, 2)
      shifted_X = X[start_frame:end_frame + 1, 34].copy()
      shifted_X = shifted_X * (2**shift_amt)
      shifted_X = shifted_X.astype('int')
      # Use - to flip the values
      polylines_pts[0:shifted_X.shape[0], 1] =  500 - shifted_X
      all_polyline_pts = get_polyline_points(X, y, all_polyline_pts, poly_pos,
          polyline_idxs, frame_idx, start_frame, end_frame)

      hist_img = np.zeros(overlay_frame.shape).astype(overlay_frame.dtype)

      # Draw polynomials
      for i in range(len(all_polyline_pts)):
        cv2.polylines(hist_img, [all_polyline_pts[i]], False, poly_colors[i],
            thickness=3)

        # Add lines to the center of each plot
        pt1 = (all_polyline_pts[i][frame_idx-start_frame, 0],
            all_polyline_pts[i][frame_idx-start_frame, 1] - 3)
        pt2 = (all_polyline_pts[i][frame_idx-start_frame, 0],
            all_polyline_pts[i][frame_idx-start_frame, 1] + 3)
        # Draw the line in hist_img for the center point
        c = poly_colors[i]
        cv2.line(hist_img, pt1, pt2, (255-c[0], 255-c[1], 255-c[2]),
            thickness=5)

      if FLAGS.overlay_confidence > 0.01:
        pred_prob = pred_probs[int(pos_frame)]
        conf_img = np.zeros(frame.shape).astype(overlay_frame.dtype)
        if np.max(pred_prob) > FLAGS.overlay_confidence:
          conf_img[:] = (0, 255, 0)  #Green
        else:
          conf_img[:] = (0, 0, 255)  #Red
        alpha = 0.8
        new_frame = cv2.addWeighted(frame, alpha, conf_img, (1-alpha), 0)
      else:
        new_frame = frame

      for i in range(len(all_polyline_pts)):
        pt = (poly_pos[i][0], poly_pos[i][1] + 50)
        cv2.putText(hist_img, polyline_text[i], pt, cv2.FONT_HERSHEY_PLAIN,
            1.0, (0, 0, 255), 2)

      for i in range(len(all_polyline_pts)):
        # It becomes verys slow rendering so many plots.
        if i >= 1:
          break
        plt.subplot(4,2,i+1)
        plt.plot(range(0,end_frame+1-start_frame),
            all_polyline_pts[i][0:end_frame+1-start_frame, 1], 'k-', lw=2)
        plt.plot(frame_idx - start_frame,
            all_polyline_pts[i][frame_idx-start_frame, 1], 'or')
        plt.title(polyline_text[i])

      # Add facial landmarks to image
      for i in range(18, 34, 2):
        cv2.circle(new_frame, (int(X[frame_idx][i]), int(X[frame_idx][i+1])),
            radius=3, color=(255,255,0), thickness=-1)

      # Add CPM joints to image
      if cpm_joints is not None:
        last_joint_pos = None
        for j_idx in joint_idx:
          joint_pos = cpm_joints[frame_idx][j_idx:j_idx+3]
          # Draw circle only if prob of joint is high
          if joint_pos[2] > 0.5:
            cv2.circle(new_frame, (int(joint_pos[0]), int(joint_pos[1])),
                radius=5, color=(0,0,255), thickness=-1)
          # Draw line between the joints
          if last_joint_pos is not None:
            cv2.line(
                new_frame,
                (int(last_joint_pos[0]), int(last_joint_pos[1])),
                (int(joint_pos[0]), int(joint_pos[1])),
                color=(0, 255, 0),
                thickness=3,
            )
          last_joint_pos = joint_pos

      # Add angular velocity text to image
      if DISPLAY_ANGULAR_VELOCITY_TEXT:
        angular_vel_txt = '({:.2f}, {:.2f}, {:.2f})'.format(
            X[frame_idx][15], X[frame_idx][16], X[frame_idx][17])
        put_angular_vel_text = False
        if put_angular_vel_text:
          cv2.putText(new_frame, angular_vel_txt,
              (trimmed_img_rect[0] + 100, trimmed_img_rect[1] + 100),
              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)

      # Display original gesture label
      cv2.putText(new_frame, 'Target: ' + class_txt,
          (trimmed_img_rect[1]+50+350, trimmed_img_rect[0]+180),
          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3
      )

      # Display original gesture label
      angle_txt = 'Nose-Forehead angle: {}'.format(X[frame_idx, 56])
      cv2.putText(new_frame, angle_txt,
          (trimmed_img_rect[1]+100, trimmed_img_rect[0]+400),
          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3
      )

      # Display probability values and corresponding rectangles
      pred_txt_color = (0, 255, 255)
      if pred_txt is not None:
        # Color is dark blue
        print(file_path)
        if "041" in os.path.basename(file_path):
            pred_txt_color = (0,252,124)

        cv2.putText(new_frame, 'Pred: ' + pred_txt,
            (trimmed_img_rect[1]+350,trimmed_img_rect[0]+250),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, pred_txt_color, 3
        )

        cv2.putText(new_frame, prob_txt,
            (trimmed_img_rect[1]+20+350,trimmed_img_rect[0]+300),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, pred_txt_color, 3
        )
        '''
        cv2.putText(new_frame, prob_txt,
            (trimmed_img_rect[1]+200,trimmed_img_rect[0]+200),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, pred_txt_color, 2
        )
        '''

        assert(pred_prob is not None)
        for i in range(len(prob_rect_pos)):
          pos = prob_rect_pos[i]
          # pos = (pos[0] + trimmed_img_rect[1] + 300, pos[1] + trimmed_img_rect[0] - 100)
          pos = (pos[0] + trimmed_img_rect[1] + 300, pos[1] + trimmed_img_rect[0])
          prob = int(pred_prob[i] * 100)  # convert to 100% scale
          end_pos = (pos[0] + 25, pos[1] + - prob)
          color = poly_colors[i]
          if i == 4:
            color = poly_colors[i+1]  # Last and first color are same otherwise
          cv2.rectangle(new_frame, pos, end_pos, poly_colors[i], thickness=-1)

      if False and beta_txt is not None:
        cv2.putText(new_frame, beta_txt,
            (trimmed_img_rect[1]+250,trimmed_img_rect[0]+250),
            cv2.FONT_HERSHEY_SIMPLEX, 2.0, pred_txt_color, 3
        )

      # Display the frame number on screen
      cv2.putText(new_frame, str(int(pos_frame)),
          (trimmed_img_rect[1]+50, trimmed_img_rect[0]+50),
          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 128, 0), 3
      )

      new_frame = new_frame[trimmed_img_rect[0]:trimmed_img_rect[2],
          trimmed_img_rect[1]:trimmed_img_rect[3],:]

      cv2.imshow('video', new_frame)
      cv2.imshow('hist', hist_img)


    else:
      # The next frame is not ready, so we try to read it again
      cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
      print "frame is not ready"
      # It is better to wait for a while for the next frame to be ready
      cv2.waitKey(NEXT_FRAME_IN_MS)

    key_pressed = cv2.waitKey(30+NEXT_FRAME_IN_MS)
    if key_pressed == 27:
      break
    elif key_pressed == ord('a'):
      NEXT_FRAME_IN_MS = max(1, NEXT_FRAME_IN_MS // 10)
    elif key_pressed == ord('s'):
      NEXT_FRAME_IN_MS = min(1000, NEXT_FRAME_IN_MS * 10)
    elif key_pressed == 2:  # Left key
      print('Left key pressed, going 10 frames back')
      cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 10)
    elif key_pressed == 3:  # Right key
      print('Right key pressed, going 10 frames ahead')
      cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame + 10)
    elif key_pressed == ord('f'):  # Forward (100 frames)
      print('f pressed, going 100 frames ahead')
      cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame + 100)
    elif key_pressed == ord('b'):
      print('b pressed, going 100 frames back')
      cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 100)
    elif key_pressed == ord('F'): # Mega Forweard ( 1000 Frames)
      print('F pressed, going 1000 frames ahead')
      cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame + 1000)
    elif key_pressed == ord('B'):
      print('B pressed, going 1000 frames back')
      cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1000)
    elif key_pressed != 255:
      print('Pressed something ', key_pressed)

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
      # If the number of captured frames is equal to the total number of frames,
      # we stop
      break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Convert json predictions to h5.')
  parser.add_argument('--video', nargs='?', type=str, const=1,
      required=True, help='Input video file to use.')
  parser.add_argument('--target_h5', nargs='?', type=str, const=1,
      required=True, help='Target h5 with labels.')
  parser.add_argument('--pred_h5', nargs='?', type=str, const=1,
          help='Pred h5.')
  parser.add_argument('--cpm_h5', nargs='?', type=str, const=1,
          default='', help='Convolution Pose Machine h5 output')
  parser.add_argument('--num_classify', nargs='?', type=int, const=1,
          default=5, help='Num classification.')
  parser.add_argument('--overlay_confidence', nargs='?', type=float, const=1,
          default=0, help='Visualize More confident values color screen ' \
                  'with green else red.')
  args = parser.parse_args()
  print(args)

  mp4_vid = args.video
  h5_vid_file = args.target_h5
  X, y = read_h5_file(h5_vid_file)
  prob, preds_type, beta = None, None, None
  if args.pred_h5 is not None and len(args.pred_h5) > 0:
    preds, prob, beta = read_pred_file(
            args.pred_h5, os.path.basename(h5_vid_file))
    preds_type = args.num_classify
  else:
    preds = None
  if args.cpm_h5 is not None and len(args.cpm_h5) > 0:
      cpm_joints = read_cpm_h5(args.cpm_h5)
      # Same number of frames.
      print('Num frames in openface annotations {}'.format(X.shape[0]))
      print('Num frames in CPM output {}'.format(cpm_joints.shape[0]))
      # assert(X.shape[0] == cpm_joints.shape[0])
  else:
      cpm_joints = None

  FLAGS = args
  show_video_with_annotations(
          mp4_vid, X, y, preds, prob, beta, cpm_joints, preds_type)

