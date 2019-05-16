import cv2
import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import sys

from scipy.signal import savgol_filter

def save_video(opt_flow_dir, video_file):
  video_file_dir = os.path.join(opt_flow_dir, video_file)

  cap = cv2.VideoCapture(os.path.join(opt_flow_dir, "flow_y_00000.jpg"))
  # cap = cv2.VideoCapture(os.path.join(opt_flow_dir, "flow_x_{0:05d}0000%d.jpg"))

  while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    print("Return status ", ret)

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
      break

  # When everything done, release the capture
  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Create video form optical flow images.')
  parser.add_argument('--video', nargs='?', type=str, const=1,
      required=True, help='Output video file to write.')
  parser.add_argument('--optical_flow_dir', nargs='?', type=str, const=1,
      required=True, help='Input optical flow dir to use.')
  args = parser.parse_args()
  print(args)

  save_video(args.optical_flow_dir, args.video)

