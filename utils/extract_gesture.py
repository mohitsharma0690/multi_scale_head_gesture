import numpy as np
import datetime
import h5py
import os
import subprocess
import sys
import pdb
import argparse
import csv
import data_utils as global_utils


'''
Given a video of gestures and its annotations in an h5 file use ffmpeg to
extract each gesture from of the video and store it in a directory.

Example usage: 
python extract_gests.py <annotation_file> <input video> <output dir> <gest name>
python extract_gests.py \
        024/024_static.mp4.txt.h5 \
        024_static.mp4 \
        024_5_videos \
        5
'''
FRAME_RATE = 24.0

def get_annotations(file_name):
    f = h5py.File(file_name)
    gests = np.array(f['annotations'])
    f.close()
    return gests

def get_gesture_sequence(gests, target_gest):
    i, gests_len = 0, len(gests)
    target_gest_seq = []
    while i < gests_len: 
        if gests[i] == target_gest:
            j = i+1
            while j < gests_len and gests[j] == target_gest:
                j = j + 1
            target_gest_seq.append([i, j])  # just like python do not include j
            i = j
        else:
            i = i + 1
    return  target_gest_seq

def convert_frame_to_timestamp(gest_seq):
    fps = FRAME_RATE
    gest_timestamps = []
    for frame_idx in gest_seq:
        st, end = frame_idx[0] / fps, frame_idx[1] / fps
        time_diff = end - st if end - st >= 1 else 1  # min time 1s
        gest_timestamps.append([st, time_diff])
    return gest_timestamps

def convert_to_ffmpeg_timestamp(val):
    hour, minute = int(val/3600), int(val/60)
    seconds = int(val - hour*3600 - minute*60)
    microseconds = val - hour*3600 - minute*60 - seconds
    assert(microseconds < 1)
    microseconds = int(microseconds * 1000000)
    d = datetime.datetime(2000, 1, 1, hour, minute, seconds, microseconds)
    # ffmpeg uses miniseconds instead of microseconds
    return d.strftime('%H:%M:%S.%f')[:-3]

def save_videos(vid_name, dir_name, gest_timestamps):
    print('Will save {} videos in {} for video {}'.format(
        len(gest_timestamps), dir_name, vid_name))
    for ts in gest_timestamps:
        start_ffmpeg_ts = convert_to_ffmpeg_timestamp(ts[0])
        duration_ffmpeg_ts = convert_to_ffmpeg_timestamp(ts[1])
        save_file_name = '{}_{}_{}'.format(
                os.path.basename(vid_name)[:-4],
                int(ts[0]*FRAME_RATE), 
                int((ts[0]+ts[1])*FRAME_RATE))
        output, error = run_ffmpeg(
                vid_name,
                dir_name + '/' + os.path.basename(save_file_name) + '.mp4',
                start_ffmpeg_ts,
                duration_ffmpeg_ts) 
        if error:
            print error
        else:
            print('Did convert frame {}'.format(int(ts[0]*FRAME_RATE)))

def run_ffmpeg(vid_name, save_file_name, start_ts, duration_ts):
    '''
     ffmpeg -ss 00:00:00.000 -i <input> -to 00:00:00.000 -c copy <output>
    '''
    bash_cmd = 'ffmpeg -ss {} -i {} -to {} -c copy {}'.format(
            start_ts,
            vid_name,
            duration_ts,
            save_file_name)
    process = subprocess.Popen(bash_cmd.split(), stdout = subprocess.PIPE)
    output, error = process.communicate()
    return output, error


def main(h5_file_name, vid_name, dir_name, target_gest, gest_list_h5,
        gest_corr_csv):
    if gest_list_h5 is None:
        gests = get_annotations(h5_file_name)
        gest_seq = get_gesture_sequence(gests, target_gest)
        gest_timestamps = convert_frame_to_timestamp(gest_seq)
        save_videos(vid_name, dir_name, gest_timestamps)
    elif gest_corr_csv is not None and len(gest_corr_csv) > 0:
        # Extract all gestures for a particular user
        user_name = os.path.basename(h5_file_name)  # corresponds to row
        csv_data = global_utils.read_csv(gest_corr_csv)
        csv_data = filter(lambda x: x['filename'] == user_name, csv_data)
        labels = [target_gest] if target_gest >= 0 else range(11)
        for label in labels:
            frames = [[int(d['start_frame']), int(d['end_frame'])] \
                    for d in csv_data if int(d['old_label']) == label]

            # Create dir for saving.
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            label_dir = os.path.join(dir_name, str(label))
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            timestamps = convert_frame_to_timestamp(frames)
            save_videos(vid_name, label_dir, timestamps)
            
    else:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        labels = [target_gest] if target_gest >= 0 else range(11)
        for label in labels:
            file_name = os.path.basename(h5_file_name)
            gest_h5 = h5py.File(gest_list_h5, 'r')
            group = 'test' if file_name in gest_h5['test'].keys() else 'train'
            all_gests = np.array(gest_h5[group][file_name][str(label)])
            label_dir = os.path.join(dir_name, str(label))
            if not os.path.exists(label_dir): os.makedirs(label_dir)
            if len(all_gests.shape) == 1: 
                print("No videos found")
            else:
                gest_timestamps = convert_frame_to_timestamp(all_gests.tolist())
                save_videos(vid_name, label_dir, gest_timestamps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Extract gestures from vides using gesture label or '
            'gesture list')
    parser.add_argument('--h5_file_name', nargs='?', type=str, const=1,
        default='', help='h5 files directory')
    parser.add_argument('--vid', nargs='?', type=str, const=1,
        required=True, help='Video file from which to extract gestures [mp4]')
    parser.add_argument('--gest_list_h5', nargs='?', type=str, const=1,
        default='', help='gesture list h5 file to be used as original gestures.')
    parser.add_argument('--extract_dir', nargs='?', type=str, const=1,
        required=True, help='Directory to extract gesture videos in')
    parser.add_argument('--gesture_label', nargs='?', type=int, const=1,
        default=-1, help='Gesture label to which to extract gestures for.')
    parser.add_argument('--gesture_correction_csv', nargs='?', type=str,
            const=1, default='',
            help='Gesture correction csv to extract gestures from.')
    
    args = parser.parse_args()
    print(args)

    h5_file_name = args.h5_file_name
    vid_name = args.vid
    gest_list_h5 = args.gest_list_h5
    extr_gest_dir = args.extract_dir
    target_gest = args.gesture_label
    gest_list_h5 = args.gest_list_h5
    gest_corr_csv = args.gesture_correction_csv
    main(h5_file_name, vid_name, extr_gest_dir, target_gest, gest_list_h5,
            gest_corr_csv)

