import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import pdb

def run_dense_flow(vid_file_path):
    cap = cv2.VideoCapture(vid_file_path)

    while not cap.isOpened():
        print "Wait for the header"
        cap = cv2.VideoCapture(vid_file_path)
        cv2.waitKey(10)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    i = 0
    print('here')
    while(1):
        i = i + 1
        ret, frame2 = cap.read()
        for j in range(4):
            ret, frame2 = cap.read()

        if ret:
            if i % 100 == 0:
                print(i)
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            # cv2.imshow('org_frame2', frame)
            
            next_frame = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs,next_frame, None, 0.5, 3,
                    20, 5, 5, 1.2, 0)

            # Change here
            horz = cv2.normalize(flow[...,0], None, 0, 255, cv2.NORM_MINMAX)     
            vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
            horz = horz.astype('uint8')
            vert = vert.astype('uint8')

            horz = horz[400:1200,400:1200]
            vert = vert[400:1200,400:1200]
            org = next_frame[400:1200,400:1200]

            # Change here too
            cv2.imshow('Horizontal Component', horz)
            cv2.imshow('Vertical Component', vert)
            cv2.imshow('original', org)

            '''
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            cv2.imshow('hsv', hsv)
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            cv2.imshow('frame2',bgr)
            cv2.imshow('org', next_frame)
            '''
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                cv2.imwrite('opticalfb.png',frame2)
                cv2.imwrite('opticalhsv.png',bgr)
            prvs = next_frame

            cv2.waitKey(10)
        else:
            print "frame is not ready"
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1.0)


    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    vid_file_path = sys.argv[1]
    print(vid_file_path)
    run_dense_flow(vid_file_path)

