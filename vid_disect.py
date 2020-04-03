import cv2
import os
import numpy as np

input_path = "/Users/ningzewang/Desktop/dataset/RGB/train/"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_path = "/Users/ningzewang/Desktop/dataset/section/train/"

listing = os.listdir(input_path)
listing.sort()

for vid in listing:
    vid_name = vid.split(".")[0]
    input_vid = '/Users/ningzewang/Desktop/dataset/RGB/train/' + vid
    cap = cv2.VideoCapture(input_vid)
    out_vid_0 = '/Users/ningzewang/Desktop/dataset/section/train/' + vid_name + "_0.mp4"
    out_0 = cv2.VideoWriter(out_vid_0, fourcc, 25.0, (960, 540), True)
    out_vid_1 = '/Users/ningzewang/Desktop/dataset/section/train/' + vid_name + "_1.mp4"
    out_1 = cv2.VideoWriter(out_vid_1, fourcc, 25.0, (960, 540), True)
    out_vid_2 = '/Users/ningzewang/Desktop/dataset/section/train/' + vid_name + "_2.mp4"
    out_2 = cv2.VideoWriter(out_vid_2, fourcc, 25.0, (960, 540), True)
    out_vid_3 = '/Users/ningzewang/Desktop/dataset/section/train/' + vid_name + "_3.mp4"
    out_3 = cv2.VideoWriter(out_vid_3, fourcc, 25.0, (960, 540), True)
    out_vid_4 = '/Users/ningzewang/Desktop/dataset/section/train/' + vid_name + "_4.mp4"
    out_4 = cv2.VideoWriter(out_vid_4, fourcc, 25.0, (960, 540), True)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if i // 25 == 0:
            if ret:
                out_0.write(frame)
                i += 1
                if cv2.waitKey(1) & 0XFF == ord('q'):
                    break
            else:
                break
        elif i // 25 == 1:

            if ret:
                out_1.write(frame)
                i += 1
                if cv2.waitKey(1) & 0XFF == ord('q'):
                    break
            else:
                break
        elif i // 25 == 2:

            if ret:
                out_2.write(frame)
                i += 1
                if cv2.waitKey(1) & 0XFF == ord('q'):
                    break
            else:
                break
        elif i // 25 == 3:

            if ret:
                out_3.write(frame)
                i += 1
                if cv2.waitKey(1) & 0XFF == ord('q'):
                    break
            else:
                break
        elif i // 25 == 4:

            if ret:
                out_4.write(frame)
                i += 1
                if cv2.waitKey(1) & 0XFF == ord('q'):
                    break
            else:
                break
        else:
            break

    cap.release()
    out_0.release()
    out_1.release()
    out_2.release()
    out_3.release()
    out_4.release()
