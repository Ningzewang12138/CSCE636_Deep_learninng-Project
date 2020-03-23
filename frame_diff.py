import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

input_path = "/Users/ningzewang/Desktop/dataset/RGB/train/"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_path = "/Users/ningzewang/Desktop/dataset/gray_diff/train/"

listing = os.listdir(input_path)
listing.sort()
listing.sort(key=lambda x: int(x[:-4]))

for vid in listing:
    input_vid = '/Users/ningzewang/Desktop/dataset/RGB/train/' + vid
    out_vid = '/Users/ningzewang/Desktop/dataset/gray_diff/train/' + vid
    cap = cv2.VideoCapture(input_vid)
    out = cv2.VideoWriter(out_vid, fourcc, 24.0, (960, 540), False)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(vid, fps, total_frames)
    frame_num = 0
    while cap.isOpened():
        catch, frame = cap.read()
        if catch:

            if not frame_num:
                previous = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_diff = cv2.absdiff(gray, previous)
            grady_diff = cv2.medianBlur(gray_diff, 3)
            ret, mask = cv2.threshold(gray_diff, 15, 255, cv2.THRESH_BINARY)
            mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4)), iterations=1)
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4)), iterations=1)
            out.write(mask)
            previous = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_num += 1
            if cv2.waitKey(1) & 0XFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

gd_input_path = "/Users/ningzewang/Desktop/dataset/gray_diff/train/"

listing2 = os.listdir(gd_input_path)
listing2.sort()
listing2.sort(key=lambda x: int(x[:-4]))

for gd_vid in listing2:
    # ext = os.path.splitext(gd_vid)[1]
    # print(ext)
    labels = []
    gd_input = gd_input_path + gd_vid
    cap2 = cv2.VideoCapture(gd_input)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    total_frames2 = cap2.get(cv2.CAP_PROP_FRAME_COUNT)
    print(gd_vid, fps2, total_frames2)
    while cap2.isOpened():
        ret, frame = cap2.read()
        summation = 0
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = frame.shape
            for i in range(height):
                for j in range(width):
                    if frame[i, j] == 255:
                        summation += 1
            if summation > 15:
                labels.append(1)
            else:
                labels.append(0)

            if cv2.waitKey(1) & 0XFF == ord('q'):
                break
        else:
            break
    print(len(labels))
    np.savetxt("/Users/ningzewang/Desktop/dataset/labels/train/train_labels" + gd_vid.split('.')[0] + ".txt", labels,
               fmt='%d')

    cap2.release()
    cv2.destroyAllWindows()
