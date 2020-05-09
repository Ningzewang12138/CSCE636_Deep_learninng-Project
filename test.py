import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import models
from keras import layers
from keras import optimizers
import json

img_rows, img_cols, num_frames = 320, 180, 120

cur_path = os.path.dirname(os.path.abspath("test.py"))

print(cur_path)

test_file = sys.argv[1]

test_data_path = cur_path +"/RGB/" + test_file
json_save_path = cur_path + "/json/test/" 
model_save_path = cur_path + "/model/827006879.h5"
plt_save_path = cur_path +"/figure/test/" + test_file + ".png" 

print("test video path: ", test_data_path)
print("json_save_path: ", json_save_path)
print("plt_save_path: ", plt_save_path)


test_frames = []
cap = cv2.VideoCapture(test_data_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
seconds = int(total_frames // fps)
print(test_data_path, fps, total_frames, str(seconds)+"s")
for k in range(seconds * 30):
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (img_rows, img_cols), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        test_frames.append(gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

test_input = np.array(test_frames)
test_ipt = np.rollaxis(test_input, 2, 1)
test_ipt = test_ipt.reshape(seconds, 30, 320,180,1)
print("test videos reading done")


model_exist = os.path.exists(model_save_path)
if model_exist:
    model = models.load_model(model_save_path)
    print("model loaded")

results = model.predict_classes(test_ipt)
results = results.reshape((1, seconds))
print(results)
json_data = {"Tremor": []}
for i in range(seconds):
    json_data["Tremor"].append([i, int(results[0][i])])
    json_path = json_save_path + test_file + ".json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f)
    time = range(0, seconds)
    plt.figure()
    plt.plot(time, results[0], 'bo')
    plt.xlabel("frames")
    plt.ylabel("Tremor: yes or no")
    plt.savefig(plt_save_path)