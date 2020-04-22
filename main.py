import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import models
from keras import layers
from keras import optimizers
import json

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

img_rows, img_cols, num_frames = 320, 180, 120
test_data_path = '/home/alex/Desktop/csce636/dataset/RGB/test/'
model_save_path = "/home/alex/Desktop/csce636/dataset/model/my_model.h5"
json_save_path = "/home/alex/Desktop/csce636/dataset/json/test/"

X_test = []
listing2 = os.listdir(test_data_path)
listing2.sort()
print("Enter test videos directory")

for vid in listing2:
    vid = test_data_path + vid
    test_frames = []
    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(vid, fps, total_frames)

    for k in range(4200):
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
    X_test.append(test_ipt)

print("test videos reading done")

X_test_array = np.array(X_test)
X_test_array = np.asarray(X_test_array).astype('float32')
X_test_array = X_test_array.astype('float32')
X_test_array -= np.mean(X_test_array)
X_test_array /= np.max(X_test_array)

test_labels = np.zeros(210)
test_labels = np.asarray(test_labels).astype('float32')
num_samples = len(X_test_array)
X_test_array = X_test_array.reshape(210, 20, img_rows, img_cols, 1)
print("test sample numbers: ", num_samples)
print("test videos shape: ", X_test_array.shape)

model_exist = os.path.exists(model_save_path)
if model_exist:
    model = models.load_model(model_save_path)
    print("model loaded")

results = model.predict_classes(X_test_array)
results = results.reshape((1, 210))
print(results)

a = model.evaluate(X_test_array, test_labels)
print(a)
model.save(model_save_path)
for k in range(1):
    json_data = {"Tremor": []}
    for i in range(210):
        json_data["Tremor"].append([i, int(results[k][i])])
    json_path = json_save_path + str(k) + ".json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f)
    time = range(1, 211)
    plt.figure()
    plt.plot(time, results[k], 'bo')
    plt.xlabel("frames")
    plt.ylabel("Tremor: yes or no")
    plt.show()
