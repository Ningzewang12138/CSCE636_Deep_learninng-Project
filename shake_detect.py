import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.applications.imagenet_utils import decode_predictions
from keras.models import load_model
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras.layers import LSTM
from keras.layers import TimeDistributed
import matplotlib.pyplot as plt
import json

# --------------------------------------------
img_rows, img_cols, num_frames = 320, 180, 140

# Get the train videos frames into the numpy array
# output:
# 1.training group: X_partial_train  X_partial_labels
# 2.validation group: X_val  val_labels
# --------------------------------------------
X_train = []
listing1 = os.listdir('/content/drive/My Drive/dataset/RGB/train/')
listing1.sort()
listing1.sort(key=lambda x: int(x[:-4]))
print("Enter training Videos directory")

for vid in listing1:
    vid = '/content/drive/My Drive/dataset/RGB/train/' + vid
    train_frames = []
    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(vid, fps, total_frames)

    for k in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (img_rows, img_cols), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            train_frames.append(gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

    train_input = np.array(train_frames)
    train_ipt = np.rollaxis(train_input, 2, 1)
    X_train.append(train_ipt)

print("Training videos reading done")

X_train_array = np.array(X_train)
num_samples = len(X_train_array)
train_labels = []
for i in range(42):
    for j in range(7):
        train_labels.append((i + 1) % 2)
trian_labels = np.asarray(train_labels).astype('float32')
X_train_array = X_train_array.reshape(num_samples * 7, 20, img_rows, img_cols, 1)
X_partial_train = X_train_array[70:]
X_val = X_train_array[:70]
val_labels = train_labels[:70]
X_partial_labels = train_labels[70:]
print("train sample numbers: ", num_samples)
print("train videos shape: ", X_train_array.shape)
# --------------------------------------------

# get the test videos frames into the numpy array
# output: X_test_array test_labels
# --------------------------------------------
X_test = []
listing2 = os.listdir('/content/drive/My Drive/dataset/RGB/test/')
listing2.sort()
print("Enter test videos directory")

for vid in listing2:
    vid = '/content/drive/My Drive/dataset/RGB/test/' + vid
    test_frames = []
    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(vid, fps, total_frames)

    for k in range(num_frames):
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
test_labels = []
for i in range(5):
    for j in range(7):
        if (i != 4):
            test_labels.append(1)
        else:
            test_labels.append(0)
test_labels = np.asarray(test_labels).astype('float32')
num_samples = len(X_test_array)
X_test_array = X_test_array.reshape(num_samples * 7, 20, img_rows, img_cols, 1)
print("test sample numbers: ", num_samples)
print("test videos shape: ", X_test_array.shape)

'''
# --------------------------------------------
# load the train and test labels
# output: train_labels, test_labels
# --------------------------------------------

train_labels = []
test_labels = []

listing3 = os.listdir("/Users/ningzewang/Desktop/dataset/labels/train/")
listing3.sort()
listing4 = os.listdir("/Users/ningzewang/Desktop/dataset/labels/test/")
listing4.sort()
print("Enter labels directory")

for label in listing3:
    a = np.loadtxt("/Users/ningzewang/Desktop/dataset/labels/train/" + label)
    train_labels.append(a)

for label in listing4:
    b = np.loadtxt("/Users/ningzewang/Desktop/dataset/labels/test/" + label)
    test_labels.append(b)

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
train_labels = np.array(train_labels).astype('float32')
test_labels = np.array(test_labels).astype('float32')

print("train_labels shape: ", train_labels.shape)
print("test_labels shape: ", test_labels.shape)

# --------------------------------------------
'''

# Neural network setting
# --------------------------------------------

X_train_array = X_train_array.astype('float32')
X_train_array -= np.mean(X_train_array)
X_train_array /= np.max(X_train_array)

'''
model_exist = os.path.exists('')
if model_exist:
    model = models.load_model('shake_detect.h5')
    print("model loaded")

else:
 '''
model = models.Sequential()

model.add(layers.Conv3D(32, (3, 3, 3), strides=(1, 1, 1), activation='relu', input_shape=(20, 320, 180, 1)))

model.add(layers.MaxPooling3D((2, 2, 2)))

model.add(layers.Conv3D(64, (3, 3, 3), strides=(1, 1, 1),activation='relu'))

model.add(layers.MaxPooling3D((1, 2, 2)))

model.add(layers.Reshape((224, -1)))

model.add(layers.LSTM(32, dropout=0.5, return_sequences=True))

model.add(layers.LSTM(32))


model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(X_partial_train,
                    X_partial_labels,
                    epochs=6,
                    batch_size=8,
                    validation_data=(X_val, val_labels))
results = model.predict_classes(X_test_array)
results = results.reshape((5, 7))
print(results)
a = model.evaluate(X_test_array, test_labels)
print(a)
model.save("/content/drive/My Drive/dataset/model/my_model.h5")
for k in range(5):
  json_data={"Tremor":[]}
  for i in range(7):
    json_data["Tremor"].append([i/25, int(results[k][i])])
  json_path = "/content/drive/My Drive/dataset/json/test" + str(k) + ".json"
  with open(json_path, 'w') as f:
    json.dump(json_data, f)
  time = range(1, 8)
  plt.figure()
  plt.plot(time, results[k],'bo')
  plt.xlabel("frames")
  plt.ylabel("class")
  plt.show()











