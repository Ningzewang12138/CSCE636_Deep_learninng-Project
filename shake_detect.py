import tensorflow as tf
import os
import cv2
import numpy as np
from keras.applications.imagenet_utils import decode_predictions
from keras.models import load_model
from keras import models
from keras import layers
from keras import optimizers
from keras.layers import LSTM
import matplotlib.pyplot as plt
import json

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# --------------------------------------------
img_rows, img_cols, num_frames = 320, 180, 120

train_dataset_path = '/home/alex/Desktop/csce636/dataset/RGB/train/'
test_data_path = '/home/alex/Desktop/csce636/dataset/RGB/test/'
model_save_path = "/home/alex/Desktop/csce636/dataset/model/my_model.h5"
json_save_path = "/home/alex/Desktop/csce636/dataset/json/test/"
# Get the train videos frames into the numpy array
# output:
# 1.training group: X_partial_train  X_partial_labels
# 2.validation group: X_val  val_labels
# --------------------------------------------
X_train = []
listing1 = os.listdir(train_dataset_path)
listing1.sort()
listing1.sort(key=lambda x: int(x[:-4]))
print("Enter training Videos directory")

for vid in listing1:
    vid = train_dataset_path + vid
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
X_train_array = X_train_array.astype('float32')
X_train_array -= np.mean(X_train_array)
X_train_array /= np.max(X_train_array)
print(X_train_array.shape)
num_samples = len(X_train_array)
train_labels = []
for i in range(42):
    for j in range(6):
        train_labels.append((i + 1) % 2)
trian_labels = np.asarray(train_labels).astype('float32')
X_train_array = X_train_array.reshape(num_samples * 6, 20, img_rows, img_cols, 1)
X_partial_train = X_train_array[60:]
X_val = X_train_array[:60]
val_labels = train_labels[:60]
X_partial_labels = train_labels[60:]
print("train sample numbers: ", num_samples)
print("train videos shape: ", X_train_array.shape)

# --------------------------------------------
# use the provided video to test
# output: test_array test_labels
# --------------------------------------------
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







'''
# --------------------------------------------
# get the test videos frames into the numpy array
# output: X_test_array test_labels
# --------------------------------------------
X_test = []
listing2 = os.listdir('/home/alex/Desktop/csce636/dataset/RGB/test/')
listing2.sort()
print("Enter test videos directory")

for vid in listing2:
    vid = '/home/alex/Desktop/csce636/dataset/RGB/test/' + vid
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
X_test_array = X_test_array.astype('float32')
X_test_array -= np.mean(X_test_array)
X_test_array /= np.max(X_test_array)

test_labels = []
for i in range(5):
    for j in range(6):
        if i != 4:
            test_labels.append(1)
        else:
            test_labels.append(0)
test_labels = np.asarray(test_labels).astype('float32')
num_samples = len(X_test_array)
X_test_array = X_test_array.reshape(num_samples * 6, 20, img_rows, img_cols, 1)
print("test sample numbers: ", num_samples)
print("test videos shape: ", X_test_array.shape)

'''

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


'''
model_exist = os.path.exists('')
if model_exist:
    model = models.load_model('shake_detect.h5')
    print("model loaded")

else:
 '''
model = models.Sequential()

model.add(layers.Conv3D(32, (2, 3, 3), strides=(1, 1, 1), padding='same', activation='relu', input_shape=(20, 320, 180, 1)))

model.add(layers.MaxPooling3D((2, 2, 2)))

model.add(layers.Conv3D(64, (2, 3, 3), padding='same', activation='relu'))

model.add(layers.MaxPooling3D((2, 2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(X_partial_train,
                    X_partial_labels,
                    epochs=2,
                    batch_size=4,
                    validation_data=(X_val, val_labels))

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
