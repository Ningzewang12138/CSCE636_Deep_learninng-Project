import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.applications.imagenet_utils import decode_predictions
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
import matplotlib.pyplot as plt
import json

# --------------------------------------------
img_rows, img_cols, num_frames = 320, 180, 149

# Get the train videos frames into the numpy array
# output: X_train_array
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

    while cap.isOpened():
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

print("train videos reading done")

X_train_array = np.array(X_train)
num_samples = len(X_train_array)
X_train_array = X_train_array.reshape(num_samples * num_frames, img_rows * img_cols)
print("train sample numbers: ", num_samples)
print("train videos shape: ", X_train_array.shape)
# --------------------------------------------

# get the test videos frames into the numpy array
# output: X_test_array
# --------------------------------------------
X_test = []
listing2 = os.listdir('/content/drive/My Drive/dataset/RGB/test/')
listing2.sort()
listing2.sort(key=lambda x: int(x[:-4]))
print("Enter test Videos directory")

for vid in listing2:
    vid = '/content/drive/My Drive/dataset/RGB/test/' + vid
    test_frames = []
    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(vid, fps, total_frames)

    while cap.isOpened():
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

num_samples = len(X_test_array)
X_test_array = X_test_array.reshape(num_samples * num_frames, img_rows * img_cols)
print("test sample numbers: ", num_samples)
print("test videos shape: ", X_test_array.shape)
# --------------------------------------------


# load the train and test labels
# output: train_labels, test_labels
# --------------------------------------------
train_labels = []
test_labels = []

listing3 = os.listdir("/content/drive/My Drive/dataset/labels/train/")
listing3.sort()
listing4 = os.listdir("/content/drive/My Drive/dataset/labels/test/")
listing4.sort()
print("Enter labels directory")

for label in listing3:
    a = np.loadtxt("/content/drive/My Drive/dataset/labels/train/" + label)
    train_labels.append(a)

for label in listing4:
    b = np.loadtxt("/content/drive/My Drive/dataset/labels/test/" + label)
    test_labels.append(b)

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
train_nums = train_labels.shape[0]
labels_nums = train_labels.shape[1]
test_nums = test_labels.shape[0]
train_labels = train_labels.reshape(( train_nums * labels_nums, ))
test_labels = test_labels.reshape((test_nums * labels_nums, ))
train_labels = np.array(train_labels).astype('float32')
test_labels = np.array(test_labels).astype('float32')

print("train_labels shape: ", train_labels.shape)
print("test_labels shape: ", test_labels.shape)
# --------------------------------------------


# Neural network setting
# --------------------------------------------

nb_pools = [3, 3, 3]

X_train_array = X_train_array.astype('float32')
X_train_array -= np.mean(X_train_array)
X_train_array /= np.max(X_train_array)

partial_X_train_array = X_train_array[447:]
X_val_array = X_train_array[:447]
partial_train_labels = train_labels[447:]
val_labels = train_labels[:447]


model_exist = os.path.exists('')
if model_exist:
    model = models.load_model('shake_detect.h5')
    print("model loaded")

else:
    model = models.Sequential()

    model.add(layers.Dense(512, activation='relu', input_shape = (img_rows * img_cols, )))

    model.add(layers.Dense(512,activation='relu'))    
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    history = model.fit(partial_X_train_array,
                        partial_train_labels,
                        epochs=8,
                        batch_size=512,
                        validation_data=(X_val_array, val_labels))
    results = model.predict_classes(X_test_array)
    results = results.reshape((test_nums, labels_nums))
    print(results)
    time = range(1,num_frames+1)
    for k in range(5):
      json_data={"Shake":[]}
      for i in range(labels_nums):
        json_data["Shake"].append([i/24,int(results[k][i])])
      json_path = "/content/drive/My Drive/dataset/json/test"+ str(k)+".json"
      with open(json_path ,'w') as f:
        json.dump(json_data, f)       
      plt.figure()
      plt.plot(time, results[k],'bo')
      plt.xlabel("frames")
      plt.ylabel("class")
      plt.show()