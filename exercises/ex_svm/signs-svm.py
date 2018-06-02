import pickle
import  matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from sklearn import svm
import random
import pandas as pd

# Load pickled data
data_file = 'data.p'
with open(data_file, mode='rb') as f:
    data_all = pickle.load(f)

X_all, y_all = data_all['images'], data_all['labels']
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=42)
assert(len(X_train) == len(y_train))
assert(len(X_test) == len(y_test))
print 'Original training samples: ' + str(len(X_train))
# convert to numpy
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# data augmentation
X_rot = []
y_rot = []
for X,y in zip(X_train,y_train):
    imrot = np.rot90(X,2)
    X_rot.append(imrot)
    y_rot.append(y)
X_train = np.append(X_train, X_rot, axis=0)
y_train = np.append(y_train, y_rot)

# check data
# Number of training examples# Numbe
n_train = X_train.shape[0]
# Number of testing examples.
n_test = X_test.shape[0]
# Shape of traffic sign image
image_shape = X_train[0].shape
# How many unique classes/labels there are in the dataset.
classes = np.unique(y_train)

print "Final training samples: " + str(n_train)
print "Final testing samples: " + str(n_test)
print "Image data shape: " + str(image_shape)
print "Classes: " + str(classes) + "\n"

# sample images
def get_sample_images(X, y, plotfig=False):
    labels = y.tolist()
    unique_labels = set(labels)
    X_samples = []
    y_samples = []
    for label in unique_labels:
        # Pick the first image for each label.
        image = X[labels.index(label)]
        X_samples.append(image)
        y_samples.append(label)

    X_samples = np.asarray(X_samples)
    y_samples = np.asarray(y_samples)

    if plotfig:
        # plot sample images
        plt.figure(figsize=(15, 15))
        for i in range(X_samples.shape[0]):
            image = X_samples[i].squeeze()
            plt.subplot(8, 8, i+1)  # A grid of 8 rows x 8 columns
            plt.axis('off')
            plt.title(y_samples[i] + "(" + str(labels.count(i)) +")")
            plt.imshow(image)
        plt.show()
    return X_samples, y_samples
X_samples, y_samples = get_sample_images(X_train, y_train, False) #True)

# Pre-process
def preprocess_grayscale(X):

    ## Grayscale
    X_gray = np.dot(X[...][...,:3],[0.299,0.587,0.114])

    ## Histogram Equalization (Improve contrast)
    X_gray_eq = np.zeros(shape=X_gray.shape)
    for i in range(X_gray.shape[0]):
        img = cv2.equalizeHist(X_gray[i].squeeze().astype(np.uint8))
        X_gray_eq[i] = img

    ## scale to [0,1]
    X_gray_eq_scale = np.divide(X_gray_eq,255.0)

    ## expand to fit dimensions
    X_prep = np.expand_dims(X_gray_eq_scale, axis=3)

    return X_prep

def equalize_Y_channel(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_output

def preprocess_rgb(X):

    ## Histogram Equalization (Improve contrast)
    X_eq = np.zeros(shape=X.shape)
    for i in range(X.shape[0]):
        img = X[i].squeeze().astype(np.uint8)
        X_eq[i] = equalize_Y_channel(img)

    ## scale to [0,1]
    X_eq_scale = np.divide(X_eq,255.0)

    return X_eq_scale

#Number of channels to process - 1:grayscale, 3:RGB
proc_num_channels = 3

if (proc_num_channels==1):
    ## Pre-Process: Grayscale
    X_train_prep = preprocess_grayscale(X_train)
    X_test_prep = preprocess_grayscale(X_test)
elif (proc_num_channels==3):
    ## Pre-Process: RGB
    X_train_prep = preprocess_rgb(X_train)
    X_test_prep = preprocess_rgb(X_test)
else:
    print('Please select ONLY 1 or 3 channels')

# check quality after pre-processing
check_quality = False
if (check_quality):
    index = random.randint(0, len(X_train))
    print("Random Test for {0}".format(y_train[index]))
    plt.figure(figsize=(5,5))

    plt.subplot(1, 2, 1)
    plt.imshow(X_train[index].squeeze())
    plt.title("Before")

    plt.subplot(1, 2, 2)
    if (proc_num_channels==1):
        plt.imshow(X_train_prep[index].squeeze(), cmap="gray")
    else:
        plt.imshow(X_train_prep[index].squeeze())
    plt.title("After")
    plt.show()

# replace data with preprocessed images
X_train = X_train_prep
X_test = X_test_prep

print('All pre-processing done.')

# Support Vector Machine
def num_white_pixels(img):
    img = np.array(img, dtype=np.float32)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(gray, 0.8, 1., cv2.THRESH_BINARY)
    return np.sum(img_bin)/(50*50)
def num_red_pixels(img):
    img = np.array(img, dtype=np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ret, img_bin = cv2.threshold(hsv[:, :, 1], .4, 1., cv2.THRESH_BINARY)
    return np.sum(img_bin)/(50*50)
def num_edges(img):
    img = np.array(img, dtype=np.float32)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.1)
    return np.sum(dst>0.01*dst.max())

# Extract features
Features_train = []
features = []
f0 = []
f1 = []
f2 = []
for x,y in zip(X_train,y_train):
    features = [num_edges(x),num_red_pixels(x)]#,num_white_pixels(x)]
    Features_train.append(features)
    if y==0:
        f0.append(features)
    elif y==1:
        f1.append(features)
    else:
        f2.append(features)

clf = svm.SVC() #(kernel='rbf')
clf.fit(Features_train,y_train)

with open('model_svm.p', 'wb') as handle:
    pickle.dump(clf, handle, protocol=2)

print('SVM training done.')

# visualize
visualize = True
if visualize:
    plt.scatter(np.asarray(f0)[:,0], np.asarray(f0)[:,1], color='g', marker='o')
    plt.scatter(np.asarray(f1)[:,0], np.asarray(f1)[:,1], color='r', marker='*')
    plt.scatter(np.asarray(f2)[:,0], np.asarray(f2)[:,1], color='b', marker='^')
    plt.show()

# test
right=0
wrong=0
for x,y in zip(X_test,y_test):
    features = [num_edges(x),num_red_pixels(x)]#,num_white_pixels(x)]
    testvec = np.asarray(features).reshape(1,-1)
    clf.predict(testvec)
    res = clf.predict(testvec)
    if res[0]==y:
        right = right + 1
    else:
        wrong = wrong + 1
        print str(res[0]) + " <- " + str(y)
print 'right = ' + str(right) + ' wrong = ' + str(wrong)
print 'accuracy = ' + str(100.*right/(right+wrong)) + ' %'
