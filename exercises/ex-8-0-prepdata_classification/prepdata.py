import numpy as np
import pickle
from PIL import Image
import glob
from sklearn.model_selection import train_test_split

images = []
labels = []

datafolder = './'
for filename in glob.glob(datafolder+'/1-stop/*.png'):
    im=Image.open(filename)
    imnp = np.array(im)
    images.append(imnp)
    labels.append(1)

for filename in glob.glob(datafolder+'/2-warn/*.png'):
    im=Image.open(filename)
    imnp = np.array(im)
    images.append(imnp)
    labels.append(2)

for filename in glob.glob(datafolder+'/0-nosign/*.png'):
    im=Image.open(filename)
    imnp = np.array(im)
    images.append(imnp)
    labels.append(0)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.20, shuffle=True, random_state=42)

traindata = {
    "images" : X_train,
    "labels" : y_train
}

testdata = {
    "images" : X_test,
    "labels" : y_test
}

# save training data
with open('data.p', 'wb') as handle:
    pickle.dump(traindata, handle, protocol=2)
print 'saved ' + str(len(traindata['images'])) + ' in data.p'

#save testing data
with open('test.p', 'wb') as handle:
    pickle.dump(testdata, handle, protocol=2)
print 'saved ' + str(len(testdata['images'])) + ' in test.p'
