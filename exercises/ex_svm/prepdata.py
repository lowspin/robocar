import numpy as np
import pickle
from PIL import Image
import glob

images = []
labels = []

for filename in glob.glob('data/stop-sign/*.png'): #assuming gif
    im=Image.open(filename)
    imnp = np.array(im) #imnp = list(im.getdata())
    images.append(imnp)
    labels.append(1)

for filename in glob.glob('data/triangle-sign/*.png'): #assuming gif
    im=Image.open(filename)
    imnp = np.array(im) #imnp = list(im.getdata())
    images.append(imnp)
    labels.append(2)

for filename in glob.glob('data/no-signs/*.png'): #assuming gif
    im=Image.open(filename)
    imnp = np.array(im) #imnp = list(im.getdata())
    images.append(imnp)
    labels.append(0)

alldata = {
    "images" : images,
    "labels" : labels
}
#print len(alldata['images'])

with open('data.p', 'wb') as handle:
    pickle.dump(alldata, handle, protocol=2)

print 'data pickled in data.p'
