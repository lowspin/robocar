import tensorflow.contrib.keras as keras
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
# from keras.models import Sequential
# from keras.layers import Convolution2D, MaxPooling2D
# from keras.layers import Dense, Dropout, Activation, Flatten

def build_model():

	model = keras.models.Sequential()

	model.add(keras.layers.Convolution2D(32, (3, 3), 3, activation='relu', 
										 input_shape=(50,50,3)))
	model.add(keras.layers.Convolution2D(32, (3, 3), 3, activation='relu',))
	model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
	model.add(keras.layers.Dropout(0.25))

	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(128, activation='relu'))
	model.add(keras.layers.Dense(128, activation='relu'))
	model.add(keras.layers.Dropout(0.5))
	model.add(keras.layers.Dense(3, activation='softmax'))

	opt = keras.optimizers.RMSprop(lr=0.001, decay=1e-6)

	# AdamOp = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

	model.compile(loss='categorical_crossentropy',
				  optimizer=opt,
				  metrics=['accuracy'])
	return model

def to_one_hot(labels):
	one_hot = np.zeros((labels.size, labels.max()+1))
	one_hot[np.arange(labels.size),labels] = 1
	return one_hot

# Load pickled data
data_all = pickle.load(open('data.p', 'rb'))

X_all, y_all = data_all['images'], data_all['labels']
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, random_state=42)
assert(len(X_train) == len(y_train))
assert(len(X_test) == len(y_test))
print('Original training samples: ' + str(len(X_train)))
# convert to numpy
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# data augmentation
X_rot = []
y_rot = []
for X,y in zip(X_train,y_train):
	for r in range(1,4):
		imrot = np.rot90(X,r)
		X_rot.append(imrot)
		y_rot.append(y)
X_train = np.append(X_train, X_rot, axis=0)
y_train = np.append(y_train, y_rot)


class TestCallback(keras.callbacks.Callback):
	def __init__(self, test_data):
		self.test_data = test_data

	def on_epoch_end(self, epoch, logs={}):
		x, y = self.test_data
		loss, acc = self.model.evaluate(x, y, verbose=0)
		print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


# X_train = np.transpose(X_train, (0, 3, 2, 1))
# X_test = np.transpose(X_test, (0, 3, 2, 1))

# check data
# Number of training examples# Numbe
n_train = X_train.shape[0]
# Number of testing examples.
n_test = X_test.shape[0]
# Shape of traffic sign image
image_shape = X_train[0].shape
# How many unique classes/labels there are in the dataset.
classes = np.unique(y_train)
print('Images loaded.')
print("Training samples: " + str(n_train))
print("Testing samples: " + str(n_test))
print("Image data shape: " + str(image_shape))
print("Classes: " + str(classes) + "\n")



y_train = to_one_hot(y_train)
y_test = to_one_hot(y_test)

print(y_train)


model = build_model()

def get_model_memory_usage(batch_size, model):
    import numpy as np
    

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([keras.backend.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([keras.backend.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

print "Memory Usage:"

model.fit(X_train, y_train, batch_size=32, epochs=250, verbose=1, callbacks=[TestCallback((X_test, y_test))])
score = model.evaluate(X_test, y_test, verbose=0)


model.save("/Users/timplump/Dropbox (MIT)/MIT/Junior/SummerClass/test_cnn.h5")
