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

# print get_model_memory_usage(32, model)





while True:
	model.predict(X_train[:1])

model.fit(X_train, y_train, batch_size=32, epochs=10000000, verbose=1, callbacks=[TestCallback((X_test, y_test))])
score = model.evaluate(X_test, y_test, verbose=0)

# Placeholder variable for the input images
# x = tf.placeholder(tf.float32, shape=[None, 28*28], name='X')
# # Reshape it into [num_images, img_height, img_width, num_channels]
# x_image = tf.reshape(x, [-1, 28, 28, 1])

# # Placeholder variable for the true labels associated with the images
# y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
# y_true_cls = tf.argmax(y_true, dimension=1)


# def new_conv_layer(input, num_input_channels, filter_size, num_filters, name):
# 	with tf.variable_scope(name) as scope:
# 		# Shape of the filter-weights for the convolution
# 		shape = [filter_size, filter_size, num_input_channels, num_filters]

# 		# Create new weights (filters) with the given shape
# 		weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))

# 		# Create new biases, one for each filter
# 		biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

# 		# TensorFlow operation for convolution
# 		layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

# 		# Add the biases to the results of the convolution.
# 		layer += biases
		
# 		return layer, weights


# def new_pool_layer(input, name):
	
# 	with tf.variable_scope(name) as scope:
# 		# TensorFlow operation for convolution
# 		layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		
# 		return layer


# def new_relu_layer(input, name):
	
# 	with tf.variable_scope(name) as scope:
# 		# TensorFlow operation for convolution
# 		layer = tf.nn.relu(input)
		
# 		return layer


# defdef  new_fc_layernew_fc_ (input, num_inputs, num_outputs, name):
	
# 	with tf.variable_scope(name) as scope:

# 		# Create new weights and biases.
# 		weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
# 		biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
		
# 		# Multiply the input and weights, and then add the bias-values.
# 		layer = tf.matmul(input, weights) + biases
		
# 		return layer

# # Convolutional Layer 1
# layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=1, filter_size=5, num_filters=6, name ="conv1")

# # Pooling Layer 1
# layer_pool1 = new_pool_layer(layer_conv1, name="pool1")

# # RelU layer 1
# layer_relu1 = new_relu_layer(layer_pool1, name="relu1")

# # Convolutional Layer 2
# layer_conv2, weights_conv2 = new_conv_layer(input=layer_relu1, num_input_channels=6, filter_size=5, num_filters=16, name= "conv2")

# # Pooling Layer 2
# layer_pool2 = new_pool_layer(layer_conv2, name="pool2")

# # RelU layer 2
# layer_relu2 = new_relu_layer(layer_pool2, name="relu2")

# # Flatten Layer
# num_features = layer_relu2.get_shape()[1:4].num_elements()
# layer_flat = tf.reshape(layer_relu2, [-1, num_features])

# # Fully-Connected Layer 1
# layer_fc1 = new_fc_layer(layer_flat, num_inputs=num_features, num_outputs=128, name="fc1")

# # RelU layer 3
# layer_relu3 = new_relu_layer(layer_fc1, name="relu3")

# # Fully-Connected Layer 2
# layer_fc2 = new_fc_layer(input=layer_relu3, num_inputs=128, num_outputs=10, name="fc2")

# # Use Softmax function to normalize the output
# with tf.variable_scope("Softmax"):
# 	y_pred = tf.nn.softmax(layer_fc2)
# 	y_pred_cls = tf.argmax(y_pred, dimension=1)



# # Use Adam Optimizer
# with tf.name_scope("optimizer"):
# 	optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)



# # Initialize the FileWriter
# writer = tf.summary.FileWriter("Training_FileWriter/")
# writer1 = tf.summary.FileWriter("Validation_FileWriter/")



# # Add the cost and accuracy to summary# Add th 
# tf.summary.scalar('loss', cost)
# tf.summary.scalar('accuracy', accuracy)

# # Merge all summaries together
# merged_summary = tf.summary.merge_all()

# num_epochs = 100
# batch_size = 100



# with tf.Session() as sess:
# 	# Initialize all variables
# 	sess.run(tf.global_variables_initializer())
	
# 	# Add the model graph to TensorBoard
# 	writer.add_graph(sess.graph)
	
# 	# Loop over number of epochs
# 	for epoch in range(num_epochs):
		
# 		start_time = time.time()
# 		train_accuracy = 0
		
# 		for batch in range(0, int(len(data.train.labels)/batch_size)):
			
# 			# Get a batch of images and labels
# 			x_batch, y_true_batch = data.train.next_batch(batch_size)
			
# 			# Put the batch into a dict with the proper names for placeholder variables
# 			feed_dict_train = {x: x_batch, y_true: y_true_batch}
			
# 			# Run the optimizer using this batch of training data.
# 			sess.run(optimizer, feed_dict=feed_dict_train)
			
# 			# Calculate the accuracy on the batch of training data
# 			train_accuracy += sess.run(accuracy, feed_dict=feed_dict_train)
			
# 			# Generate summary with the current batch of data and write to file
# 			summ = sess.run(merged_summary, feed_dict=feed_dict_train)
# 			writer.add_summary(summ, epoch*int(len(data.train.labels)/batch_size) + batch)
		
		  
# 		train_accuracy /= int(len(data.train.labels)/batch_size)
		
# 		# Generate summary and validate the model on the entire validation set
# 		summ, vali_accuracy = sess.run([merged_summary, accuracy], feed_dict={x:data.validation.images, y_true:data.validation.labels})
# 		writer1.add_summary(summ, epoch)
		

# 		end_time = time.time()
		
# 		print("Epoch "+str(epoch+1)+" completed : Time usage "+str(int(end_time-start_time))+" seconds")
# 		print("\tAccuracy:")
# 		print ("\t- Training Accuracy:\t{}".format(train_accuracy))
# 		print ("\t- Validation Accuracy:\t{}".format(vali_accuracy))





