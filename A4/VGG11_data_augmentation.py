import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import scipy
from PIL import Image, ImageFilter
import random

random.seed(69) # initialize seed for reproducability
  
batch_size = 256
num_classes = 10
epochs = 5

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# resize to 32x32, and also augment data by randomly blurring
# and rotating some of the dataset
fraction_to_rotate = 0.5
fraction_to_blur = 0.5
xreal=[]
rotations = [-45,-40,-35,-30,-25,-20,-15,-10,-5,0,5,10,15,20,25,30,35,40,45]
blurs = [0,1,2,3,4,5,6]

for x in x_train:
	x = scipy.misc.imresize(x, (32, 32))

	if random.random() >= fraction_to_rotate:
		# choose a random rotation degree
		rand_rot = rotations[random.randint(0, len(rotations)-1)]
		# rotate the image
		x = np.array(Image.fromarray(x).rotate(rand_rot))

	if random.random() >= fraction_to_blur:
		# choose a random blur radius
		rand_blur = blurs[random.randint(0, len(blurs)-1)]
		# blur the images
		x = np.array(Image.fromarray(x).filter(ImageFilter.GaussianBlur(rand_blur)))

	xreal.append(x)

x_train = np.asarray(xreal)


# resize to 32x32, and also create rotated, blurred test sets
xreal=[]
x_test_rotated = [[] for x in range(0,len(rotations))]
x_test_blurred = [[] for x in range(0,len(blurs))]

for i, x in enumerate(x_test):
	x = scipy.misc.imresize(x, (32, 32))
	xreal.append(x)

	for j, rotation in enumerate(rotations):
		x_test_rotated[j].append(np.array(Image.fromarray(x).rotate(rotation)))

	for j, blur in enumerate(blurs):
		x_test_blurred[j].append(np.array(Image.fromarray(x).filter(ImageFilter.GaussianBlur(blur))))

			   
x_test = np.asarray(xreal)

# format input image dimensions to play nice with keras,
# taken from https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
img_rows, img_cols = 32, 32

if K.image_data_format() == 'channels_first':
	x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)

	for i in range(0, len(x_test_rotated)):
		x_test_rotated[i] = np.asarray(x_test_rotated[i])
		x_test_rotated[i] = x_test_rotated[i].reshape(x_test_rotated[i].shape[0], 1, img_rows, img_cols)
	
	for i in range(0, len(x_test_blurred)):
		x_test_blurred[i] = np.asarray(x_test_blurred[i])
		x_test_blurred[i] = x_test_blurred[i].reshape(x_test_blurred[i].shape[0], 1, img_rows, img_cols)

	input_shape = (1, img_rows, img_cols)
else:
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

	for i in range(0, len(x_test_rotated)):
		x_test_rotated[i] = np.asarray(x_test_rotated[i])
		x_test_rotated[i] = x_test_rotated[i].reshape(x_test_rotated[i].shape[0], img_rows, img_cols, 1)
	
	for i in range(0, len(x_test_blurred)):
		x_test_blurred[i] = np.asarray(x_test_blurred[i])
		x_test_blurred[i] = x_test_blurred[i].reshape(x_test_blurred[i].shape[0], img_rows, img_cols, 1)

	input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
# taken from https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# Build ConvNet Configuration A, as described in https://arxiv.org/pdf/1409.1556.pdf
model = Sequential()

# Convolutional Layers
model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))

# Maxpool, Fully Connected Layers & Softmax
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
	# we use Adam, with a much smaller learning rate, instead of SGD as specified in the original paper
	# in order to get the model to converge to a decent accuracy within 5 epochs
	optimizer=keras.optimizers.Adam(lr=0.00001),
	metrics=['accuracy']
)

model.fit(x_train, y_train,
	batch_size=batch_size, # 256, as specified in the paper
	epochs=epochs, # 5, as specified in the assignment
	verbose=1,
	validation_data=(x_test, y_test)
)


score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy, regular test set:', score[1])

rot_accs=[]
blur_accs=[]

for i, rot in enumerate(rotations):
	score = model.evaluate(x_test_rotated[i], y_test, verbose=0)
	print('Test accuracy, rotated ', rot, ' degrees:', score[1])
	rot_accs.append(score[1])
  
for i, blur in enumerate(blurs):
	score = model.evaluate(x_test_blurred[i], y_test, verbose=0)
	print('Test accuracy, blurred ', blur, ' degrees:', score[1])
	blur_accs.append(score[1])

print(rot_accs)
print(blur_accs)