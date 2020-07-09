import numpy as np
import os
#import skimage.io as io
#import skimage.transform as trans
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
import tensorflow_addons as tfa
import tensorflow as tf

import os
#####################
# Global Parameters #
# ----------------- #
#####################
height		= 320
width   	=  320
channels 	= 1
numFilt 	= 32
batch_size		= 16
epochs			= 100
#steps_per_epoch 	= 17*15
#validation_steps	= 5*15
t=0.5

#name= "cross-validationv8/ktest_4.hdf5"


#train_folder	= "train"
#valid_folder	= "valid"
#test_folder		= "test"


#train_dir_input 	= os.path.join(os.getcwd(), '..','crossValidationv8','data_4', train_folder, 'input')
#valid_dir_input 	= os.path.join(os.getcwd(), '..','crossValidationv8','data_4', valid_folder, 'input')
#test_dir_input 		= os.path.join(os.getcwd(),'..','crossValidationv8','data_4', 'test', 'input')

#train_dir_output 	= os.path.join(os.getcwd(), '..','crossValidationv8','data_4', train_folder, 'output')
#valid_dir_output	= os.path.join(os.getcwd(), '..','crossValidationv8','data_4',valid_folder, 'output')
#test_dir_output 	= os.path.join(os.getcwd(),'..','crossValidationv8','data_4', 'test', 'output')

#save_dir 			= os.path.join(os.getcwd(),'model', 'save')

#fullname 			= "".join([save_dir, '/', name])





#########
# Model #
# ----- #
#########Zeros
def unet(pretrained_weights = None,input_size = (height, width, channels)):
	inputs = Input(input_size)
	conv1 = Conv2D(numFilt, 3, activation = tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(inputs)# activation = 'relu'###tf.nn.leaky_relu
	conv1 = Conv2D(numFilt, 3, activation = tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(numFilt*2, 3, activation = tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(pool1)
	conv2 = Conv2D(numFilt*2, 3, activation = tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(numFilt*4, 3, activation = tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(pool2)
	conv3 = Conv2D(numFilt*4, 3, activation = tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(numFilt*8, 3, activation = tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(pool3)
	conv4 = Conv2D(numFilt*8, 3, activation = tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(conv4)
	drop4 = Dropout(0.2)(conv4)

	pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)


	conv5 = Conv2D(numFilt*16,3, activation = tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(pool4)
	conv5 = Conv2D(numFilt*16, 3, activation = tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(conv5)
	drop5 = Dropout(0.2)(conv5)

	up6 = Conv2D(numFilt*8, 2, activation = tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
	merge6 = concatenate([drop4,up6], axis = 3)
	conv6 = Conv2D(numFilt*8, 3, activation = tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(merge6)
	conv6 = Conv2D(numFilt*8, 3, activation = tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(conv6)

	up7 = Conv2D(numFilt*4, 2, activation =tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
	merge7 = concatenate([conv3,up7], axis = 3)
	conv7 = Conv2D(numFilt*4, 3, activation = tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(merge7)
	conv7 = Conv2D(numFilt*4, 3, activation = tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(conv7)

	up8 = Conv2D(numFilt*2, 2, activation = tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
	merge8 = concatenate([conv2,up8], axis = 3)
	conv8 = Conv2D(numFilt*2, 3, activation = tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(merge8)
	conv8 = Conv2D(numFilt*2, 3, activation = tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(conv8)

	up9 = Conv2D(numFilt, 2, activation = tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
	merge9 = concatenate([conv1,up9], axis = 3)
	conv9 = Conv2D(numFilt, 3, activation = tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(merge9)
	conv9 = Conv2D(numFilt, 3, activation = tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(conv9)
	conv9 = Conv2D(2, 3, activation = tf.nn.leaky_relu, padding = 'same', kernel_initializer = 'he_normal')(conv9)
	conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

	model = Model(inputs = inputs, outputs = conv10)

	model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = [tf.keras.metrics.Precision(name='precision',thresholds=t),tf.keras.metrics.Recall(name='recall',thresholds=t),tfa.metrics.FBetaScore(num_classes=2, average="micro", threshold = t)])#[tf.keras.metrics.MeanIoU(num_classes=2)]['accuracy'],tf.keras.metrics.AUC(curve='PR')
	#binary_crossentropy
	return model
