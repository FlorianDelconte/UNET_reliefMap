import numpy as np
import os
import csv
import sys
import skimage.io as io
import cv2

#import skimage.transform as transform
#import skimage.transform as trans

#import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as img
import scipy.misc

#from sklearn.model_selection import train_test_split

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from PIL import Image
import tensorflow as tf

import model
import cv2





net = model.unet()
net.load_weights(model.fullname)

if len(sys.argv) == 3:
	dir_input= sys.argv[1]
	dir_output=sys.argv[2]
else:
	dir_input 	= model.test_dir_input
	dir_output 	= model.test_dir_output


#for i in range(20):
for path in os.listdir(dir_input):
	if path == ".DS_Store": continue
	file 	= "{}/{}".format(dir_input, path)

	print(file)
	img = io.imread(file, plugin='matplotlib')
	print("shape input :",img.shape)
    #image dimension
	h=img.shape[0]
	w=img.shape[1]
    #image dimension to fit the model
	hToPredict=model.numFilt * round(h / model.numFilt)
	#print(hToPredict)
	wToPredict=model.numFilt * round(w / model.numFilt)
	print("shape for predict :",hToPredict,wToPredict)
	#print(img.shape)
	img 	= cv2.resize(img, (wToPredict,hToPredict), interpolation =cv2.INTER_AREA)
	#print(img.shape)
	img 	= np.array([img])
	#print(img.shape)
	img 	= np.reshape(img, [1,hToPredict,wToPredict, model.channels])
	img = tf.cast(img, tf.float32)
	img /= 255.0
	#img=tf.cast(img, tf.float32)
	px 		= net.predict(img, verbose=2)

	#np.set_printoptions(threshold=sys.maxsize)
	#resize de la sortie a la taille de l'image de base
	segmentation	= px[0,:,:,0]
	segmentation 	= cv2.resize(segmentation, (w, h), interpolation =cv2.INTER_NEAREST)


	segmentation = (segmentation * 255).astype(np.uint8)
	segmentation	=Image.fromarray(segmentation,"L")

	segmentation = segmentation.resize((w,h))

	segmentation.save('{}/{}'.format(dir_output, path))
