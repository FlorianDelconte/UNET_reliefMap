import numpy as np
import os
import csv
import sys
import skimage.io as io
import cv2

#import skimage.transform as transform
import skimage.transform as trans

#import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as img
import scipy.misc
from skimage import img_as_ubyte
#from sklearn.model_selection import train_test_split
from skimage import img_as_ubyte
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

	img = io.imread(file, as_gray = True)
	h=img.shape[0]
	hToPredict=img.shape[0]
	w=img.shape[1]
	wToPredict=img.shape[1]
	if h%model.numFilt != 0 or w%model.numFilt != 0:
		print("Image input dosn't not match model shape... Resize in nearest multiple of "+str(model.numFilt))
		hToPredict = model.numFilt * round(img.shape[0] / model.numFilt)
		wToPredict = model.numFilt * round(img.shape[1] / model.numFilt)
		img = cv2.resize(img, (wToPredict,hToPredict), interpolation =cv2.INTER_AREA)
		print(hToPredict,wToPredict)

	img = img / 255
	img = np.reshape(img,img.shape+(1,))
	img = np.reshape(img,(1,)+img.shape)
	print("max : ",np.max(img),"min : ", np.min(img))
	px 	= net.predict(img, verbose=1)
	print("max : ",np.max(px),"min : ", np.min(px))
	px	= px[0,:,:,0]
	px=(px*255).astype(np.uint8)

	if h!=hToPredict or w!=wToPredict :
		print("comeback in normal size")
		px = cv2.resize(px, (w,h), interpolation =cv2.INTER_AREA)

	io.imsave('{}/{}'.format(dir_output, path),px)
