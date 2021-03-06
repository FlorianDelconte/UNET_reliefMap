import numpy as np
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import csv
import skimage.io as io
import skimage.transform as transform
#import skimage.transform as trans
import numpy as np
import matplotlib.image as img

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from tensorflow.keras.utils import to_categorical
from random_eraser import get_random_eraser  # added
from matplotlib import pyplot as plt
import model
import data
import math
# Parameters Input
numChannels = model.channels
height  = model.height
width   = model.width
batch_size = model.batch_size
epochs = model.epochs

# Folders Train, Valid, Test and Save
train_dir 	= os.path.join(os.getcwd(), '..','repartitionData9', 'train')
valid_dir   = os.path.join(os.getcwd(), '..','repartitionData9', 'valid')

save_dir        = os.path.join(os.getcwd(),'model', 'save','repartition-data9','TEST_article')
modelname       ="leakyReLu-v4-E34"
fullname        = "".join([save_dir, '/'])+modelname+".hdf5"

input_folder    = 'input'
label_folder    = 'output'

steps_per_epoch=(math.ceil((len([name for name in os.listdir(train_dir+"/input") if os.path.join(train_dir+"/input", name)]))/batch_size))*3
validation_steps=(math.ceil((len([name for name in os.listdir(valid_dir+"/input") if os.path.join(valid_dir+"/input", name)]))/batch_size))*3

#Augmentated Data Operation
data_gen_args = dict(rescale = 1.0 / 255,
                     rotation_range=20,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     zoom_range=0.3,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect',
                     preprocessing_function=get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3,
                   v_l=0, v_h=0, pixel_level=False))


# Create network
__generators    = data.trainset_generator(model.batch_size, train_dir, input_folder, label_folder,
                                 data_gen_args, save_to_dir=None)#"../crossValidationv3/data4_1st/visu_augmented_data"

__validator     = data.trainset_generator(model.batch_size, valid_dir, input_folder, label_folder,
                                          data_gen_args, save_to_dir=None)#"../crossValidationv3/data4_1st/visu_augmented_data"

model_checkpoint = ModelCheckpoint(fullname,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True)
net = model.unet()
net.summary()
try:
  net.load_weights(fullname)
except IOError:
  pass

history=net.fit(__generators,
        steps_per_epoch   = steps_per_epoch,
        validation_data   = __validator,
        validation_steps  = validation_steps,
        epochs=model.epochs,
        callbacks=[model_checkpoint])

plt.plot(history.history['precision'],'b', label='precision')
plt.plot(history.history['val_precision'],'b--',label = 'val_precision')
plt.plot(history.history['recall'],'g', label='recall')
plt.plot(history.history['val_recall'],'g--', label = 'val_recall')
plt.plot(history.history['fbeta_score'],'r', label='f1')
plt.plot(history.history['val_fbeta_score'],'r--', label = 'val_f1')
plt.plot(history.history['loss'], 'k',label='loss')
plt.plot(history.history['val_loss'],'k--', label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Measures')
plt.ylim([0.0, 1])
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=4)
plt.savefig("".join([save_dir, '/', modelname+'.png']))
plt.clf()
