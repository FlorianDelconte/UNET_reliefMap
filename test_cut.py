########################################################################################
#This script is use to cut test image insmaller image                                  #
########################################################################################

#!/usr/bin/python3
import sys, getopt
import os
import cv2
import numpy as np
from PIL import Image


#dimension of mini image
patchdim=320

def main(argv):

    inputImage = ''
    outputDirectory = ''

    try:
        opts, args = getopt.getopt(argv,"hi:g:o:",["iImage=","odir="])
    except getopt.GetoptError:
        print (' -i <inputImage>  -o <outputDirectory>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('defect_detect.py -i <inputImage> -o <outputDirectory>')
            sys.exit()
        elif opt in ("-i", "--iImage"):
            inputImage = arg
        elif opt in ("-o", "--odir"):
            outputDirectory = arg
    imageName=os.path.splitext(os.path.basename(inputImage))[0]


    print ('Path to image is ', inputImage)
    print ('Output directory is ', outputDirectory)
    print('input image name :',imageName)

    img = cv2.imread(inputImage,0)
    img=resizeImg(img)
    #crop(img)
    for k,piece in enumerate(crop(img),0):
        path=os.path.join(outputDirectory,imageName+"-%s.png" % k)
        cv2.imwrite( path, piece );




#resize img on patchDim multiple
def resizeImg(img):
    imHeight = img.shape[0]
    imWidth = img.shape[1]
    print("base width : ",imWidth)
    print("base Height : ",imHeight)
    unlargeXparam=patchdim-(imWidth%patchdim)
    unlargeYparam=patchdim-(imHeight%patchdim)
    img=cv2.resize(img,(imWidth+unlargeXparam,imHeight+unlargeYparam))
    reimHeight = img.shape[0]
    reimWidth = img.shape[1]
    print("resized width : ",reimWidth)
    print("resized Height : ",reimHeight)
    return img

#not yet finish
#add a cyl img on X axis to complete patchDim multiple
def unlargeImg(img):
    return img

#crop image in 304*304 thumbnail
def crop(im):
    imgHeight = im.shape[0]
    imgWidth = im.shape[1]
    for i in range(imgHeight//patchdim):
        for j in range(imgWidth//patchdim):
            #print("X : ",j*patchdim,(j+1)*patchdim,"Y : ",i*patchdim,(i+1)*patchdim)
            yield im[i*patchdim:(i+1)*patchdim,j*patchdim:(j+1)*patchdim]



if __name__ == "__main__":
   main(sys.argv[1:])
