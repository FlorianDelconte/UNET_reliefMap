########################################################################################
#This script is use to cut test image insmaller image                                  #
########################################################################################

#!/usr/bin/python3
import sys, getopt
import os
import cv2
import numpy as np
from PIL import Image
import cv2 as cv

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
    list_cropped_img=crop(img)
    num_image=0
    for im_crop in list_cropped_img :
        cropped_imHeight = im_crop.shape[0]
        cropped_imWidth = im_crop.shape[1]
        print(cropped_imHeight,cropped_imWidth)
        path=os.path.join(outputDirectory,imageName+"-%s.png" % num_image)
        cv2.imwrite( path, im_crop );
        num_image+=1
    #img=resizeImg(img)
    #crop(img)
    #for k,piece in enumerate(crop(img),0):
        #_,piece = cv2.threshold(piece,127,255,cv2.THRESH_BINARY)
    #    path=os.path.join(outputDirectory,imageName+"-%s.png" % k)
    #    cv2.imwrite( path, piece );




#resize img on patchDim multiple
def resizeImg(img):
    imHeight = img.shape[0]
    imWidth = img.shape[1]
    print("base width : ",imWidth)
    print("base Height : ",imHeight)
    unlargeXparam=patchdim-(imWidth%patchdim)
    unlargeYparam=patchdim-(imHeight%patchdim)
    img=cv2.resize(img,(imWidth+unlargeXparam,imHeight+unlargeYparam),cv2.INTER_NEAREST)
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
    list_img=[]
    imgHeight = im.shape[0]
    imgWidth = im.shape[1]
    shift_left=patchdim-(imgWidth%patchdim)
    shift_top=patchdim-(imgHeight%patchdim)
    print(imgHeight,imgWidth)
    print(shift_top,shift_left)


    for i in range(0, imgHeight, patchdim):
        for j in range(0, imgWidth, patchdim):
            print(i,j)
            topleftCornerI=i
            topleftCornerJ=j
            if(topleftCornerI+patchdim > imgHeight):
                print("dépasse au bot")
                topleftCornerI=topleftCornerI-shift_top
            if(topleftCornerJ+patchdim > imgWidth):
                print("dépasse a right")
                topleftCornerJ=topleftCornerJ-shift_left
            #img[y:y+h, x:x+w]
            list_img.append(im[topleftCornerI:topleftCornerI+patchdim,topleftCornerJ:topleftCornerJ+patchdim])
    return list_img

if __name__ == "__main__":
   main(sys.argv[1:])
