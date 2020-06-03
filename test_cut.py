########################################################################################
#This script is use to cut test image insmaller image                                  #
########################################################################################

#!/usr/bin/python3
import sys, getopt
import os
import cv2
import numpy as np

#dimension of mini image 
patchdim=300

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
    
    #relief image
    img=cv2.imread(inputImage)
    print('size image :'+str(img.shape))
    height = img.shape[0]
    width = img.shape[1]
    crop_img = np.zeros((patchdim,patchdim,3), np.uint8)

    for row in range(0, height, patchdim):
        for col in range(0, width, patchdim):
            print(row,col)
            crop_img = img[row:row+patchdim, col:col+patchdim]
            cv2.imwrite(outputDirectory+imageName+"_"+str(row)+"-"+str(col)+".jpg", crop_img) 
    

if __name__ == "__main__":
   main(sys.argv[1:])
