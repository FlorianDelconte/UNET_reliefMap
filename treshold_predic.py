########################################################################################
#This script is use to find the good treshold on predicted imageName                   #
#                  I:Grayscale image O:binary image                                    #
########################################################################################
#!/usr/bin/python3
import sys, getopt
import os
import cv2
import cv2
import numpy as np
from matplotlib import pyplot as plt

def main(argv):
    inputImage = ''
    outputDirectory = ''
    tresh_value=0
    try:
        opts, args = getopt.getopt(argv,"hi:o:t:",["iImage=","odir=","treshVal="])
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
        elif opt in ("-t", "--treshVal"):
            tresh_value = int(arg)


    imageName=os.path.splitext(os.path.basename(inputImage))[0]
    img=cv2.imread(inputImage,0)
    ret,thresh = cv2.threshold(img,tresh_value,255,cv2.THRESH_BINARY)
    cv2.imwrite(outputDirectory+imageName+".jpg", thresh)
    #uncomment to test different treshold type
    #displayTresholdImage(img)

def displayTresholdImage(img):
    # Apply the different thresholds
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh4 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, thresh5 = cv2.threshold(img, 0, 255, cv2.THRESH_TRIANGLE)

    titles = ['Original Image', 'BINARY',
          'TRUNC', 'TOZERO', "Otsu's Thresholding","triangle"]
    images = [img, thresh1, thresh2, thresh3, thresh4,thresh5]

    for i in range(len(titles)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

if __name__ == "__main__":
   main(sys.argv[1:])
