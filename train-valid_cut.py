########################################################################################
#This script is use to extract 300*300 mini image around center                        #
#  of connected component in groundtruthimage                                          #
########################################################################################

#!/usr/bin/python3
import sys, getopt
import os
import cv2
import numpy as np
#ratio for distribution in train dirs
ratio=70
#dimension of mini image
patchdim=320
#valid input path
validInputPath="valid/input/"
#valid output validInputPath
validOutputPath="valid/output/"
#train input path
trainInputPath="train/input/"
#train output path
trainOutputPath="train/output/"
def main(argv):
    imgPath = ''
    gtPath = ''
    outputDirectory = ''
    try:
        opts, args = getopt.getopt(argv,"hi:g:o:",["iImage=","IGT=","odir="])
    except getopt.GetoptError:
        print (' -i <inputImage> -g <inputGT> -o <outputDirectory>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('defect_detect.py -i <inputImage> -g <inputGT> -o <outputDirectory>')
            sys.exit()
        elif opt in ("-i", "--iImage"):
            imgPath = arg
        elif opt in ("-g", "--IGT"):
            gtPath = arg
        elif opt in ("-o", "--odir"):
            outputDirectory = arg

    print ('Path to image is ', imgPath)
    print ('Path to groundTruth is ', gtPath)
    print ('Output directory is ', outputDirectory)
    #cut img input in imagette (-patchdim-*-patchdim-) centered around barycentre of connected component
    #posEx,posLab=cutPositiveExample(imgPath,gtPath)
    cutNegativeExample(imgPath,gtPath)
    #########################################################################################################
    #                     DISTRIBUTE DATA IN TRAIN AND VALID DIRECTORY                                      #
    #########################################################################################################
    '''#extract image name to write file
    imageName=os.path.splitext(os.path.basename(imgPath))[0]
    #extract gt name to write file
    gtName=os.path.splitext(os.path.basename(gtPath))[0]
    #list of exemple and label need to have same size
    assert(len(posEx)==len(posLab))
    #compute how many example+label need to be distribute in training directory
    tresholdRes=round((len(posEx)/100) * ratio)
    print(str(ratio)+" % = "+str(tresholdRes)+" imagettes in training directory")
    #distribute positive label
    for i in range(len(posEx)):
        if i<=tresholdRes :
            cv2.imwrite(outputDirectory+trainInputPath+imageName+"_"+str(i)+".jpg", posEx[i])
            cv2.imwrite(outputDirectory+trainOutputPath+gtName+"_"+str(i)+".jpg", posLab[i])
        else :
            cv2.imwrite(outputDirectory+validInputPath+imageName+"_"+str(i)+".jpg", posEx[i])
            cv2.imwrite(outputDirectory+validOutputPath+gtName+"_"+str(i)+".jpg",  posLab[i])'''

def cutNegativeExample(imgPath,gtPath):
    #list of negative exemple
    negativeExample = []
    #list of negative label
    negativelabel = []
    #relief image
    image=cv2.imread(imgPath,0)
    #groundtruth image
    gt = cv2.imread(gtPath,0)
    #size gt and exemple
    h,w = gt.shape
    #extends gt and exemple (like a cylinder)
    gt_letfPart=gt[0:h,0:round(w/2)]
    gt_rightPart=gt[0:h,round(w/2):w]
    gt_extend = cv2.hconcat([gt_rightPart,gt,gt_letfPart])
    h_ext,w_ext=gt_extend.shape
    for i in range(patchdim,h_ext-patchdim):
        for j in range(patchdim,w_ext-patchdim):
            negEx=gt_extend[i-(patchdim//2):i+(patchdim//2),j-(patchdim//2):j+(patchdim//2)]
            if(255 in negEx):
                continue
            else:
                gt_extend[i-(patchdim//2):i+(patchdim//2),j-(patchdim//2):j+(patchdim//2)]=255
                cv2.imshow('image',gt_extend)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    #gt_extend[170-(patchdim//2):170+(patchdim//2),170-(patchdim//2):170+(patchdim//2)]=255


def cutPositiveExample(imgPath,gtPath):
    #list of positive exemple
    positiveExample = []
    #list of positive label
    positivelabel = []
    #relief image
    image=cv2.imread(imgPath,0)
    #groundtruth image
    gt = cv2.imread(gtPath,0)
    #find center of connected component
    centroids=find_connected_component(gt)
    for i in range(1,len(centroids)):
        print("nb imagettes : "+str(len(centroids)))
        #find top left corner of mini image
        tlc=getTopLeftCorner_Patch(gt,centroids[i])
        #create mini image
        mini_gt=gt[tlc[1]:tlc[1]+patchdim,tlc[0]:tlc[0]+patchdim]
        mini_img=image[tlc[1]:tlc[1]+patchdim,tlc[0]:tlc[0]+patchdim]
        positiveExample.append(mini_img)
        positivelabel.append(mini_gt)
    return positiveExample,positivelabel

def getTopLeftCorner_Patch(dtimg,centre):
    ct=0
    topLeftCorner=[int(centre[0]-patchdim/2),int(centre[1]-patchdim/2)]
    botRightCorner=[int(centre[0]+patchdim/2),int(centre[1]+patchdim/2)]
    ymax=dtimg.shape[0]
    xmax=dtimg.shape[1]


    print(str(xmax)+" * "+str(ymax))

    corner_output=[-1,-1]
    #tous les cas possible :
    #1.2.3
    #4.5.6
    #7.8.9
    #cas n°1
    if topLeftCorner[0]<0 and topLeftCorner[1]<0 :
        corner_output[0]=0
        corner_output[1]=0
        ct=ct+1
        print("cas n°1")
    #cas n°2
    if topLeftCorner[0]>=0 and topLeftCorner[1]<0 and botRightCorner[0]<=xmax and botRightCorner[1]<=ymax :
        corner_output[0]=topLeftCorner[0]
        corner_output[1]=0
        ct=ct+1
        print("cas n°2")
    #cas n°3
    if topLeftCorner[1]<0 and botRightCorner[0]>xmax:
        corner_output[0]=topLeftCorner[0]-abs(botRightCorner[0]-xmax)
        corner_output[1]=0
        ct=ct+1
        print("cas n°3")
    #cas n°4
    if topLeftCorner[0]<0 and topLeftCorner[1]>=0 and botRightCorner[0]<=xmax and botRightCorner[1]<=ymax:
        corner_output[0]=0
        corner_output[1]=topLeftCorner[1]
        ct=ct+1
        print("cas n°4")
    #cas n°5
    if topLeftCorner[0]>=0 and topLeftCorner[1]>=0 and botRightCorner[0]<=xmax and botRightCorner[1]<=ymax :
        corner_output=topLeftCorner
        ct=ct+1
        print("cas n°5")
    #cas n°6
    if topLeftCorner[0]>=0 and topLeftCorner[1]>=0 and botRightCorner[0]>xmax and botRightCorner[1]<=ymax :
        corner_output[0]=topLeftCorner[0]-abs(botRightCorner[0]-xmax)
        corner_output[1]=topLeftCorner[1]
        ct=ct+1
        print("cas n°6")
    #cas n°7
    if topLeftCorner[0]<0 and botRightCorner[1]>ymax :
        corner_output[0]=0
        corner_output[1]=topLeftCorner[1]-abs(botRightCorner[1]-ymax)
        ct=ct+1
        print("cas n°7")
    #cas n°8
    if topLeftCorner[0]>=0 and topLeftCorner[1]>=0 and botRightCorner[1]>ymax and botRightCorner[0]<= xmax:
        corner_output[0]=topLeftCorner[0]
        corner_output[1]=topLeftCorner[1]-abs(botRightCorner[1]-ymax)
        ct=ct+1
        print("cas n°8")
    #cas n°9
    if botRightCorner[0]>xmax and botRightCorner[1]>ymax:
        corner_output[0]=topLeftCorner[0]-abs(botRightCorner[0]-xmax)
        corner_output[1]=topLeftCorner[1]-abs(botRightCorner[1]-ymax)
        ct=ct+1
        print("cas n°9")
    print("centre :"+str(centre))
    print("top left before : "+ str(topLeftCorner))
    print("bot right before : "+ str(botRightCorner))

    print("topleftcorner after :"+str(corner_output))
    print("count : "+str(ct))
    return corner_output
def find_connected_component(gt):
    # need to choose 4 or 8 for connectivity type
    connectivity = 8
    # find connected component
    output = cv2.connectedComponentsWithStats(gt, connectivity, cv2.CV_32S)
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]
    print ("number of connected component : "+str(num_labels))
    return centroids






if __name__ == "__main__":
   main(sys.argv[1:])
