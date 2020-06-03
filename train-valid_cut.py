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
patchdim=300
#valid input path
validInputPath="valid/input/"
#valid output validInputPath
validOutputPath="valid/output/"
#train input path
trainInputPath="train/input/"
#train output path
trainOutputPath="train/output/"
def main(argv):
    inputImage = ''
    inputGT = ''
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
            inputImage = arg
        elif opt in ("-g", "--IGT"):
            inputGT = arg
        elif opt in ("-o", "--odir"):
            outputDirectory = arg
    imageName=os.path.splitext(os.path.basename(inputImage))[0]
    gtName=os.path.splitext(os.path.basename(inputGT))[0]
    print ('Path to image is ', inputImage)
    print ('Path to groundTruth is ', inputGT)
    print ('Output directory is ', outputDirectory)
    print('input image name :',imageName)
    print('input gt name :',gtName)
    #relief image
    image=cv2.imread(inputImage)
    #groundtruth image
    gt = cv2.imread(inputGT,0)
    #find center of connected component
    centroids=find_connected_component(gt)

    #print("WRITE TO : ",outputDirectory+trainInputPath+imageName+"_test"+".jpg")
    for i in range(1,len(centroids)):
        print("nb imagettes : "+str(len(centroids)))
        tresholdRes=round((len(centroids)/100) * ratio)
        print(str(ratio)+" % = "+str(tresholdRes)+" imagettes")
        #find top left corner of mini image
        tlc=getTopLeftCorner_Patch(gt,centroids[i])
        #create mini image
        mini_gt=gt[tlc[1]:tlc[1]+patchdim,tlc[0]:tlc[0]+patchdim]
        mini_img=image[tlc[1]:tlc[1]+patchdim,tlc[0]:tlc[0]+patchdim]

        if i<=tresholdRes :

            cv2.imwrite(outputDirectory+trainInputPath+imageName+"_"+str(i)+".jpg", mini_img)
            cv2.imwrite(outputDirectory+trainOutputPath+gtName+"_"+str(i)+".jpg", mini_gt)

            #cv2.imwrite("./reptest/"+imageName+"_"+str(i)+'.jpg', mini_img)
            #cv2.imwrite("./reptest/"+gtName+"_"+str(i)+".jpg", mini_gt)
        else :
            cv2.imwrite(outputDirectory+validInputPath+imageName+"_"+str(i)+".jpg", mini_img)
            cv2.imwrite(outputDirectory+validOutputPath+gtName+"_"+str(i)+".jpg", mini_gt)



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
