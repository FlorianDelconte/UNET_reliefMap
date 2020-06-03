#!/usr/bin/python3
########################################################################################
#This script is use to find the good treshold on predicted imageName                   #
#                  I:tresholdOutput of u-net  G: groundtruth                           #
#                                   O:Measure files                                    #
########################################################################################
import sys, getopt
import os
import cv2
import numpy as np
import csv
import scipy.misc
import scipy.ndimage
import skimage.filters
import sklearn.metrics

# Optional, added to ignore scipy read warnings
import warnings
def main(argv):
    inputImage = ''
    inputGT = ''
    measureFile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:g:o:",["iImage=","IGT=","-Mfile"])
    except getopt.GetoptError:
        print (' -i <inputImage> -g <inputGT> -o <measurefile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('Measure.py -i <inputImage>  -g <inputGT> -o <measurefile>')
            sys.exit()
        elif opt in ("-i", "--iImage"):
            inputImage = arg
        elif opt in ("-g", "--IGT"):
            inputGT = arg
        elif opt in ("-o", "--Mfile"):
            measureFile = arg

    imageName=os.path.splitext(os.path.basename(inputImage))[0]
    #tresholded image
    image=cv2.imread(inputImage,cv2.IMREAD_UNCHANGED)
    #groundtruth image
    gt = cv2.imread(inputGT,cv2.IMREAD_UNCHANGED)

    groundtruth_scaled = gt // 255
    predicted_scaled = image // 255
    groundtruth_list = (groundtruth_scaled).flatten().tolist()
    predicted_list = (predicted_scaled).flatten().tolist()
    #NEW
    TN,FP,FN,TP=confusion_matrix2(groundtruth_list,predicted_list)
    #OLD
    #TP,TN,FN,FP=confusion_matrix(image,gt)
    precision, recall,FPR, F1 =F1_Measure(TP,TN,FN,FP)

    #CSV WRITE
    mesures=[imageName,precision,recall,FPR,F1]
    f = open(measureFile, "a")
    csv_writer = csv.writer(f, delimiter=',')
    with f as out:
        csv_writer.writerow(mesures)
    f.close()

def F1_Measure(TP,TN,FN,FP):
    
    precision=float(TP)/(TP+FP)

    recall=float(TP)/(TP+FN)

    FPR=float(FP)/(FP+TN)

    F1=2*(precision*recall)/(precision+recall)
    return precision,recall,FPR,F1

def _assert_valid_lists(groundtruth_list, predicted_list):
    assert len(groundtruth_list) == len(predicted_list)
    for unique_element in np.unique(groundtruth_list).tolist():
        assert unique_element in [0, 1]

def _all_class_1_predicted_as_class_1(groundtruth_list, predicted_list):
     _assert_valid_lists(groundtruth_list, predicted_list)
     return np.unique(groundtruth_list).tolist() == np.unique(predicted_list).tolist() == [1]

def _all_class_0_predicted_as_class_0(groundtruth_list, predicted_list):
     _assert_valid_lists(groundtruth_list, predicted_list)
     return np.unique(groundtruth_list).tolist() == np.unique(predicted_list).tolist() == [0]

def confusion_matrix2(groundtruth_list, predicted_list):
    _assert_valid_lists(groundtruth_list, predicted_list)

    if _all_class_1_predicted_as_class_1(groundtruth_list, predicted_list) is True:
        tn, fp, fn, tp = 0, 0, 0, np.float64(len(groundtruth_list))
    elif _all_class_0_predicted_as_class_0(groundtruth_list, predicted_list) is True:
        tn, fp, fn, tp = np.float64(len(groundtruth_list)), 0, 0, 0
    else:
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(groundtruth_list, predicted_list).ravel()
        tn, fp, fn, tp = np.float64(tn), np.float64(fp), np.float64(fn), np.float64(tp)
    print("tn : "+str(tn)+"fp : "+str(fp)+"fn : "+str(fn)+"tp : "+str(tp))
    return tn, fp, fn, tp



def confusion_matrix(detections,truth):
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    positive_count = np.count_nonzero(detections)
    negative_count = truth.size - positive_count

    temp1=np.bitwise_and(truth,detections)
    TP = np.count_nonzero(temp1)

    FP = positive_count - TP
    temp2=np.bitwise_not(detections)
    temp2=np.bitwise_and(truth, temp2)
    FN = np.count_nonzero(temp2)
    TN = negative_count - FN
    #print("TP : ",TP,"TN :",TN,"FN :",FN,"FP :",FP)
    return TP,TN,FN,FP

if __name__ == "__main__":
   main(sys.argv[1:])
