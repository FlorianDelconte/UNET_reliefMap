#!/usr/bin/python3
########################################################################################
#This script is use to plot and save roc curve                                         #
#                  I : input file containing roc value                                 #
#                                   O: png file with roc curve                         #
########################################################################################
import sys, getopt
import os
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import integrate

def main(argv):
    inputFile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:",["iFile="])
    except getopt.GetoptError:
        print (' -i <inputMeasureROC> ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('roc_curve.py -i <inputMeasureROC>')
            sys.exit()
        elif opt in ("-i", "--iFile"):
            inputFile = arg

    print(inputFile)
    #CSV READ
    TPR_value1=[]
    FPR_values1=[]

    TPR_value3=[]
    FPR_values3=[]

    TPR_value4=[]
    FPR_values4=[]

    with open("repartition1/test/roc_curve/ROC", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            TPR_value1.append(float(row['Recall']))
            FPR_values1.append(float(row['FPR']))

    with open("repartition3/test/roc_curve/ROC", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            TPR_value3.append(float(row['Recall']))
            FPR_values3.append(float(row['FPR']))

    with open("repartition4/test/roc_curve/ROC", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            TPR_value4.append(float(row['Recall']))
            FPR_values4.append(float(row['FPR']))

    plt.plot(FPR_values1, TPR_value1,label='model1')
    plt.plot(FPR_values1, TPR_value1, 'ro',label='model1')

    plt.plot(FPR_values3, TPR_value3,label='model3')
    plt.plot(FPR_values3, TPR_value3, 'bo',label='model3')

    plt.plot(FPR_values4, TPR_value4,label='model4')
    plt.plot(FPR_values4, TPR_value4, 'yo',label='model4')

    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    #print(integrate.simps(FPR_values, TPR_value))
    plt.show()
if __name__ == "__main__":
   main(sys.argv[1:])
