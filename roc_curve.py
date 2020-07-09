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

    TPR_value5=[]
    FPR_values5=[]

    TPR_value6=[]
    FPR_values6=[]



    with open("crossValidationv6/roc_curve/ROCk1", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            TPR_value1.append(float(row['Recall']))
            FPR_values1.append(float(row['FPR']))

    with open("crossValidationv6/roc_curve/ROCk2", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            TPR_value3.append(float(row['Recall']))
            FPR_values3.append(float(row['FPR']))

    with open("crossValidationv6/roc_curve/ROCk3", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            TPR_value4.append(float(row['Recall']))
            FPR_values4.append(float(row['FPR']))

    with open("crossValidationv6/roc_curve/ROCk4", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            TPR_value5.append(float(row['Recall']))
            FPR_values5.append(float(row['FPR']))

    with open("crossValidationv6/roc_curve/ROCk5", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            TPR_value6.append(float(row['Recall']))
            FPR_values6.append(float(row['FPR']))

    plt.plot(FPR_values1, TPR_value1,label='k1')
    plt.plot(FPR_values1, TPR_value1, 'ro',label='k1')

    plt.plot(FPR_values3, TPR_value3,label='k2')
    plt.plot(FPR_values3, TPR_value3, 'bo',label='k2')

    plt.plot(FPR_values4, TPR_value4,label='k3')
    plt.plot(FPR_values4, TPR_value4, 'go',label='k3')

    plt.plot(FPR_values5, TPR_value5,label='k4')
    plt.plot(FPR_values5, TPR_value5, 'mo',label='k4')

    plt.plot(FPR_values6, TPR_value6,label='k5')
    plt.plot(FPR_values6, TPR_value6, 'yo',label='k5')

    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    #print(integrate.simps(FPR_values, TPR_value))
    plt.show()
if __name__ == "__main__":
   main(sys.argv[1:])
