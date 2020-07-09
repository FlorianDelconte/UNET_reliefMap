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
    Precision_value1=[]
    Recall_values1=[]

    Precision_value3=[]
    Recall_values3=[]

    Precision_value4=[]
    Recall_values4=[]

    Precision_value5=[]
    Recall_values5=[]

    Precision_value6=[]
    Recall_values6=[]

    with open("crossValidationv6/roc_curve/ROCk1", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            Precision_value1.append(float(row['Precision']))
            Recall_values1.append(float(row['Recall']))

    with open("crossValidationv6/roc_curve/ROCk2", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            Precision_value3.append(float(row['Precision']))
            Recall_values3.append(float(row['Recall']))

    with open("crossValidationv6/roc_curve/ROCk3", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            Precision_value4.append(float(row['Precision']))
            Recall_values4.append(float(row['Recall']))

    with open("crossValidationv6/roc_curve/ROCk4", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            Precision_value5.append(float(row['Precision']))
            Recall_values5.append(float(row['Recall']))

    with open("crossValidationv6/roc_curve/ROCk5", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            Precision_value6.append(float(row['Precision']))
            Recall_values6.append(float(row['Recall']))



    plt.plot(Recall_values1, Precision_value1, label='k1')
    plt.plot(Recall_values1, Precision_value1, 'ro',label='k1')

    plt.plot(Recall_values3, Precision_value3, label='k2')
    plt.plot(Recall_values3, Precision_value3, 'bo',label='k2')

    plt.plot(Recall_values4, Precision_value4, label='k3')
    plt.plot(Recall_values4, Precision_value4, 'go',label='k3')

    plt.plot(Recall_values5, Precision_value5, label='k4')
    plt.plot(Recall_values5, Precision_value5, 'mo',label='k4')

    plt.plot(Recall_values6, Precision_value6, label='k5')
    plt.plot(Recall_values6, Precision_value6, 'yo',label='k5')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # show the legend
    plt.legend()
    plt.show()
if __name__ == "__main__":
   main(sys.argv[1:])
