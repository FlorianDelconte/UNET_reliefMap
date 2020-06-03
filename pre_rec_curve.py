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

    with open("repartition1/test/roc_curve/ROC", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            Precision_value1.append(float(row['Precision']))
            Recall_values1.append(float(row['Recall']))

    with open("repartition3/test/roc_curve/ROC", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            Precision_value3.append(float(row['Precision']))
            Recall_values3.append(float(row['Recall']))

    with open("repartition4/test/roc_curve/ROC", 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            Precision_value4.append(float(row['Precision']))
            Recall_values4.append(float(row['Recall']))



    plt.plot(Recall_values1, Precision_value1, label='Logistic')
    plt.plot(Recall_values1, Precision_value1, 'ro',label='Logistic')

    plt.plot(Recall_values3, Precision_value3, label='Logistic')
    plt.plot(Recall_values3, Precision_value3, 'bo',label='Logistic')

    plt.plot(Recall_values4, Precision_value4, label='Logistic')
    plt.plot(Recall_values4, Precision_value4, 'yo',label='Logistic')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # show the legend
    plt.legend()
    plt.show()
if __name__ == "__main__":
   main(sys.argv[1:])
