#!/usr/bin/python3
########################################################################################
#This script is use to average the measure in input file and wirte it in output file   #
#                  I:inputMeasures                                                     #
#                                   O:output measure                                   #
########################################################################################
import sys, getopt
import os
import cv2
import numpy as np
import csv
import statistics

def main(argv):
    inputFileMeasure = ''
    OutputMean=''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["iMeasure=","-oMean"])
    except getopt.GetoptError:
        print (' -i <inputFileMeasure> -o <OutputMean>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('mean.py -i <inputFileMeasure> -o <OutputMean>')
            sys.exit()
        elif opt in ("-i", "--iMeasure"):
            inputFileMeasure = arg
        elif opt in ("-o", "--oMean"):
            OutputMean = arg
    print(inputFileMeasure)
    print(OutputMean)

    recall_values=[]
    FPR_values=[]
    precision_values=[]
    treshold_value=-1
    #CSV READ
    with open(inputFileMeasure, 'r') as csvfile:
        first_row=next(csvfile)
        treshold_value = int(first_row.split('#')[-1])
        reader = csv.DictReader(csvfile)
        for row in reader:
            recall_values.append(float(row['Recall']))
            FPR_values.append(float(row['FPR']))
            precision_values.append(float(row['Precision']))
    #MEAN
    mean_recall_values=statistics.mean(recall_values)
    mean_FPR_values=statistics.mean(FPR_values)
    mean_precision_values=statistics.mean(precision_values)
    print(treshold_value)
    print(mean_recall_values)
    print(mean_FPR_values)
    print(mean_precision_values)
    #CSV WRITE
    mesures=[treshold_value,mean_precision_values,mean_recall_values,mean_FPR_values]
    f = open(OutputMean, "a")
    csv_writer = csv.writer(f, delimiter=',')
    with f as out:
        csv_writer.writerow(mesures)
    f.close()
if __name__ == "__main__":
   main(sys.argv[1:])
