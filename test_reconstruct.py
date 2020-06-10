########################################################################################
#This script is use to cut test image insmaller image                                  #
########################################################################################

#!/usr/bin/python3
import sys, getopt
import os
import cv2


CR_SIZE = { "Beech": (556,920),
            "Birch": (366,922),
            "Elm": (615,904),
            "Fir1": (729,969),
            "Fir2": (778,859),
            "Redoak1": (527,978),
            "Redoak2": (627,919),
            "WildCherry1": (554,1094),
            "WildCherry2": (465,1056),
            "WildServiceTree": (898,844)}



def main(argv):
    inputDir = ''
    outputDirectory = ''

    try:
        opts, args = getopt.getopt(argv,"hi:o:",["iDir=","odir="])
    except getopt.GetoptError:
        print (' -i <inputDir>  -o <outputDirectory>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('defect_detect.py -i <inputDir> -o <outputDirectory>')
            sys.exit()
        elif opt in ("-i", "--iDir"):
            inputDir = arg
        elif opt in ("-o", "--odir"):
            outputDirectory = arg
            
        print(inputDir)
        print(outputDirectory)


if __name__ == "__main__":
   main(sys.argv[1:])
