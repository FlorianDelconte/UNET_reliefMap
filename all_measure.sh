#!/bin/bash
echo "Total Arguments:" $#
pathToModelSgm=$1
pathToReadGroundTruth=$2
RepToWriteData=$3

pathOutputFileMeasure="${RepToWriteData}measure"
pathOutputFileROC="${RepToWriteData}ROC"
pathToWriteTreshold="${RepToWriteData}tresholdTemp/"

#create a tompory directory for tresholded image
mkdir -p $pathToWriteTreshold;

echo "Path to read model segmentation -> "  $pathToModelSgm
echo "Path to read groundTruth -> "  $pathToReadGroundTruth
echo "Path to write data -> " $RepToWriteData
echo "Path to write treshold -> "  $pathToWriteTreshold

printf "tresh_value,Precision,Recall,FPR\n" >> $pathOutputFileROC
#for i in -1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190 195 200 205 210 215 220 225 230 235 240 245 250 255
for i in {-1..255}
do
  echo "___TRESHOLD___" $i
  for reliefMap in $pathToModelSgm*.png
  do
    python3 treshold_predic.py -i $reliefMap -o $pathToWriteTreshold -t $i
    echo "python3 treshold_predic.py -i" $reliefMap "-o" $pathToWriteTreshold "-t" $i
  done

  echo "___MEASURES___"
  printf "#%d\n" "$i" >> $pathOutputFileMeasure$i
  printf "Name,Precision,Recall,FPR,F1\n" >> $pathOutputFileMeasure$i
  for tresholdMap in $pathToWriteTreshold*.png
  do
    #CARE : GT file need to have "_GT.png" suffixe
    tresholdMapName=$(basename $tresholdMap)
    tresholdMapNameWithoutExt="${tresholdMapName%.*}"
    GTMapNameWithExt="${tresholdMapNameWithoutExt}_GT.png"
    pathToTresholdFile="$pathToReadGroundTruth$GTMapNameWithExt"

    python3 Measure.py -i $tresholdMap -g $pathToTresholdFile -o $pathOutputFileMeasure$i
    echo "python3 Measure.py -i" $tresholdMap "-g" $pathToTresholdFile "-o" $pathOutputFileMeasure$i
  done

    echo "Number: $i"

  echo "___MEAN___"
  python3 mean.py -i $pathOutputFileMeasure$i -o $pathOutputFileROC
  echo "python3 mean.py -i" $pathOutputFileMeasure$i "-o" $pathOutputFileROC

done
