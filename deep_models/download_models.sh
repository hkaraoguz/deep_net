#!/bin/bash

DEPLOYDIR=./caffe
mkdir -p "$DEPLOYDIR"
cd $DEPLOYDIR


FILE=vgg16_faster_rcnn_iter_40000_kth2.caffemodel.tar.gz
URL=https://www.dropbox.com/s/podka2mhub8scxu/$FILE?dl=1
if [ -f $FILE ]; then
  echo "KTH Model File already exists."
  echo "Unzipping..."
  tar zxvf $FILE
else
  echo "Downloading KTH model (508M)..."
  wget $URL -O $FILE -P ./caffe
  tar zxvf $FILE
fi


FILE=rfcn_models.tar.gz
URL=https://www.dropbox.com/s/lnz0r8uk9ax1mey/rfcn_models.tar.gz?dl=1
if [ -f $FILE ]; then
  echo "rfcn model File already exists."
  echo "Unzipping..."
  tar zxvf $FILE
  mv ./rfcn_models/* .
  rm -r ./rfcn_models
else
  echo "Downloading rfcn models..."
  wget $URL -O $FILE -P ./caffe
  tar zxvf $FILE
  mv ./rfcn_models/* .
  rm -r ./rfcn_models
fi



FILE=faster_rcnn_models.tgz
URL=http://www.cs.berkeley.edu/~rbg/faster-rcnn-data/$FILE
CHECKSUM=ac116844f66aefe29587214272054668
DIRECTORY=./faster_rcnn_models

if [ -f $FILE ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    if [ -d "$DIRECTORY" ]; then
    	mv ./faster_rcnn_models/* .
        rmdir ./faster_rcnn_models
    fi
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading Faster R-CNN demo models (695M)..."

wget $URL -O $FILE -P ./caffe

echo "Unzipping..."

tar zxvf $FILE

mv ./faster_rcnn_models/* .
rm -r ./faster_rcnn_models









