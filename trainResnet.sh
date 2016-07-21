#!/bin/bash
DEPTH=$1
GPUID=$2
OUTPUTNAME=output/${DEPTH}.out
BATCHSIZE=16
echo "Running on GPU $GPUID" > $OUTPUTNAME 
echo "PID: $$" >> $OUTPUTNAME
echo "Batch size: $BATCHSIZE" >> $OUTPUTNAME
date >> $OUTPUTNAME
CUDA_VISIBLE_DEVICES=$GPUID th main.lua -retrain weights/resnet-${DEPTH}.t7 -data ~/data/cleanedZachClassData/63KTRAINVAL/ -resetClassifier true -nClasses 7 -nThreads 2 -save checkpoints-$DEPTH -batchSize $BATCHSIZE -nEpochs 200 >> $OUTPUTNAME
date >> $OUTPUTNAME