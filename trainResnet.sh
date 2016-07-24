#!/bin/bash
DEPTH=$1
#GPUID=2,3,4,5,6,7
GPUID=$2
EXPERIMENT_NAME=$3
NGPU=$(((${#GPUID}+1)/2))
RESNETDIR=$HOME/data/fb_resnet
OUTPUTDIR=$RESNETDIR/output/$EXPERIMENT_NAME
CHECKPOINTDIR=$RESNETDIR/checkpoints/$EXPERIMENT_NAME/$DEPTH
mkdir -p $OUTPUTDIR
OUTPUTNAME=$OUTPUTDIR/${DEPTH}.out
BATCHSIZE=$((32*NGPU))
LR=0.1
#if (( "$DEPTH" <= "34" ))
#then
  LR=0.01 #Default LR was causing divergence for resnet-18 and resnet-34
#fi
echo "Running on GPU $GPUID" >> $OUTPUTNAME 
echo "PID: $$" >> $OUTPUTNAME
echo "Batch size: $BATCHSIZE" >> $OUTPUTNAME
echo "LR: $LR" >> $OUTPUTNAME
date >> $OUTPUTNAME
CUDA_VISIBLE_DEVICES=$GPUID th main.lua -tenCrop true -resume $CHECKPOINTDIR -nGPU $NGPU -retrain weights/resnet-${DEPTH}.t7 -data ~/data/cleanedZachClassData/63KTRAINVAL/ -resetClassifier true -nClasses 7 -nThreads 16 -save $CHECKPOINTDIR -batchSize $BATCHSIZE -nEpochs 100 -LR $LR >> $OUTPUTNAME
date >> $OUTPUTNAME
