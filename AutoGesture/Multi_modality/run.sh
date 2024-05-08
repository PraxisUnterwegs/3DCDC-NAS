#!/bin/sh
MODALITY=MK
LAYER=AutoGesture_RGBD_Con_shared_DiffChannels
GPU_IDS=0
FRAME=32
python -u train_AutoGesture_CDC_RGBD_sgd_12layers.py -m train -l $LAYER -t $MODALITY -g $GPU_IDS  | tee ./log/$LAYER-$MODALITY-$FRAME.log
