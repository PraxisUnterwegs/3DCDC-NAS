#!/bin/sh
MODALITY=MK
LAYER=AutoGesture_RGBD_Con_shared_DiffChannels
#LAYER=18
GPU_IDS=0,1
FRAME=32
python -u train_AutoGesture_CDC_RGBD_sgd_12layers.py -m train --config config_hpc.yml -l $LAYER -t $MODALITY -g $GPU_IDS  | tee ./log/$LAYER-$MODALITY-$FRAME.log
