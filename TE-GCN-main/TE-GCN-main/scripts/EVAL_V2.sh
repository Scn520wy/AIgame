#!/bin/bash

#RECORD=2995
#WORKDIR=work_dir/$RECORD
#MODELNAME=runs/$RECORD

WORKDIR=work_dir/uav/xsub2/agcn_joint/epoch78_test_score.npy
MODELNAME=runs/1

#CONFIG=./config/uav-cross-subjectv1/test.yaml
CONFIG=./config/uav-cross-subjectv2/test.yaml

WEIGHTS=runs/uav-77-64038.pt


BATCH_SIZE=20

python3 main.py  --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --weights $WEIGHTS
