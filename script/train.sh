#!/bin/bash
cd /Users/david/Program/PADVO/src # path to current source 
traindata=kitti
testdata=kitti

python3 padvo_train.py --imagelist ../dataset/$traindata/$traindata.train.image --motion ../dataset/$traindata/$traindata.train.pose --imagelist_test ../dataset/$testdata/$testdata.test.image --motion_test ../dataset/$testdata/$testdata.test.pose --ip http://128.237.141.242 --port 8524 --model 0913_001 --batch 60 --model_load ../saved_model//model_0905_002_030.pt

