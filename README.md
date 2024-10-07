
# Tennis Analysis

## Introduction
This project analyzes Tennis players in a video to measure their speed, ball shot speed and number of shots. This project will detect players using YOLO, the tennis ball using TrackNet and also utilizes CNNs to extract court keypoints. Poor output when camera angle is changing / input video is not captured from a static camera setup.

## Setup
* git clone https://github.com/skisurfer13/RacketVision.git
* cd RacketVision/
* conda env create -n racketvision -f environment.yml
* conda activate racketvision
* conda install -c conda-forge notebook (if you haven't installed jupyter already)
* jupyter notebook



## Models Used
* YOLO v8 for player detection
* Tracknet for tennis ball detection
* CNN for Court Key point extraction 
* Catboost Regressor for bounce detection

## Training
* Tennis ball detetcor with Tracknet: Refer TrackNet-main folder
* Tennis court keypoint with Pytorch: training/tennis_court_keypoints_training.ipynb

## Python version: 3.10.1
