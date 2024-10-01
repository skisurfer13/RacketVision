
# Tennis Analysis

## Introduction
This project analyzes Tennis players in a video to measure their speed, ball shot speed and number of shots. This project will detect players using YOLO, the tennis ball using TrackNet and also utilizes CNNs to extract court keypoints.

## Models Used
* YOLO v8 for player detection
* Tracknet for tennis ball detection
* Court Key point extraction using CNNs
* Catboost for bounce detection

## Training
* Tennis ball detetcor with Tracknet: Refer TrackNet-main folder
* Tennis court keypoint with Pytorch: training/tennis_court_keypoints_training.ipynb

## Python version: 3.10.1

## References:
https://github.com/ultralytics/ultralytics
https://github.com/yastrebksv/TrackNet
https://arxiv.org/abs/1907.03698
https://github.com/abdullahtarek/tennis_analysis