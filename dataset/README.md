# Dataset

For well-prepared images of benchmark datasets. 

## Directory Structure

```
MXNetSeg
|-- dataset
|   |-- ADE20K
|   |   |-- ADEChallengeData2016
|   |   |   |-- annotations
|   |   |   |-- images
|   |   |   |-- objectInfo150.txt
|   |   |   `-- scaneCategories.txt
|   |   `-- release_test
|   |       |-- testing
|   |       |-- list.txt
|   |       `-- readme.txt
|   |-- Aeroscapes
|   |   |-- ImageSets
|   |   |   |-- trn.txt
|   |   |   `-- val.txt
|   |   |-- JPEGImages
|   |   |-- SegmentationClass
|   |   `-- Visualizations
|   |-- BDD
|   |   `-- seg
|   |       |-- images
|   |       |   |-- test
|   |       |   |-- train
|   |       |   `-- val
|   |       `-- labels
|   |           |-- train
|   |           `-- val
|   |-- CamVid
|   |   |-- images
|   |   |-- labels
|   |   |-- labelsGray
|   |   |-- test.txt
|   |   |-- train.txt
|   |   |-- trainval.txt
|   |   `-- val.txt
|   |-- CamVidFull
|   |   |-- images
|   |   |-- labels
|   |   |-- labelsGray
|   |   |-- test.txt
|   |   |-- train.txt
|   |   |-- trainval.txt
|   |   `-- val.txt
|   |-- Cityscapes
|   |   |-- demoVideo
|   |   |   |-- stuttgart_00
|   |   |   |-- stuttgart_01
|   |   |   `-- stuttgart_02
|   |   |-- gtCoarse
|   |   |   |-- train
|   |   |   |-- train_extra
|   |   |   `-- val
|   |   |-- gtFine
|   |   |   |-- test
|   |   |   |-- train
|   |   |   `-- val
|   |   `-- leftImg8bit
|   |       |-- test
|   |       |-- train
|   |       |-- train_extra
|   |       `--val
|   |-- COCO
|   |   |-- annotations
|   |   |-- train 2017
|   |   `-- val2017
|   |-- COCOStuff
|   |   |-- annotations
|   |   |-- imageLists
|   |   `-- images
|   |-- GATECH
|   |   |-- images
|   |   |-- labels
|   |   |-- test.txt
|   |   |-- test_all.txt
|   |   `-- train.txt
|   |-- ImageNet
|   |   `--rec
|   |      |-- train.idx
|   |      |-- train.rec
|   |      |-- val.idx
|   |      `-- val.rec
|   |-- KITTI
|   |   |-- Ros
|   |   |   |-- Training_00
|   |   |   `-- Validation_07
|   |   |-- Xu
|   |   |   |-- images
|   |   |   |-- labels
|   |   |   |-- test.txt
|   |   |   `-- train.txt
|   |   `-- Zhang
|   |       |-- images
|   |       |-- labels
|   |       |-- labels_gray
|   |       |-- test.txt
|   |       `-- train.txt
|   |-- Mapillary
|   |   |-- test
|   |   |   |-- images
|   |   |   `-- labels
|   |   |-- train
|   |   |   |-- images
|   |   |   `-- labels
|   |   |-- val
|   |   |   |-- images
|   |   |   `-- labels
|   |   `-- config.json
|   |-- MHP
|   |   `-- v1
|   |       |-- annotations
|   |       |-- images
|   |       |-- test_list.txt
|   |       |-- train_list.txt
|   |       `-- visualize.m
|   |-- NYUv2
|   |   |-- images
|   |   |-- labels
|   |   |-- labels40
|   |   |-- train.txt
|   |   `-- val.txt
|   |-- PContext
|   |   |-- JPEGImages
|   |   |-- Labels_59
|   |   |-- train.txt
|   |   |-- trainval_merged.json
|   |   `-- val.txt
|   |-- SiftFlow
|   |   |-- images
|   |   |-- labels
|   |   |-- train.txt
|   |   `-- val.txt
|   |-- Stanford10
|   |   |-- images
|   |   |-- labels
|   |   |-- evalList.txt
|   |   `-- trainList.txt
|   |-- SUNRGBD
|   |   |-- images
|   |   |-- labels
|   |   |-- test.txt
|   |   `-- train.txt
|   |-- VOCdevkit
|   |   |-- SBD
|   |   |   |-- cls
|   |   |   |-- img
|   |   |   |-- inst
|   |   |   |-- train.txt
|   |   |   |-- trainval.txt
|   |   |   `-- val.txt
|   |   |-- VOC 2012
|   |   |   |-- Annotations
|   |   |   |-- ImageSets
|   |   |   |   |-- Action
|   |   |   |   |-- Layout
|   |   |   |   |-- Main
|   |   |   |   `-- Segmentation
|   |   |   |       |-- test.txt
|   |   |   |       |-- train.txt
|   |   |   |       |-- trainval.txt
|   |   |   |       `-- val.txt
|   |   |   |-- JPEGImages
|   |   |   |-- SegmentationClass
|   |   |   `-- SegmentationObject
|   |   `-- VOCAug
|   |       |-- cls_aug
|   |       |-- img_aug
|   |       |-- train.txt
|   |       |-- train_aug.txt
|   |       |-- trainval_aug.txt
|   |       `-- val.txt
|   `-- WeizHorses
|       |-- horse
|       |-- mask
|       |-- train.txt
|       `-- val.txt
|-- demo
|   |-- demo_citys.png
|   `-- repo-social-image.png
|-- mxnetseg
`-- scripts

```

## Remarks

Some benchmark datasets are not ready to use directly, and necessary transformation is needed. For example, this repo's habit is to convert both `.txt` and `.mat` annotations into grayscale images. Folder or file names can be self-defined as long as they are consistent with data loading codes in the [data-package](../mxnetseg/data).