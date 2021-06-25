# Pascal VOC Aug

- Download the original *Pascal VOC 2012* and *SBD* datasets
- Refer to [train-DeepLab](https://github.com/martinkersner/train-DeepLab) repo for `mat2png.py` & `utils.py`
### Prepare dataset

1. Convert `.mat` of SBD to `.png`: 
   - Enter the SBD directory and create `cls_aug`  folder (for example)
   - Run `python mat2png.py cls cls_aug`
2. Convert colored `.png` of Pascal VOC 2012 to gray `.png`:
   - Enter the VOC2012 directory and create `SegClassGray` folder (for example)
   - Run `python convert_labels.py SegmentationClass ImageSets/Segmentation/trainval.txt SegClassGray`
3. Move images and converted labels of the two datasets to the same folder, respectively. Just override the files with same filenames.
