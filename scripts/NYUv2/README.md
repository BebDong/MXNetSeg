# NYUv2 dataset

For dataset details, refer to the official website [NYUDv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). Note that here we do not adopt depth information.

## Prepare dataset

1. Download the [labeled dataset](http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat) (~2.8GB) from the official website.

2. Run the script `preprocess.py`

   ```shell
   # cd to this directory
   python preprocess.py
   ```