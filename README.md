# MXNetSeg

This project provides modular implementation for state-of-the-art semantic segmentation models based on the [MXNet](https://github.com/apache/incubator-mxnet) framework and [GluonCV](https://github.com/dmlc/gluon-cv) toolkit.

![](./demo/demo_citys.png)

## Bright Spots

- Ease of use and extension pipeline for the semantic segmentation task, including data pre-processing, model definition, network training and evaluation

- Parallel training on GPUs

- Multiple state-of-the-art or representative models

  - [FCN](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf), CVPR 2015
  - [AttentionToScale](http://openaccess.thecvf.com/content_cvpr_2016/papers/Chen_Attention_to_Scale_CVPR_2016_paper.pdf), CVPR 2016
  - [DeepLabv3](https://arxiv.org/pdf/1706.05587), ArXiv 2017
  - [LadderDenseNet](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w3/Kreso_Ladder-Style_DenseNets_for_ICCV_2017_paper.pdf), ICCVW 2017
  - [PSPNet](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.pdf), CVPR 2017
  - [BiSeNet](http://openaccess.thecvf.com/content_ECCV_2018/papers/Changqian_Yu_BiSeNet_Bilateral_Segmentation_ECCV_2018_paper.pdf), ECCV 2018
  - [DeepLabv3+](http://openaccess.thecvf.com/content_ECCV_2018/papers/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.pdf), ECCV 2018
  - [DenseASPP](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf), CVPR 2018
  - [SeENet](http://openaccess.thecvf.com/content_ICCV_2019/papers/Pang_Towards_Bridging_Semantic_Gap_to_Improve_Semantic_Segmentation_ICCV_2019_paper.pdf), ICCV 2019
  - [ACFNet](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_ACFNet_Attentional_Class_Feature_Network_for_Semantic_Segmentation_ICCV_2019_paper.pdf), ICCV 2019
  - [DANet](https://openaccess.thecvf.com/content_CVPR_2019/papers/Fu_Dual_Attention_Network_for_Scene_Segmentation_CVPR_2019_paper.pdf), CVPR 2019
  - [SwiftNet/SwiftNetPR](http://openaccess.thecvf.com/content_CVPR_2019/papers/Orsic_In_Defense_of_Pre-Trained_ImageNet_Architectures_for_Real-Time_Semantic_Segmentation_CVPR_2019_paper.pdf), CVPR 2019
  - [SemanticFPN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Feature_Pyramid_Networks_CVPR_2019_paper.pdf), CVPR 2019
  - [CANet](https://arxiv.org/abs/2002.12041v1), ArXiv 2020
  - [EPRNet](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9384352), T-ITS 2021
  - AttaNet, AAAI 2021
  - ViT, ICLR 2021 & SETR, CVPR 2021
- A [mirror repository](https://github.com/BebDong/MindSeg) implemented by the [HUAWEI MindSpore](https://www.mindspore.cn/en) is also provided.

## Benchmarks

We note that:

- OS is output stride of the backbone network.
- \* denotes multi-scale and flipping testing, otherwise single-scale inputs.
- No whistles and bells are adopted, e.g. OHEM or multi-grid.

### Cityscapes

| Model     | Backbone  |  OS  | #Params |                          Config                           |    TrainSet     | EvalSet | mIoU | \*mIoU |
| :-------- | :-------: | :--: | :-----: | :-------------------------------------------------------: | :-------------: | :-----: | :--: | :----: |
| BiSeNet   | ResNet18  |  32  |  13.2M  |     [silver-brook-24](./configs/silver-brook-24.yaml)     |  *train_fine*   |  *val*  | 71.6 |  74.7  |
| BiSeNet   | ResNet18  |  32  |  13.2M  |  [effortless-dust-25](./configs/effortless-dust-25.yaml)  | *trainval_fine* | *test*  |  -   |  74.8  |
| FCN-32s   | ResNet18  |  32  |  12.4M  |                             -                             |  *train_fine*   |  *val*  | 64.9 |  68.1  |
| FCN-32s   | ResNet18  |  8   |  12.4M  |       [pretty-surf-1](./configs/pretty-surf-1.yaml)       |  *train_fine*   |  *val*  | 68.3 |  69.9  |
| FCN-32s   | ResNet101 |  8   |  47.5M  |                             -                             |  *train_fine*   |  *val*  | 74.5 |   -    |
| PSPNet    | ResNet101 |  8   |  56.4M  |                             -                             |  *train_fine*   |  *val*  | 78.2 |  79.5  |
| DeepLabv3 | ResNet101 |  8   |  58.9M  |                             -                             |  *train_fine*   |  *val*  | 79.3 |  80.0  |
| DenseASPP | ResNet101 |  8   |  69.4M  | [neapolitan-pastry-1](./configs/neapolitan-pastry-1.yaml) |  *train_fine*   |  *val*  | 78.7 |  79.8  |
| DANet     | ResNet101 |  8   |  66.7M  |                             -                             |  *train_fine*   |  *val*  | 79.7 |  80.9  |

### ADE20K

| Model  | Backbone  |  OS  | Config | TrainSet | EvalSet |  PA  | mIoU | \*PA | \*mIoU |
| :----: | :-------: | :--: | :----: | :------: | :-----: | :--: | :--: | :--: | :----: |
| PSPNet | ResNet101 |  8   |   -    | *train*  |  *val*  | 80.1 | 42.9 | 80.9 |  43.7  |

### Pascal VOC 2012

| Model            | Backbone  |  OS  |                         Config                          |  TrainSet   | EvalSet |  PA  | mIoU | \*PA | \*mIoU |
| :--------------- | :-------: | :--: | :-----------------------------------------------------: | :---------: | :-----: | :--: | :--: | :--: | :----: |
| FCN-32s          | ResNet101 |  8   |       [glad-voice-1](./configs/glad-voice-1.yaml)       | *train_aug* |  *val*  | 94.4 | 74.6 | 94.5 |  75.0  |
| AttentionToScale | ResNet101 |  8   |      [peachy-snow-4](./configs/peachy-snow-4.yaml)      | *train_aug* |  *val*  | 94.8 | 77.1 |  -   |   -    |
| PSPNet           | ResNet101 |  8   |    [smart-valley-12](./configs/smart-valley-12.yaml)    | *train_aug* |  *val*  | 95.1 | 78.1 | 95.3 |  78.5  |
| DeepLabv3        | ResNet101 |  8   | [fearless-firefly-5](./configs/fearless-firefly-5.yaml) | *train_aug* |  *val*  | 95.5 | 80.1 | 95.6 |  80.4  |
| DeepLabv3+       | ResNet101 |  8   |    [silver-brook-24](./configs/silver-brook-24.yaml)    | *train_aug* |  *val*  | 95.5 | 79.9 | 95.6 |  80.1  |

### NYUv2

| Model      | Backbone  |  OS  |                          Config                           | TrainSet | EvalSet |  PA  | mIoU | *PA  | *mIoU |
| :--------- | :-------: | :--: | :-------------------------------------------------------: | :------: | :-----: | :--: | :--: | :--: | :---: |
| FCN-32s    | ResNet101 |  8   | [hopeful-snowflake-1](./configs/hopeful-snowflake-1.yaml) | *train*  |  *val*  | 69.2 | 39.7 | 70.2 | 41.0  |
| PSPNet     | ResNet101 |  8   |   [grateful-vortex-8](./configs/grateful-vortex-8.yaml)   | *train*  |  *val*  | 71.3 | 43.0 | 71.9 | 43.6  |
| DeepLabv3+ | ResNet101 |  8   |        [comfy-dew-31](./configs/comfy-dew-31.yaml)        | *train*  |  *val*  | 73.5 | 46.0 | 74.3 | 47.2  |

## Environment

We adopt python 3.6.2 and CUDA 10.1 in this project.

1. Prerequisites

   ```shell
   pip install -r requirements.txt
   ```

   Note that we employ [wandb](https://github.com/wandb/client) for log and visualization. Refer to [here](https://docs.wandb.ai/quickstart) for a QuickStart.

2. [Detail API](https://github.com/zhanghang1989/detail-api) for Pascal Context dataset

## Usage

### Training

1. Configure hyper-parameters in `./mxnetseg/config.yml`

2. Run the `./mxnetseg/train.py` script

   ```shell
   python train.py --model fcn --ctx 0 1 2 3 --wandb wandb-demo
   ```

3. During training, the program will automatically create a sub-folder `./weights/{model_name}`  to save model checkpoints/parameters.

### Inference

Simply run the `./mxnetseg/eval.py` with arguments need to be specified

```shell
python eval.py --model fcn --backbone resnet18 --checkpoint fcn_resnet18_Cityscapes_20191900_310600_best.params --ctx 0 --data cityscapes --crop 768 --base 2048 --mode val --ms
```

About the `mode`:

- `val`: to get mIoU and PA metrics on the validation set.
- `test`: to get colored predictions on the test set.
- `testval`: to get colored predictions on the validation set.

## Citations

Please kindly cite our paper if you feel our codes help in your research.

```BibTex
@article{tang2020attention,
  title={Attention-guided Chained Context Aggregation for Semantic Segmentation},
  author={Tang, Quan and Liu, Fagui and Jiang, Jun and Zhang, Yu},
  journal={arXiv preprint arXiv:2002.12041},
  year={2020}
}

@article{tang2021eprnet,
  title={EPRNet: Efficient Pyramid Representation Network for Real-Time Street Scene Segmentation},
  author={Tang, Quan and Liu, Fagui and Jiang, Jun and Zhang, Yu},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2021},
  publisher={IEEE}
}
```

