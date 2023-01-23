# [WACV2023] Dense Prediction with Attentive Feature Aggregation
This is the official implementation of our paper **"Dense Prediction with Attentive Feature Aggregation"**.

[Yung-Hsu Yang](https://royyang0714.github.io/), [Thomas E. Huang](https://www.thomasehuang.com/), [Min Sun](https://aliensunmin.github.io/), [Samuel Rota Bulò](https://research.mapillary.com/team), [Peter Kontschieder](https://research.mapillary.com/team), [Fisher Yu](https://www.yf.io/)

[[Paper](https://arxiv.org/abs/2111.00770)] [[Project](https://www.vis.xyz/pub/dla-afa/)]

## Abstract
Aggregating information from features across different layers is essential for dense prediction models. Despite its limited expressiveness, vanilla feature concatenation dominates the choice of aggregation operations. In this paper, we introduce Attentive Feature Aggregation (AFA) to fuse different network layers with more expressive non-linear operations. AFA exploits both spatial and channel attention to compute weighted averages of the layer activations. Inspired by neural volume rendering, we further extend AFA with Scale-Space Rendering (SSR) to perform a late fusion of multi-scale predictions. AFA is applicable to a wide range of existing network designs. Our experiments show consistent and significant improvements on challenging semantic segmentation benchmarks, including Cityscapes and BDD100K at negligible computational and parameter overhead. In particular, AFA improves the performance of the Deep Layer Aggregation (DLA) model by nearly 6% mIoU on Cityscapes. Our experimental analyses show that AFA learns to progressively refine segmentation maps and improve boundary details, leading to new state-of-the-art results on boundary detection benchmarks on NYUDv2 and BSDS500.

## Installation
Please refer to [INSTALL.md](./readme/INSTALL.md) for installation and to [PREPARE_DATASETS.md](./readme/PREPARE_DATASETS.md) for dataset preparation.

## Get Started
Please see [GETTING_STARTED.md](./readme/GETTING_STARTED.md) for the basic usage.

## Model Zoo
### Cityscapes
| Model | Crop Size | Batch Size | Training Epochs | mIoU (val) | mIoU (test) | config | weights | Preds | Visuals |
|:-----:|:---------:|:----------:|:---------------:|:----------:|:-----------:|:------:|:-------:|:-----:|:-------:|
| AFA-DLA (Train) | 1024x2048 | 8 | 375 | 85.14 | - | [config](./configs/cityscapes/afa_dla_up_ocr_ssr_dla102x.yaml) | [model](https://dl.cv.ethz.ch/afa/cityscapes/afa_dla_up_ocr_ssr_dla102x.ckpt) | [val](https://dl.cv.ethz.ch/afa/cityscapes/Preds/val.zip) | [val](https://dl.cv.ethz.ch/afa/cityscapes/Visuals/val.zip) |
| AFA-DLA (Train + Val) | 1024x1024 | 16 | 275 | - | 83.58 | [config](./configs/cityscapes/afa_dla_up_ocr_ssr_dla102x_cv_3.yaml) | [model](https://dl.cv.ethz.ch/afa/cityscapes/afa_dla_up_ocr_ssr_dla102x_cv_3.ckpt) | [test](https://dl.cv.ethz.ch/afa/cityscapes/Preds/test.zip) | [test](https://dl.cv.ethz.ch/afa/cityscapes/Visuals/test.zip) |

### BDD100K
| Model | Crop Size | Batch Size | Training Epochs | mIoU (val) | mIoU (test) | config | weights | Preds | Visuals |
|:-----:|:---------:|:----------:|:---------------:|:----------:|:-----------:|:------:|:-------:|:-----:|:-------:|
| AFA-DLA | 720x1280 | 16 | 200 | 67.46 | 58.70 | [config](./configs/bdd100k/afa_dla_up_ocr_ssr_dla169.yaml) | [model](https://dl.cv.ethz.ch/afa/bdd100k/afa_dla_up_ocr_ssr_dla169.ckpt) | [val](https://dl.cv.ethz.ch/afa/bdd100k/Preds/val.zip) \| [test](https://dl.cv.ethz.ch/afa/bdd100k/Preds/test.zip) | [val](https://dl.cv.ethz.ch/afa/bdd100k/Visuals/val.zip) \| [test](https://dl.cv.ethz.ch/afa/bdd100k/Visuals/test.zip) |

### NYUDv2
| Model | Crop Size | Batch Size | Training Epochs |  ODS  |  OIS  | config | weights | Preds | Visuals |
|:-----:|:---------:|:----------:|:---------------:|:-----:|:-----:|:------:|:-------:|:-----:|:-------:|
| AFA-DLA (RGB) |  480x480  | 16 | 54 | 0.762 | 0.775 | [config](./configs/nyud/afa_dla_up_dla34.yaml) | [model](https://dl.cv.ethz.ch/afa/nyud/afa_dla_up_dla34.ckpt) | [test](https://dl.cv.ethz.ch/afa/nyud/mats.zip) | [test](https://dl.cv.ethz.ch/afa/nyud/edges.zip) |
| AFA-DLA (HHA) |  480x480  | 16 | 54 | 0.718 | 0.730 | [config](./configs/nyud/afa_dla_up_dla34_hha.yaml) | [model](https://dl.cv.ethz.ch/afa/nyud_hha/afa_dla_up_dla34_hha.ckpt) | [test](https://dl.cv.ethz.ch/afa/nyud_hha/mats.zip) | [test](https://dl.cv.ethz.ch/afa/nyud_hha/edges.zip) |

### BSDS500
| Model | Crop Size | Batch Size | Training Epochs |  ODS  |  OIS  | config | weights | Preds | Visuals |
|:-----:|:---------:|:----------:|:---------------:|:-----:|:-----:|:------:|:-------:|:-----:|:-------:|
| AFA-DLA | 416x416 | 16 | 14 | 0.812 | 0.826 | [config](./configs/bsds/afa_dla_up_dla34.yaml) | [model](https://dl.cv.ethz.ch/afa/bsds/afa_dla_up_dla34.ckpt) | [test](https://dl.cv.ethz.ch/afa/bsds/mats.zip) | [test](https://dl.cv.ethz.ch/afa/bsds/edges.zip) |
| AFA-DLA (PASCAL) | 416x416 | 16 | 20 | 0.810 | 0.826 | [config](./configs/bsds/afa_dla_up_dla34_pascal.yaml) | [model](https://dl.cv.ethz.ch/afa/bsds_pascal/afa_dla_up_dla34_pascal.ckpt) | [test](https://dl.cv.ethz.ch/afa/bsds_pascal/mats.zip) | [test](https://dl.cv.ethz.ch/afa/bsds_pascal/edges.zip) |

## Citation
```
@inproceedings{yang2023dense,
    title={Dense prediction with attentive feature aggregation},
    author={Yang, Yung-Hsu and Huang, Thomas E and Sun, Min and Bul{\`o}, Samuel Rota and Kontschieder, Peter and Yu, Fisher},
    booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
    pages={97--106},
    year={2023}
}
```

## Acknowledgement
The codbase is developed from [NVIDIA segmentation](https://github.com/NVIDIA/semantic-segmentation). We deeply thank for the help of their open-sourced code.