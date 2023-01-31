# Installation Guideline

## Prerequisites
- Linux (Ubuntu)
- Python: 3.8.13
- PyTorch: 1.11 (with CUDA 11.3, torchvision 0.12.0)
- PyTorch Lightning: 1.6.5

## Installation
- Install the prerequisites
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

- Prepare these directories under the disk with large storage for dataset, checkpoints and visualization.
```bash
mkdir ${ASSET_DIR}
mkdir ${WORK_DIR}

cd ${ASSET_DIR}
mkdir data
```

- Our code will use ``asset_dirs`` and ``work_dirs`` under the repo root by default as ``${ASSET_DIR}`` and ``${WORK_DIR}``, so you might need to symlink them correctly or change the behavior with the [config](../afa/utils/opt.py).

- The experimental results will be saved under ``${WORK_DIR}/${EXP_NAME}/${VERSION}``. You need to specify the experiment name ``${EXP_NAME}`` and we will use the timestamp as the version name ``${VERSION}`` if you do not config it for every experiment.

## Inplace ABN
- We use [Inplace ABN](https://github.com/mapillary/inplace_abn) for most of our experiments.

- Please install it with the latest version.
```bash
git clone https://github.com/mapillary/inplace_abn.git
cd inplace_abn
python setup.py install
```

## Seg Fix
- We use the offset provided from [Seg Fix](https://github.com/openseg-group/openseg.pytorch/blob/master/MODEL_ZOO.md#use-offline-generated-offsets) to do the post-processing for our Cityscapes final results.

- Download the ``offset_semantic.zip`` file. Unzip, and place (or symlink) the data as below.
```bash
${ASSET_DIR}
└── data
    └── Cityscapes
        ├── leftImg8bit_trainvaltest
        ├── gtFine_trainvaltest
        ├── leftImg8bit_trainextra
        ├── gtCoarse
        ├── refinement
        └── offset_semantic
            ├── val
            └── test_offset
```