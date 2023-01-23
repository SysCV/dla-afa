# Dataset Preparation
To reproduce the results in the paper , you will need to setup these datasets.

## Cityscapes
- First of all, please request the dataset from [here](https://www.cityscapes-dataset.com/). You need multiple files.
    - leftImg8bit_trainvaltest.zip
    - gtFine_trainvaltest.zip
    - leftImg8bit_trainextra.zip
    - gtCoarse.zip

- If you prefer to use command lines (e.g., `wget`) to download the dataset,
```bash
# First step, obtain your login credentials.
Please register an account at https://www.cityscapes-dataset.com/login/.

# Second step, log into cityscapes system, suppose you already have a USERNAME and a PASSWORD.
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=USERNAME&password=PASSWORD&submit=Login' https://www.cityscapes-dataset.com/login/

# Third step, download the zip files you need.
wget -c -t 0 --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3

# The corresponding packageID is listed below,
1  -> gtFine_trainvaltest.zip (241MB)            md5sum: 4237c19de34c8a376e9ba46b495d6f66
2  -> gtCoarse.zip (1.3GB)                       md5sum: 1c7b95c84b1d36cc59a9194d8e5b989f
3  -> leftImg8bit_trainvaltest.zip (11GB)        md5sum: 0a6e97e94b616a514066c9e2adb0c97f
4  -> leftImg8bit_trainextra.zip (44GB)          md5sum: 9167a331a158ce3e8989e166c95d56d4
```

- Download the auto-labeled coarse data provided by [NVIDIA](https://github.com/NVIDIA/semantic-segmentation/blob/main/PREPARE_DATASETS.md#download-dataset)

- Unzip, and place (or symlink) the data as below.
```bash
${ASSET_DIR}
└── data
    └── Cityscapes
        ├── leftImg8bit_trainvaltest
        |   └── leftImg8bit
        |       ├── train
        |       |   ├── aachen
        |       |   |   ├── aachen_000000_000019_leftImg8bit.png
        |       |   |   ├── aachen_000001_000019_leftImg8bit.png
        |       |   |   ├── ...
        |       |   ├── bochum
        |       |   ├── ...
        |       ├── val
        |       └── test
        ├── gtFine_trainvaltest
        |    └── gtFine
        |       ├── train
        |       |   ├── aachen
        |       |   |   ├── aachen_000000_000019_gtFine_color.png
        |       |   |   ├── aachen_000000_000019_gtFine_instanceIds.png
        |       |   |   ├── aachen_000000_000019_gtFine_labelIds.png
        |       |   |   ├── aachen_000000_000019_gtFine_polygons.json
        |       |   |   ├── ...
        |       |   ├── bochum
        |       |   ├── ...
        |       ├── val
        |       └── test
        ├── leftImg8bit_trainextra
        |   └── leftImg8bit
        |       ├── train_extra
        |       |   ├── augsburg
        |       |   ├── bad-honnef
        |       |   ├── ...
        ├── gtCoarse
        |   └── gtCoarse
        |       ├── train
        |       ├── train_extra
        |       └── val
        └── refinement
            └── train_extra
                ├── augsburg
                ├── bad-honnef
                ├── ...
```

## BDD-100K
- First of all, please request the dataset from [here](https://www.bdd100k.com/). Download ``Images`` and ``Segmentation``. The downloaded files are ``bdd100k_images.zip`` and ``bdd100k_sem_seg_labels_trainval.zip``.

- Unzip, and place (or symlink) the data as below.
```bash
${ASSET_DIR}
└── data
    └── bdd100k
        ├── images
        |   └── 10k
        |       ├── train
        |       ├── ├── 0004a4c0-d4dff0ad.jpg
        |       ├── ├── 00054602-3bf57337.jpg
        |       ├── ├── ...
        |       ├── val
        |       └── test
        └── labels
            └──sem_seg
                ├── colormaps
                |    ├── train
                |    └── val
                ├── masks
                └── polygons
```

## Boundary Detection
- We use the BSDS500, PASCAL VOC Context, and NYUDv2 datasets. You can obtain augmented versions of the datasets following the instructions in the [RCF repository](https://github.com/yun-liu/rcf#testing-rcf).
- Untar the datasets and place (or symlink) the data as below.
```bash
${ASSET_DIR}
└── data
    ├── BSDS500
    |   ├── HED-BSDS
    |   |   ├── train_pair.lst
    |   |   ├── test_pair.lst
    |   |   ├── train
    |   |   |   └── ...
    |   |   └── test
    |   |       └── ...
    |   └── PASCAL
    |       ├── train_pair.lst
    |       ├── aug_data
    |       |   └── ...
    |       └── aug_gt
    |           └── ...
    └── NYUD
        ├── image-train.lst
        ├── image-test.lst
        ├── hha-train.lst
        ├── hha-test.lst
        ├── train
        |   ├── GT
        |   |   └── ...
        |   ├── HHA
        |   |   └── ...
        |   └── Images
        |       └── ...
        └── test
            ├── HHA
            |   └── ...
            └── Images
                └── ...
```
