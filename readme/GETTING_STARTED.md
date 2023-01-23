# Getting started
This guidline provides tutorials to train and evaluate our AFA model. Before getting started, make sure you have finished all the installation steps.

## Training
- Train on the dataset you want by specifying the corresponding config file.
- The first time of training, a centroid file has to be built for the dataset. The centroid file is used during training to sample from the dataset in a class-uniform way.
- You can change the scales for multi-scale inference ``n_scales_inference`` to save time or prevent from running of out GPU memory.
```bash
# Cityscapes train set only
python -m afa.trainer fit --config configs/cityscapes/afa_dla_up_ocr_ssr_dla102x.yaml

# Cityscapes on both train and val set
python -m afa.trainer fit --config configs/cityscapes/afa_dla_up_ocr_ssr_dla102x_cv_3.yaml

# BDD100K
python -m afa.trainer fit --config configs/bdd100k/afa_dla_up_ocr_ssr_dla169.yaml
```

## Inference
### Evaluation on Validation set
- It will reproduce our best results for each dataset on its validation set. Download the checkpoints from the our [model zoo](../README.md#Model-Zoo).
- You might need to change the configs according to your hardware setting.
```bash
# Cityscapes
python -m afa.trainer test --config configs/cityscapes/afa_dla_up_ocr_ssr_dla102x.yaml --weights ${CHECKPOINTS} --do_flip --seg_fix

# BDD100K
python -m afa.trainer test --config configs/bdd100k/afa_dla_up_ocr_ssr_dla169.yaml --weights ${CHECKPOINTS} --do_flip
```

### Visualization
- It will generate the predicted results under ``${WORK_DIR}/${EXP_NAME}/${VERSION}/best_images``.
- You can choose the data split by specifying ``args.mode``.
```bash
# Cityscapes
python -m afa.trainer predict --config configs/cityscapes/afa_dla_up_ocr_ssr_dla102x.yaml --weights ${CHECKPOINTS} --do_flip --seg_fix

# BDD100K
python -m afa.trainer predict --config configs/bdd100k/afa_dla_up_ocr_ssr_dla169.yaml --weights ${CHECKPOINTS} --do_flip
```

- You can also generate the predicted results for the custom folder.
```bash
# Use Cityscapes checkpoints
python -m afa.trainer predict --config configs/cityscapes/afa_dla_up_ocr_ssr_dla102x.yaml --weights ${CHECKPOINTS} --do_flip --folder ${YOUR_IMAGE_FOLDER}

# Use BDD100K checkpoints
python -m afa.trainer predict --config configs/bdd100k/afa_dla_up_ocr_ssr_dla169.yaml --weights ${CHECKPOINTS} --do_flip --folder ${YOUR_IMAGE_FOLDER}
```

### Benchmark Submission
- It will generate the submission results for Cityscapes and BDD-100K under ``${WORK_DIR}/${EXP_NAME}/${VERSION}/submit``.
```bash
# Cityscapes
python -m afa.trainer predict --config configs/cityscapes/afa_dla_up_ocr_ssr_dla102x_cv_3.yaml --weights ${CHECKPOINTS} --do_flip --seg_fix --dump_for_submission --mode test

# BDD100K
python -m afa.trainer predict --config configs/bdd100k/afa_dla_up_ocr_ssr_dla169.yaml --weights ${CHECKPOINTS} --do_flip --dump_for_submission --mode test
```

- For BDD100K, you will need to convert the submit files to [RLE](https://github.com/bdd100k/bdd100k/blob/b7e1781317784317e4e715ab325515ade73978a9/bdd100k/label/to_rle.py) format for the official evaluation.

## Boundary Detection
We additionally provide experiments on the boundary detection task.

### Training
The following commands will reproduce our best results for edge detection.
```bash
# BSDS500
python -m afa.trainer fit --config configs/bsds/afa_dla_up_dla34.yaml

# BSDS500 + Pascal
python -m afa.trainer fit --config configs/bsds/afa_dla_up_dla34_pascal.yaml

# NYUDv2 Images
python -m afa.trainer fit --config configs/nyud/afa_dla_up_dla34.yaml

# NYUDv2 HHA
python -m afa.trainer fit --config configs/nyud/afa_dla_up_dla34_hha.yaml
```

### Inference
- First run the following commands to obtain the edge prediction maps for each test image under ``${WORK_DIR}/${EXP_NAME}/${VERSION}/best_images``.

```bash
# BSDS500
python -m afa.trainer predict --config configs/bsds/afa_dla_up_dla34.yaml --weights ${CHECKPOINTS}

# BSDS500 + Pascal
python -m afa.trainer predict --config configs/bsds/afa_dla_up_dla34_pascal.yaml --weights ${CHECKPOINTS}

# NYUDv2 Images
python -m afa.trainer predict --config configs/nyud/afa_dla_up_dla34.yaml --weights ${CHECKPOINTS}

# NYUDv2 HHA
python -m afa.trainer predict --config configs/nyud/afa_dla_up_dla34_hha.yaml --weights ${CHECKPOINTS}
```

- We use the evaluation code provided in the [PiDiNet repository](https://github.com/zhuoinoulu/pidinet#evaluation). You can follow the same steps to obtain the evaluation results.
- Update the paths in each script to the saved model predictions.
