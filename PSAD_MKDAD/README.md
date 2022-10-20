# PSAD with MKDAD

This folder is one of the benchmark models (MKDAD) of our paper, A Comprehensive Real-World Photometric Stereo Dataset for Unsupervised Anomaly Detection.

## ****Datasets****

You need to download our dataset, the PSAD dataset.

Afterwards, in the `configs/config.yaml` file, change the `dataset_path` to your own path.

## Config

It is recommended to change only the plus option for PSAD part of the `configs/config.yaml` file and not touch the rest of the parameters.

```yaml
...

##### plus option for PSAD #####
last_checkpoint: 600
dataset_path : '~/your/path'
localization_test: True # True:For Localization Test --- False:For Detection
version : "multi_early_rgb" # 'single_rgb', 'single_normal', 'single_gx', 'single_gy', 'single_gauss_curvature', 'single_mean_curvature', 'multi_early_rgb', 'multi_early_rgb+normal', 'multi_early_albedo+normal', 'multi_late_mean_pooling', 'multi_late_max_pooling', 'multi_late_mean_max_pooling'
#################################
```

By changing the `version`, the experiments in our paper can be reproduced. For example, if `version : "single_normal"` is set, Table 4. surface normal experiment of the paper is reproduced.

## Train & Classification Test

If you want to run an experiment on all categories, you can run the following command:

```bash
bash PSAD_train_run.sh
```

## Localization Test

With the `localization_test` option of `configs/config.yaml` set to `True`, the localization test of the paper can be reproduced by modifying the version option appropriately.

```bash
bash PSAD_test_run.sh
```

For example, if `version : "multi_early_rgb+normal"` is set, the **Multi+Normal** part of Table 5. of the paper is reproduced.

## ****Acknowledgement****

Our **PSAD_MKDAD** implementation is baed on https://github.com/Niousha12/Knowledge_Distillation_AD. Thanks to authors for sharing the codes.