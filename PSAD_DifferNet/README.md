# PSAD with DifferNet

This folder is one of the benchmark models (DifferNet) of our paper, A Comprehensive Real-World Photometric Stereo Dataset for Unsupervised Anomaly Detection.

## ****Datasets****

You need to download our dataset, the PSAD dataset.

Afterwards, in the `config.py` file, change the `dataset_path` to your own path.

## Config

It is recommended to change only the plus option for PSAD part of the `config.py` file and not touch the rest of the parameters.

```python
...

##### plus option for PSAD #####
dataset_path = "~/your/path"
version = "single_rgb" # 'single_rgb', 'single_normal', 'single_gx', 'single_gy', 'single_gauss_curvature', 'single_mean_curvature', 'multi_early_rgb', 'multi_late_mean_pooling', 'multi_late_mean_pooling_rgb+normal', 'multi_late_mean_pooling_albedo+normal', 'multi_late_max_pooling', 'multi_late_mean_max_pooling'
###############################
```

By changing the `version`, the experiments in our paper can be reproduced. For example, if `version : "single_normal"` is set, Table 4. surface normal experiment of the paper is reproduced.

## Train & Classification Test

If you want to run an experiment on all categories, you can run the following command:

```bash
bash PSAD_train_run.sh
```

## Localization Test

If you want to see the results of the localization test in Table 5, edit the `version` option in the `config.py` file appropriately and enter the following command. 

```bash
bash PSAD_localization_test_run.sh
```

For example, if `version : "single_rgb"` is set, the **Single** part of Table 5. of the paper is reproduced.

## ****Acknowledgement****

Our **PSAD_DifferNet** implementation is baed on https://github.com/marco-rudolph/differnet. Thanks to authors for sharing the codes.