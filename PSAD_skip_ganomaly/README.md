# PSAD with Skip-GANomaly

This folder is one of the benchmark models (Skip-GANomaly) of our paper, A Comprehensive Real-World Photometric Stereo Dataset for Unsupervised Anomaly Detection.

## ****Datasets****

You need to download our dataset, the PSAD dataset.

Afterwards, in the `PSAD_config.py` file, change the `dataset_path` to your own path.

## Config

PSAD-related options are contained in the file `PSAD_config.py`. 

```python
##### plus option for PSAD #####
dataset_path = "~/your/path"
version = "single_rgb" # 'single_rgb', 'single_normal', 'single_gx', 'single_gy', 'single_gauss_curvature', 'single_mean_curvature', 'multi_early_rgb', 'multi_late_mean_pooling', 'multi_late_max_pooling', 'multi_late_mean_max_pooling', 'multi_channel_attention_rgb', 'multi_channel_attention_rgb+normal', 'multi_channel_attention_albedo+normal'
################################
```

By changing the `version`, the experiments in our paper can be reproduced. For example, if `version : "single_normal"` is set, Table 4. surface normal experiment of the paper is reproduced.

The remaining SKip-GANomaly options (batch size, image size, epoch, etc.) are configurable via argumentparser and `options.py`, but are not recommended.

## Train & Classification Test

If you want to run an experiment on all categories, you can run the following command:

```bash
bash PSAD_train_run.sh
```

## ****Acknowledgement****

Our **PSAD_Skip_GANomaly** implementation is baed on https://github.com/samet-akcay/skip-ganomaly. Thanks to authors for sharing the codes.