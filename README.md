# Xray-Anomaly-Detection
---
## Goal & Motivations
Anomaly detection is by definition _Detecting everything that is not normal_. Therefore supervised approaches are not suited and unsupervised or semi-supervised method are preferred. Moreover, there are often not enough labeled samples for a proper supervised training. The general approach is to learn the distribution of normal samples and detect element out of the distribution. It this assume that all/most of the available data comes from the normal distribution. 

The goal of this project is to detect anomalies in musculoskeletal radiograph of upper limb using unsupervised and semi-supervised methods.

## Dataset
The exploration of unsupervised and semi-supervised settings are made on the MURA dataset. A dataset of around 40'000 upper limb x-rays labeled for anomaly detection. The dataset labels are not so imbalance but we will simulate an imbalance for our research purposes.

## Git Structure


```
.
├── .gitignore
├── Code
│   ├── Figures_script
│   │   ├── data_repartition.py
│   │   ├── data_split_summary.py
│   │   ├── least_most_anomalous.py
│   │   ├── online_preprocessing_sample.py
│   │   ├── raw_data_sample.py
│   │   ├── rectangle_cropping.py
│   │   ├── results_barplot.py
│   │   ├── segmentation.py
│   │   └── sphere_diagnostic_summary_DMSAD.py
│   ├── scripts
│   │   ├── ARAE
│   │   │   ├── ARAE_train_script.py
│   │   │   └── ARAE_train_script_hands.py
│   │   ├── DROCC
│   │   │   ├── DROCC-LF_train_script.py
│   │   │   └── DROCC_train_script.py
│   │   ├── Joint_Training
│   │   │   ├── JointDSAD_frac_script_.py
│   │   │   ├── JointDeepSAD_train_script.py
│   │   │   ├── JointDeepSVDD_train_script.py
│   │   │   └── JointDeepSVDD_train_script_soft.py
│   │   ├── Joint_Training_Subspace
│   │   │   ├── JointDeepSADSubspace_train_script.py
│   │   │   └── JointDeepSVDDSubspace_train_script.py
│   │   ├── Multi-modal
│   │   │   ├── JointDMSAD_train_script.py
│   │   │   └── JointDMSVDD_train_script.py
│   │   ├── Preprocessing
│   │   │   ├── generate_data_info_script.py
│   │   │   └── preprocessing_script.py
│   │   ├── Separate_Training
│   │   │   ├── DeepSAD_train_script.py
│   │   │   └── DeepSVDD_train_script.py
│   │   └── results_processing
│   │       ├── postprocessing_diagnostic.py
│   │       ├── postprocessing_single_diagnostic.py
│   │       ├── process_ae-emded_experiment_results.py
│   │       ├── process_emded_experiment_results.py
│   │       └── sphere_diagnostic.py
│   └── src
│       ├── __init__.py
│       ├── datasets
│       │   ├── MURADataset.py
│       │   ├── __init__.py
│       │   └── transforms.py
│       ├── models
│       │   ├── ARAE.py
│       │   ├── DMSAD.py
│       │   ├── DMSVDD.py
│       │   ├── DROCC.py
│       │   ├── DeepSAD.py
│       │   ├── JointDeepSAD.py
│       │   ├── JointDeepSVDD.py
│       │   ├── __init__.py
│       │   ├── networks
│       │   │   ├── AE_ResNet18_dual.py
│       │   │   ├── AE_ResNet18_net.py
│       │   │   ├── ResNet18_binary.py
│       │   │   ├── ResNetBlocks.py
│       │   │   └── __init__.py
│       │   └── optim
│       │       ├── ARAE_trainer.py
│       │       ├── CustomLosses.py
│       │       ├── DMSAD_Joint_trainer.py
│       │       ├── DMSVDD_Joint_trainer.py
│       │       ├── DROCC_trainer.py
│       │       ├── DeepSAD_Joint_trainer.py
│       │       ├── DeepSAD_trainer.py
│       │       ├── DeepSVDD_Joint_trainer.py
│       │       ├── __init__.py
│       │       └── autoencoder_trainer.py
│       ├── postprocessing
│       │   ├── __init__.py
│       │   └── scoresCombiner.py
│       ├── preprocessing
│       │   ├── __init__.py
│       │   ├── cropping_rect.py
│       │   ├── get_data_info.py
│       │   └── segmentation.py
│       └── utils
│           ├── __init__.py
│           ├── results_processing.py
│           └── utils.py
├── Figures
│   ├── data_repartition.pdf
│   ├── least_most_anomalous_DSAD.pdf
│   ├── least_most_anomalous_DSVDD.pdf
│   ├── online_preprocessing_sample.pdf
│   ├── raw_sample.pdf
│   ├── rect_cropping_sample.pdf
│   ├── results_barplot.pdf
│   ├── segmentation_sample.pdf
│   ├── semisupervized_data_split_summary.pdf
│   └── unsupervized_data_split_summary.pdf
├── LICENSE
├── README.md
└── data
    └── data_info.csv
```
