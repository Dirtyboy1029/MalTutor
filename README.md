# MalTutor

## Overview
MalTutor is the implementation of our paper **"Understanding Model Weaknesses: A Path to Strengthening DNN-Based Android Malware Detection"**. The project focuses on:
1. Training uncertainty estimation models
2. Clustering malware samples based on uncertainty output
3. Training robust malware detection models

## Environment Setup

### Prerequisites
- Ubuntu 20.04 (recommended) or Windows
- Python 3.8.10
- Docker
- Additional dependencies listed in `requirements.txt`

### Docker Installation
1. Extract the environment package:
```bash
gzip -d maltutor.tar.gz
```

2. Load the Docker image:
```bash
docker load -i maltutor.tar
```

3. Run the container:
```bash
docker run -it maltutor /bin/bash
```

## Project Structure
```plaintext
/MalTutor/
├── dataset/                          # Dataset resources
│   ├── apk/                         # APK files
│   ├── family_source_file/         # Dataset family labels
│   └── naive_pool/                 # Naive data pool
├── dataset_reconstruction/          # Dataset processing
│   ├── encoder_model/              # Encoder models
│   ├── inter_file/                
│   ├── sample_classifier.py        # Malware classification
│   ├── uc_feature_extrctor.py     # Uncertainty features
│   ├── uc_metrics_csv/            # Metrics storage
│   └── utils.py                   
├── Model/                          # Model storage
├── requirements.txt                # Dependencies
└── Training/                       # Training scripts
    ├── comparative_model_conf.py   # Comparison configs
    ├── config/                     # App hash configs
    ├── core/                       # Core modules
    ├── evaluate_*.py              # Evaluation scripts
    ├── feature_extractor.py       
    ├── output/                    # Training results
    ├── training_*.py              # Training scripts
    └── utils.py
```

## Configuration Parameters

### Key Parameters
- `train_data_type`: Dataset source (`drebin` or `malradar`)
- `val_type`: Validation strategy (`self_val` or `cross_val`)
- `n_clusters`: Number of malware clusters (recommended: 3, 5, 7, 9, 11)
- `feature_type`: Detector type:
  - `drebin`: DeepDrebin
  - `apiseq`: Droidectc
  - `opcodeseq`: DeepDroid
- `evaluate_type`: Evaluation mode:
  - `dataset`: Cross-dataset testing
  - `ood`: Out-of-distribution testing
- `test_data_type`: Test dataset selection
- `robust_type`: Model type (`cl` for MalTutor, `ca` for random)
- `comparative_type`: Baseline models:
  - `smote`: SMOTE model
  - `sampling`: Sampling model
  - `weight`: W-Family model
  - `cls`: W-UC model

## Usage Guide

### 1. Uncertainty Estimation & Clustering

```bash
# 1. Train uncertainty model
cd Training
python3.8 training_uncertainty_model.py -train_data_type drebin -val_type self_val -feature_type drebin

# 2. Extract uncertainty features
cd dataset_reconstruction
python3.8 uc_feature_extrctor.py -train_data_type drebin -val_type self_val -feature_type drebin

# 3. Cluster malware samples
python3.8 sample_classifier.py -train_data_type drebin -val_type self_val -feature_type drebin -n_clusters 3
```

### 2. Model Training
```bash
cd Training
python3.8 training_robust_maltutor.py -train_data_type drebin -val_type self -feature_type drebin -n_clusters 3
```

### 3. Evaluation

#### Cross-Dataset Evaluation (RQ1)
```bash
# Evaluate MalTutor
python3.8 evaluate_maltutor_model.py -evaluate_type dataset -train_data_type drebin -robust_type cl -feature_type drebin -test_data_type malradar -val_type self

# Evaluate Random Model
python3.8 evaluate_maltutor_model.py -evaluate_type dataset -train_data_type drebin -robust_type ca1 -feature_type drebin -test_data_type malradar -val_type self

# Evaluate Baseline Models
python3.8 evaluate_comparative_model.py -evaluate_type dataset -train_data_type drebin -test_data_type malradar -comparative_type sampling
```

#### Out-of-Distribution Testing (RQ2 & RQ3)
Use the same commands as above but with:
- Set `evaluate_type` to `ood`
- Set `test_data_type` to one of:
  - Androzoo variants: `adware`, `smsware`, `backdoorware`, `scareware`, `ransomware`
  - CICMalDroid-2020 variants: `cic2020adware`, `cic2020smsware`, `cic2020bankware`, `cic2020riskware`

## Notes
- Model files are saved in `./Training/Model/CL_robust_model/self/drebin_drebin_3`
- Training will be skipped if model files already exist
- Dataset hashes can be found in `Training/config` folder
