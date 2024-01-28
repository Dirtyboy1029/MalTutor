# MALTUTOR


This code repository our paper titled **MALTUTOR: Enhancing Robustness in DNN-Based Android Malware Detection through Uncertainty-Guided Robust Training**.
 
## Overview
In this paper, we take the first step to explore how we can leverage the prediction uncertainty to improve DNN-based Android malware detection models.
Our key insight is if we can identify uncertainty metrics that differ greatly between correct and incorrect predictions, we can use these metrics to pinpoint the potentially incorrectly-classified samples and correct their classification results accordingly

## Dependencies:
We develop the codes on Windows operation system, and run the codes on Ubuntu 20.04. The codes depend on Python 3.8.10. Other packages (e.g., TensorFlow) can be found in the `./requirements.txt`.

##  Usage
#### 1. Estimate uncertainty
     cd Training 

     python train_base_model.py 


#### 2. malware samples clustering
     cd dataset_reconstruction

     python uc_feature_extrctor.py 

uncertainty metrics save to: Traing_robust_Malware_Detector_via_Label_uncertainty/dataset_reconstruction/uc_metrics_csv

     python sample_classifier.py  ## cluster samples 

inter file save to: Traing_robust_Malware_Detector_via_Label_uncertainty/dataset_reconstruction/inter_file

#### 3. robust training

      cd Training

      python CL_robust_model.py 

## Hyperparameters:
      
      ###  experiment_type: Type of experiment,training correction model or resultant correction.
      ###  save_model: Whether to save trained correction models
      ###  data_type: Data types for training corrective models，(small_drebin,small_multi)
      ###  train_data_size: The effect of the scale of the training data on the corrective model,[1.0,0.8,0.4,0.2,0.1]
      ###  banlance: The effect of whether the training data is balanced or not on the corrective model
