# MALTUTOR

## Overview
This code repository our paper titled **"Understanding Model Weaknesses: A Path to Strengthening DNN-Based Android Malware Detection"**. In this paper, we take the first step to train the uncertainty estimatin model. Subsequently, we clustered malware samples based on the output of the uncertainty model. Finally, we train a robust model.

## Dependencies:
We develop the codes on Windows operation system, and run the codes on Ubuntu 20.04. The codes depend on Python 3.8.10. Other packages (e.g., TensorFlow) can be found in the `./requirements.txt`.
## Datasets:
You can find the hashes of the samples used in our experiments in the `Training/config` folder.

##  0.Build Experiment Env
1. **Extract and Load the Docker Image**
   - First, decompress the MALTUTOR environment package:
     ```bash
     gzip -d maltutor.tar.gz
     ```
   - Next, load the extracted Docker image into Docker:
     ```bash
     docker load -i maltutor.tar
     ```

2. **Run the Docker Container**
   - Once loaded, start the MALTUTOR Docker container and enter an interactive Bash terminal with:
     ```bash
     docker run -it maltutor /bin/bash
     ```
   - At this point, you are inside the container environment, where you can run the code and experiments.

### 1.Hyperparameter Description 

- **train_data_type**: Type of training dataset, options are `drebin` and `malradar`, which specify the source of data used for model training.

- **val_type**: Uncertainty evaluation strategy for model assessment. Options are `self_val` and `cross_val`, representing self-validation (self) and cross-validation (cross) strategies.

- **n_clusters**: Number of malware clusters, which can be any integer greater than 1. It is recommended to try different cluster sizes and combine them with different uncertainty evaluation strategies to optimize the hard-to-learn sample set. Suggested values in this experiment are 3, 5, 7, 9, and 11.

- **evaluate_type**: Type of evaluation experiment. Options are:
  - `dataset`: Indicates that the test set is from a different source than the training set (RQ1).
  - `ood`: Evaluates different types of malware (RQ2 and RQ3).

- **test_data_type**: Type of test dataset. When `evaluate_type` is `dataset`, `test_data_type` should be the opposite of `train_data_type`. For example, if `train_data_type` is `drebin`, then `test_data_type` should be `malradar`, and vice versa.

- **feature_type**: Choice of detector type. Options are `drebin` (deepdrebin), `apiseq` (droidectc), and `opcodeseq` (deepdroid).

- **robust_type**: Type of model for robustness evaluation, specifically for evaluating the `maltutor model` and `rand model`. `cl` represents the `maltutor model`, while `ca` represents the `rand model`.

- **comparative_type**: Strategy for comparing model performance, used to evaluate `w-uc`, `w-family`, `smote`, and `sampling`. Options include:
  - `smote`: smote model
  - `sampling`: sampling model
  - `weight`: w-family model
  - `cls`: w-uc model


### 2. Uncertainty Evaluation and Malware Sample Classification 

1. **Evaluate Uncertainty**
   - Navigate to the `Training` directory:
     ```bash
     cd Training
     ```
   - Run the following command to train and evaluate the uncertainty model:
     ```bash
     python3.8 training_uncertainty_model.py -train_data_type drebin -val_type self_val -feature_type drebin
     ```
   - This step evaluates the uncertainty of samples based on the self-validation strategy, using the `drebin` dataset and the `deepdrebin` model.
   - Upon completion, `bayesian` and `vanilla` folders will be created in `./Training/output/drebin/drebin/self_val` to store the model outputs.

2. **Extract Uncertainty Features and Save Metrics**
   - Go to the `dataset_reconstruction` directory:
     ```bash
     cd dataset_reconstruction
     ```
   - Run the following command to extract features and calculate relevant metrics from the model's output:
     ```bash
     python3.8 uc_feature_extrctor.py -train_data_type drebin -val_type self_val -feature_type drebin
     ```
   - The metrics will be saved as a CSV file at `./dataset_reconstruction/uc_metrics_csv/drebin_drebin_self_val.csv`.

3. **Cluster Malware Samples**
   - Run the following command to cluster the malware samples into 3 categories based on uncertainty features:
     ```bash
     python3.8 sample_classifier.py -train_data_type drebin -val_type self_val -feature_type drebin -n_clusters 3
     ```
   - The clustering model will be saved at `./dataset_reconstruction/encoder_model/drebin_drebin_self_mal.h5`, and the clustering results will be saved to `./dataset_reconstruction/inter_file/drebin_drebin_self_3.csv`.


### 3. Train the Maltutor Model 
- Navigate to the `Training` directory:
     ```bash
     cd Training
     ```
- Run the following command to train the Maltutor Model:
     ```bash
     python3.8 training_robust_maltutor.py -train_data_type drebin -val_type self -feature_type drebin -n_clusters 3
     ```



The trained model will be saved in the directory ./Training/Model/CL_robust_model/self/drebin_drebin_3.

***Note: If the model files already exist in this directory, the training process will not be executed.***


### 4. Model Evaluation 

#### RQ1

1. **Evaluate the Maltutor Model Trained on Hard-to-Learn Samples (Self-Validation Strategy)**
   - Run the following command to evaluate the `Maltutor` model trained with self-validation strategy:
     ```bash
     python3.8 evaluate_maltutor_model.py -evaluate_type dataset -train_data_type drebin -robust_type cl -feature_type drebin -test_data_type malradar -val_type self
     ```

2. **Evaluate the Rand Model**
   - Use the command below to evaluate the Rand Model (`ca1` indicates the first of three control models):
     ```bash
     python3.8 evaluate_maltutor_model.py -evaluate_type dataset -train_data_type drebin -robust_type ca1 -feature_type drebin -test_data_type malradar -val_type self
     ```

3. **Evaluate Comparative Models**
   - Use the following command to evaluate different types of comparative models, including `sampling`, `smote`, `cls`, and `weight`:
     ```bash
     python3.8 evaluate_comparative_model.py -evaluate_type dataset -train_data_type drebin -test_data_type malradar -comparative_type sampling
     ```
   - Replace `sampling` with other comparative types (`smote`, `cls`, or `weight`) to evaluate different models.

#### RQ2 & RQ3: OOD (Out-of-Distribution) Evaluation

- Follow the same script usage as in RQ1, but set `evaluate_type` to `ood` and `test_data_type` to specific malware types such as `adware`, `smsware`, `scareware`, or `ransom`.




