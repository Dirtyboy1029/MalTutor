# -*- coding: utf-8 -*- 
# @Time : 2024/1/7 15:43 
# @Author : DirtyBoy 
# @File : CL_robust_model.py
from core.dataset_lib import build_dataset_from_numerical_data
from core.data_preprocessing import data_preprocessing, data_preprocessing_for_ood
import argparse, os, random
import numpy as np
from core.ensemble.vanilla import Vanilla
import pandas as pd
from core.tools import utils
from core.feature import feature_type_scope_dict


def sort_indices_descending(lst):
    sorted_indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=False)
    return sorted_indices


def load_curriculum(data_type, n_clusters, val_type, feature_type, malware_num):
    classifier_data = pd.read_csv(
        '../dataset_reconstruction/inter_file/' + data_type + '_' + feature_type + '_' + val_type + '_' + str(
            n_clusters) + '.csv')
    malware = classifier_data[classifier_data['gt_label'] == 1]
    curriculum = pd.read_csv(
        '../dataset_reconstruction/inter_file/' + data_type + '_' + feature_type + '_' + val_type + '_' + str(
            n_clusters) + '.csv')[
                     'label'].tolist()[0:malware_num]
    malware_df_set = [malware[malware['label'] == i] for i in range(n_clusters)]
    uc_data = pd.read_csv(
        '../dataset_reconstruction/uc_metrics_csv/' + data_type + '_' + feature_type + '_' + val_type + '_val.csv')[
        ['apk_name', 'no_kld_30', 'with_kld_label_30']]
    malware_df_set = [pd.merge(item, uc_data, on='apk_name', how='inner') for item in malware_df_set]

    malware_fitting_degree = sort_indices_descending(
        [np.sum(np.sqrt(item['no_kld_30'] ** 2 + item['with_kld_label_30'] ** 2)) / item.shape[0] for item in
         malware_df_set])

    curriculum_ = []
    for i in range(n_clusters):
        tmp = []
        for j, index in enumerate(curriculum):
            if index == malware_fitting_degree[i]:
                tmp.append(j)
        curriculum_.append(tmp)
    return curriculum_


def Deepdrebin_experiment(val_type,
                          data_type,
                          feature_type='drebin'):
    robuts_model_path = "../Model/CL_robust_model/" + val_type + '/' + data_type + '_' + feature_type + '_' + str(
        n_clusters)
    if not os.path.isdir(robuts_model_path):

        intermediate_data_saving_dir = '/home/lhd/Training_robust_Malware_Detector_via_uncertainty/Training/config'

        dataset, gt_labels, input_dim, dataX_np, data_filenames = data_preprocessing(
            feature_type=feature_type, data_type=data_type)

        samples_num = len(gt_labels)
        curriculum = load_curriculum(data_type, n_clusters, val_type, feature_type, np.sum(gt_labels))
        easy_malware_curriculum = []
        for item in curriculum[0:n_clusters - 1]:
            easy_malware_curriculum = easy_malware_curriculum + item
        difficult_curriculum = curriculum[-1]
        easy_benign_curriculum = random.choices(list(range(np.sum(gt_labels), samples_num)),
                                                k=len(easy_malware_curriculum))

        difficult_benign_curriculum = []
        for i in range(np.sum(gt_labels), samples_num):
            if i in easy_benign_curriculum:
                pass
            else:
                difficult_benign_curriculum.append(i)
        print('easy curriculum contain samples ' + str(2 * len(easy_benign_curriculum)))
        easy_trainset = build_dataset_from_numerical_data(
            (dataX_np[easy_malware_curriculum + easy_benign_curriculum],
             gt_labels[easy_malware_curriculum + easy_benign_curriculum]))

        # val_dataset, val_gt_labels, _, _, _ = data_preprocessing_for_ood(
        #     feature_type=feature_type,
        #     train_data_type=data_type,
        #     data_type='smsware')

        vanilla = Vanilla(architecture_type='dnn', model_directory=robuts_model_path)
        vanilla_prob, vanilla_training_log = vanilla.fit(train_set=easy_trainset,
                                                         validation_set=dataset,
                                                         input_dim=input_dim,
                                                         EPOCH=20,
                                                         training_predict=False)
        difficult_benign_curriculum = random.choices(difficult_benign_curriculum,
                                                     k=len(difficult_curriculum))
        difficult_trainset = build_dataset_from_numerical_data(
            (dataX_np[difficult_curriculum + difficult_benign_curriculum],
             gt_labels[difficult_curriculum + difficult_benign_curriculum]))
        print('difficult curriculum contain samples ' + str(
            len(difficult_curriculum + difficult_benign_curriculum)))
        vanilla.finetune(train_set=difficult_trainset, validation_set=dataset,
                         input_dim=input_dim,
                         EPOCH=10,
                         test_data=dataX_np,
                         training_predict=False)

        if not os.path.isdir(robuts_model_path):
            vanilla.save_ensemble_weights()


def APIseq_experiment(val_type,
                      data_type,
                      feature_type='apiseq'):
    robuts_model_path = "../Model/CL_robust_model/" + val_type + '/' + data_type + '_' + feature_type + '_' + str(
        n_clusters)
    if not os.path.isdir(robuts_model_path):
        android_features_saving_dir = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/naive_pool'
        intermediate_data_saving_dir = '/home/lhd/Training_robust_Malware_Detector_via_uncertainty/Training/config'
        save_path = os.path.join(intermediate_data_saving_dir,
                                 data_type + '_database' + '.' + feature_type)
        data_filenames, gt_labels = utils.read_joblib(save_path)

        val_data_filenames, val_gt_labels = utils.read_joblib(os.path.join(intermediate_data_saving_dir,
                                                                           'malradar_database' + '.' + feature_type))

        malware_samples_num = np.sum(gt_labels)
        samples_num = len(gt_labels)
        oos_features = [os.path.join(android_features_saving_dir, filename) for filename in data_filenames]
        val_oos_features = [os.path.join(android_features_saving_dir, filename) for filename in val_data_filenames]
        assert feature_type in feature_type_scope_dict.keys(), 'Expected {}, but {} are supported.'.format(
            feature_type, feature_type_scope_dict.keys())
        feature_extractor = feature_type_scope_dict[feature_type](android_features_saving_dir,
                                                                  intermediate_data_saving_dir,
                                                                  update=False,
                                                                  proc_number=8)

        val_dataset, _, _ = feature_extractor.feature2ipt(val_oos_features, val_gt_labels,
                                                          data_type=data_type)

        curriculum = load_curriculum(data_type, n_clusters, val_type, feature_type, malware_samples_num)
        easy_malware_curriculum = []
        for item in curriculum[0:n_clusters - 1]:
            easy_malware_curriculum = easy_malware_curriculum + item

        difficult_curriculum = curriculum[-1]
        easy_benign_curriculum = random.choices(list(range(malware_samples_num, samples_num)),
                                                k=len(easy_malware_curriculum))
        easy_oos_features = np.array(oos_features)[easy_malware_curriculum + easy_benign_curriculum]
        easy_gt_labels = gt_labels[easy_malware_curriculum + easy_benign_curriculum]
        print('easy curriculum contain samples ' + str(2 * len(easy_benign_curriculum)))
        easy_trainset, input_dim, dataX_np = feature_extractor.feature2ipt(easy_oos_features, easy_gt_labels,
                                                                           data_type=data_type)
        vanilla = Vanilla(architecture_type='droidectc', model_directory=robuts_model_path)
        vanilla.fit(train_set=easy_trainset,
                    validation_set=val_dataset,
                    input_dim=input_dim,
                    EPOCH=20,
                    training_predict=False)

        difficult_benign_curriculum = []
        for i in range(malware_samples_num, samples_num):
            if i in easy_benign_curriculum:
                pass
            else:
                difficult_benign_curriculum.append(i)

        difficult_benign_curriculum = random.choices(difficult_benign_curriculum,
                                                     k=len(difficult_curriculum))

        diff_oos_features = np.array(oos_features)[difficult_curriculum + difficult_benign_curriculum]
        diff_gt_labels = gt_labels[difficult_curriculum + difficult_benign_curriculum]

        diff_trainset, input_dim, dataX_np = feature_extractor.feature2ipt(diff_oos_features, diff_gt_labels,
                                                                           data_type=data_type)

        print('difficult curriculum contain samples ' + str(
            len(difficult_curriculum + difficult_benign_curriculum)))
        vanilla.finetune(train_set=diff_trainset, validation_set=val_dataset,
                         input_dim=input_dim,
                         EPOCH=10,
                         test_data=dataX_np,
                         training_predict=False)

        if not os.path.isdir(robuts_model_path):
            vanilla.save_ensemble_weights()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-train_data_type', '-dt', type=str, default="drebin")
    # parser.add_argument('-n_clusters', '-n', type=int, default=3)
    # args = parser.parse_args()
    # data_type = args.train_data_type
    # n_clusters = int(args.n_clusters)
    feature_type = 'apiseq'
    if feature_type == 'drebin':
        main_ = Deepdrebin_experiment
    elif feature_type == 'apiseq':
        main_ = APIseq_experiment
    else:
        main_ = None
    for data_type in ['drebin']:
        for n_clusters in [5, 7, 9, 11]:
            for val_type in ['self']:
                main_(val_type, data_type, feature_type)
