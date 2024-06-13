# -*- coding: utf-8 -*- 
# @Time : 2024/1/7 22:05 
# @Author : DirtyBoy 
# @File : comparative_training.py
from core.dataset_lib import build_dataset_from_numerical_data
from core.data_preprocessing import data_preprocessing, data_preprocessing_for_ood
import argparse, os, random
import numpy as np
from core.ensemble.vanilla import Vanilla
import pandas as pd

drebin_cross = [978, 489, 314, 212, 208, ]
malradar_cross = [573, 415, 157, 156, 152]

drebin_self = [1424, 798, 507, 439, 335, ]
malradar_self = [255, 99, 45, 60, 61]


def load_curriculum(data_type, n_clusters, val_type, malware_num):
    curriculum = pd.read_csv(
        '../dataset_reconstruction/inter_file/' + data_type + '_' + val_type + '_' + str(n_clusters) + '.csv')[
                     'label'].tolist()[0:malware_num]
    curriculum_ = []
    for i in range(n_clusters):
        tmp = []
        for j, index in enumerate(curriculum):
            if index == i:
                tmp.append(j)
        curriculum_.append(tmp)
    return curriculum_


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-train_data_type', '-dt', type=str, default="drebin")
    # parser.add_argument('-n_clusters', '-n', type=int, default=3)
    # args = parser.parse_args()
    # data_type = args.train_data_type
    # n_clusters = int(args.n_clusters)
    feature_type = 'drebin'
    val_type = 'self'
    for data_type in ['malradar']:
        for n_clusters in [3, 5, 7, 9, 11]:
            for val_type in ['self', 'cross']:
                samples_num_list = globals()[data_type + '_' + val_type]
                robuts_model_path = "../Model/comparative_model/" + val_type + '/' + data_type + '_' + feature_type + '_' + str(
                    n_clusters)
                if not os.path.isdir(robuts_model_path):

                    intermediate_data_saving_dir = '/home/lhd/Training_robust_Malware_Detector_via_uncertainty/Training/config'

                    dataset, gt_labels, input_dim, dataX_np, data_filenames = data_preprocessing(
                        feature_type=feature_type, data_type=data_type)

                    samples_num = samples_num_list[int((n_clusters - 1) / 2 - 1)]

                    diff_malware_index = random.choices(list(range(np.sum(gt_labels))), k=samples_num)

                    easy_malware_index = []
                    for i in range(np.sum(gt_labels)):
                        if i in diff_malware_index:
                            pass
                        else:
                            easy_malware_index.append(i)

                    easy_benign_index = random.choices(list(range(np.sum(gt_labels), len(dataX_np))),
                                                       k=np.sum(gt_labels) - samples_num)
                    diff_benign_index = []
                    for i in range(np.sum(gt_labels), len(dataX_np)):
                        if i in easy_benign_index:
                            pass
                        else:
                            diff_benign_index.append(i)
                    diff_benign_index = random.choices(diff_benign_index, k=len(diff_malware_index))
                    print('easy curriculum contain samples ' + str(len(easy_benign_index) + len(easy_malware_index)))
                    print(
                        'difficult curriculum contain samples ' + str(len(diff_benign_index) + len(diff_malware_index)))
                    easy_trainset = build_dataset_from_numerical_data(
                        (dataX_np[easy_malware_index + easy_benign_index],
                         gt_labels[easy_malware_index + easy_benign_index]))

                    val_dataset, val_gt_labels, _, _, _ = data_preprocessing_for_ood(
                        feature_type=feature_type,
                        train_data_type=data_type,
                        data_type='smsware')

                    vanilla = Vanilla(architecture_type='dnn', model_directory=robuts_model_path)
                    vanilla_prob, vanilla_training_log = vanilla.fit(train_set=easy_trainset,
                                                                     validation_set=val_dataset,
                                                                     input_dim=input_dim,
                                                                     EPOCH=20,
                                                                     training_predict=False)
                    difficult_trainset = build_dataset_from_numerical_data(
                        (dataX_np[diff_malware_index + diff_benign_index],
                         gt_labels[diff_malware_index + diff_benign_index]))

                    vanilla.finetune(train_set=difficult_trainset, validation_set=val_dataset,
                                     input_dim=input_dim,
                                     EPOCH=10,
                                     test_data=dataX_np,
                                     training_predict=False)

                    if not os.path.isdir(robuts_model_path):
                        vanilla.save_ensemble_weights()
