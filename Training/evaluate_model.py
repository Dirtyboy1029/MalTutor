# -*- coding: utf-8 -*- 
# @Time : 2023/12/7 16:49 
# @Author : DirtyBoy 
# @File : evaluate_model.py
from core.model_lib import _change_scaler_to_list
from core.dataset_lib import build_dataset_from_numerical_data
from core.data_preprocessing import data_preprocessing, data_preprocessing_for_test, data_preprocessing_for_adv, \
    data_preprocessing_for_ood
from tensorflow.keras import models
import tensorflow as tf
import argparse, os, json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from core.ensemble.vanilla import Vanilla
from core.ensemble.bayesian_ensemble import BayesianEnsemble

tf.config.set_visible_devices([], 'GPU')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-evaluate_type', type=str, default="ood", choices=['ood', 'adv', 'dataset', 'test'])
    parser.add_argument('-train_data_type', type=str, default="malradar")
    parser.add_argument('-robust_type', '-rt', type=str, default="weight")
    parser.add_argument('-feature_type', '-ft', type=str, default="drebin")
    parser.add_argument('-test_data_type', type=str, default="malgenome")
    parser.add_argument('-val_type', '-vt', type=str, default="self")
    parser.add_argument('-weight_type', '-wt', type=str, default="source")
    parser.add_argument('-adv_type', type=str, default="ade_ma")
    parser.add_argument('-n_clusters', '-n', type=int, default=3)
    args = parser.parse_args()
    n_clusters = int(args.n_clusters)
    evaluate_type = args.evaluate_type
    train_data_type = args.train_data_type
    test_data_type = args.test_data_type
    adv_type = args.adv_type
    weight_type = args.weight_type
    feature_type = args.feature_type
    if feature_type == 'drebin':
        architecture_type = 'dnn'
    elif feature_type == 'apiseq':
        architecture_type = 'droidectc'
    else:
        architecture_type = ''
    val_type = args.val_type
    robust_type = args.robust_type
    if robust_type == 'cl':
        model_type = 'CL_robust_model'
    elif robust_type == 'ca':
        model_type = 'comparative_model'
    else:
        model_type = ''
    robuts_model_path = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/Model/' + model_type + '/' + val_type + '/' + train_data_type + '_' + feature_type + '_'
    source_path = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/Model/base_model/' + train_data_type + '/' + feature_type + '/self_val'
    source_model = Vanilla(architecture_type=architecture_type,
                           model_directory=source_path)
    if evaluate_type == 'ood':
        print('********************************************************************************')
        print(
            '********evaluate ' + test_data_type + ' set on model which trainset is ' + train_data_type + ' set ********')
        print('********************************************************************************')
        dataset, gt_labels, input_dim, _, data_filenames = data_preprocessing_for_ood(
            feature_type=feature_type, train_data_type=train_data_type, data_type=test_data_type)
        source_model.evaluate(dataset, gt_labels)
        for n_clusters in [3, 5, 7, 9, 11]:
            print('-------' + str(n_clusters) + '-------------')
            robust_model_path_ = robuts_model_path + str(n_clusters)
            robust_model = Vanilla(architecture_type=architecture_type,
                                   model_directory=robust_model_path_)
            robust_model.evaluate(dataset, gt_labels)
    elif evaluate_type == 'dataset':
        print('********************************************************************************')
        print(
            '********evaluate ' + test_data_type + ' set on model which trainset is ' + train_data_type + ' set ********')
        print('********************************************************************************')
        dataset, gt_labels, _, _, _ = data_preprocessing_for_test(
            feature_type=feature_type, train_data_type=train_data_type, data_type=test_data_type)
        source_model.evaluate(dataset, gt_labels)
        for n_clusters in [3, 5, 7, 9, 11]:
            print('-------' + str(n_clusters) + '-------------')
            robust_model_path_ = robuts_model_path + str(n_clusters)
            robust_model = Vanilla(architecture_type=architecture_type,
                                   model_directory=robust_model_path_)
            robust_model.evaluate(dataset, gt_labels)

    elif evaluate_type == 'adv':
        print('********************************************************************************')
        print(
            '********evaluate ' + adv_type + ' set on model which trainset is ' + train_data_type + ' set ********')
        print('********************************************************************************')
        prist_data, adv_data, prist_y, adv_y, input_dim, adv_filenames = data_preprocessing_for_adv(
            feature_type=feature_type, train_data_type=train_data_type, adv_type=adv_type)
        source_model.evaluate(prist_data, prist_y)
        source_model.evaluate(adv_data, adv_y)
        for n_clusters in [3, 5, 7, 9, 11]:
            print('-------' + str(n_clusters) + '-------------')
            robust_model_path_ = robuts_model_path + str(n_clusters)
            robust_model = Vanilla(architecture_type=architecture_type,
                                   model_directory=robust_model_path_)
            print('-------pst-------------')
            robust_model.evaluate(prist_data, prist_y)
            print('-------adv-------------')
            robust_model.evaluate(adv_data, adv_y)

    else:
        model = Vanilla(architecture_type='droidectc',
                        model_directory='/home/public/rmt/heren/experiment/cl-exp/LHD_apk/Model/CL_robust_model/self/drebin_apiseq_' + str(
                            n_clusters))
        test_data = ['cic2020smsware', 'cic2020bankware', 'cic2020riskware', 'adware', 'smsware', 'backdoorware',
                     'scareware', 'ransomware',
                     'cic2020adware', ]
        for item in test_data:
            print('********************************************************************************')
            print('********' + item + '********')
            print('********************************************************************************')
            dataset, gt_labels, input_dim, _, data_filenames = data_preprocessing_for_ood(
                feature_type='apiseq', train_data_type='drebin', data_type=item)
            model.evaluate(dataset, gt_labels)
