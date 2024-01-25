# -*- coding: utf-8 -*- 
# @Time : 2023/12/10 10:31 
# @Author : DirtyBoy 
# @File : training_base_model.py
from core.model_lib import _change_scaler_to_list
from core.dataset_lib import build_dataset_from_numerical_data
from core.data_preprocessing import data_preprocessing
from tensorflow.keras import models
import tensorflow as tf
import argparse, os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from core.ensemble.vanilla import Vanilla
from core.ensemble.bayesian_ensemble import BayesianEnsemble
from core.feature.feature_extraction import DrebinFeature, APISequence
from core.config import config
from core.tools import utils


def get_feature_extor():
    android_features_saving_dir = config.get('metadata', 'naive_data_pool')
    intermediate_data_saving_dir = config.get('dataset', 'intermediate_directory')
    feature_extractor = APISequence(android_features_saving_dir, intermediate_data_saving_dir, update=False,
                                    proc_number=8)
    return feature_extractor


def save_logger(base_path, vanilla_prob, vanilla_training_log, type, fold=0):
    np.save(os.path.join(base_path, type + '/prob/fold' + str(fold + 1)), vanilla_prob)
    np.save(os.path.join(base_path, type + '/log/fold' + str(fold + 1)), vanilla_training_log)


def Deepdrebin_experiment(val_type,
                          data_type,
                          architecture_type='dnn',
                          feature_type='drebin'):
    if val_type == 'self_val':
        base_path = '/home/lhd/Training_robust_Malware_Detector_via_uncertainty/Training/output/' + data_type + '/' + feature_type + '/' + val_type

        if not os.path.isdir(base_path):
            os.makedirs(base_path)

        if not os.path.isdir(os.path.join(base_path, 'vanilla/prob')):
            os.makedirs(os.path.join(base_path, 'vanilla/prob'))
            os.makedirs(os.path.join(base_path, 'vanilla/log'))

        if not os.path.isdir(os.path.join(base_path, 'bayesian/prob')):
            os.makedirs(os.path.join(base_path, 'bayesian/prob'))
            os.makedirs(os.path.join(base_path, 'bayesian/log'))

        dataset, gt_labels, input_dim, dataX_np, data_filenames = data_preprocessing(
            feature_type=feature_type, data_type=data_type)
        vanilla = Vanilla(architecture_type=architecture_type,
                          model_directory="../Model/base_model/" + data_type + '/' + feature_type + '/' + val_type)
        vanilla_prob, vanilla_training_log = vanilla.fit(train_set=dataset, validation_set=dataset,
                                                         input_dim=input_dim,
                                                         EPOCH=30,
                                                         test_data=dataX_np,
                                                         training_predict=True)

        vanilla.save_ensemble_weights()

        save_logger(base_path, vanilla_prob, vanilla_training_log, type='vanilla')

        del vanilla_training_log
        del vanilla_prob

        bayesian = BayesianEnsemble(architecture_type=architecture_type,
                                    model_directory="../Model/base_model/" + data_type + '/' + feature_type + '/' + val_type)
        bayes_prob, bayes_training_log = bayesian.fit(train_set=dataset, validation_set=dataset,
                                                      input_dim=input_dim,
                                                      EPOCH=30,
                                                      test_data=dataX_np,
                                                      training_predict=True)

        save_logger(base_path, bayes_prob, bayes_training_log, type='bayesian')

        del bayes_training_log
        del bayes_prob
    elif val_type == 'cross_val':
        base_path = '/home/lhd/Training_robust_Malware_Detector_via_uncertainty/Training/output/' + data_type + '/' + feature_type + '/' + val_type
        if not os.path.isdir(base_path):
            os.makedirs(base_path)

        if not os.path.isdir(os.path.join(base_path, 'vanilla/prob')):
            os.makedirs(os.path.join(base_path, 'vanilla/prob'))
            os.makedirs(os.path.join(base_path, 'vanilla/log'))

        if not os.path.isdir(os.path.join(base_path, 'bayesian/prob')):
            os.makedirs(os.path.join(base_path, 'bayesian/prob'))
            os.makedirs(os.path.join(base_path, 'bayesian/log'))
        dataset, gt_labels, input_dim, dataX_np, data_filenames = data_preprocessing(
            feature_type=feature_type, data_type=data_type)

        kf = KFold(n_splits=5, shuffle=True, random_state=123)

        for fold, (train_index, test_index) in enumerate(kf.split(dataX_np)):
            print('*********processing  No.' + str(fold + 1) + ' fold training********')
            if not os.path.isdir(os.path.join(base_path, 'index')):
                os.makedirs(os.path.join(base_path, 'index'))
            np.save(os.path.join(base_path, 'index/fold' + str(fold + 1)), [train_index, test_index])

            test_data = dataX_np[test_index]
            train_set = build_dataset_from_numerical_data((dataX_np[train_index], gt_labels[train_index]))
            validation_set = build_dataset_from_numerical_data((dataX_np[test_index], gt_labels[test_index]))

            vanilla = Vanilla(architecture_type=architecture_type,
                              model_directory='../Model/base_model/' + data_type + '/' + feature_type + '/' + val_type + '/fold' + str(
                                  fold + 1))
            vanilla_prob, vanilla_training_log = vanilla.fit(train_set=train_set, validation_set=validation_set,
                                                             input_dim=input_dim,
                                                             EPOCH=30,
                                                             test_data=test_data,
                                                             training_predict=True)
            vanilla.save_ensemble_weights()
            save_logger(base_path, vanilla_prob, vanilla_training_log, type='vanilla', fold=fold)
            bayesian = BayesianEnsemble(architecture_type=architecture_type,
                                        model_directory='../Model/base_model/' + data_type + '/' + feature_type + '/' + val_type + '/fold' + str(
                                            fold + 1))
            bayes_prob, bayes_training_log = bayesian.fit(train_set=train_set, validation_set=validation_set,
                                                          input_dim=input_dim,
                                                          EPOCH=30,
                                                          test_data=test_data,
                                                          training_predict=True)
            bayesian.save_ensemble_weights()
            save_logger(base_path, bayes_prob, bayes_training_log, type='bayesian', fold=fold)


def APISeq_experiment(val_type,
                      data_type,
                      architecture_type='droidectc',
                      feature_type='apiseq'):
    if val_type == 'self_val':
        base_path = '/home/lhd/Training_robust_Malware_Detector_via_uncertainty/Training/output/' + data_type + '/' + feature_type + '/' + val_type
        if not os.path.isdir(base_path):
            os.makedirs(base_path)

        if not os.path.isdir(os.path.join(base_path, 'vanilla/prob')):
            os.makedirs(os.path.join(base_path, 'vanilla/prob'))
            os.makedirs(os.path.join(base_path, 'vanilla/log'))

        if not os.path.isdir(os.path.join(base_path, 'bayesian/prob')):
            os.makedirs(os.path.join(base_path, 'bayesian/prob'))
            os.makedirs(os.path.join(base_path, 'bayesian/log'))
        dataset, gt_labels, input_dim, testset, data_filenames = data_preprocessing(
            feature_type=feature_type, data_type=data_type)
        # android_features_saving_dir = config.get('metadata', 'naive_data_pool')
        # feature_extor = get_feature_extor()
        # oos_features = [os.path.join(android_features_saving_dir, filename) for filename in data_filenames]
        vanilla = Vanilla(architecture_type=architecture_type,
                          model_directory="../Model/base_model/" + data_type + '/' + feature_type + '/' + val_type)
        vanilla_prob, vanilla_training_log = vanilla.fit(train_set=dataset, validation_set=dataset,
                                                         input_dim=input_dim,
                                                         EPOCH=30,
                                                         test_data=testset,
                                                         training_predict=True)
        save_logger(base_path, vanilla_prob, vanilla_training_log, type='vanilla')
        del vanilla_training_log
        del vanilla_prob
        vanilla.save_ensemble_weights()

        bayesian = BayesianEnsemble(architecture_type=architecture_type,
                                    model_directory="../Model/base_model/" + data_type + '/' + feature_type + '/' + val_type)
        bayes_prob, bayes_training_log = bayesian.fit(train_set=dataset, validation_set=dataset,
                                                      input_dim=input_dim,
                                                      EPOCH=30,
                                                      test_data=testset,
                                                      training_predict=True)
        save_logger(base_path, bayes_prob, bayes_training_log, type='bayesian')
        del bayes_training_log
        del bayes_prob
    elif val_type == 'cross_val':
        base_path = '/home/lhd/Training_robust_Malware_Detector_via_uncertainty/Training/output/' + data_type + '/' + feature_type + '/' + val_type
        android_features_saving_dir = config.get('metadata', 'naive_data_pool')
        intermediate_data_saving_dir = config.get('dataset', 'intermediate_directory')
        if not os.path.isdir(base_path):
            os.makedirs(base_path)

        if not os.path.isdir(os.path.join(base_path, 'vanilla/prob')):
            os.makedirs(os.path.join(base_path, 'vanilla/prob'))
            os.makedirs(os.path.join(base_path, 'vanilla/log'))

        if not os.path.isdir(os.path.join(base_path, 'bayesian/prob')):
            os.makedirs(os.path.join(base_path, 'bayesian/prob'))
            os.makedirs(os.path.join(base_path, 'bayesian/log'))

        save_path = os.path.join(intermediate_data_saving_dir,
                                 data_type + '_database' + '.' + feature_type)
        if os.path.exists(save_path):
            print('load filename and label from ' + save_path)

            data_filenames, gt_labels = utils.read_joblib(save_path)
        else:
            print('please do self validation experiments to get database file......')
            raise FileNotFoundError
        oos_features = [os.path.join(android_features_saving_dir, filename) for filename in data_filenames]
        feature_extor = get_feature_extor()
        kf = KFold(n_splits=5, shuffle=True, random_state=123)

        for fold, (train_index, test_index) in enumerate(kf.split(oos_features)):
            print('*********processing  No.' + str(fold + 1) + ' fold training********')
            if not os.path.isdir(os.path.join(base_path, 'index')):
                os.makedirs(os.path.join(base_path, 'index'))
            np.save(os.path.join(base_path, 'index/fold' + str(fold + 1)), [train_index, test_index])

            test_data, _, _ = feature_extor.feature2ipt(oos_features[test_index])
            train_set, _, _ = feature_extor.feature2ipt((oos_features[train_index], gt_labels[train_index]))

            vanilla = Vanilla(architecture_type='droidectc',
                              model_directory="/home/lhd/Reduce_Label_noise_via_uncertainty/Training/model")
            vanilla_prob, vanilla_training_log = vanilla.fit(train_set=train_set, validation_set=train_set,
                                                             input_dim=None,
                                                             EPOCH=30,
                                                             test_data=test_data,
                                                             training_predict=True)
            save_logger(base_path, vanilla_prob, vanilla_training_log, type='vanilla', fold=fold)
            bayesian = BayesianEnsemble(architecture_type='droidectc',
                                        model_directory="/home/lhd/Reduce_Label_noise_via_uncertainty/Training/model")
            bayes_prob, bayes_training_log = bayesian.fit(train_set=train_set, validation_set=train_set,
                                                          input_dim=None,
                                                          EPOCH=30,
                                                          test_data=test_data,
                                                          training_predict=True)
            save_logger(base_path, bayes_prob, bayes_training_log, type='bayesian', fold=fold)


if __name__ == '__main__':
    # for data_type in ['malradar', ]:
    #     for val_type in ['self_val', 'cross_val']:
    #         Deepdrebin_experiment(val_type=val_type, data_type=data_type)

    for data_type in ['malradar']:
        for val_type in ['self_val']:
            APISeq_experiment(val_type=val_type, data_type=data_type)
