# conduct the group of 'out of distribution' experiments on drebin dataset
import os


def save_to_txt(goal, txt_path):
    f = open(txt_path, "w")
    for line in goal:
        f.write(line + '\n')
    f.close()


def txt_to_list(txt_path):
    f = open(txt_path, "r")
    return f.read().splitlines()


import numpy as np

from .feature import feature_type_scope_dict

from .tools import utils
from .config import config, logging


def data_preprocessing(feature_type='drebin', proc_numbers=8, data_type="drebin"):
    assert feature_type in feature_type_scope_dict.keys(), 'Expected {}, but {} are supported.'.format(
        feature_type, feature_type_scope_dict.keys())
    benware_dir = config.get('dataset', data_type + '_benware_dir')
    malware_dir = config.get('dataset', data_type + '_malware_dir')
    android_features_saving_dir = config.get('metadata', 'naive_data_pool')
    intermediate_data_saving_dir = config.get('dataset', 'intermediate_directory')
    feature_extractor = feature_type_scope_dict[feature_type](android_features_saving_dir,
                                                              intermediate_data_saving_dir,
                                                              update=False,
                                                              proc_number=proc_numbers)

    save_path = os.path.join(intermediate_data_saving_dir,
                             data_type + '_database' + '.' + feature_type)

    if os.path.exists(save_path):
        print('load filename and label from ' + save_path)
        data_filenames, gt_labels = utils.read_joblib(save_path)
        oos_features = [os.path.join(android_features_saving_dir, filename) for filename in data_filenames]

    else:
        mal_feature_list = feature_extractor.feature_extraction(malware_dir)
        ben_feature_list = feature_extractor.feature_extraction(benware_dir)

        gt_labels = np.array([1] * len(mal_feature_list) + [0] * len(ben_feature_list))
        oos_features = mal_feature_list + ben_feature_list
        data_filenames = [os.path.basename(path) for path in oos_features]

        utils.dump_joblib((data_filenames, gt_labels), save_path)
        print('save filename and label to ' + save_path)

    feature_extractor.feature_preprocess(oos_features, gt_labels, data_type)
    dataset, input_dim, dataX_np = feature_extractor.feature2ipt(oos_features, gt_labels, data_type=data_type)
    return dataset, gt_labels, input_dim, dataX_np, data_filenames


def data_preprocessing_for_test(feature_type='drebin', proc_numbers=8, train_data_type="drebin", data_type='drebin'):
    assert feature_type in feature_type_scope_dict.keys(), 'Expected {}, but {} are supported.'.format(
        feature_type, feature_type_scope_dict.keys())
    benware_dir = config.get('dataset', data_type + '_benware_dir')
    malware_dir = config.get('dataset', data_type + '_malware_dir')
    android_features_saving_dir = config.get('metadata', 'naive_data_pool')
    intermediate_data_saving_dir = config.get('dataset', 'intermediate_directory')
    feature_extractor = feature_type_scope_dict[feature_type](android_features_saving_dir,
                                                              intermediate_data_saving_dir,
                                                              update=False,
                                                              proc_number=proc_numbers)

    save_path = os.path.join(intermediate_data_saving_dir,
                             data_type + '_database' + '.' + feature_type)

    if os.path.exists(save_path):
        print('load filename and label from ' + save_path)
        data_filenames, gt_labels = utils.read_joblib(save_path)
        oos_features = [os.path.join(android_features_saving_dir, filename) for filename in data_filenames]
    else:
        mal_feature_list = feature_extractor.feature_extraction(malware_dir)
        ben_feature_list = feature_extractor.feature_extraction(benware_dir)

        gt_labels = np.array([1] * len(mal_feature_list) + [0] * len(ben_feature_list))
        oos_features = mal_feature_list + ben_feature_list
        data_filenames = [os.path.basename(path) for path in oos_features]
        utils.dump_joblib((data_filenames, gt_labels), save_path)
        print('save filename and label to ' + save_path)
    # obtain data in a format for ML algorithms
    # feature_extractor.feature_preprocess(oos_features, gt_labels, data_type)
    dataset, input_dim, dataX_np = feature_extractor.feature2ipt(oos_features, gt_labels, data_type=train_data_type)
    return dataset, gt_labels, input_dim, dataX_np, data_filenames


def data_preprocessing_for_ood(feature_type='drebin', proc_numbers=2, train_data_type="drebin", data_type='drebin'):
    assert feature_type in feature_type_scope_dict.keys(), 'Expected {}, but {} are supported.'.format(
        feature_type, feature_type_scope_dict.keys())
    malware = config.get('ood', data_type + '_ood_malware_dir')
    android_features_saving_dir = config.get('metadata', 'naive_data_pool')
    intermediate_data_saving_dir = config.get('dataset', 'intermediate_directory')
    feature_extractor = feature_type_scope_dict[feature_type](android_features_saving_dir,
                                                              intermediate_data_saving_dir,
                                                              update=False,
                                                              proc_number=proc_numbers)

    save_path = os.path.join(intermediate_data_saving_dir, data_type + '_ood_database.' + feature_type)
    if os.path.exists(save_path):
        print('load filename and label from ' + save_path)
        ood_filenames, ood_y = utils.read_joblib(save_path)
        ood_features = [os.path.join(android_features_saving_dir, filename) for filename in ood_filenames]
    else:

        ood_features = feature_extractor.feature_extraction(malware)
        if 'benignware' in data_type:
            ood_y = np.zeros(len(ood_features), dtype=np.int32)
        else:
            ood_y = np.ones(len(ood_features), dtype=np.int32)

        ood_filenames = [os.path.basename(path) for path in ood_features]
        utils.dump_joblib((ood_filenames, ood_y), save_path)
        print('save filename and label to ' + save_path)

    # obtain data in a format for ML algorithms
    dataset, input_dim, dataX_np = feature_extractor.feature2ipt(ood_features, ood_y, data_type=train_data_type)
    return dataset, ood_y, input_dim, dataX_np, ood_filenames


def data_preprocessing_for_adv(feature_type='drebin', proc_numbers=2, train_data_type="drebin", adv_type='ade_ma'):
    assert feature_type in feature_type_scope_dict.keys(), 'Expected {}, but {} are supported.'.format(
        feature_type, feature_type_scope_dict.keys())
    adv_malware = config.get('adv', adv_type + '_adv_malware_dir')
    pst_malware = config.get('adv', 'pst_malware_dir')
    android_features_saving_dir = config.get('metadata', 'naive_data_pool')
    intermediate_data_saving_dir = config.get('dataset', 'intermediate_directory')
    feature_extractor = feature_type_scope_dict[feature_type](android_features_saving_dir,
                                                              intermediate_data_saving_dir,
                                                              update=False,
                                                              proc_number=proc_numbers)

    save_path = os.path.join(intermediate_data_saving_dir, adv_type + '_adv_database.' + feature_type)
    if os.path.exists(save_path):
        prist_filenames, adv_filenames, prist_y, adv_y = utils.read_joblib(save_path)
        pst_feature_list = [os.path.join(android_features_saving_dir, filename) for filename in prist_filenames]
        adv_feature_list = [os.path.join(android_features_saving_dir, filename) for filename in adv_filenames]
    else:
        pst_feature_list = feature_extractor.feature_extraction(pst_malware)
        adv_feature_list = feature_extractor.feature_extraction(adv_malware)

        prist_filenames = [os.path.basename(path) for path in pst_feature_list]
        adv_filenames = [os.path.basename(path) for path in adv_feature_list]

        adv_y = np.ones((len(adv_feature_list),), dtype=np.int32)
        prist_y = np.ones((len(pst_feature_list),), dtype=np.int32)

        utils.dump_joblib((prist_filenames, adv_filenames, prist_y, adv_y), save_path)

    # obtain data in a format for ML algorithms
    prist_data, input_dim, _ = feature_extractor.feature2ipt(pst_feature_list, prist_y, data_type=train_data_type)
    adv_data, input_dim, _ = feature_extractor.feature2ipt(adv_feature_list, adv_y, data_type=train_data_type)

    return prist_data, adv_data, prist_y, adv_y, input_dim, adv_filenames
