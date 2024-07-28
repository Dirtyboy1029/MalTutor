# -*- coding: utf-8 -*- 
# @Time : 2024/6/1 21:32 
# @Author : DirtyBoy 
# @File : evaluate_sampling_model.py

from core.data_preprocessing import data_preprocessing_for_test, data_preprocessing_for_ood
from core.ensemble.vanilla import Vanilla
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-evaluate_type', type=str, default="ood", choices=['ood', 'dataset'])
    parser.add_argument('-random_index', '-ri', type=int, default=0)
    parser.add_argument('-test_data_type', type=str, default="drebin")
    parser.add_argument('-train_data_type', type=str, default="malradar")
    args = parser.parse_args()
    random_index = int(args.random_index)
    evaluate_type = args.evaluate_type
    test_data_type = args.test_data_type
    data_type = args.train_data_type

    for random_index in range(3):
        print('********************************************************************************')
        print(
            '********evaluate ' + test_data_type + ' set on model which trainset is ' + data_type + '_sampling' + str(
                random_index) + ' set ********')
        print('********************************************************************************')
        if evaluate_type == 'dataset':
            dataset, gt_labels, _, _, _ = data_preprocessing_for_test(
                feature_type='drebin', train_data_type=data_type + '_sampling' + str(random_index),
                data_type=test_data_type)
        elif evaluate_type == 'ood':
            dataset, gt_labels, input_dim, _, data_filenames = data_preprocessing_for_ood(
                feature_type='drebin', train_data_type=data_type + '_sampling' + str(random_index),
                data_type=test_data_type)
        else:
            dataset = None
            gt_labels = None
        print('--------------------------------' + str(random_index) + '--------------------------------')
        source_path = '../Model/smote_model/' + data_type + '_sampling' + str(
            random_index) + '/drebin'
        source_model = Vanilla(architecture_type='dnn',
                               model_directory=source_path)
        source_model.evaluate(dataset, gt_labels)
