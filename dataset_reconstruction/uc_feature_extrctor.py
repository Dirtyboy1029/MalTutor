# -*- coding: utf-8 -*- 
# @Time : 2023/12/11 9:07 
# @Author : DirtyBoy 
# @File : uc_feature_extrctor.py
import os
import numpy as np
import pandas as pd
from dataset_reconstruction.utils import read_joblib, no_label_metrics_map, with_label_metrics_map

if __name__ == '__main__':
    read_cross_val_data = True
    read_self_val_data = True

    # data_type = 'malradar'
    for data_type in ['malradar']:
        feature_type = 'drebin'
        config_path = '../Training/config/' + data_type + '_database.' + feature_type
        data_filenames, gt_labels = read_joblib(config_path)
        cv_base_path = '../Training/output/' + data_type + '/' + feature_type + '/cross_val'
        sv_base_path = '../Training/output/' + data_type + '/' + feature_type + '/self_val'

        if read_self_val_data:
            bayes_path = os.path.join(sv_base_path, 'bayesian/prob')
            vanilla_path = os.path.join(sv_base_path, 'vanilla/prob')
            index_path = os.path.join(sv_base_path, 'index')

            uc_metrics_df = pd.DataFrame()
            uc_metrics_df['apk_name'] = data_filenames
            uc_metrics_df['gt_labels'] = gt_labels

            for epoch in range(30):
                for metrics_name in ['prob', 'entropy', 'kld', 'std', 'max_max2', 'max_min', 'mean_med',
                                     'label_prob', 'nll', 'emd', 'cbs', 'mat', 'eld', 'kld_label']:
                    no_df_name = 'no_' + metrics_name + '_' + str(epoch + 1)
                    with_df_name = 'with_' + metrics_name + '_' + str(epoch + 1)
                    epoch_no_label_uc_metrics_tmp = np.zeros(int(len(data_filenames)))
                    epoch_with_label_uc_metrics_tmp = np.zeros(int(len(data_filenames)))

                    bayes_data = np.load(
                        os.path.join(bayes_path, 'fold1.npy'))
                    vanilla_data = np.load(
                        os.path.join(vanilla_path, 'fold1.npy'))

                    for i in range(len(data_filenames)):
                        if metrics_name == 'prob':
                            epoch_no_label_uc_metrics_tmp[i] = vanilla_data[epoch][i][0]
                            epoch_with_label_uc_metrics_tmp[i] = -1
                        elif metrics_name == 'label_prob':
                            epoch_no_label_uc_metrics_tmp[i] = -1
                            epoch_with_label_uc_metrics_tmp[i] = np.abs(gt_labels[i] - vanilla_data[epoch][i][0])
                        else:
                            try:
                                epoch_no_label_uc_metrics_tmp[i] = no_label_metrics_map[metrics_name](
                                    np.squeeze(bayes_data[epoch][i]))
                            except Exception:
                                epoch_no_label_uc_metrics_tmp[i] = -1
                            try:
                                epoch_with_label_uc_metrics_tmp[i] = with_label_metrics_map[metrics_name](
                                    np.squeeze(bayes_data[epoch][i]), label=gt_labels[i])
                            except Exception:
                                epoch_with_label_uc_metrics_tmp[i] = -1
                    if np.all(epoch_no_label_uc_metrics_tmp == -1):
                        pass
                    else:
                        uc_metrics_df[no_df_name] = epoch_no_label_uc_metrics_tmp
                    if np.all(epoch_with_label_uc_metrics_tmp == -1):
                        pass
                    else:
                        uc_metrics_df[with_df_name] = epoch_with_label_uc_metrics_tmp
            csv_name = 'uc_metrics_csv/' + data_type + '_' + feature_type + '_self_val.csv'
            uc_metrics_df.to_csv(csv_name)

        if read_cross_val_data:

            bayes_path = os.path.join(cv_base_path, 'bayesian/prob')
            vanilla_path = os.path.join(cv_base_path, 'vanilla/prob')
            index_path = os.path.join(cv_base_path, 'index')

            uc_metrics_df = pd.DataFrame()
            uc_metrics_df['apk_name'] = data_filenames
            uc_metrics_df['gt_labels'] = gt_labels
            for epoch in range(30):
                for metrics_name in ['prob', 'entropy', 'kld', 'std', 'max_max2', 'max_min', 'mean_med',
                                     'label_prob', 'nll', 'emd', 'cbs', 'mat', 'eld', 'kld_label']:
                    no_df_name = 'no_' + metrics_name + '_' + str(epoch + 1)
                    with_df_name = 'with_' + metrics_name + '_' + str(epoch + 1)
                    epoch_no_uc_metrics_tmp = np.zeros(int(len(data_filenames)))
                    epoch_with_uc_metrics_tmp = np.zeros(int(len(data_filenames)))
                    for fold in range(5):
                        index = \
                            np.load(os.path.join(index_path, 'fold' + str(fold + 1) + '.npy'), allow_pickle=True)[1]
                        bayes_data = np.load(
                            os.path.join(bayes_path, 'fold' + str(fold + 1) + '.npy'))
                        vanilla_data = np.load(
                            os.path.join(vanilla_path, 'fold' + str(fold + 1) + '.npy'))
                        for i in range(len(index)):
                            if metrics_name == 'prob':
                                epoch_no_uc_metrics_tmp[index[i]] = vanilla_data[epoch][i][0]
                                epoch_with_uc_metrics_tmp[index[i]] = -1
                            elif metrics_name == 'label_prob':
                                epoch_no_uc_metrics_tmp[index[i]] = -1
                                epoch_with_uc_metrics_tmp[index[i]] = np.abs(
                                    gt_labels[index[i]] - vanilla_data[epoch][i][0])
                            else:
                                try:
                                    epoch_no_uc_metrics_tmp[index[i]] = no_label_metrics_map[metrics_name](
                                        np.squeeze(bayes_data[epoch][i]))
                                except Exception:
                                    epoch_no_uc_metrics_tmp[index[i]] = -1
                                try:
                                    epoch_with_uc_metrics_tmp[index[i]] = with_label_metrics_map[metrics_name](
                                        np.squeeze(bayes_data[epoch][i]), label=gt_labels[index[i]])
                                except Exception:
                                    epoch_with_uc_metrics_tmp[index[i]] = -1
                    if np.all(epoch_no_uc_metrics_tmp == -1):
                        pass
                    else:
                        uc_metrics_df[no_df_name] = epoch_no_uc_metrics_tmp
                    if np.all(epoch_with_uc_metrics_tmp == -1):
                        pass
                    else:
                        uc_metrics_df[with_df_name] = epoch_with_uc_metrics_tmp
            csv_name = 'uc_metrics_csv/' + data_type + '_' + feature_type + '_cross_val.csv'
            uc_metrics_df.to_csv(csv_name)
