# -*- coding: utf-8 -*- 
# @Time : 2024/7/24 15:52 
# @Author : DirtyBoy 
# @File : training_sampling_model.py
from core.data_preprocessing import data_preprocessing
from core.ensemble.vanilla import Vanilla

if __name__ == '__main__':
    feature_type = 'drebin'
    for i in range(0,3):
        data_type = 'malradar_sampling' + str(i)
        dataset, gt_labels, input_dim, dataX_np, data_filenames = data_preprocessing(
            feature_type=feature_type, data_type=data_type)
        vanilla = Vanilla(architecture_type='dnn',
                          model_directory="../Model/smote_model/" + data_type + '/' + feature_type)
        vanilla_prob, vanilla_training_log = vanilla.fit(train_set=dataset, validation_set=dataset,
                                                         input_dim=input_dim,
                                                         EPOCH=30,
                                                         test_data=dataX_np,
                                                         training_predict=False)
        vanilla.save_ensemble_weights()
