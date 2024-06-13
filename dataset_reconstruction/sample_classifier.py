# -*- coding: utf-8 -*- 
# @Time : 2023/12/19 8:16 
# @Author : DirtyBoy 
# @File : sample_classifier.py
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import os, argparse
from sklearn.cluster import KMeans

uc_metrics_set = [ 'entropy', 'kld', 'std', 'max_max2', 'max_min', 'mean_med',
                   'nll',  'cbs', 'mat', 'eld', 'kld_label']

no_label_metrics_map = {
    'entropy': None,
    'kld': None,
    'std': None,
    'max_max2': None,
    'max_min': None,
    'mean_med': None
}

with_label_metrics_map = {
    'kld_label': None,
    'nll': None,
    'cbs': None,
    'mat': None,
    'eld': None
}


def Euclidean_distance(p, q):
    v1 = np.array(p)
    v2 = np.array(q)
    distance = np.linalg.norm(v1 - v2)
    return distance


def data_preprocessing(data_type, feature_type, val_type):
    EPOCH = 30
    data = pd.read_csv('uc_metrics_csv/' + data_type + '_' + feature_type + '_' + val_type + '_val.csv')
    del data['Unnamed: 0']
    feature = []
    for index, row in data.iterrows():
        row_item_feature = []
        for metrics_name in uc_metrics_set:
            tmp = []
            for epoch in range(EPOCH):
                if metrics_name in list(no_label_metrics_map.keys()):
                    columns_name = 'no_' + metrics_name + '_' + str(epoch + 1)
                    tmp.append(row[columns_name])
                elif metrics_name in list(with_label_metrics_map.keys()):
                    columns_name = 'with_' + metrics_name + '_' + str(epoch + 1)
                    tmp.append(row[columns_name])
            row_item_feature.append(tmp)
        feature.append(np.array(row_item_feature).T)
    return np.array(feature), data['apk_name'].tolist(), data['gt_labels'].tolist()


def build_autoencoder(name):
    time_steps = 30
    input_size = 14
    input_img = Input(shape=(time_steps, input_size))
    encoded = LSTM(32, activation='relu', name='lstm_' + name)(input_img)
    decoded = RepeatVector(time_steps)(encoded)
    decoded = LSTM(input_size, activation='sigmoid', return_sequences=True)(decoded)
    autoencoder = Model(input_img, decoded)
    print(autoencoder.summary())
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


def training_encoder(malware_feature, ware_type='mal', data_type='malradar', val_type='self', feature_type='apiseq'):
    malware_auto_encoder = build_autoencoder(ware_type)
    malware_auto_encoder.fit(malware_feature, malware_feature, epochs=80, batch_size=32, shuffle=True)
    malware_encoder = Model(inputs=malware_auto_encoder.input,
                            outputs=malware_auto_encoder.get_layer('lstm_' + ware_type).output)
    malware_encoder.save('encoder_model/' + data_type + '_' + feature_type + '_' + val_type + '_' + ware_type + '.h5')
    print(
        'model save to ' + 'encoder_model/' + data_type + '_' + feature_type + '_' + val_type + '_' + ware_type + '.h5')


def cluster(apk_name, malware_feature, n_clusters, ware_type='mal', data_type='drebin',
            val_type='cross', feature_type='drebin'):
    malware_encoder = tf.keras.models.load_model(
        'encoder_model/' + data_type + '_' + feature_type + '_' + val_type + '_' + ware_type + '.h5')
    print(
        'load model from ' + 'encoder_model/' + data_type + '_' + feature_type + '_' + val_type + '_' + ware_type + '.h5')
    malware_encoded_data = malware_encoder.predict(malware_feature)
    print('Cluster the samples into ' + str(n_clusters) + ' classes.')
    malware_kmeans = KMeans(n_clusters=n_clusters)
    malware_kmeans.fit(malware_encoded_data)
    cluster_centers = malware_kmeans.cluster_centers_
    dis = []
    for item in cluster_centers:
        tmp = []
        for demo in malware_encoded_data:
            tmp.append(Euclidean_distance(item, demo))
        dis.append(tmp)
    np.save(
        'inter_file/' + data_type + '_' + feature_type + '_' + val_type + '_' + ware_type + '_' + '_cluster_distance_' + str(
            n_clusters), dis)
    np.save(
        'inter_file/' + data_type + '_' + feature_type + '_' + val_type + '_' + ware_type + '_' + '_cluster_centers_' + str(
            n_clusters),
        cluster_centers)

    labels = malware_kmeans.labels_
    if ware_type == 'mal':
        gt_label = [1] * len(malware_feature)
    else:
        gt_label = [0] * len(malware_feature)

    df = pd.DataFrame({'apk_name': apk_name, 'gt_label': gt_label, 'label': labels})
    return df


'''

'''

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-data_type', '-dt', type=str, default="malradar")
    # parser.add_argument('-n_clusters', '-n', type=int, default=3)
    # args = parser.parse_args()
    # data_type = args.data_type
    # n_clusters = int(args.n_clusters)
    feature_type = 'drebin'
    for data_type in ['malradar']:
        for val_type in ['self', 'cross']:
            for n_clusters in [3, 5, 7, 9, 11, ]:
                feature, apk_name, gt_label = data_preprocessing(data_type, feature_type, val_type)
                malware_num = np.sum(gt_label)
                malware_feature = feature[:malware_num]
                benign_feature = feature[malware_num:]

                malware_apk_name = apk_name[:malware_num]
                benign_apk_name = apk_name[malware_num:]
                if os.path.isfile(
                        'encoder_model/' + data_type + '_' + feature_type + '_' + val_type + '_mal.h5') and os.path.isfile(
                    'encoder_model/' + data_type + '_' + feature_type + '_' + val_type + '_ben.h5'):
                    print('encoder model exist at folder: encoder_model')
                else:
                    mal_encoder = training_encoder(malware_feature, ware_type='mal', data_type=data_type,
                                                   val_type=val_type, feature_type=feature_type)
                    ben_encoder = training_encoder(benign_feature, ware_type='ben', data_type=data_type,
                                                   val_type=val_type, feature_type=feature_type)

                malware_df = cluster(malware_apk_name, malware_feature, n_clusters=n_clusters, ware_type='mal',
                                     data_type=data_type,
                                     val_type=val_type, feature_type=feature_type)
                benware_df = cluster(benign_apk_name, benign_feature, n_clusters=n_clusters, ware_type='ben',
                                     data_type=data_type,
                                     val_type=val_type, feature_type=feature_type)

                df = pd.concat([malware_df, benware_df], axis=0).to_csv(
                    'inter_file/' + data_type + '_' + feature_type + '_' + val_type + '_' + str(n_clusters) + '.csv')
