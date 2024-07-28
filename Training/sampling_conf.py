# -*- coding: utf-8 -*- 
# @Time : 2024/7/24 11:15 
# @Author : DirtyBoy 
# @File : sampling_conf.py
import os, random, time
import numpy as np
import pandas as pd

random.seed(int(time.time()))


def balance_dataset(malware_dict):
    counts = [len(images) for images in malware_dict.values()]
    target_count = int(np.mean(counts))
    balanced_dict = {}

    for family, malware in malware_dict.items():
        if len(malware) >= target_count:
            balanced_dict[family] = random.sample(malware, target_count)
        elif len(malware) == 1:
            balanced_dict[family] = malware + random.choices(malware, k=target_count - len(malware))
        else:
            copies_needed = target_count - len(malware)
            extended_images = malware * copies_needed
            try:
                balanced_dict[family] = random.sample(extended_images, target_count)
            except ValueError:
                balanced_dict[family] = random.choices(extended_images, k=target_count)

    return balanced_dict


def read_joblib(path):
    import joblib
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def dump_joblib(data, path):
    try:
        import joblib
        with open(path, 'wb') as wr:
            joblib.dump(data, wr)
        return
    except IOError:
        raise IOError("Dump data failed.")


if __name__ == '__main__':
    dataset = 'malradar'
    save_path = 'config/' + dataset + '_database_sampling1.drebin'

    data_filenames, labels = read_joblib('config/' + dataset + '_database.drebin')
    data_filenames = [item.replace('.drebin', '') for item in data_filenames]

    data_malware = data_filenames[:np.sum(labels)]
    data_benign = data_filenames[np.sum(labels):]
    if dataset == 'drebin':
        family_path = 'sha256_family.csv'
    elif dataset == 'malradar':
        family_path = 'malradar_family.csv'
    else:
        family_path = ''
    drebin_family = pd.read_csv(family_path)

    hash_set = set(data_filenames)

    # 过滤DataFrame
    drebin_family = drebin_family[drebin_family['sha256'].isin(hash_set)]
    print(drebin_family.shape)

    malware_family = [drebin_family[drebin_family['sha256'] == h]['family'].values[0] if not drebin_family[
        drebin_family['sha256'] == h].empty else None for h in data_malware]

    malware_family_df = pd.DataFrame({'sha256': data_malware,
                                      'family': malware_family})

    family_hash_dict = malware_family_df.groupby('family')['sha256'].apply(list).to_dict()
    balanced_dict = balance_dataset(family_hash_dict)
    malware_hash = []
    for _, malware in balanced_dict.items():
        malware_hash = malware_hash + malware
    malware_hash = [item + '.drebin' for item in malware_hash]
    benign_hash = [item + '.drebin' for item in data_benign]
    benign_hash = random.sample(benign_hash, k=len(malware_hash))
    data_filenames = malware_hash + benign_hash
    labels = np.array([1] * len(malware_hash) + [0] * len(benign_hash))

    dump_joblib((data_filenames, labels), save_path)
