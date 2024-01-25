# -*- coding: utf-8 -*- 
# @Time : 2023/11/28 8:48 
# @Author : DirtyBoy 
# @File : utils.py
import os
from Training.core.tools import metrics
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix


def save_to_txt(goal, txt_path):
    f = open(txt_path, "w")
    for line in goal:
        f.write(line + '\n')
    f.close()


def txt_to_list(txt_path):
    f = open(txt_path, "r")
    return f.read().splitlines()


def confusion(y_test_true, y_new):
    tn, fp, fn, tp = confusion_matrix(y_test_true, y_new).ravel()
    fpr = fp / float(tn + fp)
    fnr = fn / float(tp + fn)

    recall = tp / float(tp + fn)
    precision = tp / float(tp + fp)
    f1 = f1_score(y_test_true, y_new, average='binary')

    print("recall is " + str(recall * 100) + "%" + " precision  is " + str(precision * 100) + "%, F1 score is " + str(
        f1 * 100) + "%,FPR is " + str(fpr * 100) + "%")
    return tp, fp


def mkdir(target):
    try:
        if os.path.isfile(target):
            target = os.path.dirname(target)

        if not os.path.exists(target):
            os.makedirs(target)
        return 0
    except IOError as e:
        raise Exception("Fail to create directory! Error:" + str(e))


def dump_joblib(data, path):
    if not os.path.exists(os.path.dirname(path)):
        mkdir(os.path.dirname(path))

    try:
        import joblib
        with open(path, 'wb') as wr:
            joblib.dump(data, wr)
        return
    except IOError:
        raise IOError("Dump data failed.")


def read_joblib(path):
    import joblib
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


uc_metrics_set = ['prob', 'entropy', 'kld', 'std', 'max_max2', 'max_min', 'mean_med',
                  'label_prob', 'nll', 'emd', 'cbs', 'mat', 'eld', 'kld_label']

no_label_metrics_map = {
    'prob': None,
    'entropy': metrics.predictive_entropy,
    'kld': metrics.predictive_kld,
    'std': metrics.predictive_std,
    'max_max2': metrics.max_max2,
    'max_min': metrics.max_min,
    'mean_med': metrics.mean_med
}

with_label_metrics_map = {
    'label_prob': None,
    'kld_label': metrics.prob_label_kld,
    'nll': metrics.nll,
    'emd': metrics.Wasserstein_distance,
    'cbs': metrics.Chebyshev_distance,
    'mat': metrics.Manhattan_distance,
    'eld': metrics.Euclidean_distance
}


def plot_box(group1, group2):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.boxplot(group1, positions=[1], widths=0.6, patch_artist=True, showfliers=True)
    plt.boxplot(group2, positions=[2], widths=0.6, patch_artist=True, showfliers=True)

    plt.xlabel('Groups')
    plt.ylabel('Values')
    plt.title('Comparison of Two Datasets')
    plt.xticks([1, 2], ['Group 1', 'Group 2'])

    plt.grid(axis='y')
    plt.show()


def upbound(data):
    q3 = np.percentile(data, 75)  # 上四分位数 Q3
    iqr = np.percentile(data, 75) - np.percentile(data, 25)  # 四分位数间距 IQR

    # 计算上限
    upper_bound = q3 + 1.5 * iqr

    return upper_bound
