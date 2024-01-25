# -*- coding: utf-8 -*- 
# @Time : 2023/11/23 12:53 
# @Author : DirtyBoy 
# @File : feature_extractor.py
import os
from core.feature.feature_extraction import DrebinFeature, OpcodeSeq, APISequence

if __name__ == '__main__':
    android_features_saving_dir = '/home/lhd/apk/drebin1'
    intermediate_data_saving_dir = '/home/lhd/apk/drebin2'

    feature_extractor = APISequence(android_features_saving_dir, intermediate_data_saving_dir, update=False,
                                    proc_number=8)

    mal1 = '/home/lhd/apk/app'

    mal_feature_list1 = feature_extractor.feature_extraction(mal1)
