# -*- coding: utf-8 -*- 
# @Time : 2023/11/23 21:52 
# @Author : DirtyBoy 
# @File : utils.py

def save_to_txt(goal, txt_path):
    f = open(txt_path, "w")
    for line in goal:
        f.write(line + '\n')
    f.close()

def txt_to_list(txt_path):
    f = open(txt_path, "r")
    return f.read().splitlines()