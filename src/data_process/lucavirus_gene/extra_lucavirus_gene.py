#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2025/1/14 13:24
@project: LucaVirus
@file: extra_lucavirus_gene
@desc: xxxx
"""
import os
import sys
import random
sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../..")
sys.path.append("../../../src")
try:
    from .file_operator import *
except ImportError:
    from src.file_operator import *
random.seed(1111)

raw_dataset_dirpath = "../../../dataset/lucavirus/v1.0/"
dataste_save_dirpath = "../../../dataset/lucavirus-gene/v1.0/"


cnt_per_file = 100000
dataset_type_list = ["train", "dev", "test"]
for dataset_type in dataset_type_list:
    if not os.path.exists(os.path.join(dataste_save_dirpath, dataset_type)):
        os.makedirs(os.path.join(dataste_save_dirpath, dataset_type))
    print("dataset_type: %s" % dataset_type)
    filename_cnt = 0
    genes = []
    header = None
    for filename in os.listdir(os.path.join(raw_dataset_dirpath, dataset_type)):
        if not filename.endswith(".csv"):
            continue
        filename_cnt += 1
        cur_cnt = 0
        for row in csv_reader(os.path.join(raw_dataset_dirpath, dataset_type, filename), header=False, header_filter=False):
            cur_cnt += 1
            if cur_cnt == 1:
                header = row
                continue
            if row[1] == "gene":
                genes.append(row)
    print("filename_cnt: %d" % filename_cnt)
    print("dataset: %d" % len(genes))
    if dataset_type == "train":
        for _ in range(10):
            random.shuffle(genes)
    new_file_cnt = (len(genes) + cnt_per_file - 1)//cnt_per_file
    print("new_file_cnt: %d" % new_file_cnt)
    for idx in range(new_file_cnt):
        csv_writer(
            genes[idx * cnt_per_file: min(cnt_per_file + idx * cnt_per_file, len(genes))],
            handle=os.path.join(dataste_save_dirpath, dataset_type, "gene_3072_%s_%d.csv" % (dataset_type, idx + 1)),
            header=header
        )
    print("#" * 50)
"""
dataset_type: train
filename_cnt: 157
dataset: 10,465,705
new_file_cnt: 105
##################################################
dataset_type: dev
filename_cnt: 1
dataset: 6716
new_file_cnt: 1
##################################################
dataset_type: test
filename_cnt: 1
dataset: 6650
new_file_cnt: 1
##################################################
"""

