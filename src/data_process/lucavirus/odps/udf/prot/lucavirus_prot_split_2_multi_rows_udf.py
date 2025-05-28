#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2024/8/6 09:37
@project: LucaVirus
@file: lucavirus_prot_split_2_multi_rows_udf.py
@desc: 序列的keywords一行转换多行
"""
from odps.udf import annotate
from odps.udf import BaseUDTF

@annotate('string,string->string')
class split_2_multi_rows(BaseUDTF):
    def process(self, value, separator_char):
        if value:

            while True:
                idx1 = value.find("{")
                if idx1 == -1:
                    break
                idx2 = value.find("}")
                if idx2 == -1:
                    break
                value = value[0:idx1] + value[idx2+1:]

            strs = value.split(separator_char)
            for s in strs:
                self.forward(s.strip())