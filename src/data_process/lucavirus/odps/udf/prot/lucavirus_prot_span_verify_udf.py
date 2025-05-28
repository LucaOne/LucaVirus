#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2024/8/6 09:37
@project: LucaVirus
@file: lucavirus_prot_span_verify_udf.py
@desc: label中span_level进行检查，span区间(左右都闭区间)需要在seq内部
"""
import json
from odps.udf import annotate

@annotate("string,string,string->string")
class span_verify(object):
    def evaluate(self, seq_id, seq, label):
        seq_len = len(seq)
        obj = json.loads(label, encoding='UTF-8')
        span_level = obj["span_level"]
        for span in span_level.items():
            for item in span[1]:
                start, end = item[0], item[1]
                if start < 0 or end >= seq_len:
                    return span[0] + "seq_id=%s,len=%d,start=%d,end=%d" %(seq_id, seq_len, start, end)
        return None
