#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2025/5/27 14:02
@project: LucaProtGen
@file: tensorboardx
@desc: xxxx
"""
from tensorboardX import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 指定tensorboard文件路径
tb_log_dir = "../../tb-logs/lucavirus/v1.0/token_level/lucavirus/20250113063529/events.out.tfevents.1736750364.fudan-0023.79.0"

# 创建EventAccumulator对象
event_acc = EventAccumulator(tb_log_dir)

# 加载事件数据
event_acc.Reload()

# 获取所有标签
tags = event_acc.Tags()['scalars']

print("tags:")
print(tags)

# 读取事件数据
data = {}
'''
training_cur_merged_loss:
[1400000, 0.0120776342228055, 16908.6879119277, 1738988201.6360679]
logging_epoch:
[1400000, 1.0, 1400000.0, 1738988201.6372488]
logging_cur_epoch_step:
[1400000, 1400000.0, 1960000000000.0, 1738988201.6372879]
logging_cur_epoch_done_sample_num:
[1400000, 1400000.0, 1960000000000.0, 1738988201.637303]
logging_cur_epoch_avg_loss:
[1400000, 1.1700888872146606, 1638124.442100525, 1738988201.6373167]
logging_cur_batch_loss:
[1400000, 0.0120776342228055, 16908.6879119277, 1738988201.6373284]
logging_global_avg_loss:
[1400000, 1.1700888872146606, 1638124.442100525, 1738988201.637339]
logging_cur_use_time:
[1400000, 1.2011525630950928, 1681613.5883331299, 1738988201.6373498]
logging_global_step:
[1400000, 1400000.0, 1960000000000.0, 1738988201.637361]
logging_log_avg_loss:
[1400000, 1.0849182605743408, 1518885.5648040771, 1738988201.637372]
training_global_avg_token_level_gene_mask:
[1400000, 1.095523715019226, 1533733.2010269165, 1738988201.6374044]
training_global_avg_token_level_prot_mask:
[1400000, 1.3195289373397827, 1847340.5122756958, 1738988201.6374176]
training_global_steps_token_level_gene_mask:
[1400000, 933955.0, 1307537000000.0, 1738988201.637735]
training_global_steps_token_level_prot_mask:
[1400000, 466041.0, 652457400000.0, 1738988201.6377592]
training_global_avg_merged_loss:
[1400000, 2.415052652359009, 3381073.7133026123, 1738988201.637774]
training_log_avg_token_level_gene_mask:
[1400000, 1.0577375888824463, 1480832.6244354248, 1738988201.637807]
training_log_avg_token_level_prot_mask:
[1400000, 1.137446403503418, 1592424.9649047852, 1738988201.6378229]
training_log_steps_token_level_gene_mask:
[1400000, 2636.0, 3690400000.0, 1738988201.6378365]
training_log_steps_token_level_prot_mask:
[1400000, 1364.0, 1909600000.0, 1738988201.637848]
training_log_avg_merged_loss:
[1400000, 2.1951839923858643, 3073257.58934021, 1738988201.6378598]
logging_updated_lr:
[1400000, 0.00019144144607707858, 268.01802450791, 1738988203.4535303]
'''
for tag in tags:
    scalar_events = event_acc.Scalars(tag)
    print(tag + ":")
    for scalar_event in scalar_events:
        if scalar_event.step == 1400000:
            print([scalar_event.step, scalar_event.value, scalar_event.value * scalar_event.step, scalar_event.wall_time])