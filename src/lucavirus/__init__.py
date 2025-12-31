#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2025, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2025/12/30 11:32
@project: lucavirus
@file: configuration_lucavirus
@desc: configuration_lucavirus
'''

from .configuration_lucavirus import LucaVirusConfig
from .tokenization_lucavirus import LucaVirusTokenizer, LucaVirusTokenizerFast
from .modeling_lucavirus import (
    LucaVirusModel,
    LucaVirusPreTrainedModel,
    LucaVirusForMaskedLM,
    LucaVirusForSequenceClassification,
    LucaVirusForTokenClassification
)
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification
)

__all__ = [
    "LucaVirusConfig",
    "LucaVirusModel",
    "LucaVirusPreTrainedModel",
    "LucaVirusTokenizer",
    "LucaVirusTokenizerFast",
    "LucaVirusForMaskedLM",
    "LucaVirusForSequenceClassification",
    "LucaVirusForTokenClassification"
]


# 1. 注册配置类 (必选)
AutoConfig.register("lucavirus", LucaVirusConfig)

# 2. 注册基础模型 (用于 AutoModel.from_pretrained)
AutoModel.register(LucaVirusConfig, LucaVirusModel)

# 3. 注册序列分类模型 (用于 AutoModelForSequenceClassification)
AutoModelForSequenceClassification.register(LucaVirusConfig, LucaVirusForSequenceClassification)

# 4. 注册 Token 分类模型 (用于 AutoModelForTokenClassification)
AutoModelForTokenClassification.register(LucaVirusConfig, LucaVirusForTokenClassification)

# 5. 注册掩码语言模型 (用于 AutoModelForMaskedLM)
AutoModelForMaskedLM.register(LucaVirusConfig, LucaVirusForMaskedLM)