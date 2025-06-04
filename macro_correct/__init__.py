# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/29 21:41
# @author  : Mo
# @function: function of macro_correct


import logging
import sys
import os


os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
log_console = logging.StreamHandler(sys.stderr)
default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.INFO)
default_logger.addHandler(log_console)


# 默认加载FLAG_CSC_TOKEN, 为了训练也可以设置环境关闭
if os.environ.get("MACRO_CORRECT_FLAG_CSC_TOKEN", "0") == "1":
    from macro_correct.predict_csc_token_zh import MacroCSC4Token
    MODEL_CSC_TOKEN = MacroCSC4Token(logger=default_logger)
    correct_tradition = MODEL_CSC_TOKEN.func_csc_token_batch_tradition  # 繁体字字词纠错(csc)
    correct_basic = MODEL_CSC_TOKEN.model_csc.predict_batch   # 基础方法预测, 没有后处理
    correct_long = MODEL_CSC_TOKEN.func_csc_token_long        # 处理单一篇的文本
    correct = MODEL_CSC_TOKEN.func_csc_token_batch            # 批处理小于max_len的句子

if os.environ.get("MACRO_CORRECT_FLAG_CSC_PUNCT", "0") == "1":
    from macro_correct.predict_csc_punct_zh import MacroCSC4Punct
    MODEL_CSC_PUNCT = MacroCSC4Punct(logger=default_logger)
    correct_punct_basic = MODEL_CSC_PUNCT.model_csc.predict_batch   # 基础方法预测, 没有后处理
    correct_punct_long = MODEL_CSC_PUNCT.func_csc_punct_long        # 处理单一篇的文本
    correct_punct = MODEL_CSC_PUNCT.func_csc_punct_batch            # 批处理小于max_len的句子

