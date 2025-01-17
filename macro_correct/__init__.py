# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/29 21:41
# @author  : Mo
# @function: function of macro_correct


import os


FLAG_CSC_TOKEN = os.environ.get("FLAG_CSC_TOKEN", "1")  # 默认加载FLAG_CSC_TOKEN, 为了训练也可以设置环境关闭
if FLAG_CSC_TOKEN == "1":
    from macro_correct.predict_csc_token_zh import MODEL_CSC_TOKEN
    correct_basic = MODEL_CSC_TOKEN.model_csc.predict_batch   # 基础方法预测, 没有后处理
    correct_long = MODEL_CSC_TOKEN.func_csc_token_long        # 处理单一篇的文本
    correct = MODEL_CSC_TOKEN.func_csc_token_batch            # 批处理小于max_len的句子

if os.environ.get("FLAG_CSC_PUNCT") == "1":
    from macro_correct.predict_csc_punct_zh import MODEL_CSC_PUNCT
    correct_punct_basic = MODEL_CSC_PUNCT.model_csc.predict_batch   # 基础方法预测, 没有后处理
    correct_punct_long = MODEL_CSC_PUNCT.func_csc_punct_long  # 处理单一篇的文本
    correct_punct = MODEL_CSC_PUNCT.func_csc_punct_batch      # 批处理小于max_len的句子

