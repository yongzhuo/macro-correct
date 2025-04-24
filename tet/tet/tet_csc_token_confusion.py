# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/29 21:41
# @author  : Mo
# @function: tet csc of token confusion dict, 混淆词典


import os
os.environ["MACRO_CORRECT_FLAG_CSC_TOKEN"] = "1"

from macro_correct.pytorch_textcorrection.tcTrie import ConfusionCorrect
from macro_correct import MODEL_CSC_TOKEN
from macro_correct import correct


### 默认使用混淆词典
user_dict = {
    "乐而往返": "乐而忘返",
    "金钢钻": "金刚钻",
    "藤罗蔓": "藤萝蔓",
}
text_list = [
    "为什么乐而往返？",
    "没有金钢钻就不揽瓷活！",
    "你喜欢藤罗蔓吗？",
    "三周年祭日在哪举行？"
]
text_csc = correct(text_list, flag_confusion=False)
print("默认纠错(不带混淆词典):")
for res_i in text_csc:
    print(res_i)
print("#" * 128)



text_csc = correct(text_list, flag_confusion=True)
print("默认纠错(-带混淆词典-默认):")
for res_i in text_csc:
    print(res_i)
print("#" * 128)


# ---混淆词典---
### 只新增, 新增用户词典(默认混淆词典也使用)
MODEL_CSC_TOKEN.model_csc.model_confusion = ConfusionCorrect(user_dict=user_dict)
text_csc = correct(text_list, flag_confusion=True)
print("默认纠错(-带混淆词典-新增):")
for res_i in text_csc:
    print(res_i)
print("#" * 128)
### 全覆盖, 只使用用户词典(默认混淆词典废弃)
MODEL_CSC_TOKEN.model_csc.model_confusion = ConfusionCorrect(confusion_dict=user_dict)
text_csc = correct(text_list, flag_confusion=True)
print("默认纠错(-带混淆词典-全覆盖):")
for res_i in text_csc:
    print(res_i)
print("#" * 128)


# ---混淆词典文件---
### 只新增, 新增用户词典(默认混淆词典也使用), json文件, {混淆词语:正确词语} key-value;
path_user = "./user_confusion_dict.json"
MODEL_CSC_TOKEN.model_csc.model_confusion = ConfusionCorrect(path="1", path_user=path_user)
text_csc = correct(text_list, flag_confusion=True)
print("默认纠错(-带混淆词典文件-新增):")
for res_i in text_csc:
    print(res_i)
print("#" * 128)
### 全覆盖, 只使用用户词典(默认混淆词典废弃); path必须传空字符串
MODEL_CSC_TOKEN.model_csc.model_confusion = ConfusionCorrect(path="", path_user=path_user)
text_csc = correct(text_list, flag_confusion=True)
print("默认纠错(-带混淆词典文件-全覆盖):")
for res_i in text_csc:
    print(res_i)
print("#" * 128)


"""
默认纠错(不带混淆词典):
{'index': 0, 'source': '为什么乐而往返？', 'target': '为什么乐而往返？', 'errors': []}
{'index': 1, 'source': '没有金钢钻就不揽瓷活！', 'target': '没有金刚钻就不揽瓷活！', 'errors': [['钢', '刚', 3, 0.6587]]}
{'index': 2, 'source': '你喜欢藤罗蔓吗？', 'target': '你喜欢藤萝蔓吗？', 'errors': [['罗', '萝', 4, 0.8582]]}
{'index': 3, 'source': '三周年祭日在哪举行？', 'target': '三周年祭日在哪举行？', 'errors': []}
################################################################################################################################
默认纠错(-带混淆词典-默认):
{'index': 0, 'source': '为什么乐而往返？', 'target': '为什么乐而往返？', 'errors': []}
{'index': 1, 'source': '没有金钢钻就不揽瓷活！', 'target': '没有金刚钻就不揽瓷活！', 'errors': [['钢', '刚', 3, 1.0]]}
{'index': 2, 'source': '你喜欢藤罗蔓吗？', 'target': '你喜欢藤萝蔓吗？', 'errors': [['罗', '萝', 4, 0.8582]]}
{'index': 3, 'source': '三周年祭日在哪举行？', 'target': '三周年忌日在哪举行？', 'errors': [['祭', '忌', 3, 1.0]]}
################################################################################################################################
默认纠错(-带混淆词典-新增):
{'index': 0, 'source': '为什么乐而往返？', 'target': '为什么乐而忘返？', 'errors': [['往', '忘', 5, 1.0]]}
{'index': 1, 'source': '没有金钢钻就不揽瓷活！', 'target': '没有金刚钻就不揽瓷活！', 'errors': [['钢', '刚', 3, 1.0]]}
{'index': 2, 'source': '你喜欢藤罗蔓吗？', 'target': '你喜欢藤萝蔓吗？', 'errors': [['罗', '萝', 4, 1.0]]}
{'index': 3, 'source': '三周年祭日在哪举行？', 'target': '三周年忌日在哪举行？', 'errors': [['祭', '忌', 3, 1.0]]}
################################################################################################################################
默认纠错(-带混淆词典-全覆盖):
{'index': 0, 'source': '为什么乐而往返？', 'target': '为什么乐而忘返？', 'errors': [['往', '忘', 5, 1.0]]}
{'index': 1, 'source': '没有金钢钻就不揽瓷活！', 'target': '没有金刚钻就不揽瓷活！', 'errors': [['钢', '刚', 3, 1.0]]}
{'index': 2, 'source': '你喜欢藤罗蔓吗？', 'target': '你喜欢藤萝蔓吗？', 'errors': [['罗', '萝', 4, 1.0]]}
{'index': 3, 'source': '三周年祭日在哪举行？', 'target': '三周年祭日在哪举行？', 'errors': []}
################################################################################################################################
默认纠错(-带混淆词典文件-新增):
{'index': 0, 'source': '为什么乐而往返？', 'target': '为什么乐而忘返？', 'errors': [['往', '忘', 5, 1.0]]}
{'index': 1, 'source': '没有金钢钻就不揽瓷活！', 'target': '没有金刚钻就不揽瓷活！', 'errors': [['钢', '刚', 3, 1.0]]}
{'index': 2, 'source': '你喜欢藤罗蔓吗？', 'target': '你喜欢藤萝蔓吗？', 'errors': [['罗', '萝', 4, 1.0]]}
{'index': 3, 'source': '三周年祭日在哪举行？', 'target': '三周年忌日在哪举行？', 'errors': [['祭', '忌', 3, 1.0]]}
################################################################################################################################
默认纠错(-带混淆词典文件-全覆盖):
{'index': 0, 'source': '为什么乐而往返？', 'target': '为什么乐而忘返？', 'errors': [['往', '忘', 5, 1.0]]}
{'index': 1, 'source': '没有金钢钻就不揽瓷活！', 'target': '没有金刚钻就不揽瓷活！', 'errors': [['钢', '刚', 3, 1.0]]}
{'index': 2, 'source': '你喜欢藤罗蔓吗？', 'target': '你喜欢藤萝蔓吗？', 'errors': [['罗', '萝', 4, 1.0]]}
{'index': 3, 'source': '三周年祭日在哪举行？', 'target': '三周年祭日在哪举行？', 'errors': []}
################################################################################################################################
"""

