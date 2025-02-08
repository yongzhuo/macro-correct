# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/29 21:41
# @author  : Mo
# @function: transformers直接加载bert类模型测试


import traceback
import time
import sys
import os
os.environ["USE_TORCH"] = "1"
from transformers import BertConfig, BertTokenizer, BertForMaskedLM
import torch

# pretrained_model_name_or_path = "shibing624/macbert4csc-base-chinese"
# pretrained_model_name_or_path = "Macropodus/macbert4mdcspell_v1"
# pretrained_model_name_or_path = "Macropodus/macbert4csc_v1"
# pretrained_model_name_or_path = "Macropodus/macbert4csc_v2"
# pretrained_model_name_or_path = "Macropodus/bert4csc_v1"
# pretrained_model_name_or_path = "Macropodus/macbert4mdcspell_v1"
pretrained_model_name_or_path = "Macropodus/relm_v1"
# pretrained_model_name_or_path = "../../macro_correct/output/text_correction/relm_v1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 128

print("load model, please wait a few minute!")
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
bert_config = BertConfig.from_pretrained(pretrained_model_name_or_path)
model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path)
model.to(device)
print("load model success!")

texts = [
    "机七学习是人工智能领遇最能体现智能的一个分知",
    "我是练习时长两念半的鸽仁练习生蔡徐坤",
    "我是练习时长两年半的鸽人练习生蔡徐坤",
    "真麻烦你了。希望你们好好的跳无",
    "他法语说的很好，的语也不错",
    "遇到一位很棒的奴生跟我疗天",
    "我们为这个目标努力不解",
]
len_mid = min(max_len, max([len(t)+2 for t in texts])) * 2

with torch.no_grad():
    texts_relm = [list(t) + [tokenizer.sep_token] + [tokenizer.mask_token for _ in t] for t in texts]
    texts_relm = ["".join(t) for t in texts_relm]
    outputs = model(**tokenizer(texts_relm, padding=True, max_length=len_mid,
                                return_tensors="pt").to(device))

def get_errors(source, target):
    """   极简方法获取 errors   """
    len_min = min(len(source), len(target))
    errors = []
    for idx in range(len_min):
        if source[idx] != target[idx]:
            errors.append([source[idx], target[idx], idx])
    return errors

result = []
for probs, source in zip(outputs.logits, texts):
    ids = torch.argmax(probs, dim=-1)
    tokens_space = tokenizer.decode(ids, skip_special_tokens=False)
    text_new = tokens_space.split(" ")
    # print(text_new)
    target = text_new[len(source)+2:len(source)*2+2]
    errors = get_errors(list(source), target)
    target = "".join(target)
    print(source, " => ", target, errors)
    result.append([target, errors])
print(result)
"""
机七学习是人工智能领遇最能体现智能的一个分知  =>  机器学习是人工智能领域最能体现智能的一个分知 [['七', '器', 1], ['遇', '域', 10]]
我是练习时长两念半的鸽仁练习生蔡徐坤  =>  我是练习时长两年半的鸽人练习生蔡徐坤 [['念', '年', 7], ['仁', '人', 11]]
我是练习时长两年半的鸽人练习生蔡徐坤  =>  我是练习时长两年半的个人练习生蔡徐坤 [['鸽', '个', 10]]
真麻烦你了。希望你们好好的跳无  =>  真麻烦你了。希望你们好好地跳舞 [['的', '地', 12], ['无', '舞', 14]]
他法语说的很好，的语也不错  =>  他法语说得很好，德语也不错 [['的', '得', 4], ['的', '德', 8]]
遇到一位很棒的奴生跟我疗天  =>  遇到一位很棒的女生跟我聊天 [['奴', '女', 7], ['疗', '聊', 11]]
我们为这个目标努力不解  =>  我们为这个目标努力不懈 [['解', '懈', 10]]
"""


