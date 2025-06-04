# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/29 21:41
# @author  : Mo
# @function: transformers直接加载bert类模型测试


import traceback
import operator
import time
import sys
import os
os.environ["USE_TORCH"] = "1"
from transformers import BertConfig, BertTokenizer, BertForMaskedLM
import torch


# pretrained_model_name_or_path = "../../macro_correct/output/text_correction/macbert4mdcspell_v1"
# pretrained_model_name_or_path = "../../macro_correct/output/text_correction/macbert4csc_v1"
# pretrained_model_name_or_path = "../../macro_correct/output/text_correction/macbert4csc_v2"
# pretrained_model_name_or_path = "../../macro_correct/output/text_correction/bert4csc_v1"
# pretrained_model_name_or_path = "shibing624/macbert4csc-base-chinese"
# pretrained_model_name_or_path = "Macropodus/macbert4mdcspell_v1"
pretrained_model_name_or_path = "Macropodus/macbert4mdcspell_v2"
# pretrained_model_name_or_path = "Macropodus/macbert4csc_v1"
# pretrained_model_name_or_path = "Macropodus/macbert4csc_v2"
# pretrained_model_name_or_path = "Macropodus/bert4csc_v1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 128

print("load model, please wait a few minute!")
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
bert_config = BertConfig.from_pretrained(pretrained_model_name_or_path)
model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path)
model.to(device)
vocab = tokenizer.vocab
print("load model success!")

texts = [
    "机七学习是人工智能领遇最能体现智能的一个分知",
    "我是练习时长两念半的鸽仁练习生蔡徐坤",
    "真麻烦你了。希望你们好好的跳无",
    "他法语说的很好，的语也不错",
    "遇到一位很棒的奴生跟我疗天",
    "我们为这个目标努力不解",
]
len_mid = min(max_len, max([len(t)+2 for t in texts]))

with torch.no_grad():
    outputs = model(**tokenizer(texts, padding=True, max_length=len_mid,
                                return_tensors="pt").to(device))


def flag_total_chinese(text):
    """
    judge is total chinese or not, 判断是不是全是中文
    Args:
        text: str, eg. "macadam, 碎石路"
    Returns:
        bool, True or False
    """
    for word in text:
        if not "\u4e00" <= word <= "\u9fa5":
            return False
    return True

def get_errors_from_diff_length(corrected_text, origin_text, unk_tokens=[], know_tokens=[]):
    """Get errors between corrected text and origin text
    code from:  https://github.com/shibing624/pycorrector
    """
    new_corrected_text = ""
    errors = []
    i, j = 0, 0
    unk_tokens = unk_tokens or [' ', '“', '”', '‘', '’', '琊', '\n', '…', '擤', '\t', '玕', '']
    while i < len(origin_text) and j < len(corrected_text):
        if origin_text[i] in unk_tokens or origin_text[i] not in know_tokens:
            new_corrected_text += origin_text[i]
            i += 1
        elif corrected_text[j] in unk_tokens:
            new_corrected_text += corrected_text[j]
            j += 1
        # Deal with Chinese characters
        elif flag_total_chinese(origin_text[i]) and flag_total_chinese(corrected_text[j]):
            # If the two characters are the same, then the two pointers move forward together
            if origin_text[i] == corrected_text[j]:
                new_corrected_text += corrected_text[j]
                i += 1
                j += 1
            else:
                # Check for insertion errors
                if j + 1 < len(corrected_text) and origin_text[i] == corrected_text[j + 1]:
                    errors.append(('', corrected_text[j], j))
                    new_corrected_text += corrected_text[j]
                    j += 1
                # Check for deletion errors
                elif i + 1 < len(origin_text) and origin_text[i + 1] == corrected_text[j]:
                    errors.append((origin_text[i], '', i))
                    i += 1
                # Check for replacement errors
                else:
                    errors.append((origin_text[i], corrected_text[j], i))
                    new_corrected_text += corrected_text[j]
                    i += 1
                    j += 1
        else:
            new_corrected_text += origin_text[i]
            if origin_text[i] == corrected_text[j]:
                j += 1
            i += 1
    errors = sorted(errors, key=operator.itemgetter(2))
    return new_corrected_text, errors

def get_errors_from_same_length(corrected_text, origin_text, unk_tokens=[], know_tokens=[]):
        """Get new corrected text and errors between corrected text and origin text
        code from:  https://github.com/shibing624/pycorrector
        """
        errors = []
        unk_tokens = unk_tokens or [' ', '“', '”', '‘', '’', '琊', '\n', '…', '擤', '\t', '玕', '', '，']

        for i, ori_char in enumerate(origin_text):
            if i >= len(corrected_text):
                continue
            if ori_char in unk_tokens or ori_char not in know_tokens:
                # deal with unk word
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                continue
            if ori_char != corrected_text[i]:
                if not flag_total_chinese(ori_char):
                    # pass not chinese char
                    corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                    continue
                if not flag_total_chinese(corrected_text[i]):
                    corrected_text = corrected_text[:i] + corrected_text[i + 1:]
                    continue
                errors.append([ori_char, corrected_text[i], i])
        errors = sorted(errors, key=operator.itemgetter(2))
        return corrected_text, errors

def get_errors(text, probs):
    """   获取错误信息   """
    _text = tokenizer.decode(torch.argmax(probs, dim=-1), skip_special_tokens=True).replace(' ', '')
    corrected_text = _text[:len(text)]
    if len(corrected_text) == len(text):
        corrected_text, details = get_errors_from_same_length(corrected_text, text, know_tokens=vocab)
    else:
        corrected_text, details = get_errors_from_diff_length(corrected_text, text, know_tokens=vocab)
    print(text, ' => ', corrected_text, details)
    return details


for probs, source in zip(outputs.logits, texts):
    errors = get_errors(source, probs)


while 1:
    try:
        print("请输入：")
        text = input()
        text_in = [text]
        time_start = time.time()
        len_mid = min(max_len, len(text)+2)
        with torch.no_grad():
            outputs = model(**tokenizer(text_in, padding=True, max_length=len_mid,
                                        return_tensors="pt").to(device))
        result = []
        for probs, source in zip(outputs.logits, text_in):
            errors = get_errors(source, probs)
        time_end = time.time()
        print("cost", time_end-time_start)
        print("#" * 128)
    except Exception as e:
        print(traceback.print_exc())

"""
机七学习是人工智能领遇最能体现智能的一个分知  =>  机器学习是人工智能领域最能体现智能的一个分支 [['七', '器', 1], ['遇', '域', 10], ['知', '支', 21]]
我是练习时长两念半的鸽仁练习生蔡徐坤  =>  我是练习时长两年半的个人练习生蔡徐坤 [['念', '年', 7], ['鸽', '个', 10], ['仁', '人', 11]]
真麻烦你了。希望你们好好的跳无  =>  真麻烦你了。希望你们好好地跳舞 [['的', '地', 12], ['无', '舞', 14]]
他法语说的很好，的语也不错  =>  他法语说得很好，德语也不错 [['的', '得', 4], ['的', '德', 8]]
遇到一位很棒的奴生跟我疗天  =>  遇到一位很棒的女生跟我聊天 [['奴', '女', 7], ['疗', '聊', 11]]
我们为这个目标努力不解  =>  我们为这个目标努力不懈 [['解', '懈', 10]]
"""