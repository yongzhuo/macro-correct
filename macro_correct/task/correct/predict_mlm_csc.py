# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/1/18 21:34
# @author  : Mo
# @function: 中文拼写纠错, 融合macbert4mdcspell和confusion_dict


import logging as logger
import traceback
import platform
import copy
import json
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(path_root)
if platform.system().lower() == "windows":
    print(path_root)

import numpy as np
import torch

from macro_correct.pytorch_textcorrection.tcTools import load_json, load_pickle
from macro_correct.pytorch_textcorrection.tcPredict import TextCorrectPredict
from macro_correct.pytorch_textcorrection.tcTrie import ConfusionCorrect


class CscPredict:
    def __init__(self, path_config, logger=logger, flag_load_model=True, CUDA_VISIBLE_DEVICES="0", confusion_dict={}):
        self.device = "cuda:{}".format(CUDA_VISIBLE_DEVICES) if (torch.cuda.is_available()
                                and CUDA_VISIBLE_DEVICES != "-1") else "cpu"
        self.model_confusion = ConfusionCorrect(confusion_dict=confusion_dict)
        self.CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES  # "cuda:0"  # "cpu"
        self.confusion_dict = confusion_dict
        self.path_config = path_config
        self.logger = logger
        if flag_load_model:
            self.load_model()

    def load_model(self):
        """   加载模型   """
        self.model = TextCorrectPredict(self.path_config)

    def postprocess(self, errors_dict, text, errors_confusion, errors_model):
        """    后处理, 合并结果模型预测与混淆集的结果   """
        errors_merge = errors_confusion + errors_model
        # res_dict = {"source": text, "target": "", "errors": errors_merge}
        target = copy.deepcopy(text)
        errors_merge_new = []
        pos_es = set()
        for errors_i in errors_merge:
            truth = errors_i[1]
            idx = errors_i[2]
            err = errors_i[0]
            if idx not in pos_es:
                try:
                    target = target[:idx] + truth + target[idx + len(err):]
                    errors_merge_new.append(errors_i)
                    pos_es.add(idx)
                except Exception as e:
                    print(traceback.print_exc())
        errors_merge_new = list(sorted(iter(errors_merge_new), key=lambda x: x[2], reverse=False))
        errors_dict["errors"] = errors_merge_new
        errors_dict["target"] = target
        return errors_dict

    def predict_batch_score(self, texts, flag_confusion=True, flag_prob=True, **kwargs):
        """   批量预测, macbert预测, 混淆集预测   """
        resp_model_s = self.model.predict_batch(texts, flag_prob=flag_prob, **kwargs)
        if flag_confusion:
            resp_confusion_s = self.model_confusion.predict_batch(texts, flag_prob=flag_prob, **kwargs)
            resp_merge_s = []
            for idx, text in enumerate(texts):
                errors_confusion = resp_confusion_s[idx].get("errors", [])
                errors_model = resp_model_s[idx].get("errors", [])
                res_dict = self.postprocess(resp_model_s[idx], text, errors_confusion, errors_model)
                resp_merge_s.append(res_dict)
            return resp_merge_s
        return resp_model_s

    def predict_batch(self, texts, flag_confusion=True, flag_prob=False, **kwargs):
        """   批量预测, macbert预测, 混淆集预测   """
        resp_model_s = self.model.predict_batch(texts, flag_prob=flag_prob, **kwargs)
        if flag_confusion:
            resp_confusion_s = self.model_confusion.predict_batch(texts, flag_prob=flag_prob, **kwargs)
            resp_merge_s = []
            for idx, text in enumerate(texts):
                errors_confusion = resp_confusion_s[idx].get("errors", [])
                errors_model = resp_model_s[idx].get("errors", [])
                res_dict = self.postprocess(resp_model_s[idx], text, errors_confusion, errors_model)
                resp_merge_s.append(res_dict)
            return resp_merge_s
        return resp_model_s

    def predict_score(self, text, flag_confusion=True, flag_prob=True, **kwargs):
        """   预测, macbert预测, 混淆集预测   """
        resp_model = self.model.predict(text, flag_prob=flag_prob, **kwargs)
        if flag_confusion:
            resp_confusion = self.model_confusion.predict(text, flag_prob=flag_prob, **kwargs)
            errors_confusion = resp_confusion.get("errors", [])
            errors_model = resp_model.get("errors", [])
            res_dict = self.postprocess(resp_model, text, errors_confusion, errors_model)
            return res_dict
        return resp_model

    def predict(self, text, flag_confusion=True, flag_prob=False, **kwargs):
        """   预测, macbert预测, 混淆集预测   """
        resp_model = self.model.predict(text, flag_prob=flag_prob, **kwargs)
        if flag_confusion:
            resp_confusion = self.model_confusion.predict(text, flag_prob=flag_prob, **kwargs)
            errors_confusion = resp_confusion.get("errors", [])
            errors_model = resp_model.get("errors", [])
            res_dict = self.postprocess(resp_model, text, errors_confusion, errors_model)
            return res_dict
        return resp_model


if __name__ == '__main__':
    yz = 0

    path_config = "../../output/text_correction/macbert4mdcspell_v1/csc.config"
    # path_config = "../../output/text_correction/macbert4csc_v1/csc.config"
    # path_config = "../../output/text_correction/macbert4csc_v2/csc.config"
    # path_config = "../../output/text_correction/bert4csc_v1/csc.config"
    # path_config = "shibing624/macbert4csc-base-chinese"  # 将csc.config放入即可
    CUDA_VISIBLE_DEVICES = "0"
    flag_load_model = True
    confusion_dict = {}
    model = CscPredict(path_config, flag_load_model=flag_load_model, confusion_dict=confusion_dict,
                       CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES)
    # 真麻烦你了。希望你们好好地跳舞
    texts = ["一个分知,陌光回聚,莪受打去,祢爱带馀",
        "馀额还有100w",
        '放在陌光下',
        '真麻烦你了。希望你们好好的跳无',
        '少先队员因该为老人让坐',
        '机七学习是人工智能领遇最能体现智能的一个分知',
        '一只小鱼船浮在平净的河面上'
        ]
    params = {
        "threshold": 0.382,      # token阈值过滤
        "batch_size": 32,        # 批大小
        "max_len": 128,          # 自定义的长度, 如果截断了, 则截断部分不参与纠错, 后续直接一模一样的补回来
        "rounded": 4,            # 保存4位小数
        "flag_confusion": True,  # 是否使用默认的混淆词典
        "flag_prob": True,       # 是否返回纠错token处的概率
    }
    texts_predict = model.predict_batch_score(texts, **params)
    print(texts_predict)
    print("#"*128)
    for texts_i in texts:
        texts_i_predict = model.predict_score(texts_i, **params)
        print(texts_i_predict)
    print("#"*128)
    texts_predict = model.predict_batch(texts, **params)
    print(texts_predict)
    print("#" * 128)
    for texts_i in texts:
        texts_i_predict = model.predict(texts_i, **params)
        print(texts_i_predict)

    while True:
        try:
            print("请输入:")
            text = input()
            texts_i_predict = model.predict_score(text, **params)
            print(texts_i_predict)
        except Exception as e:
            print(traceback.print_exc())


"""
################################################################################################################################
{'source': '一个分知,陌光回聚,莪受打去,祢爱带馀', 'target': '一个分支,阳光汇聚,我受打击,你爱戴余', 'errors': [['知', '支', 3, 1.0], ['陌', '阳', 5, 1.0], ['回', '汇', 7, 1.0], ['莪', '我', 10, 1.0], ['去', '击', 13, 1.0], ['祢', '你', 15, 1.0], ['带', '戴', 17, 1.0], ['馀', '余', 18, 1.0]]}
{'source': '馀额还有100w', 'target': '余额还有100w', 'errors': [['馀', '余', 0, 1.0]]}
{'source': '放在陌光下', 'target': '放在阳光下', 'errors': [['陌', '阳', 2, 1.0]]}
{'source': '真麻烦你了。希望你们好好的跳无', 'target': '真麻烦你了。希望你们好好地跳舞', 'errors': [['的', '地', 12, 0.6584], ['无', '舞', 14, 1.0]]}
{'source': '少先队员因该为老人让坐', 'target': '少先队员应该为老人让坐', 'errors': [['因', '应', 4, 0.995]]}
{'source': '机七学习是人工智能领遇最能体现智能的一个分知', 'target': '机器学习是人工智能领域最能体现智能的一个分支', 'errors': [['七', '器', 1, 0.9998], ['遇', '域', 10, 0.9999], ['知', '支', 21, 1.0]]}
{'source': '一只小鱼船浮在平净的河面上', 'target': '一只小鱼船浮在平静的河面上', 'errors': [['净', '静', 8, 0.9961]]}
################################################################################################################################
{'source': '一个分知,陌光回聚,莪受打去,祢爱带馀', 'target': '一个分支,阳光汇聚,我受打击,你爱戴余', 'errors': [['知', '支', 3], ['陌', '阳', 5], ['回', '汇', 7], ['莪', '我', 10], ['去', '击', 13], ['祢', '你', 15], ['带', '戴', 17], ['馀', '余', 18]]}
{'source': '馀额还有100w', 'target': '余额还有100w', 'errors': [['馀', '余', 0]]}
{'source': '放在陌光下', 'target': '放在阳光下', 'errors': [['陌', '阳', 2]]}
{'source': '真麻烦你了。希望你们好好的跳无', 'target': '真麻烦你了。希望你们好好地跳舞', 'errors': [['的', '地', 12], ['无', '舞', 14]]}
{'source': '少先队员因该为老人让坐', 'target': '少先队员应该为老人让坐', 'errors': [['因', '应', 4]]}
{'source': '机七学习是人工智能领遇最能体现智能的一个分知', 'target': '机器学习是人工智能领域最能体现智能的一个分支', 'errors': [['七', '器', 1], ['遇', '域', 10], ['知', '支', 21]]}
{'source': '一只小鱼船浮在平净的河面上', 'target': '一只小鱼船浮在平静的河面上', 'errors': [['净', '静', 8]]}
"""

