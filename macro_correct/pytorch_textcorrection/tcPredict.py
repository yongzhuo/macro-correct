# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/7/25 9:30
# @author  : Mo
# @function: predict model, 预测模块


# 适配linux
import logging as logger
import traceback
import platform
import copy
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(path_root)
if platform.system().lower() == "windows":
    print(path_root)
# os.environ["CUDA_VISIBLE_DEVICES"] = model_config.get("CUDA_VISIBLE_DEVICES", "0")
from macro_correct.pytorch_textcorrection.tcTools import get_errors_for_same_length, get_errors_for_difflib
from macro_correct.pytorch_textcorrection.tcData import TextCorrectionDataCollator
from macro_correct.pytorch_textcorrection.tcData import TextCorrectionDataset
from macro_correct.pytorch_textcorrection.tcConfig import model_config
from macro_correct.pytorch_textcorrection.tcTools import get_logger
from macro_correct.pytorch_textcorrection.tcTools import load_json
from macro_correct.pytorch_textcorrection.tcTools import txt_read
from macro_correct.pytorch_textcorrection.tcOffice import Office

from torch.utils.data import DataLoader
from argparse import Namespace
import torch


class TextCorrectPredict:
    def __init__(self, path_config, logger=logger, flag_load_model=True, CUDA_VISIBLE_DEVICES="0"):
        """ 初始化 """
        self.CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES
        self.logger = logger
        if flag_load_model:
            self.load_config(path_config)
            self.load_model()

    def load_config(self, path_config):
        """ 加载超参数  """
        model_save_path_real = os.path.split(path_config)[0]
        ### csc.json如果没有, 则取得本地macro-correct的, 这样可以跑其他训练的bert类模型
        if os.path.exists(model_save_path_real) and not os.path.exists(path_config):
            path_config_local = os.path.join(path_root, "macro_correct/output/csc.config")
            config = load_json(path_config_local)
        else:
            config = load_json(path_config)
        self.config = Namespace(**config)
        # 将模型目录替换为实际目录, 方便移动的时候不影响; bert_tokenizer/bert_config等都从模型里边读取
        self.config.CUDA_VISIBLE_DEVICES = self.CUDA_VISIBLE_DEVICES
        self.config.pretrained_model_name_or_path = model_save_path_real
        self.config.model_save_path = model_save_path_real
        # path_log_dir = os.path.join(model_save_path_real, "log")
        # self.logger = get_logger(path_log_dir)
        self.l2i, self.i2l = self.config.l2i, self.config.i2l
        # 数据预处理类
        self.tc_dataset = TextCorrectionDataset(path="", config=self.config, flag_shuffle=False)
        self.tc_collator = TextCorrectionDataCollator(config=self.config)

    def load_model(self):
        """ 加载模型  """
        self.office = Office(config=self.config, tokenizer=self.tc_collator.tokenizer, logger=self.logger)
        try:
            try:
                ### 加载transformers的标准模型, 即支持训练好的pycorrector的macbert4csc模型, 普通的bert模型也可以
                self.logger.info("****** Load BertForMaskedLM Std Start! ******")
                path_model = os.path.join(self.office.config.model_save_path, self.office.config.model_name)
                # model_dict_std = self.model.state_dict()
                model_dict_new = torch.load(path_model, map_location=torch.device(self.office.device))
                model_dict_new = {"pretrain_model." + k if not k.startswith("pretrain_model.") else k: v
                                  for k, v in model_dict_new.items()}
                self.office.model.load_state_dict(model_dict_new, strict=False)
                self.office.model.to(self.office.device)
                self.office.logger.info("****** Load BertForMaskedLM Std Success ******")
                self.office.logger.info("****** self.device: {} ******".format(self.office.device))
            except Exception as e:
                ### 加载本项目训练的模型, 只保存为权重
                self.logger.info(traceback.print_exc())
                self.logger.info("****** load_model_state Start! ******")
                # self.office.load_model()
                self.office.load_model_state()
                self.logger.info("****** load_model_state Success! ******")
        except Exception as e:
            ### 加载本项目下载的模型, 保存为权重+模型架构
            self.logger.info(traceback.print_exc())
            self.logger.info("****** load_model Success! ******")
            # self.office.load_model_state()
            self.office.load_model()
            self.logger.info("****** load_model Success! ******")
        return 1

    def process(self, texts, max_len=None, batch_size=None):
        """ 数据预处理, process """
        # token 转 idx, 训练集/验证集
        self.tc_dataset.read_dict(texts)
        if max_len:  # 动态变化max_len
            self.tc_collator.max_len = min(max_len, self.tc_collator.config.max_len)
        else:
            self.tc_collator.max_len = self.tc_collator.config.max_len
        if not batch_size:
            batch_size_dy = self.config.batch_size
        else:
            batch_size_dy = batch_size

        pred_data_loader = DataLoader(num_workers=self.config.num_workers,
                                      collate_fn=self.tc_collator,
                                      batch_size=batch_size_dy,
                                      dataset=self.tc_dataset,
                                      # pin_memory=True
                                      )
        return pred_data_loader

    def predict_loop(self):
        while 1:
            print("请输入：")
            text = input()
            res = self.predict([text])
            print(res)

    def predict(self, text, **kwargs):
        """   只预测一个句子   """
        outputs = self.predict_batch([text], **kwargs)
        return outputs[0] if outputs else []

    def predict_batch(self, texts, threshold=0.6, batch_size=32, max_len=128, rounded=4,
                      flag_prob=True, flag_logits=False, flag_print=False, **kwargs):
        """  分类模型预测
        config:
            texts      : list<dict>, inputs of text, eg. ["你是谁"], [{"source":"你是谁"}], [{"original_text":"你是谁"}]
            threshold  : float, threshold of token prob, eg. 0.75, 0.6
            flag_prob  : bool, output prob or not, eg. True, False
            rounded    : int, rounded of float, eg. 3, 4, 6
            flag_logits: bool, reture logits or softmax, eg. True
            flag_print : bool, print sentence(org/tgt/prd) or not, eg. False
        Returns:
            res        : list<dict>, output of label-score, eg. [{}]
        """
        texts_new = []
        if texts and type(texts[0]) == str:
            for data_i in texts:
                data_i_dict = {}
                data_i_dict[self.config.xy_keys_predict[0]] = data_i
                data_i_dict[self.config.xy_keys_predict[1]] = ""
                data_i_dict[self.config.xy_keys_predict[2]] = []
                texts_new.append(data_i_dict)
            texts = texts_new
        elif texts and type(texts[0]) == dict and "source" in texts[0]:  # {"source":"", "target":""}
            for data_i in texts:
                data_i_dict = {}
                data_i_dict[self.config.xy_keys_predict[0]] = data_i.get("source", "")
                data_i_dict[self.config.xy_keys_predict[1]] = ""
                data_i_dict[self.config.xy_keys_predict[2]] = []
                texts_new.append(data_i_dict)
            texts = texts_new
        dataset = self.process(texts, max_len=max_len, batch_size=batch_size)
        ys_pred_id, probs_pred_id = self.office.predict(dataset, flag_logits=flag_logits, **kwargs)
        new_corrected_sentences = []
        corrected_details = []
        for idx, y in enumerate(ys_pred_id):
            # y_orginal = list(texts[idx].get(self.config.xy_keys_predict[0], ""))
            y_orginal = texts[idx].get(self.config.xy_keys_predict[0], "")
            y_decode = self.tc_collator.tokenizer.convert_ids_to_tokens(y, skip_special_tokens=False)
            y_decode = y_decode[1: len(y_orginal)+1]
            y_prob = probs_pred_id[idx][1: len(y_orginal)+1]
            y_new = ""
            for yo, yd, yp in zip(y_orginal, y_decode, y_prob):
                if yd in self.tc_collator.special_tokens:
                    y_new += yo
                elif yp >= threshold:
                    y_new += yd
                else:
                    y_new += yo
            new_corrected_sent, sub_details = get_errors_for_same_length(y_new, y_orginal)
            # new_corrected_sent, sub_details = get_errors_for_difflib(y_new, y_orginal)
            if flag_prob and sub_details:
               sub_details = [s+[round(y_prob[s[-1]], rounded)] for s in sub_details]
            new_corrected_sentences.append(new_corrected_sent)
            corrected_details.append(sub_details)
        outputs = []
        for s, c, e in zip(texts, new_corrected_sentences, corrected_details):
            original_text = s.get("original_text", "")
            correct_text = copy.deepcopy(c)
            len_correct_text = len(correct_text)
            len_original_text = len(original_text)
            ### 如果预测截断了, 则会补全
            len_mid = len_original_text - len_correct_text
            if len_mid <= 0:
                correct_text = correct_text[:len_original_text]
            else:
                correct_text += "".join(list(original_text)[-len_mid:])
            if flag_print and len(correct_text) != len(original_text):
                print("#"*128)
                print("original_text: ", original_text)
                print("1correct_text: ", c)
                print("2correct_text:", correct_text)
            line_dict = {"source": original_text, "target": correct_text, "errors":  e}
            outputs.append(line_dict)
        return outputs


if __name__ == "__main__":
    # BERT-base = 8109M
    path_config = "../output/text_correction/macbert4mdcspell_v1/csc.config"
    # path_config = "../output/text_correction/macbert4csc_v1/csc.config"
    # path_config = "../output/text_correction/macbert4csc_v2/csc.config"
    # path_config = "../output/text_correction/bert4csc_v1/csc.config"
    # path_config = "shibing624/macbert4csc-base-chinese"  # 将csc.config放入即可
    tcp = TextCorrectPredict(path_config)
    texts = [{"original_text": "真麻烦你了。希望你们好好的跳无"},
             {"original_text": "少先队员因该为老人让坐"},
             {"original_text": "机七学习是人工智能领遇最能体现智能的一个分知"},
             {"original_text": "一只小鱼船浮在平净的河面上"},
             {"original_text": "我的家乡是有明的渔米之乡"},
             ]

    res = tcp.predict_batch(texts, flag_logits=False, threshold=0.01, max_len=128)
    # print(res)
    for r in res:
        print(r)
    # tcp.office.config.model_save_path = tcp.office.config.model_save_path + "_state"
    # tcp.office.save_model_state()

    while True:
        print("请输入:")
        question = input()
        res = tcp.predict_batch([{"source": question}], flag_logits=False, threshold=0.01)
        # print(res)
        print(res)

"""
今天我写这张信球球妳帮我买一间房子。
可是你现在不在宿舍，所以我留了一枝条。
所以我在“义大利面方子”已经定位了
我以前想要告诉你，可是我忘了，我真户秃。
真麻烦你了。希望你们好好的跳无。

Pytorch → ONNX → TensorRT
Pytorch → ONNX → TVM

"""
# macbert-now
# 30 72 0.4166666666666667



"""
文本纠错：\n今天我写这张信球球妳帮我买一间房子。
文本纠错：\n机七学习是人工智能领遇最能体现智能的一个分知。
"""

"""

请输入:
机七学习是人工智能领遇最能体现智能的一个分知。
[{'source': '机七学习是人工智能领遇最能体现智能的一个分知。', 
'target': '机器学习是人工智能领域最能体现智能的一个分支。', 
'errors': [('七', '器', 1, 1.0), ('遇', '域', 10, 1.0), ('知', '支', 21, 0.9685)]}]
"""


