# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/1/18 21:34
# @author  : Mo
# @function: relm
# @paper   : [MDCSpell: A Multi-task Detector-Corrector Framework for Chinese Spelling Correction](https://aclanthology.org/2022.findings-acl.98/).


from __future__ import absolute_import, division, print_function
from argparse import Namespace
import traceback
import argparse
import logging
import random
import time
import math
import sys
import os

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(path_root)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import SchedulerType, get_scheduler
from transformers import BertForMaskedLM
from transformers import AutoTokenizer
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch

from macro_correct.pytorch_user_models.csc.macbert4mdcspell.dataset import DataSetProcessor, sent_mertic_det, sent_mertic_cor
from macro_correct.pytorch_user_models.csc.macbert4mdcspell.dataset import save_json, load_json
from macro_correct.pytorch_user_models.csc.macbert4mdcspell.graph import Macbert4MDCSpell as Graph


# adapt the input for ReLM
def convert_examples_to_features(examples, max_seq_length, tokenizer, prompt_length, anchor=None, mask_rate=0.2):
    features = []
    for i, example in tqdm(enumerate(examples), desc="data"):
        src, trg, block_flag, trg_ref = convert_examples_to_prompts(example.src, example.trg, prompt_length,
                                                                    max_seq_length, tokenizer, anchor, mask_rate)
        example.src = src
        example.trg = trg
        encoded_inputs = tokenizer(example.src,
                                   max_length=max_seq_length,
                                   padding="max_length",
                                   truncation=True,
                                   return_token_type_ids=True,
                                   is_split_into_words=True)

        trg_ids = tokenizer(example.trg,
                            max_length=max_seq_length,
                            padding="max_length",
                            truncation=True,
                            return_token_type_ids=True,
                            is_split_into_words=True)["input_ids"]

        trg_ref_ids = tokenizer(trg_ref,
                                max_length=max_seq_length,
                                padding="max_length",
                                truncation=True,
                                return_token_type_ids=True,
                                is_split_into_words=True)["input_ids"]

        src_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"]
        block_flag = ([0] + block_flag)[: max_seq_length]
        ## zero padding
        if len(block_flag) < max_seq_length:
            block_flag = block_flag + [0] * max(0, max_seq_length - len(block_flag))

        assert len(src_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(trg_ids) == max_seq_length
        assert len(trg_ref_ids) == max_seq_length
        assert len(block_flag) == max_seq_length

        # if i < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % example.guid)
        #     logger.info("src_tokens: %s" % " ".join(example.src))
        #     logger.info("trg_tokens: %s" % " ".join(example.trg))
        #     logger.info("src_ids: %s" % " ".join([str(x) for x in src_ids]))
        #     logger.info("trg_ids: %s" % " ".join([str(x) for x in trg_ids]))
        #     logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #     logger.info("block_flag: %s" % " ".join([str(x) for x in block_flag]))

        features.append(
            InputFeatures(src_ids=src_ids,
                          attention_mask=attention_mask,
                          trg_ids=trg_ids,
                          trg_ref_ids=trg_ref_ids,
                          block_flag=block_flag)
        )
    return features


# adapt the input for ReLM
def convert_examples_to_prompts(src, trg, prompt_length, max_seq_length, tokenizer, anchor=None, mask_rate=0.2):
    def truncate(x, max_length):
        return x[: max_length]

    ## here max_seq = tokenizer.max_seq_length//2, we need to truncate
    src = truncate(src, max_seq_length - prompt_length - 2)
    if anchor is not None:
        prompt_src = [tokenizer.cls_token] * prompt_length + src + anchor + [tokenizer.sep_token] * prompt_length
    else:
        prompt_src = [tokenizer.cls_token] * prompt_length + src + [tokenizer.sep_token] * prompt_length
    return prompt_src, [], [], []


class InputFeatures(object):
    def __init__(self, src_ids, attention_mask, trg_ids, trg_ref_ids, block_flag):
        self.src_ids = src_ids
        self.attention_mask = attention_mask
        self.trg_ids = trg_ids
        self.trg_ref_ids = trg_ref_ids
        self.block_flag = block_flag


class MDCSpellPredict:
    def __init__(self, path_pretrain_model_dir="bert-base-chinese",
                 path_trained_model_dir="./output/pytorch_model.bin",
                 device="cuda:0"):
        self.path_pretrain_model_dir = path_pretrain_model_dir
        self.path_trained_model_dir = path_trained_model_dir
        self.device = device
        self.load_trained_model()

    def load_trained_model(self):
        """   加载训练好的模型   """
        csc_config = load_json(os.path.join(self.path_trained_model_dir, "csc.config"))
        self.csc_config = Namespace(**csc_config)
        self.csc_config.pretrained_model_name_or_path = self.path_pretrain_model_dir \
            if self.path_pretrain_model_dir else self.csc_config.pretrained_model_name_or_path
        self.csc_config.flag_train = False
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.csc_config.pretrained_model_name_or_path,
                                                       do_lower_case=self.csc_config.do_lower_case,
                                                       use_fast=self.csc_config.flag_fast_tokenizer)
        self.processor = DataSetProcessor()
        self.model = Graph(config=self.csc_config)
        state_dict = torch.load(os.path.join(self.path_trained_model_dir, "pytorch_model.bin"))
        state_dict_keys_0 = list(state_dict.keys())[0]
        if "pretrain_model.bert." in state_dict_keys_0:
            state_dict = {k.replace("pretrain_model.", "bert."): v for k, v in state_dict.items()}
        elif "bert.bert." not in state_dict:
            state_dict = {"bert." + k: v for k, v in state_dict.items()}
        self.model.model.load_state_dict(state_dict, strict=False)
        # self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, texts):
        """   数据预处理   """
        pred_examples = self.processor._create_predicts(texts, set_type="pred")
        pred_features = convert_examples_to_features(pred_examples, self.csc_config.max_seq_length, self.tokenizer,
                                                     self.csc_config.prompt_length, anchor=self.csc_config.anchor)
        all_input_ids = torch.tensor([f.src_ids for f in pred_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.attention_mask for f in pred_features], dtype=torch.long)
        pred_data = TensorDataset(all_input_ids, all_input_mask)
        pred_sampler = SequentialSampler(pred_data)
        pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=self.csc_config.eval_batch_size)
        return pred_dataloader

    def decode(self, input_ids):
        """   解码   """
        return self.tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=self.csc_config.flag_skip)

    def predict(self, texts):
        """   推理   """
        # texts = [{"original_text":"", "correct_text":""}]
        all_inputs, all_labels, all_predictions = [], [], []
        res = []
        predict_dataloader = self.preprocess(texts)
        for batch in tqdm(predict_dataloader, desc="pred"):
            batch = tuple(t.to(self.device) for t in batch)
            src_ids, attention_mask = batch
            with torch.no_grad():
                outputs = self.model(input_ids=src_ids,
                                attention_mask=attention_mask,
                                )
                # tmp_eval_loss = outputs.loss
                prd_ids = outputs[-1]
                prd_prob = outputs[-2]

            for s, p, pb in zip(src_ids, prd_ids, prd_prob):
                mapped_src = []
                mapped_prd = []
                ##src: [CLS]+[CLS]...+src+[SEP]...
                ##trg: [CLS]+[CLS]...+src+[SEP]...
                # for st, pt in zip(s[self.config.prompt_length+1:-self.config.prompt_length-1],
                #                   p[self.config.prompt_length+1:-self.config.prompt_length-1]):
                #     mapped_src += [st]
                #     if st == pt:
                #         mapped_prd += [st]
                #     else:
                #         mapped_prd += [pt]
                for st, pt, pbt in zip(s, p, pb):
                    if st in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]:
                        continue
                    mapped_src += [st]
                    # 阈值
                    if st != pt and pbt >= self.csc_config.threshold:
                        mapped_prd += [pt]
                    else:
                        mapped_prd += [st]
                if self.csc_config.anchor is not None:
                    ##src: [CLS]+[CLS]...+src+[SEP]+anchor+[SEP]...+[mask]
                    ##trg: [CLS]+[CLS]...+src+[SEP]+anchor+[SEP]...+trg
                    ## remove the anchor tokens from the src
                    anchor_length = len(self.csc_config.anchor)
                    del mapped_prd[:anchor_length]
                ## we skip special tokens including '[UNK]'
                all_inputs += [self.decode(mapped_src)]
                all_predictions += [self.decode(mapped_prd)]
        for o, t in zip(texts, all_predictions):
            if "original_text" in o:
                source = o.get("original_text", "")
            else:
                source = o.get("source", "")
            target = "".join(t)
            # if len(source) != len(target):
            #     print(source)
            #     print(target)
            #     print(len(source))
            #     print(len(target))
            len_real = min(len(source), len(target))
            errors = []
            for idx in range(len_real):
                if source[idx].lower() != target[idx].lower():
                    errors.append([source[idx], target[idx], idx])
            line_dict = {"source": source, "target": target, "errors": errors}
            res.append(line_dict)
        return res


if __name__ == "__main__":
    # main()

    path_trained_model_dir = "../../../output/text_correction/espell_law_of_macbert4mdcspell"

    model = MDCSpellPredict(path_trained_model_dir, path_trained_model_dir)
    model.csc_config.threshold = 0
    texts = [{"source": "真麻烦你了。希望你们好好的跳无"},
             {"source": "少先队员因该为老人让坐"},
             {"source": "机七学习是人工智能领遇最能体现智能的一个分知"},
             {"source": "一只小鱼船浮在平净的河面上"},
             {"source": "我的家乡是有明的渔米之乡"},
             {"source": "chuxī到了,兰兰和妈妈一起包jiǎo子,兰兰包的jiǎo子十分可爱,\n今天,兰兰和妈妈dù过了一个&快乐的chúxī。"},
             {"source": "今天早上我吃了以个火聋果"},
             {"source": "我是联系时长两年半的个人练习生蔡徐鲲，喜欢唱跳RAP蓝球"},
             {"source": "眼睛蛇咬了"},
             {"source": "输暖管手术投保"},
             {"source": "白然语言处理"},

             ]
    res = model.predict(texts)
    for r in res:
        print(r)

    while 1:
        try:
            print("请输入：")
            text = input()
            text_in = [{"original_text": text}]
            time_start = time.time()
            res = model.predict(text_in)
            time_end = time.time()
            print(res)
            print(time_end - time_start)
        except Exception as e:
            print(traceback.print_exc())


"""
python predict.py


"""


