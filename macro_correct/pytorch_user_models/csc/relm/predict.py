# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/1/18 21:34
# @author  : Mo
# @function: relm
# @paper   : [Chinese Spelling Correction as Rephrasing Language Model](https://arxiv.org/abs/2308.08796).
# @code    : most code copy from https://github.com/Claude-Liu/ReLM, small modfix
# @modfiy  : remove the Network Architecture of P-tuning(BiLSTM)


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

from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import torch

from macro_correct.pytorch_user_models.csc.relm.dataset import DataSetProcessor, sent_mertic_det, sent_mertic_cor
from macro_correct.pytorch_user_models.csc.relm.dataset import save_json, load_json, txt_write
from macro_correct.pytorch_user_models.csc.relm.graph import RELM as Graph


# adapt the input for ReLM
def convert_examples_to_features(examples, max_seq_length, tokenizer, prompt_length, anchor=None, mask_rate=0.2):
    features = []
    for i, example in tqdm(enumerate(examples), desc="data"):
        src, trg, block_flag, trg_ref = convert_examples_to_prompts(example.src, example.trg, prompt_length, max_seq_length // 2, tokenizer, anchor, mask_rate)
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
    src = truncate(src, max_seq_length-prompt_length-2)
    if anchor is not None:
        prompt_src = [tokenizer.cls_token] * prompt_length + src + anchor + [tokenizer.sep_token] + [tokenizer.mask_token for _ in src] + [tokenizer.sep_token] * prompt_length
    else:
        prompt_src = [tokenizer.cls_token] * prompt_length + src + [tokenizer.sep_token] + [tokenizer.mask_token for _ in src] + [tokenizer.sep_token] * prompt_length
    return prompt_src, [], [], []


def dynamic_mask_token(inputs, targets, tokenizer, device, mask_mode="noerror", noise_probability=0.2):
    '''
    the masked-FT proposed in 'Rethinking Masked Language Model for Chinese Spelling Correction'
    '''
    #src:[CLS]...[CLS],x1,x2,...,xn,[SEP],...,[SEP],m1,m2,...,mn
    #trg:[CLS]...[CLS],t1,t2,...,tn,[SEP],...,[SEP],t1,t2,...,tn
    
    inputs = inputs.clone()
    probability_matrix = torch.full(inputs.shape, noise_probability).to(device)
    #do not mask sepcail tokens
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool).to(device)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    # mask_mode in ["all","error","noerror"]
    if mask_mode == "noerror":
        probability_matrix.masked_fill_(inputs!=targets, value=0.0)
    elif mask_mode == "error":
        probability_matrix.masked_fill_(inputs==targets, value=0.0)
    else:
        assert mask_mode == "all"
    masked_indices = torch.bernoulli(probability_matrix).bool()
    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs
    

class InputFeatures(object):
    def __init__(self, src_ids, attention_mask, trg_ids, trg_ref_ids, block_flag):
        self.src_ids = src_ids
        self.attention_mask = attention_mask
        self.trg_ids = trg_ids
        self.trg_ref_ids = trg_ref_ids
        self.block_flag = block_flag


class RelmPredict:
    def __init__(self, path_pretrain_model_dir="bert-base-chinese",
                 path_trained_model_dir="./output/step-9000_f1-0.50.bin",
                 device="cuda:0"):
        self.path_pretrain_model_dir = path_pretrain_model_dir
        self.path_trained_model_dir = path_trained_model_dir
        self.flag_fast_tokenizer = True
        self.device = device
        self.load_trained_model()

    def load_trained_model(self, path_trained_model_dir=None):
        """   加载训练好的模型   """
        csc_config = load_json(os.path.join(self.path_trained_model_dir, "csc.config"))
        self.csc_config = Namespace(**csc_config)
        self.csc_config.pretrained_model_name_or_path = self.path_pretrain_model_dir \
            if self.path_pretrain_model_dir else self.csc_config.pretrained_model_name_or_path
        self.csc_config.flag_train = False
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.csc_config.pretrained_model_name_or_path,
                                                       do_lower_case=self.csc_config.do_lower_case,
                                                       use_fast=self.csc_config.flag_fast_tokenizer)
        # self.model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=self.path_pretrain_model_dir, return_dict=True)
        # state_dict = torch.load(os.path.join(self.path_trained_model_dir, "pytorch_model.bin"))
        # state_dict = {k.replace("bert.", ""): v for k, v in state_dict.items()}
        self.processor = DataSetProcessor()
        self.model = Graph(config=self.csc_config)
        state_dict = torch.load(os.path.join(self.path_trained_model_dir, "pytorch_model.bin"))
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, texts):
        """   数据预处理   """
        pred_examples = self.processor._create_predicts(texts, set_type="pred")
        pred_features = convert_examples_to_features(pred_examples, self.csc_config.max_seq_length,
                                self.tokenizer, self.csc_config.prompt_length, anchor=self.csc_config.anchor)
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
                # logits = outputs.logits
                prd_cor_probs = outputs[0]
                prd_ids = outputs[-1]
                prd_prob = outputs[-2]
            # src_ids = src_ids.tolist()
            # _, prd_ids = torch.max(logits, -1)
            # prd_ids = prd_ids.masked_fill(attention_mask == 0, 0).tolist()
            # pred_prob, pred_ids = torch.max(logits, -1)
            src_ids = src_ids.detach().cpu().numpy().tolist()
            prd_ids = prd_ids.detach().cpu().numpy().tolist()
            prd_prob = prd_prob.detach().cpu().numpy().tolist()
            for s, p, pb in zip(src_ids, prd_ids, prd_prob):
                mapped_src = []
                mapped_prd = []
                flag = False  ## if we pass to the target part
                ##src: [CLS]+[CLS]...+src+[SEP]...+trg+[SEP]
                ##trg: [CLS]+[CLS]...+src+[SEP]...+trg+[SEP]
                for st, pt, pbt in zip(s, p, pb):
                    if st == self.tokenizer.sep_token_id:
                        flag = True
                    if not flag:
                        mapped_src += [st]
                    else:
                        ## threshold; and we only predict the masked tokens
                        if st == self.tokenizer.mask_token_id and pbt >= self.csc_config.threshold:
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
            len_real = min(len(source), len(target))
            len_max = max(len(source), len(target))
            source = source[:len_max]
            target = target[:len_max]
            errors = []
            ### 可能截断了
            for idx in range(len_real):
                if source[idx].lower() != target[idx].lower():
                    errors.append([source[idx], target[idx], idx])
            line_dict = {"source": source, "target": target, "errors": errors}
            res.append(line_dict)
        return res


if __name__ == "__main__":
    yz = 0

    path_model_dir = "../../../output/text_correction/espell_law_of_relm"

    model = RelmPredict(path_model_dir, path_model_dir)
    texts = [
             {"source": "真麻烦你了。希望你们好好的跳无"},
             {"source": "少先队员因该为老人让坐"},
             {"source": "机七学习是人工智能领遇最能体现智能的一个分知"},
             {"source": "一只小鱼船浮在平净的河面上"},
             {"source": "我的家乡是有明的渔米之乡"},
             {"source": "chuxī到了,兰兰和妈妈一起包jiǎo子,兰兰包的jiǎo子十分可爱,\n今天,兰兰和妈妈dù过了一个&快乐的chúxī。"},
             {"source": "哪里有上大学，不想念书的道理？"},
             {"source": "从那里，我们可以走到纳福境的新光三钺百货公司逛一逛"},
             {"source": "他主动拉了姑娘的手，心里很高心，嘴上故作生气"},
             {"source": "我这一次写信给你是想跟你安排一下关以我们要见面的。"},
             {"source": "为了减少急遍的生孩子率，需要呼吁适当的生育政策。"},
             {"source": "他主动拉了姑娘的手，心里很高心，嘴上故作生气"},
             {"source": "这种千亿市值的板块龙头，昨天都尽然可以这么杀？"},
             {"source": "关于刑法的解释，下列说法挫误的是"},
             {"source": "购买伟造的增值税专用发票罪"},
             {"source": "下裂选项有关劳动争议仲裁的表述中，正确的有:"},
             {"source": "本票上有关票面金页、收款人名称以及出票日期的记载不得余改，否作该票无效"},
             {"source": "包销分为全额包销和余额包销两种形式"},
             {"source": "如果“教酸他人犯罪”中的“他人”必须达到刑事法定年龄，则教锁15周岁的人盗窃的，不能适用第29条第一款后半段的规定"},

        {"source": "从那里，我们可以走到纳福境的新光三钺百货公司逛一逛"},
        {"source": "哪里有上大学，不想念书的道理？"},
        {"source": "他主动拉了姑娘的手，心里很高心，嘴上故作生气"},
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
            print(time_end-time_start)
        except Exception as e:
            print(traceback.print_exc())



"""
python predict.py

pred: 100%|██████████| 1/1 [00:00<00:00,  3.71it/s]
{'original_text': '真麻烦你了。希望你们好好的跳无', 'target_text': '真麻烦你了。希望你们好好的跳无', 'errors': []}
{'original_text': '少先队员因该为老人让坐', 'target_text': '少先队员因该为老人让座', 'errors': [['坐', '座', 10]]}
{'original_text': '机七学习是人工智能领遇最能体现智能的一个分知', 'target_text': '机器学习是人工智能领域最能体现智能的一个分享', 'errors': [['七', '器', 1], ['遇', '域', 10], ['知', '享', 21]]}
{'original_text': '一只小鱼船浮在平净的河面上', 'target_text': '一只小鱼船浮在干净的河面上', 'errors': [['平', '干', 7]]}
{'original_text': '我的家乡是有明的渔米之乡', 'target_text': '我的家乡是有名的渔米之乡', 'errors': [['明', '名', 6]]}
请输入：
"""


