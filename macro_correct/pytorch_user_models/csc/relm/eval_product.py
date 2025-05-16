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
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(path_root)
print(path_root)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import SchedulerType, get_scheduler
from transformers import BertForMaskedLM
from transformers import AutoTokenizer
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch

from macro_correct.pytorch_user_models.csc.relm.dataset import transfor_chinese_symbol_to_english, tradition_to_simple, string_q2b
from macro_correct.pytorch_user_models.csc.relm.dataset import DataSetProcessor, sent_mertic_det, sent_mertic_cor
from macro_correct.pytorch_user_models.csc.relm.dataset import save_json, load_json, txt_write, txt_read
from macro_correct.pytorch_user_models.csc.relm.graph import RELM as Graph


# adapt the input for ReLM
def convert_examples_to_features(examples, max_seq_length, tokenizer, prompt_length, anchor=None, mask_rate=0.2):
    features = []
    # for i, example in tqdm(enumerate(examples), desc="data"):
    for i, example in enumerate(examples):
        src, trg, block_flag, trg_ref = convert_examples_to_prompts(example.src, example.trg, prompt_length, max_seq_length // 2, tokenizer, anchor, mask_rate)
        example.src = src
        example.trg = trg
        encoded_inputs = tokenizer(example.src,
                                   max_length=max_seq_length,
                                   padding="max_length",
                                   truncation=True,
                                   return_token_type_ids=True,
                                   is_split_into_words=True)

        src_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"]
        block_flag = ([0] + block_flag)[: max_seq_length]
        ## zero padding
        if len(block_flag) < max_seq_length:
            block_flag = block_flag + [0] * max(0, max_seq_length - len(block_flag))

        assert len(src_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(block_flag) == max_seq_length

        # if i < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % example.guid)
        #     logger.info("src_tokens: %s" % " ".join(example.src))
        #     logger.info("src_ids: %s" % " ".join([str(x) for x in src_ids]))
        #     logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #     logger.info("block_flag: %s" % " ".join([str(x) for x in block_flag]))
        
        features.append(
                InputFeatures(src_ids=src_ids,
                              attention_mask=attention_mask,
                              trg_ids=[],
                              trg_ref_ids=[],
                              block_flag=block_flag)
        )
    return features


# adapt the input for ReLM
def convert_examples_to_prompts(src, trg, prompt_length, max_seq_length, tokenizer, anchor=None, mask_rate=0.2):
    def truncate(x, max_length):
        return x[: max_length]
    ## here max_seq = tokenizer.max_seq_length//2, we need to truncate
    # src = truncate(src, max_seq_length-prompt_length-2)
    src = truncate(src, max_seq_length-prompt_length-2)
    ### 中间必须要有一个sep
    if anchor is not None:
        prompt_src = [tokenizer.cls_token] * prompt_length + src + anchor + [tokenizer.sep_token] + [tokenizer.mask_token for _ in src] + [tokenizer.sep_token] * prompt_length
        # prompt_trg = [tokenizer.cls_token] * prompt_length + src + anchor + [tokenizer.sep_token] + trg + [tokenizer.sep_token] * prompt_length
        block_flag = [1] * prompt_length + [0 for _ in src] + [0 for _ in anchor] + [1] + [0 for _ in src] + [0] * prompt_length
        # trg_ref = [tokenizer.cls_token] * prompt_length + trg + anchor + [tokenizer.sep_token] + trg + [tokenizer.sep_token] * prompt_length
        # src_ref = [tokenizer.cls_token] * prompt_length + src + anchor + [tokenizer.sep_token] * (prompt_length+1) + src + [tokenizer.sep_token] * prompt_length
    else:
        prompt_src = [tokenizer.cls_token] * prompt_length + src + [tokenizer.sep_token] + [tokenizer.mask_token for _ in src] + [tokenizer.sep_token] * prompt_length
        # prompt_trg = [tokenizer.cls_token] * prompt_length + src + [tokenizer.sep_token] + trg + [tokenizer.sep_token] * prompt_length
        block_flag = [1] * prompt_length + [0 for _ in src] + [1] + [0 for _ in src] + [0] * prompt_length
        # trg_ref = [tokenizer.cls_token] * prompt_length + trg + [tokenizer.sep_token] + trg + [tokenizer.sep_token] * prompt_length
        # src_ref = [tokenizer.cls_token] * prompt_length + src + [tokenizer.sep_token] * (prompt_length + 1) + src + [tokenizer.sep_token] * prompt_length
    return prompt_src, [], block_flag, []  # , src_ref


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
                 path_trained_model_dir="../output",
                 device="cuda:0"
                 ):
        self.path_pretrain_model_dir = path_pretrain_model_dir
        self.path_trained_model_dir = path_trained_model_dir
        self.device = device
        self.load_model()

    def load_model(self, path_trained_model_dir=None):
        """   加载训练好的模型   """
        csc_config = load_json(os.path.join(self.path_trained_model_dir, "csc.config"))
        self.csc_config = Namespace(**csc_config)
        self.csc_config.pretrained_model_name_or_path = self.path_pretrain_model_dir \
            if self.path_pretrain_model_dir else self.csc_config.pretrained_model_name_or_path
        self.csc_config.flag_train = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.csc_config.pretrained_model_name_or_path,
            do_lower_case=self.csc_config.do_lower_case,
            use_fast=self.csc_config.flag_fast_tokenizer)
        # self.model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=self.path_pretrain_model_dir, return_dict=True)
        # state_dict = torch.load(os.path.join(self.path_trained_model_dir, "pytorch_model.bin"))
        # state_dict = {k.replace("bert.", ""): v for k, v in state_dict.items()}
        self.processor = DataSetProcessor()
        self.model = Graph(config=self.csc_config)
        state_dict = torch.load(os.path.join(self.path_trained_model_dir, "pytorch_model.bin"))
        # self.model.load_state_dict(state_dict)
        state_dict_keys_0 = list(state_dict.keys())[0]
        if "pretrain_model.bert." in state_dict_keys_0:
            state_dict = {k.replace("pretrain_model.", "bert."): v for k, v in state_dict.items()}
        elif "bert.bert." not in state_dict:
            state_dict = {"bert." + k: v for k, v in state_dict.items()}
        self.model.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, texts):
        """   数据预处理   """
        pred_examples = self.processor._create_predicts(texts, set_type="pred")
        pred_features = convert_examples_to_features(pred_examples, self.csc_config.max_seq_length, self.tokenizer, self.csc_config.prompt_length, anchor=self.csc_config.anchor)
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
        # texts = [{"source":"", "correct_text":""}]
        all_inputs, all_labels, all_predictions = [], [], []
        res = []
        predict_dataloader = self.preprocess(texts)
        # for batch in tqdm(predict_dataloader, desc="pred"):
        for batch in predict_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            src_ids, attention_mask = batch
            with torch.no_grad():
                outputs = self.model(input_ids=src_ids,
                                attention_mask=attention_mask,
                                # labels=trg_ids,
                                # prompt_mask=block_flag,
                                # apply_prompt=args.apply_prompt
                                )
                # tmp_eval_loss = outputs.loss
                # prd_ids = outputs[-1]
                prd_ids = outputs[-1]
                prd_prob = outputs[-2]
            #     logits = outputs.logits
            # src_ids = src_ids.tolist()
            # _, prd_ids = torch.max(logits, -1)
            # prd_ids = prd_ids.masked_fill(attention_mask == 0, 0).tolist()
            # _, prd_ids = torch.max(logits, -1)
            # prd_ids = prd_ids.masked_fill(attention_mask == 0, 0).tolist()
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
                # s_decode = self.decode(s)
                # p_decode = self.decode(p)
                # src_decode = self.decode(mapped_src)
                # pred_decode = self.decode(mapped_prd)
                # print("".join(s_decode))
                # print("".join(src_decode))
                # print("".join(p_decode))
                # print("".join(pred_decode))
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
            # source = o.get("source", "")
            if "original_text" in o:
                source = o.get("original_text", "")
            else:
                source = o.get("source", "")
            target = "".join(t)
            len_real = min(len(source), len(target))
            source = source[:len_real]
            target = target[:len_real]
            errors = []
            ### 可能截断了
            for idx in range(len_real):
                if source[idx] != target[idx]:
                    errors.append([source[idx], target[idx], idx])
            line_dict = {"source": source, "target": target, "errors": errors}
            res.append(line_dict)
        return res


def eval_std_list(path_tet=None, path_model_dir=None, threshold=0.5, args=None):
    """   测评   """
    processor = DataSetProcessor()
    if path_model_dir and os.path.exists(path_model_dir):
        csc_config = load_json(os.path.join(path_model_dir, "csc.config"))
        args = Namespace(**csc_config)
        args.pretrained_model_name_or_path = path_model_dir
        args.model_save_path = path_model_dir
        args.flag_train = False
        args.threshold = threshold
    else:
        print("no path_model_dir, fail")
    processor.path_train = args.path_train
    processor.task_name = args.task_name
    processor.path_dev = args.path_dev
    processor.path_tet = path_tet or args.path_tet
    # task_name = args.task_name

    ### 加载模型
    model = RelmPredict(path_model_dir, path_model_dir)


    # args.model_save_path = os.path.join(args.model_save_path, task_name)
    # if not os.path.exists(args.model_save_path):
    #     os.makedirs(args.model_save_path)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO,
                        filename=os.path.join(args.model_save_path, "eval_std.py.log"))
    logger = logging.getLogger(__name__)
    # 创建一个handler用于输出到控制台
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    datas = load_json(path_tet)
    all_predictions = []
    all_labels = []
    all_inputs = []
    datas_bs = []
    for d in tqdm(datas, desc="data"):
        datas_bs.append(d)
        if len(datas_bs) == model.csc_config.eval_batch_size:
            predict_bs = model.predict(datas_bs)
            for s, t in zip(datas_bs, predict_bs):
                # len_pred = len(t["target_text"])
                # all_predictions.append(t["target_text"])
                # all_labels.append(s["target"][:len_pred])
                # all_inputs.append(s["source"][:len_pred])
                if "source" in d:
                    len_pred = len(t["target"])
                    all_predictions.append(t["target"])
                    all_labels.append(s["target"][:len_pred])
                    all_inputs.append(t["source"])
                else:
                    len_pred = len(t["target"])
                    all_predictions.append(t["target"])
                    all_labels.append(s["correct_text"][:len_pred])
                    all_inputs.append(t["source"])
            datas_bs = []
    if datas_bs:
        predict_bs = model.predict(datas_bs)
        for s, t in zip(datas_bs, predict_bs):
            # len_pred = len(t["target_text"])
            # all_predictions.append(t["target_text"])
            # all_labels.append(s["target"][:len_pred])
            # all_inputs.append(s["source"][:len_pred])
            if "source" in datas[0]:
                len_pred = len(t["target"])
                all_predictions.append(t["target"])
                all_labels.append(s["target"][:len_pred])
                all_inputs.append(t["source"])
            else:
                len_pred = len(t["target"])
                all_predictions.append(t["target"])
                all_labels.append(s["correct_text"][:len_pred])
                all_inputs.append(t["source"])
    print("#" * 128)
    common_det_acc, common_det_precision, common_det_recall, common_det_f1 = sent_mertic_det(
        all_inputs, all_predictions, all_labels, logger)
    common_cor_acc, common_cor_precision, common_cor_recall, common_cor_f1 = sent_mertic_cor(
        all_inputs, all_predictions, all_labels, logger)

    common_det_mertics = f'common Sentence Level detection: acc:{common_det_acc:.4f}, precision:{common_det_precision:.4f}, recall:{common_det_recall:.4f}, f1:{common_det_f1:.4f}'
    common_cor_mertics = f'common Sentence Level correction: acc:{common_cor_acc:.4f}, precision:{common_cor_precision:.4f}, recall:{common_cor_recall:.4f}, f1:{common_cor_f1:.4f}'
    print("#" * 128)
    logger.info(f'flag_eval: common')
    print(f'flag_eval: common')
    logger.info("path_tet: " + str(processor.path_tet))
    print("path_tet: " + str(processor.path_tet))
    logger.info(common_det_mertics)
    print(common_det_mertics)
    logger.info(common_cor_mertics)
    print(common_cor_mertics)
    print("#" * 128)

    strict_det_acc, strict_det_precision, strict_det_recall, strict_det_f1 = sent_mertic_det(
        all_inputs, all_predictions, all_labels, logger, flag_eval="strict")
    strict_cor_acc, strict_cor_precision, strict_cor_recall, strict_cor_f1 = sent_mertic_cor(
        all_inputs, all_predictions, all_labels, logger, flag_eval="strict")

    strict_det_mertics = f'strict Sentence Level detection: acc:{strict_det_acc:.4f}, precision:{strict_det_precision:.4f}, recall:{strict_det_recall:.4f}, f1:{strict_det_f1:.4f}'
    strict_cor_mertics = f'strict Sentence Level correction: acc:{strict_cor_acc:.4f}, precision:{strict_cor_precision:.4f}, recall:{strict_cor_recall:.4f}, f1:{strict_cor_f1:.4f}'
    print("#" * 128)
    logger.info(f'flag_eval: strict')
    print(f'flag_eval: strict')
    logger.info("path_tet: " + str(processor.path_tet))
    print("path_tet: " + str(processor.path_tet))
    logger.info(strict_det_mertics)
    print(strict_det_mertics)
    logger.info(strict_cor_mertics)
    print(strict_cor_mertics)
    print("#" * 128)

    result_mertics = {
        "eval_loss": None,

        "common_det_acc": common_det_acc,
        "common_det_precision": common_det_precision,
        "common_det_recall": common_det_recall,
        "common_det_f1": common_det_f1,

        "common_cor_acc": common_cor_acc,
        "common_cor_precision": common_cor_precision,
        "common_cor_recall": common_cor_recall,
        "common_cor_f1": common_cor_f1,

        "strict_det_acc": strict_det_acc,
        "strict_det_precision": strict_det_precision,
        "strict_det_recall": strict_det_recall,
        "strict_det_f1": strict_det_f1,

        "strict_cor_acc": strict_cor_acc,
        "strict_cor_precision": strict_cor_precision,
        "strict_cor_recall": strict_cor_recall,
        "strict_cor_f1": strict_cor_f1,
        "task_name": os.path.split(path_tet)[-1]
    }

    output_eval_file = os.path.join(args.model_save_path, "eval_product.results.txt")
    text_log_list = ["#"*128 + "\n", "path_tet: " + path_tet.strip() + "\n"]
    for key in sorted(result_mertics.keys()):
        text_log = "Global step: %s,  %s = %s\n" % (str(-1), key, str(result_mertics[key]))
        # logger.info(text_log)
        text_log_list.append(text_log)
    txt_write(text_log_list, output_eval_file, mode="a+")

    return common_det_mertics, common_cor_mertics, strict_det_mertics, strict_cor_mertics, result_mertics
def tet_test_dataset(path_model_dir=None, path_tet=None):
    """   模型测试数据集, 指定原来的本地的   """
    path_model_dir = path_model_dir or  "../../../output/text_correction/espell_law_of_relm"  # 预训练模型的config.json和tokenizer也存在该目录

    # model = RelmPredict(path_model_dir, path_model_dir)
    # path_tet = path_tet or model.csc_config.path_tet  # 或者自己指定位置
    path_csc_config = os.path.join(path_model_dir, "csc.config")
    csc_config = load_json(path_csc_config)
    args = Namespace(**csc_config)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO,
                        filename=os.path.join(path_model_dir, "eval.py.log"))
    logger = logging.getLogger(__name__)
    # 创建一个handler用于输出到控制台
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    ### 全量测试
    path_corpus_dir = os.path.join(path_root, "macro_correct", "corpus", "text_correction")
    path_tet1 = os.path.join(path_corpus_dir, "public/gen_de3.json")
    path_tet2 = os.path.join(path_corpus_dir, "public/lemon_v2.tet.json")
    path_tet3 = os.path.join(path_corpus_dir, "public/acc_rmrb.tet.json")
    path_tet4 = os.path.join(path_corpus_dir, "public/acc_xxqg.tet.json")

    path_tet5 = os.path.join(path_corpus_dir, "public/gen_passage.tet.json")
    path_tet6 = os.path.join(path_corpus_dir, "public/textproof.tet.json")
    path_tet7 = os.path.join(path_corpus_dir, "public/gen_xxqg.tet.json")

    path_tet8 = os.path.join(path_corpus_dir, "public/faspell.dev.json")
    path_tet9 = os.path.join(path_corpus_dir, "public/lomo_tet.json")
    path_tet10 = os.path.join(path_corpus_dir, "public/mcsc_tet_5k.json")
    path_tet11 = os.path.join(path_corpus_dir, "public/ecspell.dev.json")
    path_tet12 = os.path.join(path_corpus_dir, "public/sighan2013.dev.json")
    path_tet13 = os.path.join(path_corpus_dir, "public/sighan2014.dev.json")
    path_tet14 = os.path.join(path_corpus_dir, "public/sighan2015.dev.json")
    path_tet15 = os.path.join(path_corpus_dir, "public/mcsc_tet.json")

    path_tet_list = [path_tet1, path_tet2, path_tet3, path_tet4, path_tet5,
                     path_tet6, path_tet7, path_tet8, path_tet9, path_tet10,
                     path_tet11, path_tet12, path_tet13, path_tet14,
                     # path_tet15,
                     ][-2:-1]

    string_mertics_list = [path_model_dir + "\n"]
    res_mertics = {}  # 最终结果
    threshold = 0  # 阈值
    for path_tet_i in tqdm(path_tet_list, desc="path_tet_list"):
        common_det_mertics, common_cor_mertics, strict_det_mertics, strict_cor_mertics, result_mertics \
            = eval_std_list(path_tet=path_tet_i, path_model_dir=path_model_dir, threshold=threshold, args=args)
        print(path_tet_i)
        string_mertics_list.append("path_tet_i: " + os.path.split(path_tet_i)[-1] + "\n")
        string_mertics_list.append(common_det_mertics + "\n")
        string_mertics_list.append(common_cor_mertics + "\n")
        string_mertics_list.append(strict_det_mertics + "\n")
        string_mertics_list.append(strict_cor_mertics + "\n")
        string_mertics_list.append("#" * 128 + "\n")
        res_mertics[path_tet_i] = result_mertics
    txt_write(string_mertics_list, os.path.join(path_model_dir, "eval_std.pred_mertics.txt"))
    save_json(res_mertics, os.path.join(path_model_dir, "eval_std.pred_mertics.json"))


if __name__ == "__main__":
    # yz = 0

    ### 测试全量验证集
    path_model_dir = "../../../output/text_correction/espell_law_of_relm"
    tet_test_dataset(path_model_dir=path_model_dir)

    #### 测试单个样例
    path_model_dir = "../../../output/text_correction/espell_law_of_relm"
    model = RelmPredict(path_model_dir, path_model_dir)
    path_tet = model.csc_config.path_tet  # 或者自己指定位置

    texts = [{"source": "真麻烦你了。希望你们好好的跳无"},
             {"source": "少先队员因该为老人让坐"},
             {"source": "机七学习是人工智能领遇最能体现智能的一个分知"},
             {"source": "一只小鱼船浮在平净的河面上"},
             {"source": "我的家乡是有明的渔米之乡"},
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

             {"source": "我今天吃苹果，明天吃香姣"},
            {"source": "这位外国网友说：这是地球最安全的锅加，在这里，我从未感到过害啪！"},
            {"source": "智能识别并纠正句子中的语法、拼写、用词等错误，确保文本的准确性和可读性。"},
            {"source": "实现祖国完全统一，是全体中华儿女共同愿望，解决台湾问题，是中华民族根本利益所在。推动两岸关系和平发展，必须继续坚持“和平统一、一郭两制”方针，退进祖国和平统一。"},
            {"source": "百度是一家人工只能公司"},
            {"source": "我门是中国点信的源工"},
            {"source": "通过机七学习可以实现中文语句的智能纠错。我的朋友法语说的很好，的语也不错。"},
            {"source": "天行数据是一个友秀的只能平台"},
            {"source": "传承和弘扬中华优秀传统文化既是增强文华自信、建设社会主义文化强国的应然之义，也是全面建设社会注意现代化国家、推进实现中华民族伟大复兴的实践前提。"},


            {"source": "刚开始我也收不撩这样。"},
            {"source": "没想到跳着跳着就照过半夜了。"},

            {"source": "团长将士兵布署在城外，让他们安兵不动。"},
            {"source": "任何因难,都不能下倒有坚强意志的对员们。"},

            {"source": "患者需要按照剂量服用甲苯米坐片。"},
            {"source": "人民监查员依法审察案件。"},

            {"source": "希望您帮我素取公平。"},
            {"source": "我们为这个目标努力不解。"},

            {"source": "我门看美国电影，我觉得很有意事。"},
            {"source": "他派了很多照片。"},
            {"source": "他门才开始考试。"},
            {"source": "大明说: 请座，请座。"},
            {"source": "可是你现在不在宿舍，所以我留了一枝条。"},
            {"source": "我以前想要高诉你，可是我忘了。我真户秃。"},

            {"source": "老师进教师来了。"},
            {"source": "我带上运动鞋出门。"},
             ]

    res = model.predict(texts)
    for r in res:
        print(r)




"""
python predict.py

pred: 100%|██████████| 1/1 [00:00<00:00,  3.71it/s]
{'source': '真麻烦你了。希望你们好好的跳无', 'target_text': '真麻烦你了。希望你们好好的跳无', 'errors': []}
{'source': '少先队员因该为老人让坐', 'target_text': '少先队员因该为老人让座', 'errors': [['坐', '座', 10]]}
{'source': '机七学习是人工智能领遇最能体现智能的一个分知', 'target_text': '机器学习是人工智能领域最能体现智能的一个分享', 'errors': [['七', '器', 1], ['遇', '域', 10], ['知', '享', 21]]}
{'source': '一只小鱼船浮在平净的河面上', 'target_text': '一只小鱼船浮在干净的河面上', 'errors': [['平', '干', 7]]}
{'source': '我的家乡是有明的渔米之乡', 'target_text': '我的家乡是有名的渔米之乡', 'errors': [['明', '名', 6]]}
请输入：

### output_relm2  后处理---2
Sentence Level detection: acc:0.9052, precision:0.8524, recall:0.9167, f1:0.8834
Sentence Level correction: acc:0.8972, precision:0.8376, recall:0.9008, f1:0.8681

### output_relm2  后处理---1
Sentence Level detection: acc:0.9021, precision:0.8479, recall:0.9139, f1:0.8797
Sentence Level correction: acc:0.8938, precision:0.8327, recall:0.8975, f1:0.8639

### output_relm2  eval_batch_size=32
Sentence Level detection: acc:0.8604, precision:0.7810, recall:0.8807, f1:0.8279
Sentence Level correction: acc:0.8521, precision:0.7664, recall:0.8642, f1:0.8124

### output_relm2  eval_batch_size=8
Sentence Level detection: acc:0.8629, precision:0.7845, recall:0.8845, f1:0.8315
Sentence Level correction: acc:0.8548, precision:0.7703, recall:0.8685, f1:0.8165
"""


