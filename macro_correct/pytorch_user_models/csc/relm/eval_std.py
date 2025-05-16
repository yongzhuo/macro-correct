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
import argparse
import logging
import random
import math
import copy
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(path_root)
print(path_root)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import SchedulerType, get_scheduler
from transformers import BertForMaskedLM
from transformers import AutoTokenizer
from tensorboardX import SummaryWriter
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch

from macro_correct.pytorch_user_models.csc.relm.dataset import transfor_english_symbol_to_chinese, transfor_bert_unk_pun_to_know, tradition_to_simple, string_q2b
from macro_correct.pytorch_user_models.csc.relm.dataset import DataSetProcessor, sent_mertic_det, sent_mertic_cor, char_mertic_det_cor
from macro_correct.pytorch_user_models.csc.relm.dataset import save_json, load_json, txt_write
from macro_correct.pytorch_user_models.csc.relm.config import csc_config as args
from macro_correct.pytorch_user_models.csc.relm.graph import RELM as Graph


class InputFeatures(object):
    def __init__(self, src_ids, attention_mask, trg_ids, trg_ref_ids, block_flag, src_ref_ids=None):
        self.attention_mask = attention_mask
        self.trg_ref_ids = trg_ref_ids
        self.src_ref_ids = src_ref_ids
        self.block_flag = block_flag
        self.src_ids = src_ids
        self.trg_ids = trg_ids


# adapt the input for ReLM
def convert_examples_to_features(examples, max_seq_length, tokenizer, prompt_length, anchor=None, mask_rate=0.2, logger=None):
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

        # src_ref_ids = tokenizer(src_ref,
        #                     max_length=max_seq_length,
        #                     padding="max_length",
        #                     truncation=True,
        #                     return_token_type_ids=True,
        #                     is_split_into_words=True)["input_ids"]

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
        # assert len(src_ref_ids) == max_seq_length

        if i < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("src_tokens: %s" % " ".join(example.src))
            logger.info("trg_tokens: %s" % " ".join(example.trg))
            logger.info("src_ids: %s" % " ".join([str(x) for x in src_ids]))
            logger.info("trg_ids: %s" % " ".join([str(x) for x in trg_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("block_flag: %s" % " ".join([str(x) for x in block_flag]))
        
        features.append(
                InputFeatures(src_ids=src_ids,
                              attention_mask=attention_mask,
                              trg_ids=trg_ids,
                              trg_ref_ids=trg_ref_ids,
                              block_flag=block_flag,
                              # src_ref_ids=src_ref_ids
                              )
        )
    return features


# adapt the input for ReLM
def convert_examples_to_prompts(src, trg, prompt_length, max_seq_length, tokenizer, anchor=None, mask_rate=0.2):
    def truncate(x, max_length):
        return x[: max_length]
    ## here max_seq = tokenizer.max_seq_length//2, we need to truncate
    src = truncate(src, max_seq_length - prompt_length - 2)
    trg = truncate(trg, max_seq_length - prompt_length - 2)
    assert(len(src) == len(trg))
    ### 中间必须要有一个sep
    if anchor is not None:
        prompt_src = [tokenizer.cls_token] * prompt_length + src + anchor + [tokenizer.sep_token] + [tokenizer.mask_token for _ in trg] + [tokenizer.sep_token] * prompt_length
        prompt_trg = [tokenizer.cls_token] * prompt_length + src + anchor + [tokenizer.sep_token] + trg + [tokenizer.sep_token] * prompt_length
        block_flag = [1] * prompt_length + [0 for _ in src] + [0 for _ in anchor] + [1] + [0 for _ in trg] + [0] * prompt_length
        trg_ref = [tokenizer.cls_token] * prompt_length + trg + anchor + [tokenizer.sep_token] + trg + [tokenizer.sep_token] * prompt_length
        # src_ref = [tokenizer.cls_token] * prompt_length + src + anchor + [tokenizer.sep_token] * (prompt_length+1) + src + [tokenizer.sep_token] * prompt_length
    else:
        prompt_src = [tokenizer.cls_token] * prompt_length + src + [tokenizer.sep_token] + [tokenizer.mask_token for _ in trg] + [tokenizer.sep_token] * prompt_length
        prompt_trg = [tokenizer.cls_token] * prompt_length + src + [tokenizer.sep_token] + trg + [tokenizer.sep_token] * prompt_length
        block_flag = [1] * prompt_length + [0 for _ in src] + [1] + [0 for _ in trg] + [0] * prompt_length
        trg_ref = [tokenizer.cls_token] * prompt_length + trg + [tokenizer.sep_token] + trg + [tokenizer.sep_token] * prompt_length
        # src_ref = [tokenizer.cls_token] * prompt_length + src + [tokenizer.sep_token] * (prompt_length + 1) + src + [tokenizer.sep_token] * prompt_length
    return prompt_src, prompt_trg, block_flag, trg_ref  # , src_ref


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
    

def eval_std(path_tet=None, path_model_dir=None, threshold=0.5, args=None):
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

    device = torch.device("cuda" if torch.cuda.is_available() and args.flag_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, "Unsupported", args.flag_fp16))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps##

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    # if args.do_train:
    #     torch.save(args, os.path.join(args.model_save_path, "train_args.bin"))

    # tensorboardx_witer = SummaryWriter(logdir=args.model_save_path)

    # task_name = args.task_name.lower()
    # if task_name not in processors:
    #     raise ValueError("Task not found: %s" % task_name)
    #
    # processor = processors[task_name]()

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path,
                                              do_lower_case=args.do_lower_case,
                                              # cache_dir=cache_dir,
                                              use_fast=args.flag_fast_tokenizer)
    
    anchor=None
    if args.anchor is not None:
        anchor=[tokenizer.sep_token]+[t for t in args.anchor]


    if args.do_test:
        eval_examples = processor.get_test_examples()
        eval_features = convert_examples_to_features(eval_examples, args.max_seq_length, tokenizer, args.prompt_length, anchor=anchor, logger=logger)

        all_input_ids = torch.tensor([f.src_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.trg_ids for f in eval_features], dtype=torch.long)
        all_block_flag = torch.tensor([f.block_flag for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_block_flag)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        ### 存储为标准模式的文件, 最好的结果
        path_model_best = os.path.join(args.model_save_path, "pytorch_model.bin")
        model = Graph(config=args)
        # model = PTuningWrapper(model, args.prompt_length)
        model.to(device)
        model.load_state_dict(torch.load(path_model_best))
        model.eval()
        # model = BertForMaskedLM.from_pretrained(args.load_model_path,
        #                                         return_dict=True,
        #                                         cache_dir=cache_dir)
        # model = PTuningWrapper(model, args.prompt_length)
        # model.to(device)
        # if args.load_state_dict:
        #     model.load_state_dict(torch.load(args.load_state_dict))
        # if n_gpu > 1:
        #     model = torch.nn.DataParallel(model)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        def decode(input_ids):
            input_tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)
            # return [t.lower() for t in input_tokens]
            return input_tokens

        eval_loss = 0
        eval_steps = 0
        all_inputs, all_labels, all_predictions = [], [], []
        for batch in tqdm(eval_dataloader, desc="Evaluation"):
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            src_ids, attention_mask, trg_ids, block_flag = batch
            # ### unk转mask
            # src_ids = src_ids.masked_fill(src_ids == tokenizer.unk_token_id, tokenizer.mask_token_id)
            with torch.no_grad():
                outputs = model(input_ids=src_ids,
                                attention_mask=attention_mask,
                                labels=trg_ids,
                                )
                # tmp_eval_loss = outputs.loss
                # logits = outputs.logits
                tmp_eval_loss = outputs[0]
                logits = outputs[-2]

            src_ids = src_ids.detach().cpu().numpy().tolist()
            trg_ids = trg_ids.detach().cpu().numpy().tolist()
            eval_loss += tmp_eval_loss.mean().item()
            prd_prob, prd_ids = torch.max(logits, -1)
            prd_prob = prd_prob.detach().cpu().numpy().tolist()
            prd_ids = prd_ids.masked_fill(attention_mask == 0, 0).tolist()
            for s, t, p, pb in zip(src_ids, trg_ids, prd_ids, prd_prob):
                mapped_src = []
                mapped_trg = []
                mapped_prd = []
                flag = False  ## if we pass to the target part
                ##src: [CLS]+[CLS]...+src+[SEP]...+trg+[SEP]
                ##trg: [CLS]+[CLS]...+src+[SEP]...+trg+[SEP]
                for st, tt, pt, pbt in zip(s, t, p, pb):
                    if st == tokenizer.sep_token_id:
                        flag = True
                    if not flag:
                        mapped_src += [st]
                    else:
                        mapped_trg += [tt]
                        if st == tokenizer.mask_token_id and pbt >= args.threshold:##we only predict the masked tokens
                        # if st == tokenizer.mask_token_id:  ##we only predict the masked tokens
                            mapped_prd += [pt]
                        else:
                            mapped_prd += [st]
                if anchor is not None:
                    ##src: [CLS]+[CLS]...+src+[SEP]+anchor+[SEP]...+[mask]
                    ##trg: [CLS]+[CLS]...+src+[SEP]+anchor+[SEP]...+trg
                    ## remove the anchor tokens from the src
                    anchor_length = len(anchor)
                    del mapped_trg[:anchor_length]
                    del mapped_prd[:anchor_length]
                ## we skip special tokens including '[UNK]'
                all_inputs += [decode(mapped_src)]
                all_labels += [decode(mapped_trg)]
                all_predictions += [decode(mapped_prd)]

                print("".join(all_inputs[-1]))
                print("".join(all_labels[-1]))
                print("".join(all_predictions[-1]))
                print("#"*128)

            eval_steps += 1

        eval_loss = eval_loss / eval_steps

        print("#" * 128)
        det_acc, det_precision, det_recall, det_f1 = sent_mertic_det(all_inputs, all_predictions, all_labels, logger)
        cor_acc, cor_precision, cor_recall, cor_f1 = sent_mertic_cor(all_inputs, all_predictions, all_labels, logger)
        logger.info(
            f'Sentence Level detection: acc:{det_acc:.4f}, precision:{det_precision:.4f}, recall:{det_recall:.4f}, f1:{det_f1:.4f}')
        print(
            f'Sentence Level detection: acc:{det_acc:.4f}, precision:{det_precision:.4f}, recall:{det_recall:.4f}, f1:{det_f1:.4f}')
        logger.info(
            f'Sentence Level correction: acc:{cor_acc:.4f}, precision:{cor_precision:.4f}, recall:{cor_recall:.4f}, f1:{cor_f1:.4f}')

        print(
            f'Sentence Level correction: acc:{cor_acc:.4f}, precision:{cor_precision:.4f}, recall:{cor_recall:.4f}, f1:{cor_f1:.4f}')
        print("#" * 128)


        result = {
            "eval_loss": eval_loss,

            "det_acc": det_acc,
            "det_precision": det_precision,
            "det_recall": det_recall,
            "det_f1": det_f1,

            "cor_acc": cor_acc,
            "cor_precision": cor_precision,
            "cor_recall": cor_recall,
            "cor_f1": cor_f1,
        }

        output_eval_file = os.path.join(args.model_save_path, "eval_std.results.txt")
        text_log_list = []
        for key in sorted(result.keys()):
            text_log = "Global step: %s,  %s = %s\n" % (str(-1), key, str(result[key]))
            logger.info(text_log)
            text_log_list.append(text_log)
        text_log_list.append("\n")
        txt_write(text_log_list, output_eval_file, mode="a+")


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

    device = torch.device("cuda" if torch.cuda.is_available() and args.flag_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, "Unsupported", args.flag_fp16))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps  ##

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    # if args.do_train:
    #     torch.save(args, os.path.join(args.model_save_path, "train_args.bin"))

    # tensorboardx_witer = SummaryWriter(logdir=args.model_save_path)

    # task_name = args.task_name.lower()
    # if task_name not in processors:
    #     raise ValueError("Task not found: %s" % task_name)
    #
    # processor = processors[task_name]()

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path,
                                              do_lower_case=args.do_lower_case,
                                              # cache_dir=cache_dir,
                                              use_fast=args.flag_fast_tokenizer)

    anchor = None
    if args.anchor is not None:
        anchor = [tokenizer.sep_token] + [t for t in args.anchor]

    common_det_mertics = None
    common_cor_mertics = None
    strict_det_mertics = None
    strict_cor_mertics = None
    result_mertics = {}
    if args.do_test:
        eval_examples = processor.get_test_examples()
        eval_features = convert_examples_to_features(eval_examples, args.max_seq_length, tokenizer, args.prompt_length,
                                                     anchor=anchor, logger=logger)

        all_input_ids = torch.tensor([f.src_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.trg_ids for f in eval_features], dtype=torch.long)
        all_block_flag = torch.tensor([f.block_flag for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_block_flag)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        ### 存储为标准模式的文件, 最好的结果
        path_model_best = os.path.join(args.model_save_path, "pytorch_model.bin")
        model = Graph(config=args)
        # model = PTuningWrapper(model, args.prompt_length)
        # model.load_state_dict(torch.load(path_model_best))
        # ### relm-m0.3.bin
        # state_dict = torch.load("E:/DATA/bert-model/00_pytorch/CSC/relm-m0.3.bin")
        # state_dict = {"bert." + k: v for k, v in state_dict.items()}
        # model.load_state_dict(state_dict, strict=False)
        ### train_model
        state_dict = torch.load(path_model_best)
        state_dict_keys_0 = list(state_dict.keys())[0]
        if "pretrain_model.bert." in state_dict_keys_0:
            state_dict = {k.replace("pretrain_model.", "bert."): v for k, v in state_dict.items()}
        elif "bert.bert." not in state_dict:
            state_dict = {"bert." + k: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        # model = BertForMaskedLM.from_pretrained(args.load_model_path,
        #                                         return_dict=True,
        #                                         cache_dir=cache_dir)
        # model = PTuningWrapper(model, args.prompt_length)
        # model.to(device)
        # if args.load_state_dict:
        #     model.load_state_dict(torch.load(args.load_state_dict))
        # if n_gpu > 1:
        #     model = torch.nn.DataParallel(model)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        def decode(input_ids):
            # input_tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False)
            input_tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)
            # return [t.lower() for t in input_tokens]
            return input_tokens

        eval_loss = 0
        eval_steps = 0
        all_inputs, all_labels, all_predictions = [], [], []
        for batch in tqdm(eval_dataloader, desc="Evaluation-"+os.path.split(path_tet)[-1]):
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            src_ids, attention_mask, trg_ids, block_flag = batch
            # ### unk转mask
            # src_ids = src_ids.masked_fill(src_ids == tokenizer.unk_token_id, tokenizer.mask_token_id)
            with torch.no_grad():
                outputs = model(input_ids=src_ids,
                                attention_mask=attention_mask,
                                labels=trg_ids,
                                )
                # tmp_eval_loss = outputs.loss
                # logits = outputs.logits
                tmp_eval_loss = outputs[0]
                logits = outputs[-2]

            src_ids = src_ids.detach().cpu().numpy().tolist()
            trg_ids = trg_ids.detach().cpu().numpy().tolist()
            eval_loss += tmp_eval_loss.mean().item()
            prd_prob, prd_ids = torch.max(logits, -1)
            prd_prob = prd_prob.detach().cpu().numpy().tolist()
            prd_ids = prd_ids.masked_fill(attention_mask == 0, 0).tolist()
            for s, t, p, pb in zip(src_ids, trg_ids, prd_ids, prd_prob):
                mapped_src = []
                mapped_trg = []
                mapped_prd = []
                flag = False  ## if we pass to the target part
                ##src: [CLS]+[CLS]...+src+[SEP]...+trg+[SEP]
                ##trg: [CLS]+[CLS]...+src+[SEP]...+trg+[SEP]
                for st, tt, pt, pbt in zip(s, t, p, pb):
                    if st == tokenizer.sep_token_id:
                        flag = True
                    if not flag:
                        mapped_src += [st]
                    else:
                        mapped_trg += [tt]
                        if st == tokenizer.mask_token_id and pbt >= args.threshold:  ##we only predict the masked tokens
                            # if st == tokenizer.mask_token_id:  ##we only predict the masked tokens
                            mapped_prd += [pt]
                        else:
                            mapped_prd += [st]
                if anchor is not None:
                    ##src: [CLS]+[CLS]...+src+[SEP]+anchor+[SEP]...+[mask]
                    ##trg: [CLS]+[CLS]...+src+[SEP]+anchor+[SEP]...+trg
                    ## remove the anchor tokens from the src
                    anchor_length = len(anchor)
                    del mapped_trg[:anchor_length]
                    del mapped_prd[:anchor_length]
                ## we skip special tokens including '[UNK]'

                # all_inputs += [decode(mapped_src)]
                # all_labels += [decode(mapped_trg)]
                # all_predictions += [decode(mapped_prd)]

                ### 长度约束, 必须等长
                mapped_src_d = decode(mapped_src)[1:]
                mapped_trg_d = decode(mapped_trg)[1:]
                mapped_prd_d = decode(mapped_prd)[1:]
                len_mapped_src_d = len(mapped_src_d)
                len_mapped_trg_d = len(mapped_trg_d)
                len_mapped_prd_d = len(mapped_prd_d)
                len_min = min(len_mapped_src_d, len_mapped_trg_d)
                mapped_src_d = mapped_src_d[:len_min]
                mapped_trg_d = mapped_trg_d[:len_min]
                mapped_prd_d = mapped_prd_d[:len_min]
                len_mapped_src_d = len(mapped_src_d)
                len_mapped_trg_d = len(mapped_trg_d)
                len_mapped_prd_d = len(mapped_prd_d)
                ### 如果预测截断了, 则会补全
                len_mid = len_mapped_src_d - len_mapped_prd_d
                if len_mid <= 0:
                    mapped_prd_d = mapped_prd_d[:len_mapped_src_d]
                else:
                    # mapped_prd_d += "".join(list(mapped_prd_d)[-len_mid:])
                    mapped_prd_d += mapped_src_d[-len_mid:]
                all_inputs += [mapped_src_d]
                all_labels += [mapped_trg_d]
                all_predictions += [mapped_prd_d]

                # print("".join(all_inputs[-1]))
                # print("".join(all_labels[-1]))
                # print("".join(all_predictions[-1]))
                # print("#" * 128)

            eval_steps += 1

        print(len(all_inputs[-1]), "".join(all_inputs[-1]))
        print(len(all_labels[-1]), "".join(all_labels[-1]))
        print(len(all_predictions[-1]), "".join(all_predictions[-1]))
        print("#" * 128)

        eval_loss = eval_loss / eval_steps

        srcs, preds, tgts = all_inputs, all_predictions, all_labels
        print("#" * 128)
        common_det_acc, common_det_precision, common_det_recall, common_det_f1 = sent_mertic_det(
            srcs, preds, tgts)
        common_cor_acc, common_cor_precision, common_cor_recall, common_cor_f1 = sent_mertic_cor(
            srcs, preds, tgts)

        common_det_mertics = f'common Sentence Level detection: acc:{common_det_acc:.4f}, precision:{common_det_precision:.4f}, recall:{common_det_recall:.4f}, f1:{common_det_f1:.4f}'
        common_cor_mertics = f'common Sentence Level correction: acc:{common_cor_acc:.4f}, precision:{common_cor_precision:.4f}, recall:{common_cor_recall:.4f}, f1:{common_cor_f1:.4f}'
        print("#" * 128)
        print(f'flag_eval: common')
        print(common_det_mertics)
        print(common_cor_mertics)
        print("#" * 128)

        strict_det_acc, strict_det_precision, strict_det_recall, strict_det_f1 = sent_mertic_det(
            srcs, preds, tgts, flag_eval="strict")
        strict_cor_acc, strict_cor_precision, strict_cor_recall, strict_cor_f1 = sent_mertic_cor(
            srcs, preds, tgts, flag_eval="strict")

        strict_det_mertics = f'strict Sentence Level detection: acc:{strict_det_acc:.4f}, precision:{strict_det_precision:.4f}, recall:{strict_det_recall:.4f}, f1:{strict_det_f1:.4f}'
        strict_cor_mertics = f'strict Sentence Level correction: acc:{strict_cor_acc:.4f}, precision:{strict_cor_precision:.4f}, recall:{strict_cor_recall:.4f}, f1:{strict_cor_f1:.4f}'
        print("#" * 128)
        print(f'flag_eval: strict')
        print(strict_det_mertics)
        print(strict_cor_mertics)
        print("#" * 128)

        sent_mertics = {
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
        }

        detection_precision, detection_recall, detection_f1, \
        correction_precision, correction_recall, correction_f1 = char_mertic_det_cor(srcs, preds, tgts)
        token_result = {"det_precision": detection_precision, "det_recall": detection_recall, "det_f1": detection_f1,
                        "cor_precision": correction_precision, "cor_recall": correction_recall, "cor_f1": correction_f1,
                        }

        token_det_mertics = f'common Token Level correction: precision:{detection_precision:.4f}, recall:{detection_recall:.4f}, f1:{detection_f1:.4f}'
        token_cor_mertics = f'common TOken Level correction: precision:{correction_precision:.4f}, recall:{correction_recall:.4f}, f1:{correction_f1:.4f}'
        print("#" * 128)
        print(f'flag_eval: token')
        print(token_det_mertics)
        print(token_cor_mertics)
        print("#" * 128)

        result_mertics = {"task": os.path.split(path_tet)[-1], "sent": sent_mertics, "token": token_result}

    return common_det_mertics, common_cor_mertics, strict_det_mertics, strict_cor_mertics, result_mertics


def tet_test_dataset(path_model_dir):
    """   多个测试集   """
    path_model_dir = path_model_dir or "../../../output/text_correction/espell_law_of_relm"

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
                     ]  # [-2:-1]

    string_mertics_list = [path_model_dir + "\n"]
    res_mertics = {}  # 最终结果
    # threshold = 0  # 阈值
    threshold = 0.75  # 阈值
    for path_tet_i in tqdm(path_tet_list, desc="path_tet_list"):
        common_det_mertics, common_cor_mertics, strict_det_mertics, strict_cor_mertics, result_mertics \
            = eval_std_list(path_tet=path_tet_i, path_model_dir=path_model_dir, threshold=threshold, args=args)
        print(path_tet_i)
        string_mertics_list.append("path_tet_i: " + os.path.split(path_tet_i)[-1] + "\n")
        string_mertics_list.append(common_det_mertics+ "\n")
        string_mertics_list.append(common_cor_mertics+ "\n")
        string_mertics_list.append(strict_det_mertics+ "\n")
        string_mertics_list.append(strict_cor_mertics+ "\n")
        string_mertics_list.append("#" * 128 + "\n")
        res_mertics[path_tet_i] = result_mertics
    txt_write(string_mertics_list, os.path.join(path_model_dir, "eval_std.pred_mertics.txt"))
    save_json(res_mertics, os.path.join(path_model_dir, "eval_std.pred_mertics.json"))



if __name__ == "__main__":
    yz = 0


    ### 一次性全量测试
    # path_model_dir = "../../../output/text_correction/espell_law_of_relm"
    path_model_dir = "../../../output/text_correction/relm_v1"
    tet_test_dataset(path_model_dir=path_model_dir)


    # ### 单个测试集的测试
    # path_model_dir = "../../../output/text_correction/espell_law_of_relm"
    # path_corpus_dir = os.path.join(path_root, "macro_correct", "corpus", "text_correction")
    # path_tet1 = os.path.join(path_corpus_dir, "public/gen_de3.json")
    # path_tet2 = os.path.join(path_corpus_dir, "public/lemon_v2.tet.json")
    # path_tet3 = os.path.join(path_corpus_dir, "public/acc_rmrb.tet.json")
    # path_tet4 = os.path.join(path_corpus_dir, "public/acc_xxqg.tet.json")
    #
    # path_tet5 = os.path.join(path_corpus_dir, "public/gen_passage.tet.json")
    # path_tet6 = os.path.join(path_corpus_dir, "public/textproof.tet.json")
    # path_tet7 = os.path.join(path_corpus_dir, "public/gen_xxqg.tet.json")
    #
    # path_tet8 = os.path.join(path_corpus_dir, "public/faspell.dev.json")
    # path_tet9 = os.path.join(path_corpus_dir, "public/lomo_tet.json")
    # path_tet10 = os.path.join(path_corpus_dir, "public/mcsc_tet_5k.json")
    # path_tet11 = os.path.join(path_corpus_dir, "public/ecspell.dev.json")
    # path_tet12 = os.path.join(path_corpus_dir, "public/sighan2013.dev.json")
    # path_tet13 = os.path.join(path_corpus_dir, "public/sighan2014.dev.json")
    # path_tet14 = os.path.join(path_corpus_dir, "public/sighan2015.dev.json")
    # path_tet15 = os.path.join(path_corpus_dir, "public/mcsc_tet.json")
    # threshold = 0.0
    # eval_std(path_tet=path_tet1, path_model_dir=path_model_dir, threshold=threshold, args=args)
    # print(path_tet1)




"""
CUDA_VISIBLE_DEVICES=0

nohup python run_relm_wang271k_xs_lomo.py \
 --do_train \
 --do_eval \
 --do_test \
 --mft \
 --prompt_length 1 \
 --mask_mode "all" \
 --mask_rate 0.15 \
 --task_name "ecspell" \
 --train_on "law"  \
 --eval_on 'law' \
 --save_steps 3000  \
 --learning_rate 3e-5 \
 --num_train_epochs 10 \
 --train_batch_size 32 \
 --gradient_accumulation_steps 4 \
 --eval_batch_size 32 \
 --pretrained_model_name_or_path "hfl_chinese-macbert-base" \
 --model_save_path "model/model_wang271k_xs_lomo"  > tc.run_relm_wang271k_xs_lomo.py.log 2>&1 &
 
 
 
nohup python run_relm_wang271k_xs_lomo.py \
--do_train \
--do_eval \
--do_test \
--mft \
--prompt_length 1 \
--mask_mode "all" \
--mask_rate 0.3 \
--task_name "ecspell" \
--train_on "law"  \
--eval_on 'law' \
--save_steps 3000  \
--learning_rate 3e-5 \
--num_train_epochs 10 \
--train_batch_size 32 \
--gradient_accumulation_steps 4 \
--eval_batch_size 32 \
--pretrained_model_name_or_path "hfl_chinese-macbert-base" \
--model_save_path "model/model_wang271k_xs_lomo_v2"  > tc.run_relm_wang271k_xs_lomo.py.v2.log 2>&1 &
"""


"""
std-offical
89.9 94.5 91.2↑10.6


../output_weights/wang271k_org_std_of_relm
path_tet_i: test.json
common Sentence Level detection: acc:0.8000, precision:0.8629, recall:0.7072, f1:0.7773
common Sentence Level correction: acc:0.7709, precision:0.8523, recall:0.6483, f1:0.7364
strict Sentence Level detection: acc:0.8000, precision:0.7245, recall:0.7072, f1:0.7158
strict Sentence Level correction: acc:0.7709, precision:0.6642, recall:0.6483, f1:0.6561
################################################################################################################################
path_tet_i: lomo_tet.json
common Sentence Level detection: acc:0.4758, precision:0.4199, recall:0.3617, f1:0.3886
common Sentence Level correction: acc:0.4554, precision:0.3884, recall:0.3174, f1:0.3493
strict Sentence Level detection: acc:0.4758, precision:0.3076, recall:0.3617, f1:0.3325
strict Sentence Level correction: acc:0.4554, precision:0.2699, recall:0.3174, f1:0.2918
################################################################################################################################
path_tet_i: sighan2015.dev.json
common Sentence Level detection: acc:0.7918, precision:0.8501, recall:0.7011, f1:0.7685
common Sentence Level correction: acc:0.7627, precision:0.8386, recall:0.6421, f1:0.7273
strict Sentence Level detection: acc:0.7918, precision:0.7050, recall:0.7011, f1:0.7031
strict Sentence Level correction: acc:0.7627, precision:0.6456, recall:0.6421, f1:0.6438
################################################################################################################################
path_tet_i: lemon_v2_1k.tet.json
common Sentence Level detection: acc:0.2780, precision:0.9886, recall:0.2663, f1:0.4196
common Sentence Level correction: acc:0.2180, precision:0.9853, recall:0.2051, f1:0.3395
strict Sentence Level detection: acc:0.2780, precision:0.4516, recall:0.2663, f1:0.3350
strict Sentence Level correction: acc:0.2180, precision:0.3478, recall:0.2051, f1:0.2580
################################################################################################################################
path_tet_i: lemon_v2.tet.json
common Sentence Level detection: acc:0.4680, precision:0.4069, recall:0.2648, f1:0.3208
common Sentence Level correction: acc:0.4399, precision:0.3475, recall:0.2056, f1:0.2584
strict Sentence Level detection: acc:0.4680, precision:0.2714, recall:0.2648, f1:0.2680
strict Sentence Level correction: acc:0.4399, precision:0.2107, recall:0.2056, f1:0.2081
################################################################################################################################
path_tet_i: ecspell.dev.json
common Sentence Level detection: acc:0.6480, precision:0.7254, recall:0.4657, f1:0.5672
common Sentence Level correction: acc:0.5853, precision:0.6580, recall:0.3392, f1:0.4476
strict Sentence Level detection: acc:0.6480, precision:0.4866, recall:0.4657, f1:0.4759
strict Sentence Level correction: acc:0.5853, precision:0.3544, recall:0.3392, f1:0.3466
################################################################################################################################
path_tet_i: pt_mask_4_passage_130w_10epoch.tet.json
common Sentence Level detection: acc:0.3363, precision:0.7418, recall:0.2415, f1:0.3643
common Sentence Level correction: acc:0.2503, precision:0.6115, recall:0.1323, f1:0.2175
strict Sentence Level detection: acc:0.3363, precision:0.2637, recall:0.2415, f1:0.2521
strict Sentence Level correction: acc:0.2503, precision:0.1444, recall:0.1323, f1:0.1381
################################################################################################################################
path_tet_i: csc_130w_to_de3.tet.json
common Sentence Level detection: acc:0.3093, precision:0.9994, recall:0.3092, f1:0.4723
common Sentence Level correction: acc:0.2844, precision:0.9994, recall:0.2843, f1:0.4427
strict Sentence Level detection: acc:0.3093, precision:0.4176, recall:0.3092, f1:0.3553
strict Sentence Level correction: acc:0.2844, precision:0.3840, recall:0.2843, f1:0.3267
################################################################################################################################
path_tet_i: csc_TextProofreadingCompetition.train.json
common Sentence Level detection: acc:0.4914, precision:0.7253, recall:0.2933, f1:0.4177
common Sentence Level correction: acc:0.4713, precision:0.7015, recall:0.2611, f1:0.3806
strict Sentence Level detection: acc:0.4914, precision:0.4112, recall:0.2933, f1:0.3424
strict Sentence Level correction: acc:0.4713, precision:0.3660, recall:0.2611, f1:0.3048
################################################################################################################################
path_tet_i: csc_xf_2w_nopun.tet.json
common Sentence Level detection: acc:0.2922, precision:0.9454, recall:0.2971, f1:0.4522
common Sentence Level correction: acc:0.2078, precision:0.9249, recall:0.2113, f1:0.3440
strict Sentence Level detection: acc:0.2922, precision:0.4196, recall:0.2971, f1:0.3479
strict Sentence Level correction: acc:0.2078, precision:0.2984, recall:0.2113, f1:0.2474
################################################################################################################################
"""


"""
../output_weights/wang271k_clean_std_of_relm
path_tet_i: test.json
common Sentence Level detection: acc:0.8091, precision:0.8675, recall:0.7238, f1:0.7892
common Sentence Level correction: acc:0.7764, precision:0.8561, recall:0.6575, f1:0.7437
strict Sentence Level detection: acc:0.8091, precision:0.7387, recall:0.7238, f1:0.7312
strict Sentence Level correction: acc:0.7764, precision:0.6711, recall:0.6575, f1:0.6642
################################################################################################################################
path_tet_i: lomo_tet.json
common Sentence Level detection: acc:0.5088, precision:0.4592, recall:0.3739, f1:0.4122
common Sentence Level correction: acc:0.4882, precision:0.4278, recall:0.3291, f1:0.3720
strict Sentence Level detection: acc:0.5088, precision:0.3339, recall:0.3739, f1:0.3527
strict Sentence Level correction: acc:0.4882, precision:0.2939, recall:0.3291, f1:0.3105
################################################################################################################################
path_tet_i: sighan2015.dev.json
common Sentence Level detection: acc:0.8018, precision:0.8553, recall:0.7196, f1:0.7816
common Sentence Level correction: acc:0.7700, precision:0.8432, recall:0.6550, f1:0.7373
strict Sentence Level detection: acc:0.8018, precision:0.7209, recall:0.7196, f1:0.7202
strict Sentence Level correction: acc:0.7700, precision:0.6562, recall:0.6550, f1:0.6556
################################################################################################################################
path_tet_i: lemon_v2_1k.tet.json
common Sentence Level detection: acc:0.2870, precision:0.9890, recall:0.2755, f1:0.4310
common Sentence Level correction: acc:0.2270, precision:0.9859, recall:0.2143, f1:0.3521
strict Sentence Level detection: acc:0.2870, precision:0.4663, recall:0.2755, f1:0.3464
strict Sentence Level correction: acc:0.2270, precision:0.3627, recall:0.2143, f1:0.2694
################################################################################################################################
path_tet_i: lemon_v2.tet.json
common Sentence Level detection: acc:0.4722, precision:0.4147, recall:0.2735, f1:0.3296
common Sentence Level correction: acc:0.4441, precision:0.3570, recall:0.2144, f1:0.2679
strict Sentence Level detection: acc:0.4722, precision:0.2800, recall:0.2735, f1:0.2767
strict Sentence Level correction: acc:0.4441, precision:0.2195, recall:0.2144, f1:0.2169
################################################################################################################################
path_tet_i: ecspell.dev.json
common Sentence Level detection: acc:0.6560, precision:0.7360, recall:0.4764, f1:0.5784
common Sentence Level correction: acc:0.5920, precision:0.6701, recall:0.3472, f1:0.4574
strict Sentence Level detection: acc:0.6560, precision:0.4979, recall:0.4764, f1:0.4869
strict Sentence Level correction: acc:0.5920, precision:0.3629, recall:0.3472, f1:0.3549
################################################################################################################################
path_tet_i: pt_mask_4_passage_130w_10epoch.tet.json
common Sentence Level detection: acc:0.3424, precision:0.7520, recall:0.2464, f1:0.3712
common Sentence Level correction: acc:0.2546, precision:0.6242, recall:0.1349, f1:0.2219
strict Sentence Level detection: acc:0.3424, precision:0.2704, recall:0.2464, f1:0.2579
strict Sentence Level correction: acc:0.2546, precision:0.1481, recall:0.1349, f1:0.1412
################################################################################################################################
path_tet_i: csc_130w_to_de3.tet.json
common Sentence Level detection: acc:0.3170, precision:0.9989, recall:0.3172, f1:0.4814
common Sentence Level correction: acc:0.2931, precision:0.9988, recall:0.2932, f1:0.4533
strict Sentence Level detection: acc:0.3170, precision:0.4264, recall:0.3172, f1:0.3637
strict Sentence Level correction: acc:0.2931, precision:0.3941, recall:0.2932, f1:0.3362
################################################################################################################################
path_tet_i: csc_TextProofreadingCompetition.train.json
common Sentence Level detection: acc:0.4927, precision:0.7255, recall:0.2967, f1:0.4211
common Sentence Level correction: acc:0.4734, precision:0.7029, recall:0.2656, f1:0.3855
strict Sentence Level detection: acc:0.4927, precision:0.4159, recall:0.2967, f1:0.3463
strict Sentence Level correction: acc:0.4734, precision:0.3723, recall:0.2656, f1:0.3100
################################################################################################################################
path_tet_i: csc_xf_2w_nopun.tet.json
common Sentence Level detection: acc:0.2915, precision:0.9453, recall:0.2964, f1:0.4513
common Sentence Level correction: acc:0.2073, precision:0.9248, recall:0.2108, f1:0.3434
strict Sentence Level detection: acc:0.2915, precision:0.4192, recall:0.2964, f1:0.3472
strict Sentence Level correction: acc:0.2073, precision:0.2982, recall:0.2108, f1:0.2470
################################################################################################################################

"""


"""
我今天吃苹果，明天吃香姣
这位外国网友说：这是地球最安全的锅加，在这里，我从未感到过害啪！
We always rent a little cottages from a sheep farmer and now we know his family.
智能识别并纠正句子中的语法、拼写、用词等错误，确保文本的准确性和可读性。
实现祖国完全统一，是全体中华儿女共同愿望，解决台湾问题，是中华民族根本利益所在。推动两岸关系和平发展，必须继续坚持“和平统一、一郭两制”方针，退进祖国和平统一。
百度是一家人工只能公司
我门是中国点信的源工
通过机七学习可以实现中文语句的智能纠错。我的朋友法语说的很好，的语也不错。
天行数据是一个友秀的只能平台 
传承和弘扬中华优秀传统文化既是增强文华自信、建设社会主义文化强国的应然之义，也是全面建设社会注意现代化国家、推进实现中华民族伟大复兴的实践前提。


刚开始我也收不撩这样。
没想到跳着跳着就照过半夜了。

团长将士兵布署在城外，让他们安兵不动。
任何因难(nán)都不能下倒有坚强意志的对员们。

ECSPELL
患者需要按照剂量服用甲苯米坐片。
人民监查员依法审察案件。

ECOPO
希望您帮我素取公平。
我们为这个目标努力不解。

ECGM
我门看美国电影，我觉得很有意事。
他派了很多照片。
他门才开始考试。
大明说: 请座，请座。
可是你现在不在宿舍，所以我留了一枝条。
我以前想要高诉你，可是我忘了。我真户秃。

RERIC
老师进教师来了。
我带上运动鞋出门。

C-LLM
有胆量的图片曝光。
这也更新，让软件更加方便。
可查询详析数据信息。
关注微信火下载都有机会。

LEAD
铁轨上有一列或车在行驶。
炉子上正绕着水。
那时天起非常好。
他们正举办一个误会。
要永于面对逆境。
秋天己经无声的来到了。
迎接每一个固难，并克服它。

MDCSpell
哪里有上大学，不想念书的道理？
从那里，我们可以走到纳福境的新光三钺百货公司逛一逛。
他主动拉了姑娘的手，心里很高心，嘴上故作生气。
我这一次写信给你是想跟你安排一下关以我们要见面的。
为了减少急遍的生孩子率，需要呼吁适当的生育政策。

PLOME
今天的夕阳真是太没了。
必须持有门票才能进人场馆。
于下的都是垃圾。

SCOPE
我以前想要高诉你，可我忘了。我真户秃。
可是福物生对我们很客气。
他再也不会撤扬。
幸运地，隔天她带着辞典来学校。
我觉得你们会好好的完。
我以前想要高诉你。
他收到山上的时候。
行为都被蓝控设备录影。


SDCL
农民伯伯们正在X稼地里干农活。
这是一个很好的返例。
这是一个很好的友例。
 
SoftMaskBERT
埃及有金子塔。
他的求胜欲很强，为了越狱在挖洞。
我会说一点儿，不过一个汉子也看不懂，所以我迷路了。
他主动拉了姑娘的手, 心里很高心, 嘴上故作生气。
芜湖:女子落入青戈江,众人齐救援。

pycorrector
重定位后宁蒗M5.5主震震中位置为，震源深度为13.71km。
沂沭断裂带是郯庐断裂带在山东境内的区段，由昌邑—大店断裂、安丘—莒县断裂等构成，组成了“两堑夹一垒”的构造格局。
 
own-pinyin
如果识别不了图形，那么平面几何和例题集合得题目基本没法做。
 
PTCSpell
今天教师里面很热。 室
操场上有一于个人。 千
我的录影机在哪？ 影
晚上的花园很安静。 宁
乍么才能让孩子对绘画蝉声兴趣呢？ 怎么才能让孩子对绘画产生兴趣呢？
什么才能让孩子对绘画蚊身兴趣呢？ 怎么才能让孩子对绘画产生兴趣呢？
我们学校购买了十台录影机。 我们学校购买了十台录影机。
我们学校购买了十台录音机。 我们学校购买了十台录影机。
我反对这个延议,学校里不要装绿音机。  我反对这个建议,学校里不要装录影机。


PHMOSpell
不惜娱弄大臣。  不惜愚弄大臣。
那别人的欢说是没办法改变你的。  那别人的劝说是没办法改变你的。
他们有时候，有一点捞到。  他们有时候，有一点唠叨。
天将降大任于斯人也，必先苦其心智，劳其筋骨。  天将降大任于斯人也，必先苦其心志，劳其筋骨。
人们必生去追求的目标。  人们毕生去追求的目标。
迎接每一个固难。 迎接每一个困难。
 
SpellBERT
我喜欢吃蛋高。我喜欢吃蛋糕。
我喜欢吃蛋达。我喜欢吃蛋挞。

ThinkTwice
我很喜欢吃水果。  误纠:我很喜欢吃苹果。
 
Eval-GCSC
迈近新目标。迈进新目标。
我轻触这些压力对于我目前来说是不避的。 我清楚这些压力对于我目前来说是不必的。
他题很多问题。 他提很多问题。  他有很多问题。
        error以后我回来我的国家。     以后我回我的国家。
脸百百的。 脸白白的。
水虑可机。   水过滤机。

DISC
肌肉酸痛是运动过读导致的。 肌肉酸痛是运动过度导致的。
浓荫蔽空，郁郁苍苍。  浓荫蔽日，郁郁苍苍。
记得戴眼睛。  记得戴眼镜。
从商场的人口进去。 从商场的入口进去。

Visual3C
新的政策设计千家万付。  新的政策涉及千家万户。
北京企业将迁入宣新区。  北京企业将迁入雄安新区。
徐夕阳行业概念股名单。  徐翔行业概念股名单。





https://www.gugudata.com/api/details/nlpcorrect

ali: 128
tencent: 150
baidu: 3000
ctyun： 1000 天翼云


0: 错误在句中的位置[l, r)，左闭右开
1: 推荐意⻅(list)
    0: string 推荐词
    1: int 推荐程度
        1: 表⽰“低概率错误，⼀般推荐”
        2: 表⽰“⾼概率错误，强烈推荐”
        3: 系统默认敏感词
        4: ⽤⼾⾃定义敏感词
        5: ⽤⼾⾃定义错词
        6: 共享词典敏感词
        7: 共享词典错
        8: 标点符号错误
    2: 推荐类别, 格式”x-x”
        “0-x”: 默认分类 (没有对应分类)
        “1-“: 表⽰同⾳错误，建议替换
        “2-“: 常⻅谐⾳错误，建议替换
        “3-“: 遗漏字词错误，建议补充
        “4-“: 冗余字词错误，建议删减
        “5-“: 其他谐⾳、近形错误，建议替换
        “7-“: 语序错误，建议调整语序
        “8-x”: 敏感词错误，建议删减
            8-1: 未分类（默认分类）
            8-2: ⻩赌毒
            8-3: 司法、政治
            8-4: 宗教、迷信
            8-5: ⾔语 辱骂
            8-6: ⾮法信息
            8-7: 宣传、⼴告
        “9-1”: 地址归属地错误
        “10-x”:
            10-1: 中英类型错⽤
            10-2: 成对标点缺失或⽤反
            10-3: 多余标点
    3: 0/1 命名实体标志。0: ⽆命名实体；1: 有命名实体。
2: 空
论文解读
"""


# shell
# nohup python train.py > tc.train.py.log 2>&1 &
# tail -n 1000  -f tc.train.py.log
# |myz|


"""
生成的评估指标在模型文件目录下
"eval_std.pred_mertics.json"
"eval_std.pred_mertics.txt"

## 个人感受
```
1.relm模型泛化性好, 适合小样本数据集, 使用少量领域数据就能取得不错的效果;
2.relm模型性能不行(最大文本长度翻倍), 指标也不太好, 如wank271k效果就不太行;
```

"""