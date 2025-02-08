# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/1/18 21:34
# @author  : Mo
# @function: MDCSpell
# @paper   : [MDCSpell: A Multi-task Detector-Corrector Framework for Chinese Spelling Correction](https://aclanthology.org/2022.findings-acl.98/).


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

from macro_correct.pytorch_user_models.csc.macbert4mdcspell.dataset import DataSetProcessor, sent_mertic_det, sent_mertic_cor, char_mertic_det_cor
from macro_correct.pytorch_user_models.csc.macbert4mdcspell.dataset import save_json, load_json, txt_write
from macro_correct.pytorch_user_models.csc.macbert4mdcspell.config import csc_config as args
from macro_correct.pytorch_user_models.csc.macbert4mdcspell.graph import Macbert4MDCSpell as Graph


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
        src, trg, block_flag, trg_ref = convert_examples_to_prompts(example.src, example.trg, prompt_length, max_seq_length, tokenizer, anchor, mask_rate)
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
    src = truncate(src, max_seq_length-prompt_length-2)
    trg = truncate(trg, max_seq_length-prompt_length-2)
    assert(len(src) == len(trg))
    if anchor is not None:
        prompt_src = [tokenizer.cls_token] * prompt_length + src + anchor + [tokenizer.sep_token] * prompt_length
        prompt_trg = [tokenizer.cls_token] * prompt_length + trg + anchor + [tokenizer.sep_token] * prompt_length
        block_flag = [1] * prompt_length + [0 for _ in src] + [0 for _ in anchor] + [1] * prompt_length
        trg_ref = [tokenizer.cls_token] * prompt_length + trg + anchor + [tokenizer.sep_token] * prompt_length
    else:
        prompt_src = [tokenizer.cls_token] * prompt_length + src + [tokenizer.sep_token] * prompt_length
        prompt_trg = [tokenizer.cls_token] * prompt_length + trg + [tokenizer.sep_token] * prompt_length
        block_flag = [1] * prompt_length + [0 for _ in src] + [1] * prompt_length
        trg_ref = [tokenizer.cls_token] * prompt_length + trg + [tokenizer.sep_token] * prompt_length
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
        print("no path_model_dir, fail!")
    processor.path_train = args.path_train
    processor.task_name = args.task_name
    processor.path_dev = args.path_dev
    processor.path_tet = path_tet or args.path_tet
    # processor.path_tet = args.path_tet  # org
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

    # args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps##

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

        model.eval()
        eval_loss = 0
        eval_steps = 0
        all_inputs, all_labels, all_predictions = [], [], []
        for batch in tqdm(eval_dataloader, desc="Evaluation"):
            batch = tuple(t.to(device) for t in batch)
            src_ids, attention_mask, trg_ids, block_flag = batch
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
                for st, tt, pt, pbt in zip(s, t, p, pb):
                    if st in [tokenizer.cls_token_id, tokenizer.sep_token_id]:
                        continue
                    mapped_src += [st]
                    mapped_trg += [tt]
                    # 阈值
                    if st != pt and pbt > args.threshold:
                        mapped_prd += [pt]
                    else:
                        mapped_prd += [st]
                if anchor is not None:
                    ## remove the anchor tokens from the src
                    anchor_length = len(anchor)
                    del mapped_trg[:anchor_length]
                    del mapped_prd[:anchor_length]
                ## we skip special tokens including '[UNK]'
                all_inputs += [decode(mapped_src)]
                all_labels += [decode(mapped_trg)]
                all_predictions += [decode(mapped_prd)]

            eval_steps += 1

        print(len(all_inputs[-1]), "".join(all_inputs[-1]))
        print(len(all_labels[-1]), "".join(all_labels[-1]))
        print(len(all_predictions[-1]), "".join(all_predictions[-1]))
        print("#" * 128)

        eval_loss = eval_loss / eval_steps

        print("#" * 128)
        det_acc, det_precision, det_recall, det_f1 = sent_mertic_det(all_inputs, all_predictions,
                                                                     all_labels, logger)
        cor_acc, cor_precision, cor_recall, cor_f1 = sent_mertic_cor(all_inputs, all_predictions,
                                                                     all_labels, logger)
        logger.info(
            f'flag_eval: common')
        print(
            f'flag_eval: common')
        logger.info(
            f'Sentence Level detection: acc:{det_acc:.4f}, precision:{det_precision:.4f}, recall:{det_recall:.4f}, f1:{det_f1:.4f}')
        print(
            f'Sentence Level detection: acc:{det_acc:.4f}, precision:{det_precision:.4f}, recall:{det_recall:.4f}, f1:{det_f1:.4f}')
        logger.info(
            f'Sentence Level correction: acc:{cor_acc:.4f}, precision:{cor_precision:.4f}, recall:{cor_recall:.4f}, f1:{cor_f1:.4f}')

        print(
            f'Sentence Level correction: acc:{cor_acc:.4f}, precision:{cor_precision:.4f}, recall:{cor_recall:.4f}, f1:{cor_f1:.4f}')
        det_acc, det_precision, det_recall, det_f1 = sent_mertic_det(all_inputs, all_predictions,
                                                                     all_labels, logger, flag_eval="strict")
        cor_acc, cor_precision, cor_recall, cor_f1 = sent_mertic_cor(all_inputs, all_predictions,
                                                                     all_labels, logger, flag_eval="strict")
        print("#" * 128)
        logger.info(
            f'flag_eval: strict')
        print(
            f'flag_eval: strict')
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
            # logger.info(text_log)
            text_log_list.append(text_log)
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
        # model.to(device)
        # model.load_state_dict(torch.load(path_model_best))
        state_dict = torch.load(path_model_best)
        if "pretrain_model.bert." in state_dict:
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
                ##src: [CLS]+[CLS]...+src+[SEP]
                for st, tt, pt, pbt in zip(s, t, p, pb):
                    if st in [tokenizer.cls_token_id, tokenizer.sep_token_id]:
                        continue
                    mapped_src += [st]
                    mapped_trg += [tt]
                    # 阈值
                    if st != pt and pbt > args.threshold:
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


def tet_test_dataset(path_model_dir=""):
    """   多个测试集   """
    ### 全量测试
    path_model_dir = path_model_dir or "../../../output/text_correction/espell_law_of_macbert4mdcspell"

    ### 数据集(只用了.train训练(大概1000w数据集), dev/test都没有参与训练)
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
                     ]

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
    yz = 0

    ### 一次性全量测试
    path_model_dir = "../../../output/text_correction/espell_law_of_macbert4mdcspell"
    tet_test_dataset(path_model_dir=path_model_dir)

    # ### 单个测试集的测试
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
    #
    # path_model_dir = "../../../output/text_correction/espell_law_of_macbert4mdcspell"
    # threshold = 0
    # eval_std(path_tet=path_tet1, path_model_dir=path_model_dir, threshold=threshold, args=args)
    # print(path_tet1)


"""
生成的评估指标在模型文件目录下
"eval_std.pred_mertics.json"
"eval_std.pred_mertics.txt"

mdcspell
mdcspell在训练语料的领域效果较好;
但是因为新加了网络层, 所以模型需要比较多的数据, 或者先预训练, 否则效果不太好, 迁移性/few-shot能力比较差
"""


# shell
# nohup python eval_std.py > tc.eval_std.py.log 2>&1 &
# tail -n 1000  -f tc.eval_std.py.log
# |myz|



