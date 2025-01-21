# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/1/18 21:34
# @author  : Mo
# @function: relm
# @paper   : [Chinese Spelling Correction as Rephrasing Language Model](https://arxiv.org/abs/2308.08796).
# @code    : most code copy from https://github.com/Claude-Liu/ReLM, small modfix
# @modfiy  : remove the Network Architecture of P-tuning(BiLSTM)


from __future__ import absolute_import, division, print_function
import traceback
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USE_TORCH"] = "1"
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import SchedulerType, get_scheduler
from transformers import BertForMaskedLM
from transformers import AutoTokenizer
from tensorboardX import SummaryWriter
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch

from macro_correct.pytorch_user_models.csc.relm.dataset import DataSetProcessor, sent_mertic_det, sent_mertic_cor
from macro_correct.pytorch_user_models.csc.relm.dataset import save_json, load_json, txt_write
from macro_correct.pytorch_user_models.csc.relm.config import csc_config as args
from macro_correct.pytorch_user_models.csc.relm.graph import RELM as Graph


# adapt the input for ReLM
def convert_examples_to_features(examples, max_seq_length, tokenizer, prompt_length, anchor=None, mask_rate=0.2, logger=None):
    features = []
    for i, example in tqdm(enumerate(examples), desc="data"):
        # src, trg, block_flag, trg_ref, src_ref = convert_examples_to_prompts(example.src, example.trg, prompt_length, max_seq_length // 2, tokenizer, anchor, mask_rate)
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
    src = truncate(src, max_seq_length-prompt_length-2)  # cls-source-sep-target-sep
    trg = truncate(trg, max_seq_length-prompt_length-2)
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


class InputFeatures(object):
    def __init__(self, src_ids, attention_mask, trg_ids, trg_ref_ids, block_flag, src_ref_ids=None):
        self.src_ids = src_ids
        self.attention_mask = attention_mask
        self.trg_ids = trg_ids
        self.trg_ref_ids = trg_ref_ids
        self.block_flag = block_flag
        self.src_ref_ids = src_ref_ids


def train_csc():
    processor = DataSetProcessor()
    processor.path_train = args.path_train
    processor.task_name = args.task_name
    processor.path_dev = args.path_dev
    processor.path_tet = args.path_tet
    task_name = args.task_name

    args.model_save_path = os.path.join(args.model_save_path, task_name)
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

    tensorboardx_witer = SummaryWriter(logdir=args.model_save_path)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO,
                        filename=os.path.join(args.model_save_path, "train.log"))
    logger = logging.getLogger(__name__)


    device = torch.device("cuda" if torch.cuda.is_available() and args.flag_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, "Unsupported", args.flag_fp16))

    # args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps  ##

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path,
                                              do_lower_case=args.do_lower_case,
                                              # cache_dir=cache_dir,
                                              use_fast=args.flag_fast_tokenizer)
    
    anchor = None
    if args.anchor is not None:
        anchor = [tokenizer.sep_token]+[t for t in args.anchor]


    def data_collate_fn(batch, tokenizer=tokenizer, args=args):
        """   数据处理等   """
        # self.logger.info("max_len:  ", max_len)
        def truncate(x, max_length):
            """   最大截断   """
            return x[: max_length]

        max_seq_length = args.max_seq_length
        prompt_length = args.prompt_length
        batch_attention_mask = []
        batch_trg_ref_ids = []
        batch_block_flag = []
        batch_src_ids = []
        batch_trg_ids = []
        max_len = copy.deepcopy(max_seq_length)
        if args.flag_dynamic_encode:  # 使用动态文本长度编码
            max_len_bs = max([len(b.src) for b in batch])
            max_len = min(max_seq_length, max_len_bs + prompt_length + 2)
        count = 0
        for ba in batch:
            count += 1
            src = ba.src
            trg = ba.trg
            ## here max_seq = tokenizer.max_seq_length//2, we need to truncate
            src = truncate(src, max_len - args.prompt_length - 2)
            trg = truncate(trg, max_len - args.prompt_length - 2)
            assert (len(src) == len(trg))
            ### 中间必须要有一个sep
            if anchor is not None:
                prompt_src = [tokenizer.cls_token] * prompt_length + src + anchor + [tokenizer.sep_token] \
                             + [tokenizer.mask_token for _ in trg] + [tokenizer.sep_token] * prompt_length
                prompt_trg = [tokenizer.cls_token] * prompt_length + src + anchor + [tokenizer.sep_token] \
                             + trg + [tokenizer.sep_token] * prompt_length
                block_flag = [1] * prompt_length + [0 for _ in src] + [0 for _ in anchor] + \
                             [1] + [0 for _ in trg] + [0] * prompt_length
                trg_ref = [tokenizer.cls_token] * prompt_length + trg + anchor + [tokenizer.sep_token] \
                          + trg + [tokenizer.sep_token] * prompt_length
                # src_ref = [tokenizer.cls_token] * prompt_length + src + anchor + [tokenizer.sep_token] * (prompt_length+1) + src + [tokenizer.sep_token] * prompt_length
            else:
                prompt_src = [tokenizer.cls_token] * prompt_length + src + [tokenizer.sep_token] \
                             + [tokenizer.mask_token for _ in trg] + [tokenizer.sep_token] * prompt_length
                prompt_trg = [tokenizer.cls_token] * prompt_length + src + [tokenizer.sep_token] \
                             + trg + [tokenizer.sep_token] * prompt_length
                block_flag = [1] * prompt_length + [0 for _ in src] + [1] + [0 for _ in trg] \
                             + [0] * prompt_length
                trg_ref = [tokenizer.cls_token] * prompt_length + trg + [tokenizer.sep_token] \
                          + trg + [tokenizer.sep_token] * prompt_length

            encoded_inputs = tokenizer(prompt_src,
                                       max_length=max_len,
                                       padding="max_length",
                                       truncation=True,
                                       return_token_type_ids=True,
                                       is_split_into_words=True)

            trg_ids = tokenizer(prompt_trg,
                                max_length=max_len,
                                padding="max_length",
                                truncation=True,
                                return_token_type_ids=True,
                                is_split_into_words=True)["input_ids"]

            trg_ref_ids = tokenizer(trg_ref,
                                    max_length=max_len,
                                    padding="max_length",
                                    truncation=True,
                                    return_token_type_ids=True,
                                    is_split_into_words=True)["input_ids"]

            src_ids = encoded_inputs["input_ids"]
            attention_mask = encoded_inputs["attention_mask"]
            block_flag = ([0] + block_flag)[: args.max_seq_length]
            ## zero padding
            if len(block_flag) < args.max_seq_length:
                block_flag = block_flag + [0] * max(0, args.max_seq_length - len(block_flag))

            batch_attention_mask.append(attention_mask)
            # batch_token_type.append(token_type_id)
            batch_trg_ref_ids.append(trg_ref_ids)
            batch_block_flag.append(block_flag)  # for eval and pred
            batch_src_ids.append(src_ids)
            batch_trg_ids.append(trg_ids)

        # tensor
        tensor_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        tensor_trg_ref_ids = torch.tensor(batch_trg_ref_ids, dtype=torch.long)
        tensor_block_flag = torch.tensor(batch_block_flag, dtype=torch.long)
        tensor_trg_ids = torch.tensor(batch_trg_ids, dtype=torch.long)
        tensor_src_ids = torch.tensor(batch_src_ids, dtype=torch.long)
        # tensor_det = torch.tensor(batch_det, dtype=torch.long)
        return tensor_src_ids, tensor_attention_mask, tensor_trg_ids, tensor_trg_ref_ids, tensor_block_flag


    if args.do_train:
        train_examples = processor.get_train_examples()
        train_dataloader = DataLoader(num_workers=args.num_workers,
                                      batch_size=args.train_batch_size,
                                      # shuffle=True,
                                      collate_fn=data_collate_fn,
                                      dataset=train_examples,
                                      sampler=RandomSampler(train_examples),
                                      pin_memory=args.flag_pin_memory,
                                      )

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

        if args.max_train_steps is None:
            args.max_train_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        ##we use BERTMLM as the backbone for ReLM
        model = Graph(config=args)
        if args.path_relm and os.path.exists(args.path_relm):
            try:
                state_dict = torch.load(args.path_relm)
                state_dict = {"bert." + k: v for k, v in state_dict.items()}
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(traceback.print_exc())

        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)  ##It is recommended to use DistributedDataParallel

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        ## apply weight decay
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]
        ## set the Adam optimizer
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        ## set the scheduler
        num_warmup_steps = args.num_warmup_steps if args.num_warmup_steps \
            else int(num_update_steps_per_epoch*args.warmup_proportion)  # 一个epoch的0.1
        scheduler = get_scheduler(optimizer=optimizer, name=args.lr_scheduler_type,
                                  num_warmup_steps=num_warmup_steps,
                                  num_training_steps=args.max_train_steps)

        scaler = None
        if args.flag_fp16:  ##use half precision to reduce the memory usage of neural networks
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
        
        if args.do_eval:
            eval_examples = processor.get_dev_examples()
            eval_dataloader = DataLoader(num_workers=args.num_workers,
                                         batch_size=args.eval_batch_size,
                                         # shuffle=True,
                                         collate_fn=data_collate_fn,
                                         dataset=eval_examples,
                                         sampler=SequentialSampler(eval_examples),
                                         pin_memory=args.flag_pin_memory,
                                         )
            # eval_features = convert_examples_to_features(eval_examples, args.max_seq_length,  tokenizer, args.prompt_length, anchor=anchor, logger=logger)##never mask the source during evaluation
            #
            # all_input_ids = torch.tensor([f.src_ids for f in eval_features], dtype=torch.long)
            # all_input_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
            # all_label_ids = torch.tensor([f.trg_ids for f in eval_features], dtype=torch.long)
            # all_block_flag = torch.tensor([f.block_flag for f in eval_features], dtype=torch.long)
            #
            # eval_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_block_flag)
            # eval_sampler = SequentialSampler(eval_data)
            # eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", args.max_train_steps)

        progress_bar = tqdm(range(args.max_train_steps))
        best_result = list()
        global_step = 0
        wrap = False
        for idx_num in range(int(args.num_train_epochs)):
            num_train_examples = 0
            train_det_loss = 0
            train_cor_loss = 0
            train_cpo_loss = 0
            train_loss = 0
            if wrap:
                break
            for step, batch in enumerate(train_dataloader):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                # src_ids, attention_mask, trg_ids, trg_ref_ids, block_flag, src_ref_ids = batch
                src_ids, attention_mask, trg_ids, trg_ref_ids, block_flag = batch
                # ### cpo-mask not use mft, only src, trg
                # trg_ref_det = copy.deepcopy(trg_ref_ids)
                # trg_ref_det[(src_ref_ids != trg_ref_det)] = 1  ## ignore index = -100
                # trg_ref_det[(src_ref_ids == trg_ref_det)] = 0  ## ignore index = -100
                # if args.flag_mft:
                #     src_ids = dynamic_mask_token(src_ids, trg_ref_ids, tokenizer,
                #                                  device, args.mask_mode, args.mask_rate)
                ### 50%随机mask, 并且最后一轮不mask
                thr_mft = random.random()
                # thr_std = 0.5  # 随机概率/50%
                # if args.flag_mft and thr_mft <= thr_std and idx_num < int(args.num_train_epochs-1):
                if args.flag_mft and thr_mft <= 0.5:
                    src_ids = dynamic_mask_token(src_ids, trg_ref_ids, tokenizer, device,
                                                 args.mask_mode, args.mask_rate)

                ### 训练4/5以后再采用 focal-loss + cpo_loss
                if args.flag_loss_period and global_step < args.num_train_epochs * 4 / 5:
                    model.loss_type = "BCE_MULTI"
                    model.flag_cpo_loss = False
                else:
                    # model.flag_cpo_loss = args.flag_cpo_loss
                    model.flag_cpo_loss = False
                    model.loss_type = args.loss_type
                ### unk转mask
                src_ids = src_ids.masked_fill(src_ids == tokenizer.unk_token_id, tokenizer.mask_token_id)
                ## only loss on the masked positions are included when calculating loss
                trg_ids[(src_ids == trg_ids)] = -100  ## ignore index = -100, 就是source部分不参与训练loss计算
                if args.flag_fp16:
                    with autocast():
                        # you can deactivate the prompt by
                        # setting prompt length as 1, and apply_prompt as False
                        outputs = model(input_ids=src_ids,
                                        attention_mask=attention_mask,
                                        labels=trg_ids,
                                        )
                else:
                    # you can deactivate the prompt by
                    # setting prompt length as 1, and apply_prompt as False
                    outputs = model(input_ids=src_ids,
                                    attention_mask=attention_mask,
                                    labels=trg_ids,
                                    )
                # loss = outputs.loss
                tmp_train_det_loss = outputs[1]
                tmp_train_cor_loss = outputs[2]
                tmp_train_cpo_loss = outputs[3]
                loss = outputs[0]

                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.flag_fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                train_det_loss += tmp_train_det_loss.mean().item()
                train_cor_loss += tmp_train_cor_loss.mean().item()
                train_cpo_loss += tmp_train_cpo_loss.mean().item()
                train_loss += loss.mean().item()

                num_train_examples += src_ids.size(0)
                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    if args.flag_fp16:
                        scaler.unscale_(optimizer)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    progress_bar.update(1)
                    global_step += 1

                if args.do_eval and global_step % args.save_steps == 0 and (step + 1) % args.gradient_accumulation_steps == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    def decode(x):
                        return tokenizer.convert_ids_to_tokens(x, skip_special_tokens=True)

                    model.eval()
                    eval_det_loss = 0
                    eval_cor_loss = 0
                    eval_cpo_loss = 0
                    eval_loss = 0
                    eval_steps = 0
                    all_inputs, all_labels, all_predictions = [], [], []
                    for batch in tqdm(eval_dataloader, desc="Evaluation"):
                        batch = tuple(t.to(device) for t in batch)
                        # src_ids, attention_mask, trg_ids, block_flag = batch
                        src_ids, attention_mask, trg_ids, trg_ref_ids, block_flag = batch
                        ### unk转mask
                        src_ids = src_ids.masked_fill(src_ids == tokenizer.unk_token_id, tokenizer.mask_token_id)

                        with torch.no_grad():
                            outputs = model(input_ids=src_ids,
                                            attention_mask=attention_mask,
                                            labels=trg_ids,
                                            # prompt_mask=block_flag,
                                            # apply_prompt=args.apply_prompt
                                            )
                            # tmp_eval_loss = outputs.loss
                            # logits = outputs.logits
                            tmp_eval_det_loss = outputs[1]
                            tmp_eval_cor_loss = outputs[2]
                            tmp_eval_cpo_loss = outputs[3]
                            tmp_eval_loss = outputs[0]
                            logits = outputs[-2]

                        src_ids = src_ids.cpu().numpy().tolist()
                        trg_ids = trg_ids.cpu().numpy().tolist()
                        # eval_loss += tmp_eval_loss.mean().item()
                        eval_loss += tmp_eval_loss.mean().item()
                        eval_det_loss += tmp_eval_det_loss.mean().item()
                        eval_cor_loss += tmp_eval_cor_loss.mean().item()
                        eval_cpo_loss += tmp_eval_cpo_loss.mean().item()

                        _, prd_ids = torch.max(logits, -1)  # (batch,seq)
                        prd_ids = prd_ids.masked_fill(attention_mask == 0, 0).tolist()  # set the padding part to 0
                        for s, t, p in zip(src_ids, trg_ids, prd_ids):
                            mapped_src = []
                            mapped_trg = []
                            mapped_prd = []
                            flag = False  # if we arrive at the target part
                            # src: [CLS]+[CLS]...+src+[SEP]...+[mask]
                            # trg: [CLS]+[CLS]...+src+[SEP]...+trg
                            for st, tt, pt in zip(s, t, p):
                                if st == tokenizer.sep_token_id:
                                    flag = True
                                if not flag:
                                    mapped_src += [st]
                                else:
                                    mapped_trg += [tt]
                                    if st == tokenizer.mask_token_id:  # we only predict the masked tokens
                                        mapped_prd += [pt]
                                    else:
                                        mapped_prd += [st]
                            if anchor is not None:
                                # src: [CLS]+[CLS]...+src+anchor+[SEP]...+[mask]
                                # trg: [CLS]+[CLS]...+src+anchor+[SEP]...+trg
                                # remove the anchor tokens from the src
                                anchor_length = len(anchor)
                                del mapped_trg[:anchor_length]
                                del mapped_prd[:anchor_length]
                            # we skip special tokens like '[UNK]','[SEP]'
                            all_inputs += [decode(mapped_src)]
                            all_labels += [decode(mapped_trg)]
                            all_predictions += [decode(mapped_prd)]

                        eval_steps += 1
    
                    # train_loss = train_loss / (step+1)
                    # eval_loss = eval_loss / eval_steps
                    train_loss = train_loss / (step+1)
                    train_det_loss = train_det_loss / (step+1)
                    train_cor_loss = train_cor_loss / (step+1)
                    train_cpo_loss = train_cpo_loss / (step+1)

                    eval_loss = eval_loss / (eval_steps+1)
                    eval_det_loss = eval_det_loss / (eval_steps+1)
                    eval_cor_loss = eval_cor_loss / (eval_steps+1)
                    eval_cpo_loss = eval_cpo_loss / (eval_steps+1)

                    show_topk = 5
                    for s_i, s_t, s_p in zip(all_inputs[:show_topk], all_labels[:show_topk],
                                             all_predictions[:show_topk]):
                        print("s_i: " + "".join(s_i))
                        print("s_t: " + "".join(s_t))
                        print("s_p: " + "".join(s_p))

                    print(step)
                    print(eval_steps)

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
                        "global_step": global_step,
                        "train_loss": train_loss,
                        "eval_loss": eval_loss,
                        "eval_det_loss": eval_det_loss,
                        "eval_cor_loss": eval_cor_loss,
                        "eval_cpo_loss": eval_cpo_loss,
                        "train_det_loss": train_det_loss,
                        "train_cor_loss": train_cor_loss,
                        "train_cpo_loss": train_cpo_loss,

                        "det_acc": det_acc,
                        "det_precision": det_precision,
                        "det_recall": det_recall,
                        "det_f1": det_f1,

                        "cor_acc": cor_acc,
                        "cor_precision": cor_precision,
                        "cor_recall": cor_recall,
                        "cor_f1": cor_f1,

                        "lr": scheduler.get_lr()[-1]
                    }

                    for k, v in result.items():
                        if k and "det_" in k:
                            k = "det/" + k
                        elif k and "cor_" in k:
                            k = "cor/" + k
                        elif k and "loss" in k:
                            k = "loss/" + k
                        elif k and "lr" in k:
                            k = "lr/" + k
                        else:
                            continue
                        tensorboardx_witer.add_scalar(k, v, global_step)
                        print(f"eval---epoch:{idx_num}; global_step:{global_step}; {k}:{v}")

                    model_to_save = model.module if hasattr(model, "module") else model
                    output_model_file = os.path.join(args.model_save_path, "step-%s_f1-%.4f.bin" % (str(global_step), result["cor_f1"]))
                    torch.save(model_to_save.state_dict(), output_model_file)  ## save the model
                    output_task_config = os.path.join(args.model_save_path, "csc.config")
                    model_to_save.config.save_pretrained(save_directory=args.model_save_path)
                    tokenizer.save_pretrained(save_directory=args.model_save_path)
                    args.flag_train = False
                    save_json(vars(args), output_task_config, mode="w")
                    ### only for check
                    if best_result and result["cor_f1"] > best_result[0][0]:
                        torch.save(model.state_dict(), os.path.join(args.model_save_path, "pytorch_model.bin"))
                    ## sort by f1 and remove model whose f1 is the fourth biggest
                    best_result.append((result["cor_f1"], output_model_file))
                    best_result.sort(key=lambda x: x[0], reverse=True)
                    if len(best_result) > 3:
                        _, model_to_remove = best_result.pop()
                        os.remove(model_to_remove)
                    ### save mertics of save_steps
                    output_eval_file = os.path.join(args.model_save_path, "train_results.txt")
                    text_log_list = []
                    for key in sorted(result.keys()):
                        text_log = "Global epoch: %s, step: %s,  %s = %s\n" % (str(idx_num), str(global_step), key, str(result[key]))
                        logger.info(text_log)
                        text_log_list.append(text_log)
                    txt_write(text_log_list, output_eval_file, mode="a+")

                if global_step >= args.max_train_steps:
                    wrap = True
                    break

    if args.do_test:
        eval_examples = processor.get_test_examples()
        eval_dataloader = DataLoader(num_workers=args.num_workers,
                                      batch_size=args.eval_batch_size,
                                      # shuffle=True,
                                      collate_fn=data_collate_fn,
                                      dataset=eval_examples,
                                      sampler=SequentialSampler(eval_examples),
                                      pin_memory=args.flag_pin_memory,
                                      )

        ### 存储为标准模式的文件, 最好的结果
        if args.do_train and best_result:
            best_result.sort(key=lambda x: x[0], reverse=True)
            path_model_best = best_result[0][-1]
            model.load_state_dict(torch.load(path_model_best))
            output_model_file = os.path.join(args.model_save_path, "pytorch_model.bin")
            torch.save(model.state_dict(), output_model_file)  ##save the model

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        def decode(input_ids):
            return tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)

        model.eval()
        eval_det_loss = 0
        eval_cor_loss = 0
        eval_cpo_loss = 0
        eval_steps = 0
        eval_loss = 0
        all_inputs, all_labels, all_predictions = [], [], []
        for batch in tqdm(eval_dataloader, desc="Evaluation"):
            batch = tuple(t.to(device) for t in batch)
            # src_ids, attention_mask, trg_ids, block_flag = batch
            src_ids, attention_mask, trg_ids, trg_ref_ids, block_flag = batch
            ### unk转mask
            src_ids = src_ids.masked_fill(src_ids == tokenizer.unk_token_id, tokenizer.mask_token_id)

            with torch.no_grad():
                outputs = model(input_ids=src_ids,
                                attention_mask=attention_mask,
                                labels=trg_ids,
                                # prompt_mask=block_flag,
                                # apply_prompt=args.apply_prompt
                                )
                # tmp_eval_loss = outputs.loss
                # logits = outputs.logits
                # tmp_eval_loss = outputs[0]
                # logits = outputs[-2]
                tmp_eval_det_loss = outputs[1]
                tmp_eval_cor_loss = outputs[2]
                tmp_eval_cpo_loss = outputs[3]
                tmp_eval_loss = outputs[0]
                logits = outputs[-2]

            src_ids = src_ids.tolist()
            trg_ids = trg_ids.cpu().numpy()
            eval_loss += tmp_eval_loss.mean().item()
            eval_det_loss += tmp_eval_det_loss.mean().item()
            eval_cor_loss += tmp_eval_cor_loss.mean().item()
            eval_cpo_loss += tmp_eval_cpo_loss.mean().item()

            _, prd_ids = torch.max(logits, -1)
            prd_ids = prd_ids.masked_fill(attention_mask == 0, 0).tolist()
            for s, t, p in zip(src_ids, trg_ids, prd_ids):
                mapped_src = []
                mapped_trg = []
                mapped_prd = []
                flag = False## if we pass to the target part
                ##src: [CLS]+[CLS]...+src+[SEP]...+trg+[SEP]
                ##trg: [CLS]+[CLS]...+src+[SEP]...+trg+[SEP]
                for st, tt, pt in zip(s, t, p):
                    if st == tokenizer.sep_token_id:
                        flag = True
                    if not flag:
                        mapped_src += [st]
                    else:
                        mapped_trg += [tt]
                        if st == tokenizer.mask_token_id:##we only predict the masked tokens
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
                '''
                print(all_inputs[-1])
                print(all_labels[-1])
                print(all_predictions[-1])
                print("--------------------\n")
                '''
            eval_steps += 1

        # eval_loss = eval_loss / eval_steps
        eval_loss = eval_loss / (eval_steps + 1)
        eval_det_loss = eval_det_loss / (eval_steps + 1)
        eval_cor_loss = eval_cor_loss / (eval_steps + 1)
        eval_cpo_loss = eval_cpo_loss / (eval_steps + 1)

        show_topk = 5
        for s_i, s_t, s_p in zip(all_inputs[:show_topk], all_labels[:show_topk],
                                 all_predictions[:show_topk]):
            print("s_i: " + "".join(s_i))
            print("s_t: " + "".join(s_t))
            print("s_p: " + "".join(s_p))

        print(step)
        print(eval_steps)

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
            "eval_det_loss": eval_det_loss,
            "eval_cor_loss": eval_cor_loss,
            "eval_cpo_loss": eval_cpo_loss,

            "det_acc": det_acc,
            "det_precision": det_precision,
            "det_recall": det_recall,
            "det_f1": det_f1,

            "cor_acc": cor_acc,
            "cor_precision": cor_precision,
            "cor_recall": cor_recall,
            "cor_f1": cor_f1,

            "lr": scheduler.get_lr()[-1]
        }

        output_eval_file = os.path.join(args.model_save_path, "eval_results.txt")
        text_log_list = []
        for key in sorted(result.keys()):
            text_log = "Global step: %s,  %s = %s" % (str(-1), key, str(result[key]))
            logger.info(text_log)
            text_log_list.append(text_log)
        txt_write(text_log_list, output_eval_file, mode="a+")


if __name__ == "__main__":
    train_csc()


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

law-16epoch-2e5-prompt1-macbert
################################################################################################################################
Sentence Level detection: acc:0.8840, precision:0.8328, recall:0.8819, f1:0.8566
Sentence Level correction: acc:0.8560, precision:0.7840, recall:0.8303, f1:0.8065
################################################################################################################################
100%|██████████| 4900/4900 [16:43<00:00,  4.88it/s]


100%|██████████| 7840/7840 [26:07<00:00,  5.00it/s]
law-32epoch-3e5-prompt1-macbert
Sentence Level detection: acc:0.8780, precision:0.8173, recall:0.9077, f1:0.8601
Sentence Level correction: acc:0.8660, precision:0.7973, recall:0.8856, f1:0.8392

100%|██████████| 7840/7840 [26:07<00:00,  5.00it/s]
law-32epoch-3e5-prompt5-macbert
Sentence Level detection: acc:0.8740, precision:0.8115, recall:0.9270, f1:0.8654
Sentence Level correction: acc:0.8660, precision:0.7987, recall:0.9124, f1:0.8518

100%|██████████| 3936/3936 [22:07<00:00,  2.97it/s]
law-32epoch-5e5-prompt3-bertbase
Sentence Level detection: acc:0.8780, precision:0.8143, recall:0.9124, f1:0.8606
Sentence Level correction: acc:0.8540, precision:0.7752, recall:0.8686, f1:0.8193

100%|██████████| 3936/3936 [22:07<00:00,  2.97it/s]
law-6epoch-5e5-prompt3-bertbase
Sentence Level detection: acc:0.8220, precision:0.7485, recall:0.9114, f1:0.8220
Sentence Level correction: acc:0.8120, precision:0.7333, recall:0.8930, f1:0.8053


100%|██████████| 3936/3936 [22:07<00:00,  2.97it/s]
law-32epoch-5e5-prompt3-bertbase-0.2mask-noerror
Sentence Level detection: acc:0.8900, precision:0.8362, recall:0.8942, f1:0.8642
Sentence Level correction: acc:0.8700, precision:0.8020, recall:0.8577, f1:0.8289


100%|██████████| 5000/5000 [57:39<00:00,  1.45it/s]
law-32epoch-32bs-5e5-prompt1-macbert-0.3mask-noerror
Sentence Level detection: acc:0.9040, precision:0.8617, recall:0.8967, f1:0.8788
Sentence Level correction: acc:0.8980, precision:0.8511, recall:0.8856, f1:0.8680


 42%|████▏     | 2100/5000 [24:13<33:32,  1.44it/s]
law-5000step-32bs-3e5-prompt3-macbert-0.3mask-noerror
Sentence Level detection: acc:0.9120, precision:0.8741, recall:0.9124, f1:0.8929
Sentence Level correction: acc:0.9040, precision:0.8601, recall:0.8978, f1:0.8786

 96%|█████████▌| 4800/5000 [54:43<02:07,  1.57it/s]
law-5000step-32bs-2e5-prompt1-macbert-0.3mask-noerror
Sentence Level detection: acc:0.8940, precision:0.8428, recall:0.9299, f1:0.8842
Sentence Level correction: acc:0.8880, precision:0.8328, recall:0.9188, f1:0.8737

 64%|████▏     | 3200/5000 [24:13<33:32,  1.44it/s]
law-5000step-32bs-3e5-prompt0-macbert-0.3mask-noerror--flag_loss_period---accu=4---0.3-det-loss
Sentence Level detection: acc:0.9040, precision:0.8567, recall:0.9262, f1:0.8901
Sentence Level correction: acc:0.8980, precision:0.8464, recall:0.9151, f1:0.8794

law-5000step-32bs-3e5-prompt0-macbert-0.3mask-noerror--flag_loss_period---accu=4---no-det-loss
Sentence Level detection: acc:0.9120, precision:0.8690, recall:0.9299, f1:0.8984
Sentence Level correction: acc:0.9100, precision:0.8655, recall:0.9262, f1:0.8948


law-5000step-16*4bs-3e5-prompt0-macbert-0.3mask-error--flag_loss_period---accu=4---no-det-loss
Sentence Level detection: acc:0.8440, precision:0.7945, recall:0.7417, f1:0.7672
Sentence Level correction: acc:0.7420, precision:0.5929, recall:0.5535, f1:0.5725


law-5000step-32*4bs-3e5-prompt0-macbert-0.3mask-noerror--flag_loss_period---accu=4---no-det-loss
Sentence Level detection: acc:0.9160, precision:0.8713, recall:0.9294, f1:0.8994
Sentence Level correction: acc:0.9080, precision:0.8566, recall:0.9137, f1:0.8843

.54
{'loss': 3.428, 'lr': 3.544098020613618e-12, 'epoch': 3.0}
{'train_runtime': 42384.3568, 'train_samples_per_second': 17.825, 'train_steps_per_second': 0.139, 'train_loss': 3.527541259167013, 'epoch': 3.0}

"""


"""
################################################################################################################################
flag_eval: common
Sentence Level detection: acc:0.8113, precision:0.9708, recall:0.7855, f1:0.8684
Sentence Level correction: acc:0.7312, precision:0.9667, recall:0.6845, f1:0.8015
################################################################################################################################
flag_eval: strict
Sentence Level detection: acc:0.8113, precision:0.8372, recall:0.7855, f1:0.8106
Sentence Level correction: acc:0.7312, precision:0.7296, recall:0.6845, f1:0.7063
################################################################################################################################
eval---epoch:1; global_step:122000; loss/train_loss:0.0035272446099549187
eval---epoch:1; global_step:122000; loss/eval_loss:8.47018651688284
eval---epoch:1; global_step:122000; det/eval_det_loss:1.1078668854097509
eval---epoch:1; global_step:122000; cor/eval_cor_loss:11.62546653532276
eval---epoch:1; global_step:122000; loss/eval_cpo_loss:11.62546653532276
eval---epoch:1; global_step:122000; det/train_det_loss:0.0006470593258879975
eval---epoch:1; global_step:122000; cor/train_cor_loss:0.01987837265292519
eval---epoch:1; global_step:122000; loss/train_cpo_loss:0.01987837265292519
eval---epoch:1; global_step:122000; det/det_acc:0.8112825307776219
eval---epoch:1; global_step:122000; det/det_precision:0.8372322330958661
eval---epoch:1; global_step:122000; det/det_recall:0.7855309110252793
eval---epoch:1; global_step:122000; det/det_f1:0.8105579685933846
eval---epoch:1; global_step:122000; cor/cor_acc:0.7312194630072693
eval---epoch:1; global_step:122000; cor/cor_precision:0.7295831587250721
eval---epoch:1; global_step:122000; cor/cor_recall:0.6845294539399126
eval---epoch:1; global_step:122000; cor/cor_f1:0.7063385995895184
eval---epoch:1; global_step:122000; lr/lr:1.3341185018062286e-05
"""


# shell
# nohup python train_yield.py > tc.train_yield.py.log 2>&1 &
# tail -n 1000  -f tc.train_yield.py.log
# |myz|


"""
relm在少样本的数据集上效果不错, 但数据量重组的时候
"""


