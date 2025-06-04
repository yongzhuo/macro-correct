# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/1/18 21:34
# @author  : Mo
# @function: MDCSpell
# @paper   : [MDCSpell: A Multi-task Detector-Corrector Framework for Chinese Spelling Correction](https://aclanthology.org/2022.findings-acl.98/).
# @code    : most code copy from https://github.com/gingasan/lemon, small modfix
# @notice  : 训练时候计算macbert4csc的loss和mdspell的loss, 推理的时候只使用BertForMaskedLM


from __future__ import absolute_import, division, print_function
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
from macro_correct.pytorch_user_models.csc.macbert4mdcspell.config import csc_config as args
os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES or "0"
os.environ["USE_TORCH"] = args.USE_TORCH or "1"
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import get_linear_schedule_with_warmup
from transformers import SchedulerType, get_scheduler
from transformers import BertForMaskedLM
from transformers import AutoTokenizer
from tensorboardX import SummaryWriter
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch

from macro_correct.pytorch_user_models.csc.macbert4mdcspell.dataset import DataSetProcessor, sent_mertic_det, sent_mertic_cor
from macro_correct.pytorch_user_models.csc.macbert4mdcspell.dataset import save_json, load_json, txt_write
from macro_correct.pytorch_user_models.csc.macbert4mdcspell.graph import Macbert4MDCSpell as Graph


# adapt the input for ReLM
def convert_examples_to_features(examples, max_seq_length, tokenizer, prompt_length, anchor=None, mask_rate=0.2,
                                 logger=None):
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
                          block_flag=block_flag)
        )
    return features


# adapt the input for ReLM
def convert_examples_to_prompts(src, trg, prompt_length, max_seq_length, tokenizer, anchor=None, mask_rate=0.2):
    def truncate(x, max_length):
        return x[: max_length]

    ## here max_seq = tokenizer.max_seq_length//2, we need to truncate
    src = truncate(src, max_seq_length - prompt_length - 2)
    trg = truncate(trg, max_seq_length - prompt_length - 2)
    assert (len(src) == len(trg))
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

    return prompt_src, prompt_trg, block_flag, trg_ref


def dynamic_mask_token(inputs, targets, tokenizer, device, mask_mode="noerror",
                       replace_mode="mask", noise_probability=0.2):
    '''
    the masked-FT proposed in 'Rethinking Masked Language Model for Chinese Spelling Correction'
    '''
    # src:[CLS]...[CLS],x1,x2,...,xn,[SEP],...,[SEP],m1,m2,...,mn
    # trg:[CLS]...[CLS],t1,t2,...,tn,[SEP],...,[SEP],t1,t2,...,tn

    inputs = inputs.clone()
    probability_matrix = torch.full(inputs.shape, noise_probability).to(device)
    # do not mask sepcail tokens
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool).to(device)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # mask_mode in ["all","error","noerror"]
    if mask_mode == "noerror":
        probability_matrix.masked_fill_(inputs != targets, value=0.0)
    elif mask_mode == "error":
        probability_matrix.masked_fill_(inputs == targets, value=0.0)
    else:
        assert mask_mode == "all"
    masked_indices = torch.bernoulli(probability_matrix).bool()
    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    # if replace_mode == "mask":
    #     inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    # elif replace_mode == "random":
    #     random_words = torch.randint(low=0, high=vocab_size, size=inputs.size()).to(device)
    #     random_words.masked_fill_(~masked_indices, value=0)  # only change the masked positions
    #     inputs = torch.where(masked_indices, random_words, inputs)
    # elif replace_mode == "id_map":
    #     if id_map is None:
    #         raise ValueError("id_map must be provided when replace_mode is 'id_map'")
    #
    #     # Create a tensor of mapped ids based on the id_map dictionary
    #     mapped_ids = torch.tensor([id_map.get(idx.item(), idx.item()) for row in inputs for idx in row]).view(
    #         inputs.size()).to(device)
    #     inputs = torch.where(masked_indices, mapped_ids, inputs)
    # else:
    #     raise ValueError(f"Unsupported replace_mode: {replace_mode}")
    return inputs


class InputFeatures(object):
    def __init__(self, src_ids, attention_mask, trg_ids, trg_ref_ids, block_flag):
        self.src_ids = src_ids
        self.attention_mask = attention_mask
        self.trg_ids = trg_ids
        self.trg_ref_ids = trg_ref_ids
        self.block_flag = block_flag


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
        anchor = [tokenizer.sep_token] + [t for t in args.anchor]


    def data_collate_fn(batch, tokenizer=tokenizer, args=args):
        """   数据处理collate_fn   """
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
            src = src[: max_len - args.prompt_length - 2]
            trg = trg[: max_len - args.prompt_length - 2]
            assert (len(src) == len(trg))
            ### 中间必须要有一个sep
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

        model = Graph(config=args)
        model.to(device)
        # if args.load_state_dict:
        #     model.load_state_dict(torch.load(args.load_state_dict))
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

        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
        #                                             num_training_steps=args.num_train_epochs)
        # 先args.num_warmup_steps 后args.warmup_proportion
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
            train_det_macbert_loss = 0
            train_cor_macbert_loss = 0
            train_det_loss = 0
            train_cor_loss = 0
            train_loss = 0
            if wrap:
                break
            for step, batch in enumerate(train_dataloader):
                model.csc_config.flag_train = True
                model.train()
                batch = tuple(t.to(device) for t in batch)
                src_ids, attention_mask, trg_ids, trg_ref_ids, block_flag = batch
                ### 50%随机mask, 并且最后一轮不mask
                thr_mft = random.random()
                # thr_std = 0.5  # 随机概率/50%
                # if args.flag_mft and thr_mft <= thr_std and idx_num < int(args.num_train_epochs-1):
                if args.flag_mft and thr_mft <= 0.7:
                    src_ids = dynamic_mask_token(src_ids, trg_ref_ids, tokenizer,
                                                 device, args.mask_mode, args.mask_rate)
                ### 原句子预测占比15%, 防止过度纠错
                elif thr_mft <= 0.85:
                    src_ids = copy.deepcopy(trg_ids)
                # ### 训练4/5以后再采用 focal-loss + cpo_loss
                # if args.flag_loss_period and global_step < args.num_train_epochs * 4 / 5:
                #     model.loss_type = "BCE_MULTI"
                #     model.flag_cpo_loss = False
                #     # if global_step < args.num_train_epochs * 2 / 5:
                #     #     model.loss_type = "BCE_MULTI"
                #     #     model.flag_cpo_loss = False
                # else:
                #     model.flag_cpo_loss = args.flag_cpo_loss
                #     model.loss_type = args.loss_type
                ### unk转mask
                src_ids = src_ids.masked_fill(src_ids == tokenizer.unk_token_id, tokenizer.mask_token_id)
                ## only loss on the masked positions are included when calculating loss
                # trg_ids[(src_ids == trg_ids)] = -100  ##ignore index = -100
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
                tmp_train_det_macbert_loss = outputs[3]
                tmp_train_cor_macbert_loss = outputs[4]
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
                train_det_macbert_loss += tmp_train_det_macbert_loss.mean().item()
                train_cor_macbert_loss += tmp_train_cor_macbert_loss.mean().item()
                train_loss += loss.mean().item()

                num_train_examples += src_ids.size(0)
                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    # 裁剪
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

                if args.do_eval and global_step % args.save_steps == 0 and (
                        step + 1) % args.gradient_accumulation_steps == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    def decode(x):
                        return tokenizer.convert_ids_to_tokens(x, skip_special_tokens=True)

                    model.eval()
                    eval_det_macbert_loss = 0
                    eval_cor_macbert_loss = 0
                    eval_det_loss = 0
                    eval_cor_loss = 0
                    eval_loss = 0
                    eval_steps = 0
                    all_inputs, all_labels, all_predictions = [], [], []
                    for batch in tqdm(eval_dataloader, desc="Evaluation"):
                        batch = tuple(t.to(device) for t in batch)
                        src_ids, attention_mask, trg_ids, trg_ref_ids, block_flag = batch
                        ### unk转mask
                        src_ids = src_ids.masked_fill(src_ids == tokenizer.unk_token_id, tokenizer.mask_token_id)

                        model.csc_config.flag_train = False
                        with torch.no_grad():
                            outputs = model(input_ids=src_ids,
                                            attention_mask=attention_mask,
                                            labels=trg_ids,
                                            )
                            # tmp_eval_loss = outputs.loss
                            # logits = outputs.logits
                            tmp_eval_det_loss = outputs[1]
                            tmp_eval_cor_loss = outputs[2]
                            tmp_eval_det_macbert_loss = outputs[3]
                            tmp_eval_cor_macbert_loss = outputs[4]
                            tmp_eval_loss = outputs[0]
                            logits = outputs[-2]

                        src_ids = src_ids.cpu().numpy().tolist()
                        trg_ids = trg_ids.cpu().numpy().tolist()
                        eval_loss += tmp_eval_loss.mean().item()
                        eval_det_loss += tmp_eval_det_loss.mean().item()
                        eval_cor_loss += tmp_eval_cor_loss.mean().item()
                        eval_det_macbert_loss += tmp_eval_det_macbert_loss.mean().item()
                        eval_cor_macbert_loss += tmp_eval_cor_macbert_loss.mean().item()

                        _, prd_ids = torch.max(logits, -1)  # (batch,seq)
                        prd_ids = prd_ids.masked_fill(attention_mask == 0, 0).tolist()  # set the padding part to 0
                        for s, t, p in zip(src_ids, trg_ids, prd_ids):
                            mapped_src = []
                            mapped_trg = []
                            mapped_prd = []
                            # flag = False  # if we arrive at the target part
                            # src: [CLS]+[CLS]...+src+[SEP]...+[mask]
                            # trg: [CLS]+[CLS]...+src+[SEP]...+trg
                            # ### tokenizer会+1
                            # for st, tt, pt in zip(s[args.prompt_length+1:-args.prompt_length-1],
                            #                       t[args.prompt_length+1:-args.prompt_length-1],
                            #                       p[args.prompt_length+1:-args.prompt_length-1]):
                            #     mapped_trg += [tt]
                            #     mapped_src += [st]
                            #     if st == pt:
                            #         mapped_prd += [st]
                            #     else:
                            #         mapped_prd += [pt]
                            for st, tt, pt in zip(s, t, p):
                                if st in [tokenizer.sep_token_id, tokenizer.cls_token_id]:
                                    continue
                                mapped_trg += [tt]
                                mapped_src += [st]
                                if st == pt:
                                    mapped_prd += [st]
                                else:
                                    mapped_prd += [pt]
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
                    train_loss = train_loss / (step + 1)
                    train_det_loss = train_det_loss / (step + 1)
                    train_cor_loss = train_cor_loss / (step + 1)
                    train_det_macbert_loss = train_det_macbert_loss / (step + 1)
                    train_cor_macbert_loss = train_cor_macbert_loss / (step + 1)

                    eval_loss = eval_loss / (eval_steps + 1)
                    eval_det_loss = eval_det_loss / (eval_steps + 1)
                    eval_cor_loss = eval_cor_loss / (eval_steps + 1)
                    eval_det_macbert_loss = eval_det_macbert_loss / (eval_steps + 1)
                    eval_cor_macbert_loss = eval_cor_macbert_loss / (eval_steps + 1)

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
                        "eval_det_macbert_loss": eval_det_macbert_loss,
                        "eval_cor_macbert_loss": eval_cor_macbert_loss,

                        "train_det_loss": train_det_loss,
                        "train_cor_loss": train_cor_loss,
                        "train_det_macbert_loss": train_det_macbert_loss,
                        "train_cor_macbert_loss": train_cor_macbert_loss,

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
                        logger.info(f"eval---epoch:{idx_num}; global_step:{global_step}; {k}:{v}")
                        print(f"eval---epoch:{idx_num}; global_step:{global_step}; {k}:{v}")

                    model_to_save = model.module if hasattr(model, "module") else model
                    output_model_file = os.path.join(args.model_save_path,
                                                     "step-%s_f1-%.4f.bin" % (str(global_step), result["cor_f1"]))
                    torch.save(model_to_save.state_dict(), output_model_file)  ##save the model
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
                        text_log = "Global epoch: %s, step: %s,  %s = %s\n" % (
                        str(idx_num), str(global_step), key, str(result[key]))
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
        eval_det_macbert_loss = 0
        eval_cor_macbert_loss = 0
        eval_det_loss = 0
        eval_cor_loss = 0
        eval_loss = 0
        eval_steps = 0
        all_inputs, all_labels, all_predictions = [], [], []
        for batch in tqdm(eval_dataloader, desc="Evaluation"):
            batch = tuple(t.to(device) for t in batch)
            src_ids, attention_mask, trg_ids, trg_ref_ids, block_flag = batch
            ### unk转mask
            src_ids = src_ids.masked_fill(src_ids == tokenizer.unk_token_id, tokenizer.mask_token_id)

            model.csc_config.flag_train = False
            with torch.no_grad():
                outputs = model(input_ids=src_ids,
                                attention_mask=attention_mask,
                                labels=trg_ids,
                                )
                # tmp_eval_loss = outputs.loss
                # logits = outputs.logits
                tmp_eval_det_loss = outputs[1]
                tmp_eval_cor_loss = outputs[2]
                tmp_eval_det_macbert_loss = outputs[3]
                tmp_eval_cor_macbert_loss = outputs[4]
                tmp_eval_loss = outputs[0]
                logits = outputs[-2]

            src_ids = src_ids.cpu().numpy().tolist()
            trg_ids = trg_ids.cpu().numpy().tolist()
            eval_loss += tmp_eval_loss.mean().item()
            eval_det_loss += tmp_eval_det_loss.mean().item()
            eval_cor_loss += tmp_eval_cor_loss.mean().item()
            eval_det_macbert_loss += tmp_eval_det_macbert_loss.mean().item()
            eval_cor_macbert_loss += tmp_eval_cor_macbert_loss.mean().item()

            _, prd_ids = torch.max(logits, -1)  # (batch,seq)
            prd_ids = prd_ids.masked_fill(attention_mask == 0, 0).tolist()  # set the padding part to 0
            for s, t, p in zip(src_ids, trg_ids, prd_ids):
                mapped_src = []
                mapped_trg = []
                mapped_prd = []
                # flag = False  # if we arrive at the target part
                # src: [CLS]+[CLS]...+src+[SEP]...+[mask]
                # trg: [CLS]+[CLS]...+src+[SEP]...+trg
                # ### tokenizer会+1
                # for st, tt, pt in zip(s[args.prompt_length+1:-args.prompt_length-1],
                #                       t[args.prompt_length+1:-args.prompt_length-1],
                #                       p[args.prompt_length+1:-args.prompt_length-1]):
                #     mapped_trg += [tt]
                #     mapped_src += [st]
                #     if st == pt:
                #         mapped_prd += [st]
                #     else:
                #         mapped_prd += [pt]
                for st, tt, pt in zip(s, t, p):
                    if st in [tokenizer.sep_token_id, tokenizer.cls_token_id]:
                        continue
                    mapped_trg += [tt]
                    mapped_src += [st]
                    if st == pt:
                        mapped_prd += [st]
                    else:
                        mapped_prd += [pt]
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
        # train_loss = train_loss / (step + 1)
        # train_det_loss = train_det_loss / (step + 1)
        # train_cor_loss = train_cor_loss / (step + 1)
        # train_det_macbert_loss = train_det_macbert_loss / (step + 1)
        # train_cor_macbert_loss = train_cor_macbert_loss / (step + 1)

        eval_loss = eval_loss / (eval_steps + 1)
        eval_det_loss = eval_det_loss / (eval_steps + 1)
        eval_cor_loss = eval_cor_loss / (eval_steps + 1)
        eval_det_macbert_loss = eval_det_macbert_loss / (eval_steps + 1)
        eval_cor_macbert_loss = eval_cor_macbert_loss / (eval_steps + 1)

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
            "eval_det_macbert_loss": eval_det_macbert_loss,
            "eval_cor_macbert_loss": eval_cor_macbert_loss,

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


# shell
# nohup python train.py > tc.train.py.log 2>&1 &
# tail -n 1000  -f tc.train.py.log
# |myz|


# shell
# nohup python train_yield.py > tc.train_yield_w.py.log 2>&1 &
# tail -n 1000  -f tc.train_yield_w.py.log
# |myz|

