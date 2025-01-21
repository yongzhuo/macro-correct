# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/1/18 21:34
# @author  : Mo
# @function: Macbert4CSC
# @github   : [Macbert4CSC](https://github.com/shibing624/pycorrector/tree/master/examples/macbert)
# @paper   :  [SoftMaskedBert4CSC](https://arxiv.org/abs/2004.13922)


from __future__ import absolute_import, division, print_function
import argparse
import logging
import random
import math
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(path_root)
print(path_root)

from macro_correct.pytorch_user_models.csc.macbert4csc.config import csc_config as args

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

from macro_correct.pytorch_user_models.csc.macbert4csc.dataset import DataSetProcessor, sent_mertic_det, sent_mertic_cor
from macro_correct.pytorch_user_models.csc.macbert4csc.dataset import save_json, load_json, txt_write
from macro_correct.pytorch_user_models.csc.macbert4csc.graph import Macbert4CSC as Graph


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


def dynamic_mask_token(inputs, targets, tokenizer, device, mask_mode="noerror", noise_probability=0.2):
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

    return inputs


class InputFeatures(object):
    def __init__(self, src_ids, attention_mask, trg_ids, trg_ref_ids, block_flag):
        self.src_ids = src_ids
        self.attention_mask = attention_mask
        self.trg_ids = trg_ids
        self.trg_ref_ids = trg_ref_ids
        self.block_flag = block_flag


def train_csc():
    """   训练   """
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

    if args.do_train:
        train_examples = processor.get_train_examples()

        train_features = convert_examples_to_features(train_examples, args.max_seq_length, tokenizer,
                                                      args.prompt_length, anchor=anchor, logger=logger)
        all_input_ids = torch.tensor([f.src_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.trg_ids for f in train_features], dtype=torch.long)
        all_trg_ref_ids = torch.tensor([f.trg_ref_ids for f in train_features], dtype=torch.long)
        all_block_flag = torch.tensor([f.block_flag for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_trg_ref_ids, all_block_flag)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

        if args.max_train_steps is None:
            args.max_train_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        ##we use BERTMLM as the backbone for ReLM
        # model = BertForMaskedLM.from_pretrained(args.pretrained_model_name_or_path,
        #                                         return_dict=True,
        #                                         # cache_dir=cache_dir
        #                                         )
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

            eval_features = convert_examples_to_features(eval_examples, args.max_seq_length, tokenizer,
                                                         args.prompt_length, anchor=anchor,
                                                         logger=logger)  ##never mask the source during evaluation

            all_input_ids = torch.tensor([f.src_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.trg_ids for f in eval_features], dtype=torch.long)
            all_block_flag = torch.tensor([f.block_flag for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_block_flag)
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

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
                src_ids, attention_mask, trg_ids, trg_ref_ids, block_flag = batch
                # if args.flag_mft and thr_mft <= thr_std and idx_num < int(args.num_train_epochs-1):
                # thr_std = 0.5  # 随机概率/50%
                thr_mft = random.random()
                if args.flag_mft and thr_mft <= 0.5:
                    src_ids = dynamic_mask_token(src_ids, trg_ref_ids, tokenizer, device,
                                                 args.mask_mode, args.mask_rate)
                # if args.flag_mft:
                #     src_ids = dynamic_mask_token(src_ids, trg_ref_ids, tokenizer,
                #                                  device, args.mask_mode, args.mask_rate)
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
                # ### only loss on the masked positions are included when calculating loss
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
                    eval_det_loss = 0
                    eval_cor_loss = 0
                    eval_cpo_loss = 0
                    eval_loss = 0
                    eval_steps = 0
                    all_inputs, all_labels, all_predictions = [], [], []
                    for batch in tqdm(eval_dataloader, desc="Evaluation"):
                        batch = tuple(t.to(device) for t in batch)
                        src_ids, attention_mask, trg_ids, block_flag = batch
                        ### unk转mask
                        src_ids = src_ids.masked_fill(src_ids == tokenizer.unk_token_id, tokenizer.mask_token_id)

                        with torch.no_grad():
                            outputs = model(input_ids=src_ids,
                                            attention_mask=attention_mask,
                                            labels=trg_ids,
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
                                if st == tokenizer.sep_token_id or st == tokenizer.cls_token_id:
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
                    train_cpo_loss = train_cpo_loss / (step + 1)

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
                        logger.info(f"eval---epoch:{idx_num}; global_step:{global_step}; {k}:{v}")
                        # print(f"eval---epoch:{idx_num}; global_step:{global_step}; {k}:{v}")

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
        if args.do_train and best_result:
            best_result.sort(key=lambda x: x[0], reverse=True)
            path_model_best = best_result[0][-1]
            model.load_state_dict(torch.load(path_model_best))
            output_model_file = os.path.join(args.model_save_path, "pytorch_model.bin")
            torch.save(model.state_dict(), output_model_file)  ##save the model

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
            src_ids, attention_mask, trg_ids, block_flag = batch
            with torch.no_grad():
                outputs = model(input_ids=src_ids,
                                attention_mask=attention_mask,
                                labels=trg_ids,
                                )
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
                # flag = False  # if we arrive at the target part
                # src: [CLS]+[CLS]...+src+[SEP]...+[mask]
                # trg: [CLS]+[CLS]...+src+[SEP]...+trg
                for st, tt, pt in zip(s, t, p):
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
                '''
                print(all_inputs[-1])
                print(all_labels[-1])
                print(all_predictions[-1])
                print("--------------------\n")
                '''
            eval_steps += 1

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
std-offical
89.9 94.5 91.2 ↑10.6


mdcspell-law-5000step-scheduler=linear-32bs-accu=4-3e5-prompt1-macbert-0.3mask-maskmode=noerror-detrate0.15
Sentence Level detection: acc:0.7900, precision:0.6918, recall:0.8627, f1:0.7679
Sentence Level correction: acc:0.7740, precision:0.6667, recall:0.8314, f1:0.7400

################################################################################################################################
mdcspell-law-5000step-scheduler=linear-bs=16-accu=1-lr=3e5-prompt=0-macbert-0.15mask-maskmode=noerror-detrate0.15-loss=bce_multi
 94%|█████████▍| 4700/5000 [34:25<01:53,  2.64it/s]
Sentence Level detection: acc:0.8240, precision:0.7406, recall:0.8510, f1:0.7920
Sentence Level correction: acc:0.8080, precision:0.7133, recall:0.8196, f1:0.7628

################################################################################################################################
mdcspell-law-10000step-scheduler=cosine-weightdecay=5e4-bs=16-accu=1-lr=5e5-prompt=0-macbert-0.15mask-maskmode=noerror-detrate0.15-loss=bce_multi
 36%|███▌      | 3600/10000 [26:22<40:53,  2.61it/s]
Sentence Level detection: acc:0.7940, precision:0.7045, recall:0.8510, f1:0.7709
Sentence Level correction: acc:0.7780, precision:0.6786, recall:0.8196, f1:0.7425


################################################################################################################################
mdcspell-law-5000step-scheduler=linear-weightdecay=1e2-bs=16-accu=1-lr=2e5-prompt=0-macbert-0.3mask-maskmode=noerror-detrate0.3-loss=bce_multi
100%|██████████| 5000/5000 [36:02<00:00,  2.31it/s]
Sentence Level detection: acc:0.8060, precision:0.7162, recall:0.8510, f1:0.7778
Sentence Level correction: acc:0.7880, precision:0.6865, recall:0.8157, f1:0.7455

################################################################################################################################
mdcspell-law-6000step-scheduler=cosine-weightdecay=1e2-bs=16-accu=1-lr=3e5-prompt=0-macbert-0.15mask-maskmode=noerror-detrate0.15-loss=bce_multi
 95%|█████████▌| 5700/6000 [41:43<01:54,  2.62it/s]
Sentence Level detection: acc:0.7920, precision:0.6968, recall:0.8471, f1:0.7646
Sentence Level correction: acc:0.7780, precision:0.6742, recall:0.8196, f1:0.7398


################################################################################################################################
mdcspell-law-6000step-scheduler=cosine-weightdecay=1e2-bs=16-accu=1-lr=3e5-prompt=0-macbert-0.3mask-maskmode=noerror-detrate0.15-loss=circleloss
Sentence Level detection: acc:0.8060, precision:0.7185, recall:0.8510, f1:0.7792
Sentence Level correction: acc:0.7820, precision:0.6788, recall:0.8039, f1:0.7361

################################################################################################################################
mdcspell-law-6000step-scheduler=cosine-weightdecay=1e4-bs=16-accu=1-lr=3e5-prompt=0-macbert-0.15mask-maskmode=noerror-detrate0.15-loss=focalloss
Sentence Level detection: acc:0.8100, precision:0.7205, recall:0.8392, f1:0.7754
Sentence Level correction: acc:0.7980, precision:0.7003, recall:0.8157, f1:0.7536

################################################################################################################################
mdcspell-law-6000step-scheduler=linear-weightdecay=1e2-bs=16-accu=1-lr=3e5-prompt=0-macbert-0.15mask-maskmode=noerror-detrate0.15-loss=focalloss
Sentence Level detection: acc:0.8100, precision:0.7205, recall:0.8392, f1:0.7754
Sentence Level correction: acc:0.7980, precision:0.7003, recall:0.8157, f1:0.7536

100%|██████████| 5000/5000 [1:14:17<00:00,  1.12it/s]
Sentence Level detection: acc:0.4100, precision:0.3059, recall:0.4667, f1:0.3696
Sentence Level correction: acc:0.3580, precision:0.2391, recall:0.3647, f1:0.2888
"""



# shell
# nohup python train.py > tc.train.py.log 2>&1 &
# tail -n 1000  -f tc.train.py.log
# |myz|


