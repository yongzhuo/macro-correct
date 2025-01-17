# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/23 21:34
# @author  : Mo
# @function: main programing


# 适配linux
import numpy as np
import traceback
import platform
import random
import codecs
import copy
import json
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(path_root)
if platform.system().lower() == "windows":
    print(path_root)
from macro_correct.pytorch_sequencelabeling.slTools import get_pos_from_common, get_pos_from_span, mertics_report_sequence_labeling, chinese_extract_extend, get_logger
from macro_correct.pytorch_sequencelabeling.slConfig import _SL_MODEL_SOFTMAX, _SL_MODEL_GRID, _SL_MODEL_SPAN, _SL_MODEL_CRF
from macro_correct.pytorch_sequencelabeling.slConfig import _SL_DATA_CONLL, _SL_DATA_SPAN, model_config
from macro_correct.pytorch_sequencelabeling.slData import SeqLabelingDataCollator
from macro_correct.pytorch_sequencelabeling.slData import SeqlabelingDataset
from macro_correct.pytorch_sequencelabeling.slData import CreateDataLoader
from macro_correct.pytorch_sequencelabeling.slTqdm import tqdm, trange
from macro_correct.pytorch_sequencelabeling.slAdversarial import FGM
from macro_correct.pytorch_sequencelabeling.slOffice import Office
from macro_correct.pytorch_sequencelabeling.slGraph import Graph


# 预训练模型地址, 本地win10默认只跑2步就评估保存模型
if platform.system().lower() == "windows":
    pretrained_model_dir = "E:/DATA/bert-model/00_pytorch"
    evaluate_steps = 128  # 评估步数
    save_steps = 128  # 存储步数
else:
    # pretrained_model_dir = "/pretrain_models/pytorch"
    # evaluate_steps = 320  # 评估步数
    # save_steps = 320  # 存储步数
    pretrained_model_dir = "/opt/pretrain_models/pytorch"
    evaluate_steps = 3200  # 评估步数
    save_steps = 3200  # 存储步数
    ee = 0

# 预训练模型适配的class
model_type = ["BERT", "ERNIE", "BERT_WWM", "ALBERT", "ROBERTA", "XLNET", "ELECTRA"]
pretrained_model_name_or_path = {
    "BERT_WWM": pretrained_model_dir + "/chinese_wwm_pytorch",
    "ROBERTA": pretrained_model_dir + "/chinese_roberta_wwm_ext_pytorch",
    "ALBERT": pretrained_model_dir + "/albert_base_v1",
    "XLNET": pretrained_model_dir + "/chinese_xlnet_mid_pytorch",
    # "ERNIE": pretrained_model_dir + "/ERNIE_stable-1.0.1-pytorch",
    "ERNIE": pretrained_model_dir + "/ernie-tiny",  # 小模型
    # "BERT": pretrained_model_dir + "/uer_roberta-base-word-chinese-cluecorpussmall",
    # "BERT": pretrained_model_dir + "/bert-base-chinese",
    # "BERT": pretrained_model_dir + "/uer_roberta-base-wwm-chinese-cluecorpussmall",
    "BERT": pretrained_model_dir + "/LLM/hfl_chinese-macbert-base"
    # "ELECTRA": pretrained_model_dir + "/hfl_chinese-electra-180g-base-discriminator",
}


if __name__ == '__main__':

    task_version = "ner_chinese_symbol"

    path_corpus_dir_data = os.path.join(os.path.split(path_root)[0], "corpus", "sequence_labeling")
    # path_train = os.path.join(path_corpus_dir_data, "chinese_symbol/chinese_symbol.train.conll")
    # path_train = os.path.join(path_corpus_dir_data, "chinese_symbol/chinese_symbol.dev.conll")
    # path_train = os.path.join(path_corpus_dir_data, "chinese_symbol/chinese_symbol.tet.conll")
    path_train = os.path.join(path_corpus_dir_data, "chinese_symbol/chinese_symbol.dev.conll")
    path_dev = os.path.join(path_corpus_dir_data, "chinese_symbol/chinese_symbol.dev.conll")
    path_tet = os.path.join(path_corpus_dir_data, "chinese_symbol/chinese_symbol.dev.conll")

    model_config["evaluate_steps"] = evaluate_steps  # 评估步数
    model_config["save_steps"] = save_steps  # 存储步数
    model_config["path_train"] = path_train  # 训练模语料, 必须
    model_config["path_dev"] = path_dev  # 验证语料, 可为None
    model_config["path_tet"] = path_tet  # 测试语料, 可为None
    # model_config["CUDA_VISIBLE_DEVICES"] = "0"
    # 一种格式 文件以.conll结尾, 或者corpus_type=="DATA-CONLL"
    # 另一种格式 文件以.span结尾, 或者corpus_type=="DATA-SPAN"
    # 任务类型, "SL-SOFTMAX", "SL-CRF", "SL-SPAN", "SL-GRID", "sequence_labeling"
    # model_config["task_type"] = "SL-CRF"
    model_config["task_type"] = "SL-SPAN"
    # model_config["task_type"] = "SL-GRID"
    # model_config["task_type"] = "SL-SOFTMAX"
    # model_config["corpus_type"] = "DATA-SPAN"  # 语料数据格式, "DATA-CONLL", "DATA-SPAN", 如果有.conll/.span格式就不需要指定, 否则需要指定
    # model_config["loss_type"] = "FOCAL_LOSS"
    model_config["loss_type"] = "CIRCLE_LOSS"  # 因为0比较多所以必须使用circle_loss, 否则不收敛
    # 损失函数类型, 可选 None(BCE), BCE, MSE, FOCAL_LOSS,
    # multi-label:  MARGIN_LOSS, PRIOR_MARGIN_LOSS, CIRCLE_LOSS等
    # 备注: "SL-GRID"类型不要用BCE、PRIOR_MARGIN_LOSS
    model_config["flag_dynamic_encode"] = False  # False  # True
    model_config["padding_side"] = "RIGHT"  # "LEFT"  # "RIGHT"
    model_config["grad_accum_steps"] = 1
    model_config["warmup_steps"] = 4  # 1024  # 0.01  # 预热步数, -1为取 0.5 的epoch步数
    model_config["batch_size"] = 32
    model_config["max_len"] = 128
    model_config["epochs"] = 12

    model_config["xy_keys"] = ["text", "label"]  # SPAN格式的数据, text, label在file中对应的keys
    model_config["sl_ctype"] = "BIO"  # 数据格式sl-type, BIO, BMES, BIOES, BO, 只在"corpus_type": "MYX", "task_type": "SL-CRL"或"SL-SOFTMAX"时候生效
    model_config["dense_lr"] = 5e-5  # CRF层学习率/全连接层学习率, 1e-5, 1e-4, 1e-3
    model_config["lr"] = 5e-5  # 学习率, 1e-5, 2e-5, 5e-5, 8e-5, 1e-4, 4e-4
    model_config["num_workers"] = 0

    idx = 1  # 0   # 选择的预训练模型类型---model_type, 0为BERT,
    model_config["pretrained_model_name_or_path"] = pretrained_model_name_or_path[model_type[idx]]
    # model_config["model_save_path"] = "../output/sequence_labeling/model_{}".format(model_type[idx] + "_" + str(get_current_time()))
    model_config["model_save_path"] = "../output/sequence_labeling/model_{}_{}_lr-{}_bs-{}_epoch-{}"\
        .format(task_version, model_type[idx], model_config["lr"],
                model_config["batch_size"], model_config["epochs"])
    model_config["model_type"] = model_type[idx]
    logger = get_logger(os.path.join(model_config["model_save_path"], "log"))

    # data
    cdl = CreateDataLoader(model_config, logger=logger)
    train_data_loader, dev_data_loader, tet_data_loader, \
                tokenizer, data_config = cdl.create_for_train(model_config)
    # train
    office = Office(data_config, tokenizer, logger=logger)
    office.train(train_data_loader, dev_data_loader)
    office.evaluate(dev_data_loader)
    office.evaluate(tet_data_loader)


"""
# shell
# nohup python slRun.py > tc.slRun.py.log 2>&1 &
# tail -n 1000  -f tc.slRun.py.log
# |myz|
"""

