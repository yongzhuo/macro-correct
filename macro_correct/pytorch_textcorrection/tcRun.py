# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/23 21:34
# @author  : Mo
# @function: main programing, "训练时候logger不需要考虑"


# 适配linux
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
from macro_correct.pytorch_textcorrection.tcTools import chinese_extract_extend, get_logger
from macro_correct.pytorch_textcorrection.tcData import TextCorrectionDataCollator
from macro_correct.pytorch_textcorrection.tcData import TextCorrectionDataset
from macro_correct.pytorch_textcorrection.tcData import CreateDataLoader
from macro_correct.pytorch_textcorrection.tcConfig import model_config
from macro_correct.pytorch_textcorrection.tcTqdm import tqdm, trange
from macro_correct.pytorch_textcorrection.tcAdversarial import FGM
from macro_correct.pytorch_textcorrection.tcOffice import Office
from macro_correct.pytorch_textcorrection.tcGraph import Graph


# 预训练模型地址, 本地win10默认只跑2步就评估保存模型
if platform.system().lower() == "windows":
    pretrained_model_dir = "E:/DATA/bert-model/00_pytorch"
    path_corpus_dir = os.path.join(path_root, "macro_correct", "corpus", "text_correction")
    evaluate_steps = 320  # 评估步数
    save_steps = 320  # 存储步数
else:
    pretrained_model_dir = "/pretrain_models/pytorch"
    path_corpus_dir = os.path.join(path_root, "macro_correct", "corpus", "text_correction")
    evaluate_steps = 3000  # 评估步数
    save_steps = 3000  # 存储步数
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
    # "BERT": pretrained_model_dir + "/bert-base-chinese",
    "BERT": pretrained_model_dir + "/LLM/hfl_chinese-macbert-base",
    # "BERT": pretrained_model_dir + "/uer_roberta-base-word-chinese-cluecorpussmall",
    # "BERT": pretrained_model_dir + "/bert-base-chinese",
    # "BERT": pretrained_model_dir + "/uer_roberta-base-wwm-chinese-cluecorpussmall",
    "ELECTRA": pretrained_model_dir + "/hfl_chinese-electra-180g-base-discriminator",
}


if __name__ == '__main__':


    # task_version = "csc_sighanall_det3_nomft"
    task_version = "csc_sighanall_det3_mft_v2"
    # task_version = "csc_sighanall_det3_mft"
    # task_version = "csc_sighanall_det15_mft"

    path_train = os.path.join(path_corpus_dir, "sighan", "sighan2015.train.json")
    path_dev = os.path.join(path_corpus_dir, "sighan", "sighan2015.dev.json")
    path_tet = os.path.join(path_corpus_dir, "sighan", "sighan2015.dev.json")

    # task_version = "espell_law"
    # path_train = os.path.join(path_corpus_dir, "espell", "csc_espell_law.train")
    # path_dev = os.path.join(path_corpus_dir, "espell", "csc_espell_law.test")
    # path_tet = os.path.join(path_corpus_dir, "espell", "csc_espell_law.test")

    model_config["evaluate_steps"] = evaluate_steps  # 评估步数
    model_config["save_steps"] = save_steps  # 存储步数
    model_config["path_train"] = path_train  # 训练模语料, 必须
    model_config["path_dev"] = path_dev  # 验证语料, 可为None
    model_config["path_tet"] = path_tet  # 测试语料, 可为None
    # model_config["CUDA_VISIBLE_DEVICES"] = "0"
    model_config["loss_type"] = "BCE"
    # model_config["loss_type"] = "FOCAL_LOSS"
    # model_config["loss_type"] = "CIRCLE_LOSS"  # 因为0比较多所以必须使用circle_loss, 否则不收敛
    # 损失函数类型, 可选 None(BCE), BCE, MSE, FOCAL_LOSS,
    # multi-label:  MARGIN_LOSS, PRIOR_MARGIN_LOSS, CIRCLE_LOSS等
    # 备注: "SL-GRID"类型不要用BCE、PRIOR_MARGIN_LOSS
    model_config["flag_dynamic_encode"] = False  # False  # True
    model_config["scheduler_name"] = "linear"  # "linear", "cosine"
    model_config["padding_side"] = "RIGHT"  # "LEFT"  # "RIGHT"
    model_config["flag_dropout"] = False  # fc-dropout
    model_config["flag_active"] = False  # fc-active-mish
    model_config["flag_mft"] = True  # fc-active-mish
    model_config["grad_accum_steps"] = 1
    model_config["loss_det_rate"] = 0.3  # 0.15 # 0.3  # 0.7  # det-loss-rate
    model_config["warmup_steps"] = 0.1  # 3  # 1024  # 0.01  # 预热步数, -1为取 0.5 的epoch步数
    model_config["num_workers"] = 0
    model_config["batch_size"] = 16
    model_config["max_len"] = 128
    model_config["epochs"] = 50

    model_config["xy_keys"] = ["original_text", "correct_text", "wrong_ids"]  # keys
    # model_config["dense_lr"] = 5e-5  # CRF层学习率/全连接层学习率, 1e-5, 1e-4, 1e-3
    # model_config["lr"] = 5e-5  # 学习率, 1e-5, 2e-5, 5e-5, 8e-5, 1e-4, 4e-4
    # model_config["dense_lr"] = 2e-5  # CRF层学习率/全连接层学习率, 1e-5, 1e-4, 1e-3
    model_config["lr"] = 3e-5  # 学习率, 1e-5, 2e-5, 5e-5, 8e-5, 1e-4, 4e-4

    idx = 0  # 1  # 0   # 选择的预训练模型类型---model_type, 0为BERT,
    model_config["pretrained_model_name_or_path"] = pretrained_model_name_or_path[model_type[idx]]
    # model_config["model_save_path"] = "../output/text_correction/model_{}".format(model_type[idx] + "_" + str(get_current_time()))
    model_config["model_save_path"] = "../output/text_correction/model_public_{}_{}_lr-{}_bs-{}_epoch-{}"\
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


# shell
# nohup python  tcRun.py > tc.tcRun.py.log 2>&1 &
# tail -n 1000  -f tc.tcRun.py.log
# |myz|

"""
model_public_csc_sighanall_BERT_lr-2e-05_bs-16_epoch-32--det0.7
sentence-eval_loss：3.191731905564666
sentence-common_det_acc：0.7854545454545454
sentence-common_det_precision：0.8509174311926605
sentence-common_det_recall：0.6845018450184502
sentence-common_det_f1：0.7586912065439673
sentence-common_cor_acc：0.7136363636363636
sentence-common_cor_precision：0.8179271708683473
sentence-common_cor_recall：0.5387453874538746
sentence-common_cor_f1：0.649610678531702
sentence-strict_det_acc：0.7854545454545454
sentence-strict_det_precision：0.7053231939163498
sentence-strict_det_recall：0.6845018450184502
sentence-strict_det_f1：0.6947565543071162
sentence-strict_cor_acc：0.7136363636363636
sentence-strict_cor_precision：0.5551330798479087
sentence-strict_cor_recall：0.5387453874538746
sentence-strict_cor_f1：0.5468164794007492
token-det_precision：0.7627840909090909
token-det_recall：0.8136363636363636
token-det_f1：0.7873900293255132
token-cor_precision：0.750465549348231
token-cor_recall：0.696027633851468
token-cor_f1：0.7222222222222222
val-det_loss：0.08761642128229141
val-mlm_loss：0.0380692295730114
val-all_loss：0.04625698413861835

model_public_csc_sighanall_det0.3_nomft_BERT_lr-2e-05_bs-16_epoch-32---det0.3
sentence-eval_loss：2.717403970658779
sentence-common_det_acc：0.7727272727272727
sentence-common_det_precision：0.8705583756345178
sentence-common_det_recall：0.6328413284132841
sentence-common_det_f1：0.732905982905983
sentence-common_cor_acc：0.7018181818181818
sentence-common_cor_precision：0.8386075949367089
sentence-common_cor_recall：0.488929889298893
sentence-common_cor_f1：0.6177156177156177
sentence-strict_det_acc：0.7727272727272727
sentence-strict_det_precision：0.7130977130977131
sentence-strict_det_recall：0.6328413284132841
sentence-strict_det_f1：0.6705767350928641
sentence-strict_cor_acc：0.7018181818181818
sentence-strict_cor_precision：0.5509355509355509
sentence-strict_cor_recall：0.488929889298893
sentence-strict_cor_f1：0.5180840664711632
token-det_precision：0.6931818181818182
token-det_recall：0.8356164383561644
token-det_f1：0.7577639751552796
token-cor_precision：0.7295081967213115
token-cor_recall：0.6846153846153846
token-cor_f1：0.7063492063492064
val-det_loss：0.03736872971057892
val-mlm_loss：0.06033508852124214
val-all_loss：0.03938266624143158


model_public_csc_sighanall_det0.3_BERT_lr-2e-05_bs-16_epoch-16---det0.3
sentence-eval_loss：1.7931031184270978
sentence-common_det_acc：0.7818181818181819
sentence-common_det_precision：0.8578199052132701
sentence-common_det_recall：0.6678966789667896
sentence-common_det_f1：0.7510373443983402
sentence-common_cor_acc：0.72
sentence-common_cor_precision：0.8305084745762712
sentence-common_cor_recall：0.5424354243542435
sentence-common_cor_f1：0.65625
sentence-strict_det_acc：0.7818181818181819
sentence-strict_det_precision：0.7084148727984344
sentence-strict_det_recall：0.6678966789667896
sentence-strict_det_f1：0.6875593542260208
sentence-strict_cor_acc：0.72
sentence-strict_cor_precision：0.5753424657534246
sentence-strict_cor_recall：0.5424354243542435
sentence-strict_cor_f1：0.5584045584045583
token-det_precision：0.7414772727272727
token-det_recall：0.8130841121495327
token-det_f1：0.775631500742942
token-cor_precision：0.7681992337164751
token-cor_recall：0.7097345132743362
token-cor_f1：0.7378104875804967
val-det_loss：0.07103697210550308
val-mlm_loss：0.02858821116387844
val-all_loss：0.02598700171633475

"""


