# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/1/18 21:34
# @author  : Mo
# @function: Macbert4CSC
# @github   : [Macbert4CSC](https://github.com/shibing624/pycorrector/tree/master/examples/macbert)
# @paper   :  [SoftMaskedBert4CSC](https://arxiv.org/abs/2004.13922)


from __future__ import absolute_import, division, print_function
from copy import deepcopy
import argparse
import logging
import random
import math
import sys
import os

from transformers import BertConfig, BertModel
from transformers import BertPreTrainedModel
from transformers import BertForMaskedLM

import torch.nn as nn
import torch

from macro_correct.pytorch_textcorrection.tcLayer import MultiLabelCircleLoss, LabelSmoothingCrossEntropy
from macro_correct.pytorch_textcorrection.tcLayer import FocalLoss, DiceLoss


class Macbert4CSC(BertPreTrainedModel):
    def __init__(self, config):
        self.config = BertConfig.from_pretrained(pretrained_model_name_or_path=config.pretrained_model_name_or_path)
        super().__init__(self.config)
        self.csc_config = config
        if self.csc_config.flag_train:
            self.config.update({"gradient_checkpointing": True})
            self.bert = BertForMaskedLM.from_pretrained(config.pretrained_model_name_or_path, config=self.config)
        else:
            self.bert = BertForMaskedLM(self.config)
        # detect
        self.detect = nn.Linear(self.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        # 损失函数, loss
        self.flag_cpo_loss = self.csc_config.flag_cpo_loss  # 是否使用cpo_loss
        self.loss_type = self.csc_config.loss_type if self.csc_config.loss_type else "BCE_MULTI"
        self.loss_ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_mlsm = torch.nn.MultiLabelSoftMarginLoss()  # like BCEWithLogitsLoss
        self.loss_bcelog = torch.nn.BCEWithLogitsLoss()
        self.loss_bce = torch.nn.BCELoss()
        self.loss_mse = torch.nn.MSELoss()
        self.loss_focal = FocalLoss(activation_type="sigmoid")
        self.loss_lsce = LabelSmoothingCrossEntropy()
        self.loss_circle = MultiLabelCircleLoss()
        self.loss_dice = DiceLoss()
        # self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            labels=labels, output_hidden_states=True, return_dict=True, )
        if labels is not None:
            ### cor
            cor_loss = outputs.loss
            cor_probs = self.softmax(outputs.logits)
            pred_prob, pred_ids = torch.max(cor_probs, -1)
            pred_ids = pred_ids.masked_fill(attention_mask == 0, 0)
            ### det
            det_probs = self.detect(outputs.hidden_states[-1])
            det_labels = (input_ids != labels).float()
            det_probs = det_probs.squeeze(-1) * attention_mask

            if self.loss_type.upper() == "SOFT_MARGIN_LOSS":  # 边缘损失pytorch版, 划分距离
                det_loss = self.loss_mlsm(det_probs, det_labels)
            elif self.loss_type.upper() == "FOCAL_LOSS":       # 聚焦损失(学习难样本, 2-0.25, 负样本多的情况)
                det_loss = self.loss_focal(det_probs, det_labels)
            elif self.loss_type.upper() == "CIRCLE_LOSS":      # 圆形损失(均衡, 统一 triplet-loss 和 softmax-ce-loss)
                det_loss = self.loss_circle(det_probs, det_labels)
            elif self.loss_type.upper() == "DICE_LOSS":        # 切块损失(图像)
                det_loss = self.loss_dice(det_probs, det_labels.long())
            elif self.loss_type.upper() == "LABEL_SMOOTH":     # 交叉熵平滑
                det_loss = self.loss_lsce(det_probs, det_labels.long())
            elif self.loss_type.upper() == "BCE_LOGITS":       # 二元交叉熵平滑连续计算型pytorch版
                det_loss = self.loss_bcelog(det_probs, det_labels)
            elif self.loss_type.upper() == "BCE_MULTI":        # 二元交叉熵的pytorch版, 多标签
                active_probs_sigmoid = self.sigmoid(det_probs)
                det_loss = self.loss_bce(active_probs_sigmoid, det_labels)
            elif self.loss_type.upper() == "MSE":              # 均方误差
                det_loss = self.loss_mse(det_probs, det_labels)
            else:  # 二元交叉熵的pytorch版, 多标签(BCE
                active_probs_sigmoid = self.sigmoid(det_probs)
                det_loss = self.loss_bce(active_probs_sigmoid, det_labels)

            loss = (1-self.csc_config.loss_det_rate) * cor_loss + self.csc_config.loss_det_rate * det_loss
            # print(cor_loss)
            # print(det_loss)
            # print(loss)
            # print("#"*128)
            outputs = (loss,
                       det_loss,
                       cor_loss,
                       cor_loss,
                       det_probs,
                       cor_probs,
                       pred_ids,
                       )
        else:
            # det
            cor_probs = self.softmax(outputs.logits)
            pred_prob, pred_ids = torch.max(cor_probs, -1)
            pred_ids = pred_ids.masked_fill(attention_mask == 0, 0)
            outputs = (cor_probs, pred_prob, pred_ids)
        return outputs

