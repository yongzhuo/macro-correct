# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/1/18 21:34
# @author  : Mo
# @function: relm
# @paper   : [Chinese Spelling Correction as Rephrasing Language Model](https://arxiv.org/abs/2308.08796).
# @code    : most code copy from https://github.com/Claude-Liu/ReLM


from __future__ import absolute_import, division, print_function
import random
import json
import sys
import os

from transformers import BertPreTrainedModel
from transformers import BertForMaskedLM
from transformers import BertConfig
import torch.nn as nn
import torch

from macro_correct.pytorch_textcorrection.tcLayer import MultiLabelCircleLoss, LabelSmoothingCrossEntropy
from macro_correct.pytorch_textcorrection.tcLayer import FocalLoss, DiceLoss


class RELM(BertPreTrainedModel):
    def __init__(self, config):
        # super(ECOPO, self).__init__()
        self.config = BertConfig.from_pretrained(pretrained_model_name_or_path=config.pretrained_model_name_or_path)
        super(RELM, self).__init__(self.config)
        self.csc_config = config
        if config.flag_train:
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
        self.post_init()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            labels=labels, output_hidden_states=True, return_dict=True, )
        ### pred
        cor_probs = self.softmax(outputs.logits)
        if labels is not None:
            pred_prob, pred_ids = torch.max(cor_probs, -1)
            pred_ids = pred_ids.masked_fill(attention_mask == 0, 0)
            ### det
            prob = self.detect(outputs.hidden_states[-1])
            det_labels = (input_ids != labels).float()
            det_probs = prob.squeeze(-1) * attention_mask
            ### cor
            cor_loss = outputs.loss
            # ### cpo
            # cpo_loss = self.loss_cpo(outputs.logits, labels, mask=(input_ids != labels))
            # loss
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

            # loss = cor_loss + cpo_loss
            loss = (1-self.csc_config.loss_det_rate) * cor_loss + self.csc_config.loss_det_rate * det_loss
            # loss = cor_loss
            # print(cor_loss)
            # print(det_loss)
            output = (loss, det_loss, cor_loss, cor_loss,
                       det_probs, cor_probs, pred_ids,)
        else:
            pred_prob, pred_ids = torch.max(cor_probs, -1)
            pred_ids = pred_ids.masked_fill(attention_mask == 0, 0)
            output = (cor_probs, pred_prob, pred_ids)
        return output


class RelmPtuning(nn.Module):
    def __init__(self, path_pretrain_model_dir, prompt_length=1, flag_cuda=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() and flag_cuda else "cpu")
        model = BertForMaskedLM.from_pretrained(path_pretrain_model_dir, return_dict=True)
        model.to(self.device)

        self.prompt_length = prompt_length
        self.config = model.config

        self.model_type = self.config.model_type.split("-")[0]
        self.word_embeddings = getattr(model, self.model_type).embeddings.word_embeddings
        self.model = model

        self.hidden_size = self.config.embedding_size if hasattr(self.config, "embedding_size") else self.config.hidden_size
        self.prompt_embeddings = nn.Embedding(2*self.prompt_length, self.hidden_size)
        self.prompt_lstm = nn.LSTM(input_size=self.hidden_size,  hidden_size=self.hidden_size,
                                   num_layers=2, bidirectional=True, batch_first=True)
        self.prompt_linear = nn.Sequential(nn.Linear(2 * self.hidden_size, self.hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_size, self.hidden_size))

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        token_type_ids=None,
        prompt_mask=None, ##(batch,msl)
        labels=None,
        apply_prompt=False,
    ):
        if inputs_embeds==None:
            inputs_embeds = self.word_embeddings(input_ids)##inputs_embeds(batch,seq,hidden)
        if apply_prompt:
            replace_embeds = self.prompt_embeddings(torch.LongTensor(list(range(2*self.prompt_length))).to(input_ids.device))
            replace_embeds = replace_embeds.unsqueeze(0)##(1,2*prompt_length,hidden_size)
            replace_embeds = self.prompt_lstm(replace_embeds)[0]##(2*prompt_length,2*hidden_size)
            replace_embeds = self.prompt_linear(replace_embeds).squeeze()##(2*prompt_length,hidden_size)
            blocked_indices = (prompt_mask == 1).nonzero().reshape((input_ids.shape[0], 2*self.prompt_length, 2))[:, :, 1]##indices of the prompts p, 
            for i in range(input_ids.shape[0]):
                for j in range(blocked_indices.shape[1]):
                    inputs_embeds[i, blocked_indices[i, j], :] = replace_embeds[j, :]

        outputs = self.model(
            inputs_embeds=inputs_embeds,  ##take inputs_embeds as input instead of inputs_ids
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        return outputs


