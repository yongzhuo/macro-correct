# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/11/17 21:35
# @author  : Mo
# @function: graph of pre-train model


# torch
from transformers import BertPreTrainedModel
import torch

from macro_correct.pytorch_textcorrection.tcLayer import PriorMultiLabelSoftMarginLoss, MultiLabelCircleLoss, LabelSmoothingCrossEntropy
from macro_correct.pytorch_textcorrection.tcLayer import FCLayer, FocalLoss, DiceLoss
from macro_correct.pytorch_textcorrection.tcConfig import PRETRAINED_MODEL_CLASSES
from macro_correct.pytorch_textcorrection.tcLayer import FCLayer


class Graph(BertPreTrainedModel):
    def __init__(self, graph_config, tokenizer):
        """
        Pytorch Graph of TextClassification, Pre-Trained Model based
        config:
            config: json, params of graph, eg. {"num_labels":17, "model_type":"BERT"}
        Returns:
            output: Tuple, Tensor of logits and loss
        """
        # 预训练语言模型读取
        self.graph_config = graph_config
        pretrained_config, pretrained_tokenizer, pretrained_model = PRETRAINED_MODEL_CLASSES[graph_config.model_type]
        self.pretrained_config = pretrained_config.from_pretrained(graph_config.pretrained_model_name_or_path, output_hidden_states=graph_config.output_hidden_states)
        self.pretrained_config.update({"gradient_checkpointing": True})
        super(Graph, self).__init__(self.pretrained_config)
        if self.graph_config.flag_train:
            self.bert = pretrained_model.from_pretrained(graph_config.pretrained_model_name_or_path, config=self.pretrained_config)
            self.bert.resize_token_embeddings(len(tokenizer))
        else:
            self.bert = pretrained_model(self.pretrained_config)  # 推理时候只需要加载超参数, 不需要预训练模型的权重
            self.bert.resize_token_embeddings(len(tokenizer))

        self.dense = FCLayer(self.pretrained_config.hidden_size, self.graph_config.num_labels, flag_dropout=self.graph_config.flag_dropout,
                             flag_active=self.graph_config.flag_active, active_type=self.graph_config.active_type)

        self.detection = FCLayer(self.pretrained_config.hidden_size, 1, flag_dropout=False,
                                 flag_active=False, active_type=self.graph_config.active_type)
        # 池化层
        self.global_maxpooling = torch.nn.AdaptiveMaxPool1d(1)
        self.global_avgpooling = torch.nn.AdaptiveAvgPool1d(1)
        # 激活层/随即失活层
        self.softmax = torch.nn.Softmax()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout
        # 损失函数, loss
        self.loss_pmlsm = PriorMultiLabelSoftMarginLoss(prior=self.graph_config.prior, num_labels=self.graph_config.num_labels)
        self.loss_type = self.graph_config.loss_type if self.graph_config.loss_type else "BCE"
        self.loss_ce = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.loss_mlsm = torch.nn.MultiLabelSoftMarginLoss()  # like BCEWithLogitsLoss
        self.loss_bcelog = torch.nn.BCEWithLogitsLoss()
        self.loss_bce = torch.nn.BCELoss()
        self.loss_mse = torch.nn.MSELoss()

        self.loss_focal = FocalLoss(activation_type="sigmoid")
        self.loss_lsce = LabelSmoothingCrossEntropy()
        self.loss_circle = MultiLabelCircleLoss()
        self.loss_dice = DiceLoss()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                text_labels=None, det_labels=None):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids, labels=text_labels,
                                 output_hidden_states=True,
                                 return_dict=True,
                                 )
        prob = None
        if text_labels is None:
            # 检错输出，纠错输出
            outputs = (prob, bert_outputs.logits)
        else:
            # 检错概率
            prob = self.detection(bert_outputs.hidden_states[-1])
            # det_loss_fct = FocalLoss(num_labels=None, activation_type='sigmoid')
            # pad部分不计算损失
            active_loss = attention_mask.view(-1, prob.shape[1]) == 1
            active_probs = prob.view(-1, prob.shape[1])[active_loss]
            active_labels = det_labels[active_loss].float()
            # det_loss = det_loss_fct(active_probs, active_labels.float())
            if self.loss_type.upper() == "PRIOR_MARGIN_LOSS":  # 带先验的边缘损失
                det_loss = self.loss_pmlsm(active_probs, active_labels)
            elif self.loss_type.upper() == "SOFT_MARGIN_LOSS": # 边缘损失pytorch版, 划分距离
                det_loss = self.loss_mlsm(active_probs, active_labels)
            elif self.loss_type.upper() == "FOCAL_LOSS":       # 聚焦损失(学习难样本, 2-0.25, 负样本多的情况)
                det_loss = self.loss_focal(active_probs, active_labels)
            elif self.loss_type.upper() == "CIRCLE_LOSS":      # 圆形损失(均衡, 统一 triplet-loss 和 softmax-ce-loss)
                det_loss = self.loss_circle(active_probs, active_labels)
            elif self.loss_type.upper() == "DICE_LOSS":        # 切块损失(图像)
                det_loss = self.loss_dice(active_probs, active_labels.long())
            elif self.loss_type.upper() == "LABEL_SMOOTH":     # 交叉熵平滑
                det_loss = self.loss_lsce(active_probs, active_labels.long())
            elif self.loss_type.upper() == "BCE_LOGITS":       # 二元交叉熵平滑连续计算型pytorch版
                det_loss = self.loss_bcelog(active_probs, active_labels)
            elif self.loss_type.upper() == "BCE_MULTI":        # 二元交叉熵的pytorch版, 多标签
                active_probs_sigmoid = self.sigmoid(active_probs)
                det_loss = self.loss_bce(active_probs_sigmoid, active_labels)
            elif self.loss_type.upper() == "MSE":              # 均方误差
                det_loss = self.loss_mse(active_probs, active_labels)
            else:  # 二元交叉熵的pytorch版, 多类
                # active_probs_softmax = self.softmax(active_probs)
                # det_loss = self.loss_bce(active_probs_softmax, active_labels)
                active_probs_sigmoid = self.sigmoid(active_probs)
                det_loss = self.loss_bce(active_probs_sigmoid, active_labels)
            # 检错loss，纠错loss，检错输出，纠错输出
            outputs = (det_loss,
                       bert_outputs.loss,
                       self.sigmoid(prob).squeeze(-1),
                       bert_outputs.logits)
        return outputs

