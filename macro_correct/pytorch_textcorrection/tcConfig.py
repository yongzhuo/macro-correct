# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/11/19 21:48
# @author  : Mo
# @function: config of transformers and graph-model


_TC_MULTI_CLASS = "TC-MULTI-CLASS"
_TC_MULTI_LABEL = "TC-MULTI-LABEL"


# model算法超参数
model_config = {
    "CUDA_VISIBLE_DEVICES": "0",  # 环境, GPU-CPU, "-1"/"0"/"1"/"2"...
    "USE_TORCH": "1",             # transformers使用torch, 因为脚本是torch写的
    "output_hidden_states": None,  # [6,11]  # 输出层, 即取第几层transformer的隐藏输出, list
    "pretrained_model_name_or_path": "",  # 预训练模型地址
    "model_save_path": "output",  # 训练模型保存-训练完毕模型目录
    "config_name": "csc.config",  # 训练模型保存-超参数文件名
    "model_name": "pytorch_model.bin",  # 训练模型保存-全量模型
    "path_train": None,  # 验证语料地址, 必传, string
    "path_dev": None,  # 验证语料地址, 必传, 可为None
    "path_tet": None,  # 验证语料地址, 必传, 可为None

    "scheduler_name": "cosine",  # "linear", "cosine", "cosine_with_restarts",
                                 # "polynomial", "constant", "constant_with_warmup",
                                 # "inverse_sqrt", "reduce_lr_on_plateau"
    "tokenizer_type": "CHAR",  # "BASE",  # tokenizer解析的类型, 默认transformers自带的, 可设"CHAR"(单个字符的, 不使用bpe等词根的)
    "padding_side": "RIGHT",  # tokenizer, 左右padding, LEFT, RIGHT
    "active_type": "RELU",  # 最后几层输出使用的激活函数, 可填写RELU/SIGMOID/TANH/MISH/SWISH/GELU
    "task_type": "CSC",  # 任务类型, 依据数据类型自动更新
    "model_type": "BERT",  # 预训练模型类型, 如bert, roberta, ernie
    "loss_type": "BCE",  # 损失函数类型, 可选 None(BCE), BCE, MSE, FOCAL_LOSS,
                         # multi-label:  MARGIN_LOSS, PRIOR_MARGIN_LOSS, CIRCLE_LOSS等
                         # 备注: "SL-GRID"类型不要用BCE、PRIOR_MARGIN_LOSS

    "loss_det_rate": 0.3,  # det-loss的权重
    "max_len_limit": 512,  # 最大文本长度, 一般为510的bert-base的水准
    "batch_size": 32,  # 批尺寸
    "num_labels": 0,  # 类别数, 自动更新
    "max_len": 128,  # 最大文本长度(不超过512), -1则为自动获取覆盖0.95数据的文本长度, 0为取得最大文本长度作为maxlen
    "epochs": 21,  # 训练轮次
    "lr": 2e-5,    # 学习率

    "grad_accum_steps": 1,  # 梯度积累多少步
    "max_grad_norm": 1.0,  # 最大标准化梯度
    "weight_decay": 1e-2,  # 模型参数l2正则化, 1e-2/5e-4/1e-4
    "dropout_rate": 0.1,  # 随即失活概率
    "adam_eps": 1e-8,  # adam优化器超参
    "seed": 42,  # 随机种子, 3407, 2021, 2023

    "evaluate_steps": 320,  # 评估步数
    "warmup_steps": -1,  # 预热步数, -1为取 0.1 的一个epoch步数
    "ignore_index": 0,  # 忽略的index, mask部分
    "save_steps": 320,  # 存储步数
    "stop_epochs": 4,  # 连续N轮无增长早停轮次
    "num_workers": 0,  # data_loader加载的进程数, 加快数据处理
    "max_steps": -1,  # 最大步数, -1表示取满epochs

    "flag_save_model_state": True,  # 是否只保存model_state, False为网络结构也保存
    "flag_dynamic_encode": False,  # 是否在一个batch中使用动态文本长度编码, 否则就使用max_len
    "flag_tokenizer_char": True,  # one-char2one-char, 不适用bpe
    "flag_soft_label": True,  # 是否使用软标签, soft-label
    "flag_save_best": True,  # 只保留最优模型(一个), False为需要保留每一次优化的模型
    "flag_dropout": False,  # 最后几层输出是否使用随即失活
    "flag_shuffle": True,  # 数据集是否shufle, 同时用于data_的shuffle
    "flag_active": False,  # 最后几层输出是否使用激活函数, 如FCLayer/SpanLayer层
    "flag_train": True,  # 是否训练, 另外一个人不是(而是预测)
    "flag_cuda": True,  # 是否使用gpu, 另外一个不是gpu(而是cpu)
    "flag_mft": True,   # 训练时是否使用动态mask非错误区域
    "flag_adv": False,  # 是否使用对抗训练(默认FGM)

    "xy_keys_predict": ["original_text", "correct_text", "wrong_ids"],  # text,label在file中对应的keys
    "keys": ["original_text", "correct_text", "wrong_ids"],  # text,label在file中对应的keys
    "save_best_mertics_key": ["sentence", "strict_cor_f1"],  # 模型存储的判别指标,
    "label_sep": "|myz|",  # "|myz|" 多标签数据分割符, 用于多标签分类语料中
    "multi_label_threshold": 0.5,  # 多标签分类时候生效, 大于该阈值则认为预测对的
    "len_rate": 1,  # 训练数据和验证数据占比, float, 0-1闭区间

    # 对抗学习
    "adv_emb_name": "word_embeddings.",  # emb_name这个参数要换成你模型中embedding的参数名, model.embeddings.word_embeddings.weight
    "adv_eps": 1.0,  # 梯度权重epsilon

    "additional_special_tokens": [],  # 新增特殊字符
    "len_corpus": None,  # 训练语料长度
    "prior_count": None,  # 每个类别样本频次
    "prior": None,  # 类别先验分布, 自动设置, 为一个label_num类别数个元素的list, json无法保存np.array
    # "l2i": None,
    # "i2l": None,
}


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = model_config.get("CUDA_VISIBLE_DEVICES", "2")
os.environ["USE_TORCH"] = model_config.get("USE_TORCH", "1")
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer, XLNetTokenizer, ElectraTokenizer, XLMTokenizer, AutoTokenizer
from transformers import ErnieConfig, BertConfig, RobertaConfig, AlbertConfig, XLNetConfig, ElectraConfig, XLMConfig, AutoConfig
from transformers import ErnieForMaskedLM, BertForMaskedLM, RobertaForMaskedLM, AlbertModel, XLNetModel, ElectraModel, XLMModel, AutoModel, AutoModelForMaskedLM
# from transformers import LongformerTokenizer, LongformerConfig, LongformerModel
from transformers import DebertaTokenizer, DebertaConfig, DebertaModel
from transformers import GPT2Tokenizer, GPT2Config, GPT2Model
from transformers import T5Tokenizer, T5Config, T5Model


# transformers类等from transformers import BertTokenizerFast, BertForMaskedLM
PRETRAINED_MODEL_CLASSES = {
    # "LONGFORMER": (LongformerConfig, LongformerTokenizer, LongformerModel),
    "ROBERTA": (RobertaConfig, RobertaTokenizer, RobertaForMaskedLM),  # (RobertaConfig, RobertaTokenizer, RobertaModel),  #
    "MACBERT": (BertConfig, BertTokenizer, BertForMaskedLM),
    "ERNIE3": (ErnieConfig, BertTokenizer, ErnieForMaskedLM),
    "ERNIE": (BertConfig, BertTokenizer, BertForMaskedLM),
    "NEZHA": (BertConfig, BertTokenizer, BertForMaskedLM),
    "BERT": (BertConfig, BertTokenizer, BertForMaskedLM),
    "AUTO": (AutoConfig, AutoTokenizer, AutoModel),
}

