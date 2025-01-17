# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/7/8 20:42
# @author  : Mo
# @function: config of sequence-labeling, 超参数/类


# 模型类型标识符
_SL_MODEL_SOFTMAX = "SL-SOFTMAX"
_SL_MODEL_GRID = "SL-GRID"  # 网格, 即矩阵, Global-Pointer
_SL_MODEL_SPAN = "SL-SPAN"
_SL_MODEL_CRF = "SL-CRF"
# 数据类型标识符
_SL_DATA_CONLL = "DATA-CONLL"  # conll
_SL_DATA_SPAN = "DATA-SPAN"  # span


# model算法超参数
model_config = {
    "CUDA_VISIBLE_DEVICES": "0",  # 环境, GPU-CPU, "-1"/"0"/"1"/"2"...
    "USE_TORCH": "1",             # transformers使用torch, 因为脚本是torch写的
    "output_hidden_states": None,  # 输出层, 即取第几层transformer的隐藏输出, list, eg. [6,11], None, [-1]
    "pretrained_model_name_or_path": "",  # 预训练模型地址
    "model_save_path": "model",  # 训练模型保存-训练完毕模型目录
    "config_name": "sl.config",  # 训练模型保存-超参数文件名
    "model_name": "pytorch_model.bin",  # 训练模型保存-全量模型

    "path_train": None,  # 验证语料地址, 必传, string
    "path_dev": None,    # 验证语料地址, 必传, 可为None
    "path_tet": None,    # 验证语料地址, 必传, 可为None

    "scheduler_name": "linear",  # "linear", "cosine", "cosine_with_restarts",
                                 # "polynomial", "constant", "constant_with_warmup",
                                 #  "inverse_sqrt", "reduce_lr_on_plateau"
    "loss_type": "CIRCLE_LOSS",  # 损失函数类型, 可选 None(BCE), BCE, MSE, FOCAL_LOSS,
    # multi-label:  MARGIN_LOSS, PRIOR_MARGIN_LOSS, CIRCLE_LOSS等
    # 备注: "SL-GRID"类型不要用BCE、PRIOR_MARGIN_LOSS
    "corpus_type": "DATA-SPAN",  # 语料数据格式, "DATA-CONLL", "DATA-SPAN"
    "padding_side": "RIGHT",  # tokenizer, 左右padding, LEFT, RIGHT
    "active_type": "GELU",  # 最后几层输出使用的激活函数, 可填写RELU/SIGMOID/TANH/MISH/SWISH/GELU
    "task_type": "SL-CRF",  # 任务类型, "SL-SOFTMAX", "SL-CRF", "SL-SPAN", "SL-GRID", "sequence_labeling"
    "model_type": "BERT",   # 预训练模型类型, 如BERT/ROBERTA/ERNIE/ELECTRA/ALBERT

    "max_len_limit": 512,  # 最大文本长度, 一般为510的bert-base的水准
    "batch_size": 32,  # 批尺寸
    "num_labels": 0,   # 类别数, 自动更新
    "dense_lr": 1e-5,  # CRF层学习率/全连接层学习率, CRF时候与lr保持100-1000倍的大小差距
    "max_len": 128,    # 最大文本长度, None和-1则为自动获取覆盖0.95数据的文本长度, 0则取训练语料的最大长度, 具体的数值就是强制padding到max_len
    "epochs": 16,      # 训练轮次
    "lr": 1e-5,        # 学习率

    "grad_accum_steps": 1,  # 梯度积累多少步
    "max_grad_norm": 1.0,  # 最大标准化梯度
    "weight_decay": 5e-4,  # lr学习率衰减系数
    "dropout_rate": 0.1,  # 随机失活概率
    "adam_eps": 1e-8,  # adam优化器超参
    "seed": 2023,  # 随机种子

    "evaluate_steps": 320,  # 评估步数
    "warmup_steps": -1,  # 预热步数, -1为取 0.5 的epoch步数
    "ignore_index": 0,  # 忽略的index, mask部分
    "save_steps": 320,  # 存储步数
    "stop_epochs": 4,  # 连续N轮无增长早停轮次
    "num_workers": 0,  # data_loader加载的进程数, 加快数据处理
    "max_steps": -1,  # 最大步数, -1表示取满epochs

    "flag_save_model_state": True,  # 是否只保存model_state, False为网络结构也保存
    "flag_dynamic_encode": True,  # 是否在一个batch中使用动态文本长度编码, 否则就使用max_len
    "flag_tokenizer_char": True,  # one-char2one-char, 不适用bpe
    "flag_soft_label": True,  # 是否使用软标签, soft-label
    "flag_save_best": True,  # 只保留最优模型(一个), False为需要保留每一次优化的模型
    "flag_dropout": True,  # 最后几层输出是否使用随即失活
    "flag_shuffle": True,  # 数据集是否shufle, 同时用于data_的shuffle
    "flag_active": True,  # 最后几层输出是否使用激活函数, 如FCLayer/SpanLayer层
    "flag_train": True,  # 是否训练, 另外一个人不是(而是预测)
    "flag_cuda": True,  # 是否使用gpu, 另外一个不是gpu(而是cpu)
    "flag_adv": False,  # 是否使用对抗训练(默认FGM)

    "save_best_mertics_key": ["micro_avg", "f1-score"],  # 模型存储的判别指标, index-1可选: [micro_avg, macro_avg, weighted_avg],
                                                                          # index-2可选: [precision, recall, f1-score]
    "multi_label_threshold": 0.5,  # 多标签分类时候生效, 大于该阈值则认为预测对的
    "grid_pointer_threshold": 0,  # 网格(全局)指针网络阈值, 大于该阈值则认为预测对的
    "xy_keys_predict": ["text", "label"],  # 读取数据的格式, predict预测的时候用
    # "xy_keys": ["text", "label"],  # SPAN格式的数据, text, label在file中对应的keys
    "xy_keys": [0, 1],     # CONLL格式的数据, text, label在file中对应的keys, conll时候选择[0,1]等integer
    "label_sep": "|myz|",  # "|myz|" 多标签数据分割符, 用于多标签分类语料中
    "sl_ctype": "BIO",   #  数据格式sl-type, BIO, BMES, BIOES, BO, 只在"corpus_type": "MYX", "task_type": "SL-CRL"或"SL-SOFTMAX"时候生效
    "head_size": 64,  # task_type=="SL-GRID"用

    # 是否对抗学习
    "adv_emb_name": "word_embeddings.",  # emb_name这个参数要换成你模型中embedding的参数名, model.embeddings.word_embeddings.weight
    "adv_eps": 1.0,  # 梯度权重epsilon

    "additional_special_tokens": [],  # 新增字典, 默认不新增; eg.["<macropodus>", "<macadam>"],
    "keys": ["text", "label"],
    "row_sep": " ",
    "prior": None,  # 类别先验分布, 自动设置, 为一个label_num类别数个元素的list, json无法保存np.array
    "l2i_conll": {},
    "l2i": {},
    "i2l": {},
}


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = model_config.get("CUDA_VISIBLE_DEVICES", "0")
os.environ["USE_TORCH"] = model_config.get("USE_TORCH", "1")
from transformers import BertTokenizer, RobertaTokenizer, AlbertTokenizer, XLNetTokenizer, ElectraTokenizer, XLMTokenizer, AutoTokenizer
from transformers import BertConfig, RobertaConfig, AlbertConfig, XLNetConfig, ElectraConfig, XLMConfig, AutoConfig
from transformers import BertModel, RobertaModel, AlbertModel, XLNetModel, ElectraModel, XLMModel, AutoModel
# from transformers import LongformerTokenizer, LongformerConfig, LongformerModel
from transformers import GPT2Tokenizer, GPT2Config, GPT2Model
from transformers import T5Tokenizer, T5Config, T5Model


# transformers类等
PRETRAINED_MODEL_CLASSES = {
    # "LONGFORMER": (LongformerConfig, LongformerTokenizer, LongformerModel),
    "ELECTRA": (ElectraConfig, ElectraTokenizer, ElectraModel),
    "ROBERTA": (RobertaConfig, RobertaTokenizer, RobertaModel),
    "ALBERT": (AlbertConfig, AlbertTokenizer, AlbertModel),
    "ERNIE3": (BertConfig, BertTokenizer, ElectraModel),
    "XLNET": (XLNetConfig, XLNetTokenizer, XLNetModel),
    "ERNIE": (BertConfig, BertTokenizer, BertModel),
    "NEZHA": (BertConfig, BertTokenizer, BertModel),
    "BERT": (BertConfig, BertTokenizer, BertModel),
    "GPT2": (GPT2Config, GPT2Tokenizer, GPT2Model),
    "AUTO": (AutoConfig, AutoTokenizer, AutoModel),
    "XLM": (XLMConfig, XLMTokenizer, XLMModel),
    "T5": (T5Config, T5Tokenizer, T5Model)
}

