# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/1/18 21:34
# @author  : Mo
# @function: deal corpus(data preprocess), label转成list形式, 或者span的形式


from collections import Counter
from argparse import Namespace
import logging as logger
import argparse
import random
import json
import os


from torch.utils.data import Dataset, DataLoader
import torch

from macro_correct.pytorch_sequencelabeling.slConfig import _SL_MODEL_SOFTMAX, _SL_MODEL_GRID, _SL_MODEL_SPAN, _SL_MODEL_CRF
from macro_correct.pytorch_sequencelabeling.slConfig import _SL_DATA_CONLL, _SL_DATA_SPAN
from macro_correct.pytorch_sequencelabeling.slConfig import PRETRAINED_MODEL_CLASSES
from macro_correct.pytorch_sequencelabeling.slTools import transform_span_to_conll
from macro_correct.pytorch_sequencelabeling.slTools import get_pos_from_common
from macro_correct.pytorch_sequencelabeling.slTqdm import tqdm


__all__ = [
    "SeqLabelingDataCollator",
    "SeqlabelingDataset",
    "CreateDataLoader",
    "create_data_loader",
]


class SeqlabelingDataset(Dataset):
    def __init__(self, path, config, flag_shuffle=True, logger=logger):
        self.logger = logger
        if type(config) == dict:
            self.config = argparse.Namespace(**config)
        else:
            self.config = config
        self.max_len_limit = self.config.max_len_limit  # BERT类语言模型的最大文本长度限制
        self.max_len = self.config.max_len  # 实际自定义的最大文本长度
        if path:  # 训练用
            self.data = self.read_corpus_from_list(path)
            if flag_shuffle:
                random.shuffle(self.data)

    def read_corpus_from_conll(self, path, keys=[0, 1], encoding="utf-8", row_sep=" "):
        """
        从定制化的标准txt文件中读取初始语料, read corpus from conll
        args:
            path: str, path of ner-corpus
            encoding: str, file encoding type, eg. "utf-8", "gbk"
            keys: list<int>, selected key-index of json, eg. [0,1], [1,3] 
            row_sep: str, str sep of split, eg. " ", "\t"
        returns:
            (xs, ys): tuple
        examples:
            中 B-CONT
            国 M-CONT
            国 M-CONT
            籍 E-CONT
            ， O
        """
        with open(path, "r", encoding=encoding) as fo:
            line_list = []
            len_maxs = []
            x, y = [], []
            count = 0
            for line in fo:
                count += 1
                # if count > 32:
                #     break
                line_sp = line.strip().split(row_sep)
                if len(line_sp) == 1:
                    line_json = {"label": y, "text": "".join(x)}
                    line_list.append(line_json)
                    len_maxs.append(len(x))
                    x, y = [], []
                else:
                    x.append(line_sp[keys[0]])
                    y.append(line_sp[keys[1]])
            fo.close()
            # 如果自动则选择覆盖0.95的长度, 即max_len为None, -1的情形
            len_maxs.sort()
            len_max_100 = len_maxs[-1]
            len_max_000 = len_maxs[0]
            len_max_95 = len_maxs[int(len(len_maxs) * 0.95)]
            len_max_90 = len_maxs[int(len(len_maxs) * 0.90)]
            len_max_60 = len_maxs[int(len(len_maxs) * 0.60)]
            len_max_50 = len_maxs[int(len(len_maxs) * 0.50)]
            self.logger.info("len_max_100: {}".format(len_max_100))
            self.logger.info("len_max_000: {}".format(len_max_000))
            self.logger.info("len_max_95: {}".format(len_max_95))
            self.logger.info("len_max_90: {}".format(len_max_90))
            self.logger.info("len_max_60: {}".format(len_max_60))
            self.logger.info("len_max_50: {}".format(len_max_50))
            if not self.config.max_len or self.config.max_len == -1:  # 自动化覆盖95%的文本长度
                self.max_len = min(len_max_95, 510)
            elif self.config.max_len == 0:  # 即max_len为0则是强制获取语料中的最大文本长度
                self.max_len = min(len_max_100, 510)
            else:
                self.max_len = min(self.max_len, 510)
            return line_list

    def read_corpus_from_span(self, path, keys=["text", "label"], encoding="utf-8"):
        """
        从定制化的标准jsonl文件中读取初始语料, read corpus from myz
        config:
            path_json: str, path of corpus
            encoding: str, file encoding type, eg. "utf-8", "gbk"
            keys: list, selected key of json
        Returns:
            (xs, ys): tuple
        """
        with open(path, "r", encoding=encoding) as fo:
            line_list = []
            len_maxs = []
            count = 0
            for line in fo:
                count += 1
                # if count > 32:
                #     break
                if not line:
                    continue
                #  最初想定义可配置化, 但是后期实验较多, 还是设置成一般形式, 可自己定义
                line_json = json.loads(line.strip())
                x, y = line_json.get(keys[0], ""), line_json.get(keys[1], [])
                len_maxs.append(len(x))
                line_list.append(line_json)
            fo.close()
            # 如果自动则选择覆盖0.95的长度, 即max_len为None, -1的情形
            len_maxs.sort()
            len_max_100 = len_maxs[-1]
            len_max_000 = len_maxs[0]
            len_max_95 = len_maxs[int(len(len_maxs) * 0.95)]
            len_max_90 = len_maxs[int(len(len_maxs) * 0.90)]
            len_max_50 = len_maxs[int(len(len_maxs) * 0.50)]
            self.logger.info("len_max_100: {}".format(len_max_100))
            self.logger.info("len_max_000: {}".format(len_max_000))
            self.logger.info("len_max_95: {}".format(len_max_95))
            self.logger.info("len_max_90: {}".format(len_max_90))
            self.logger.info("len_max_50: {}".format(len_max_50))
            if not self.config.max_len or self.config.max_len == -1:  # 自动化覆盖95%的文本长度
                self.max_len = min(len_max_95, 510)
            elif self.config.max_len == 0:  # 即max_len为0则是强制获取语料中的最大文本长度
                self.max_len = min(len_max_100, 510)
            else:
                self.max_len = min(self.max_len, 510)
            return line_list

    def read_corpus_from_json(self, path, keys=["text", "label"], encoding="utf-8"):
        """
        从定制化的标准json文件中读取初始语料, read corpus from myz
        config:
            path_json: str, path of corpus
            encoding: str, file encoding type, eg. "utf-8", "gbk"
            keys: list, selected key of json
        Returns:
            (xs, ys): tuple
        """
        with open(path, "r", encoding=encoding) as fo:
            line_list = json.load(fo)
            len_maxs = []
            for line_json in line_list:
                text = line_json.get(keys[0], "")
                len_maxs.append(len(text))
            fo.close()
            # 如果自动则选择覆盖0.95的长度, 即max_len为None, -1的情形
            len_maxs.sort()
            len_max_100 = len_maxs[-1]
            len_max_000 = len_maxs[0]
            len_max_95 = len_maxs[int(len(len_maxs) * 0.95)]
            len_max_90 = len_maxs[int(len(len_maxs) * 0.90)]
            len_max_50 = len_maxs[int(len(len_maxs) * 0.50)]
            self.logger.info("len_max_100: {}".format(len_max_100))
            self.logger.info("len_max_000: {}".format(len_max_000))
            self.logger.info("len_max_95: {}".format(len_max_95))
            self.logger.info("len_max_90: {}".format(len_max_90))
            self.logger.info("len_max_50: {}".format(len_max_50))
            if not self.config.max_len or self.config.max_len == -1:  # 自动化覆盖95%的文本长度
                self.max_len = min(len_max_95, 510)
            elif self.config.max_len == 0:  # 即max_len为0则是强制获取语料中的最大文本长度
                self.max_len = min(len_max_100, 510)
            else:
                self.max_len = min(self.max_len, 510)
            return line_list

    def read_corpus_from_list(self, path):
        """  读取多个文件  """
        if type(path) == list:  # 可传list
            data = []
            for path_i in path:
                data_p = self.read_corpus_from_one(path_i)
                data.extend(data_p)
        else:
            data = self.read_corpus_from_one(path)
        if self.config.flag_shuffle:
            random.shuffle(data)
            random.shuffle(data)
            random.shuffle(data)
        return data

    def read_corpus_from_one(self, path):
        """  只读一个文件  """
        if not (os.path.exists(path) and os.path.isfile(path)):
            raise ValueError("error, function read_corpus_from_span error! " + "path is not right, path: " + path)
        if path.endswith(".conll"):
            data = self.read_corpus_from_conll(path=path, keys=[0, 1], row_sep=self.config.row_sep)
            self.config.corpus_type = _SL_DATA_CONLL
        elif path.endswith(".span"):
            data = self.read_corpus_from_span(path=path, keys=self.config.keys)
            self.config.corpus_type = _SL_DATA_SPAN
        elif path.endswith(".jsonl"):
            data = self.read_corpus_from_span(path=path, keys=self.config.keys)
            self.config.corpus_type = _SL_DATA_SPAN
        elif path.endswith(".json"):
            data = self.read_corpus_from_json(path=path, keys=self.config.keys)
            self.config.corpus_type = _SL_DATA_SPAN
        else:
            raise ValueError("error, function read_corpus_from_span error! " + "file type is not right!")
        return data

    def analysis_corpus(self, data=None):
        """  分析语料, 且获取l2i,prior,label2conll等信息  """
        self.logger.info("analysis_corpus start!")
        if data:
            self.data = data
        # 排序, 样本多的类别排在前面
        if self.config.corpus_type == _SL_DATA_CONLL:
            # {"label": ["B-i",...], "text": ""}
            ys = dict()
            for data_i in tqdm(self.data, desc="analysis_corpus: "+_SL_DATA_CONLL):
                ys.update(dict(Counter(data_i.get("label", ""))))
            ys_sort = sorted(ys.items(), key=lambda x: x[1], reverse=True)
        elif self.config.corpus_type == _SL_DATA_SPAN:
            ys_type = []
            for data_i in tqdm(self.data, desc="analysis_corpus: "+_SL_DATA_SPAN):
                if self.config.task_type.upper() in [_SL_MODEL_SOFTMAX, _SL_MODEL_CRF]:
                    label = data_i.get("label", [])
                    text = data_i.get("text", "")
                    label_str = transform_span_to_conll(label, label_id=[0]*len(text), l2i_conll=None,
                                    flag_to_id=False, sl_ctype=self.config.sl_ctype.upper())
                    ys_type.extend([s for s in label_str if s != "O"])
                else:
                    ys_type.extend([ysti.get("type", "") for ysti in data_i.get("label", []) if ysti])
            ys = dict(Counter(ys_type))
            ys_sort = sorted(ys.items(), key=lambda x: x[1], reverse=True)
        else:
            raise ValueError("### error, analysis_corpus error, invalid line of data type, must be "
                             + _SL_DATA_CONLL + " or " + _SL_DATA_SPAN)

        # 处理标签, S-city, "O"必须为0
        # SOFTMAX、CRF、SPAN的情况下需要加 "O", GRID不能放0, 必须从1开始, eg. start_id = [0,0,0,2,0,0]
        if self.config.task_type.upper() not in [_SL_MODEL_GRID]:
            self.config.l2i["O"] = 0
            self.config.i2l[str(0)] = "O"
        ys_sort_dict = {}
        for y, y_count in ys_sort:
            # "SL-SPAN" 只保存 Label-Type, 其他保存BIO/BIOES(即BILOU-BMEWO-)/BMES/IOB(即IOB-1)/
            if self.config.task_type.upper() in [_SL_MODEL_SPAN, _SL_MODEL_GRID]:
                y = y.split("-")[-1]
                if y not in ys_sort_dict:  # 处理该情况, [('I-ORG', 5), ('B-ORG', 2)]
                    ys_sort_dict[y] = y_count
                else:
                    ys_sort_dict[y] += y_count
            else:
                ys_sort_dict[y] = y_count
            if y not in self.config.l2i and "O" != y:
                self.config.l2i[str(y)] = len(self.config.l2i)
                self.config.i2l[str(len(self.config.l2i) - 1)] = y
        # 类别先验分布
        len_corpus = sum([count for key, count in ys_sort_dict.items()])

        # 重构BIO字典统计prior, span转conll的时候用得上, 要求O必须在第一个label_id, 即0
        l2i_conll = {"O": 0}
        prior = [0.5]  # SL-CRF or SL-SOFTMAX
        # prior_count_dict = {key: count for key, count in ys_sort_new}
        for idx, (k, v) in enumerate(self.config.l2i.items()):
            if k != "O":
                l2i_conll[k] = len(l2i_conll)
                k_count = ys_sort_dict.get(k, 1)
                prior.append(k_count / len_corpus)

        # SL-SPAN
        if self.config.task_type.upper() in [_SL_MODEL_SPAN]:
            # prior = [0.5] + [count / len_corpus for key, count in ys_sort_dict.items() if key != "O"]
            prior = prior * 2  # [start_pos, end_pos] 两倍
        # # SL-GRID
        # if self.config.task_type.upper() in [_SL_MODEL_GRID]:
        #     prior = [0.5] + [count / len_corpus for key, count in ys_sort_dict.items() if key != "O"]
        #     # prior = [prior] * self.config.max_len

        # 参数更新
        self.config.num_labels = len(self.config.l2i)  # len(l2i_conll)  # len(self.config.l2i)
        self.config.max_len = self.max_len
        self.config.l2i_conll = l2i_conll
        self.config.prior = prior
        self.logger.info("train-data-lines: " + str(self.data.__len__()))
        self.logger.info("max_len: " + str(self.max_len))
        self.logger.info("l2i_conll: " + str(l2i_conll))
        self.logger.info("prior-label: " + str(prior))
        self.logger.info("analysis_corpus end!")

    def read_dict(self, data):
        """   推理时候传入List<dict>, 如[{"label": [], "text": "你是谁"}]   """
        data_new = []
        for data_i in data:
            # for k in self.config.keys:
            #     if k not in data_i:
            #         data_i[k] = []
            # data_new.append({"text": data_i.get("text", ""), "label": ""})
            data_i["label"] = []
            data_new.append(data_i)
        self.data = data_new

    def __getitem__(self, id):
        return tuple(self.data[id][k] for k in self.config.keys)

    def __len__(self):
        return len(self.data)


class SeqLabelingDataCollator:
    def __init__(self, config):
        self.tokenizer = self.load_tokenizer(config)
        self.config = config

    def preprocess_common(self, data_iter, label2idx, max_len=512, sl_ctype="BIO", l2i_conll=None):
        """  sequence-labeling, 序列标注任务
        pre-process with x(sequence)
        config:
            data_iter: iter, iter of (x, y), eg. ("你是谁", "问句")
            label2idx: dict, dict of label to number, eg. {"问句":0}
            max_len: int, max length of text, eg. 512
            l2i_conll: dict, label-to-index of conll-label-datatype
            sl_ctype: str, corpus-data-type of sequence-labeling, eg.BIO, IOB, BMES, BIOES
        Returns:
            inputs of bert-like model
        """
        batch_attention_mask = []
        batch_token_type = []
        batch_input = []
        batch_label = []
        batch_text = []
        if self.config.flag_dynamic_encode:  # 使用动态文本长度编码
            max_len_bs = max([len(d[0]) for d in data_iter])
            max_len = min(max_len, max_len_bs+2)
        # self.logger.info("max_len:  ", max_len)
        count = 0
        for di in data_iter:
            count += 1
            x, y = di
            token = self.tokenizer.tokenize(x)
            input_id = self.tokenizer.convert_tokens_to_ids(token)
            # 先计算好 input_id, label_id, 然后padding
            # ner-label 全部转为 onehot, 0是最多的O
            label_id = [0] * len(input_id)
            if self.config.corpus_type == _SL_DATA_CONLL:  # conll格式, 如已经存在的BMES、BIO等, eg. {"text": "南京", "label":["B-city", "I-city"]}
                for i, yi in enumerate(y):
                    if yi in label2idx and i <= len(input_id) - 1:
                        label_id[i] = label2idx[yi]
            elif self.config.corpus_type == _SL_DATA_SPAN:  # myx格式, 嵌套为json, eg. {"text": "南京", "label":[{"ent":"南京", "type": "city", "pos":[0,1]}]}
                # 数据格式, sl-type, BIO, BMES, BIOES, pos_id是实际id没有+1的
                if y:
                    label_id = transform_span_to_conll(y, label_id, l2i_conll, sl_ctype)
            # padding到最大文本长度
            pad_len = max_len - len(input_id) - 2
            token_type_id = [0] * max_len
            pad_input_id = 0
            pad_out_id = 0
            if pad_len >= 0:
                if self.config.padding_side.upper() == "LEFT":
                    ### cls_id 必须在 pos_id=0, 否则0开头 torch-crf 报错
                    input_id = [self.tokenizer.cls_token_id] + [pad_input_id] * pad_len + input_id + [self.tokenizer.sep_token_id]
                    attention_mask_id = [1] + [0] * (pad_len) + [1] * (max_len - pad_len - 1)
                    label_id = [pad_out_id] + [pad_out_id]*(pad_len) + label_id + [pad_out_id]
                else:
                    input_id = [self.tokenizer.cls_token_id] + input_id + [self.tokenizer.sep_token_id] + [pad_input_id] * pad_len
                    attention_mask_id = [1] * (max_len - pad_len) + [0] * (pad_len)
                    label_id = [pad_out_id] + label_id + [pad_out_id]*(pad_len+1)
            else:
                input_id = [self.tokenizer.cls_token_id] + input_id[:max_len - 2] + [self.tokenizer.sep_token_id]
                label_id = [pad_out_id] + label_id[: max_len-2] + [pad_out_id]
                attention_mask_id = [1] * max_len

            batch_attention_mask.append(attention_mask_id)
            batch_token_type.append(token_type_id)
            batch_input.append(input_id)
            batch_label.append(label_id)
            batch_text.append(x)  # for eval and pred
        # tensor
        tensor_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        tensor_token_type = torch.tensor(batch_token_type, dtype=torch.long)
        tensor_label = torch.tensor(batch_label, dtype=torch.float32)
        tensor_input = torch.tensor(batch_input, dtype=torch.long)
        return tensor_input, tensor_attention_mask, tensor_token_type, tensor_label, batch_text

    def preprocess_span(self, data_iter, label2idx, max_len=512, sl_ctype="BIO", l2i_conll=None):
        """  sequence-labeling, 序列标注任务
        pre-process with x(sequence)
        config:
            data_iter: iter, iter of (x, y), eg. ("你是谁", [{"ent":"北京", "type":"LOC", "pos":[0,1]}])
            label2idx: dict, dict of label to number, eg. {"问句":0}
            max_len: int, max length of text, eg. 512
            flag_seconds: bool, either use [SEP] separate texts2 or not, eg.True
            is_multi: bool, either sign sentence in texts with multi or not, eg. True
            label_sep: str, sign of multi-label split, eg. "#", "|@|"
        Returns:
            inputs of bert-like model
        """
        batch_attention_mask = []
        batch_token_type = []
        batch_input = []
        batch_start = []
        batch_end = []
        batch_text = []
        if self.config.flag_dynamic_encode:  # 使用动态文本长度编码
            max_len_bs = max([len(d[0]) for d in data_iter])
            max_len = min(max_len, max_len_bs)
        # self.logger.info("max_len:  ", max_len)
        count = 0
        for di in data_iter:
            count += 1
            x, y = di
            token = self.tokenizer.tokenize(x)
            input_id = self.tokenizer.convert_tokens_to_ids(token)
            # ner-label 全部转为 onehot
            start_id = [0] * len(input_id)
            end_id = [0] * len(input_id)
            if self.config.corpus_type == _SL_DATA_CONLL:  # conll格式, eg. {"text": "南京", "label":["B-city", "I-city"]}
                for i, yi in enumerate(y):
                    sep = "-"
                    yi_sp = yi.split(sep)
                    yi_type = yi_sp[-1]
                    if yi_type in label2idx and i <= len(input_id)-1:  # 存成span格式, 即开头和结尾 / 有-的类别, 如S-city, B-LOC
                        if i == 0 or (i > 0 and y[i-1].split(sep)[-1] != yi_type):  # 开始时候, 或者前一个类别不等于当前的情况, 记一start
                            start_id[i] = label2idx[yi_type]
                        if i == len(y)-1 or (i <= len(y)-1 and y[i+1].split(sep)[-1] != yi_type):  # 结尾时候, 或者后一个类别不等于当前的情况, 记一end
                            end_id[i] = label2idx[yi_type]
            elif self.config.corpus_type == _SL_DATA_SPAN:  # myx格式, 嵌套为json, eg. {"text": "南京", "label":[{"ent":"南京", "type": "city", "pos":[0,1]}]}
                for i, yi in enumerate(y):
                    yi_pos = yi.get("pos", [0, 1])
                    yi_type = yi.get("type", "")
                    # yi_e = yi.get("ent", "")
                    if yi_type in label2idx and yi_pos[1] <= len(input_id)-1:
                        start_id[yi_pos[0]] = label2idx[yi_type]
                        end_id[yi_pos[1]] = label2idx[yi_type]
            # padding到最大文本长度
            pad_len = max_len - len(input_id) - 2
            token_type_id = [0] * max_len
            pad_input_id = 0
            pad_out_id = 0
            if pad_len >= 0:
                if self.config.padding_side.upper() == "LEFT":
                    ### cls_id 必须在 pos_id=0, 否则0开头 torch-crf 报错
                    input_id = [self.tokenizer.cls_token_id] + [pad_input_id] * pad_len + input_id + [self.tokenizer.sep_token_id]
                    attention_mask_id = [1] + [0] * (pad_len) + [1] * (max_len - pad_len - 1)
                    start_id = [pad_out_id] + [pad_out_id] * (pad_len) + start_id + [pad_out_id]
                    end_id = [pad_out_id] + [pad_out_id] * (pad_len) + end_id + [pad_out_id]
                else:
                    input_id = [self.tokenizer.cls_token_id] + input_id + [self.tokenizer.sep_token_id] + [pad_input_id] * pad_len
                    attention_mask_id = [1] * (max_len - pad_len) + [0] * (pad_len)
                    start_id = [pad_out_id] + start_id + [pad_out_id] * (pad_len + 1)
                    end_id = [pad_out_id] + end_id + [pad_out_id] * (pad_len + 1)
            else:
                input_id = [self.tokenizer.cls_token_id] + input_id[:max_len - 2] + [self.tokenizer.sep_token_id]
                start_id = [pad_out_id] + start_id[: max_len - 2] + [pad_out_id]
                end_id = [pad_out_id] + end_id[: max_len - 2] + [pad_out_id]
                attention_mask_id = [1] * max_len

            batch_attention_mask.append(attention_mask_id)
            batch_token_type.append(token_type_id)
            batch_input.append(input_id)
            batch_start.append(start_id)
            batch_end.append(end_id)
            batch_text.append(x)   # for eval and pred
        # tensor, batch
        tensor_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        tensor_token_type = torch.tensor(batch_token_type, dtype=torch.long)
        tensor_input = torch.tensor(batch_input, dtype=torch.long)
        tensor_start = torch.tensor(batch_start, dtype=torch.float32)
        tensor_end = torch.tensor(batch_end, dtype=torch.float32)
        return tensor_input, tensor_attention_mask, tensor_token_type, tensor_start, tensor_end, batch_text

    def preprocess_grid(self, data_iter, label2idx, max_len=512, sl_ctype="BIO", l2i_conll=None):
        """  sequence-labeling, 序列标注任务
        pre-process with x(sequence)
        config:
            data_iter: iter, iter of (x, y), eg. ("你是谁", [{"ent":"北京", "type":"LOC", "pos":[0,1]}])
            label2idx: dict, dict of label to number, eg. {"问句":0}
            max_len: int, max length of text, eg. 512
            flag_seconds: bool, either use [SEP] separate texts2 or not, eg.True
            is_multi: bool, either sign sentence in texts with multi or not, eg. True
            label_sep: str, sign of multi-label split, eg. "#", "|@|"
        Returns:
            inputs of bert-like model
        """
        batch_attention_mask = []
        batch_token_type = []
        batch_input = []
        batch_grid = []
        batch_text = []
        if self.config.flag_dynamic_encode:  # 使用动态文本长度编码
            max_len_bs = max([len(d[0]) for d in data_iter])
            max_len = min(max_len, max_len_bs)
        # self.logger.info("max_len:  ", max_len)
        count = 0
        for di in data_iter:
            count += 1
            x, y = di
            token = self.tokenizer.tokenize(x)
            input_id = self.tokenizer.convert_tokens_to_ids(token)  # 全部当成中文处理, 杜绝 "雷吉纳vsac米兰" 问题
            # padding到最大文本长度
            pad_len = max_len - len(input_id) - 2
            token_type_id = [0] * max_len
            pad_len_grid = 1  # 偏移, padding导致的偏移. 如果没有padding则为1, 因为cls占用一个
            pad_input_id = 0
            pad_out_id = 0
            if pad_len >= 0:
                if self.config.padding_side.upper() == "LEFT":
                    ### cls_id 必须在 pos_id=0, 否则0开头 torch-crf 报错
                    input_id = [self.tokenizer.cls_token_id] + [pad_input_id] * pad_len + input_id + [self.tokenizer.sep_token_id]
                    attention_mask_id = [1] + [0] * (pad_len) + [1] * (max_len - pad_len - 1)
                    pad_len_grid = pad_len_grid + pad_len
                else:
                    input_id = [self.tokenizer.cls_token_id] + input_id + [self.tokenizer.sep_token_id] + [pad_input_id] * pad_len
                    attention_mask_id = [1] * (max_len - pad_len) + [0] * (pad_len)
            else:
                input_id = [self.tokenizer.cls_token_id] + input_id[:max_len - 2] + [self.tokenizer.sep_token_id]
                attention_mask_id = [1] * max_len
            # ner-label 全部转为 onehot
            grid = [[[0 for _ in range(max_len)] for _ in range(max_len)] for _ in range(len(label2idx))]
            if self.config.corpus_type == _SL_DATA_CONLL:  # conll格式, eg. {"text": "南京", "label":["B-city", "I-city"]}
                y_conll = get_pos_from_common(x, y, sep="-")  # 支持BIO, BIOES, BMES
                for i, yi in enumerate(y_conll):
                    yi_pos = yi.get("pos", [0, 1])
                    yi_type = yi.get("type", "")
                    if yi_type in label2idx and yi_pos[1] < len(input_id) - 1:
                        grid[label2idx[yi_type]][yi_pos[0] + pad_len_grid][yi_pos[1] + pad_len_grid] = 1
            elif self.config.corpus_type == _SL_DATA_SPAN:  # myx格式, 嵌套为json, eg. {"text": "南京", "label":[{"ent":"南京", "type": "city", "pos":[0,1]}]}
                for i, yi in enumerate(y):
                    yi_pos = yi.get("pos", [0, 1])
                    yi_type = yi.get("type", "")
                    # yi_e = yi.get("ent", "")
                    if yi_type in label2idx and yi_pos[1] < len(input_id) - 1:
                        grid[label2idx[yi_type]][yi_pos[0] + pad_len_grid][yi_pos[1] + pad_len_grid] = 1
            batch_attention_mask.append(attention_mask_id)
            batch_token_type.append(token_type_id)
            batch_input.append(input_id)
            batch_grid.append(grid)
            batch_text.append(x)  # for eval and pred
        # tensor
        tensor_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        tensor_token_type = torch.tensor(batch_token_type, dtype=torch.long)
        tensor_input = torch.tensor(batch_input, dtype=torch.long)
        tensor_grid = torch.tensor(batch_grid, dtype=torch.float32)
        return tensor_input, tensor_attention_mask, tensor_token_type, tensor_grid, batch_text

    def load_tokenizer(self, config):
        """
        加载标记器, load tokenizer, char-to-char
        config:
            config: dict, enum of parms
        Returns:
            tokenizer: class
        """
        class PretrainedTokenizer(PRETRAINED_MODEL_CLASSES[config.model_type][1]):
            """ char2char, 剔除BPE; 避免自带的tokenize删除空白、或者是其他特殊字符的情况 """
            def tokenize(self, text):
                tokens = []
                for t in text:
                    if self.do_lower_case:
                        t = t.lower()
                    if t in self.vocab:
                        tokens.append(t)
                    else:
                        tokens.append(self.unk_token)
                return tokens

        if config.flag_tokenizer_char:  # 使用char-to-char, 不用bpe的词根模式
            tokenizer = PretrainedTokenizer.from_pretrained(config.pretrained_model_name_or_path)
        else:
            tokenizer = PRETRAINED_MODEL_CLASSES[config.model_type][1].from_pretrained(config.pretrained_model_name_or_path)
        if config.additional_special_tokens:  # 新增字符串
            tokenizer.add_special_tokens({"additional_special_tokens": config.additional_special_tokens})
        tokenizer.padding_side == config.padding_side.lower()
        return tokenizer

    def __call__(self, data):
        # 数据转化, 转化成输入训练的数据格式, SL-SPAN, SL-CRF, SL-SOFTMAX, _SL_MODEL_GRID
        if self.config.task_type.upper() in [_SL_MODEL_SOFTMAX, _SL_MODEL_CRF]:
            sl_preprocess = self.preprocess_common
        elif self.config.task_type.upper() in [_SL_MODEL_GRID]:
            sl_preprocess = self.preprocess_grid
        else:
            sl_preprocess = self.preprocess_span

        data_prepro = sl_preprocess(data_iter=data, label2idx=self.config.l2i, max_len=self.config.max_len,
                                    sl_ctype=self.config.sl_ctype, l2i_conll=self.config.l2i_conll)
        return data_prepro


class CreateDataLoader:
    def __init__(self, config, logger=logger):
        self.config = config
        self.logger = logger

    def create_for_train(self, config):
        """  加载数据, data_loader, 为方便统一所以写成class的形式
            数据转为迭代器iter的形式, 同时需要分析数据, 获取label2conll/max_len/l2i/等信息;
        """
        if type(config) == dict:
            config = Namespace(**config)
        train_data_loader = None
        dev_data_loader = None
        tet_data_loader = None
        train_data = SeqlabelingDataset(config.path_train, config, logger=self.logger)
        train_data.analysis_corpus()
        data_config = train_data.config
        collate_fn_sl = SeqLabelingDataCollator(data_config)
        train_data_loader = DataLoader(num_workers=config.num_workers,
                                       batch_size=config.batch_size,
                                       shuffle=config.flag_shuffle,
                                       collate_fn=collate_fn_sl,
                                       dataset=train_data,
                                       )
        tokenizer = collate_fn_sl.tokenizer

        if config.path_dev and os.path.exists(config.path_dev):
            dev_data = SeqlabelingDataset(config.path_dev, config, logger=self.logger)
            collate_fn_sl = SeqLabelingDataCollator(data_config)
            dev_data_loader = DataLoader(num_workers=config.num_workers,
                                         batch_size=config.batch_size,
                                         shuffle=config.flag_shuffle,
                                         collate_fn=collate_fn_sl,
                                         dataset=dev_data,
                                         )
        if config.path_tet and os.path.exists(config.path_tet):
            tet_data = SeqlabelingDataset(config.path_tet, config, logger=self.logger)
            collate_fn_sl = SeqLabelingDataCollator(data_config)
            tet_data_loader = DataLoader(num_workers=config.num_workers,
                                         batch_size=config.batch_size,
                                         shuffle=config.flag_shuffle,
                                         collate_fn=collate_fn_sl,
                                         dataset=tet_data,
                                         )
        return train_data_loader, dev_data_loader, tet_data_loader, tokenizer, data_config


def create_data_loader(config):
    """  加载数据, data_loader
        数据转为迭代器iter的形式, 同时需要分析数据, 获取label2conll/max_len/l2i/等信息;
    """
    if type(config) == dict:
        config = Namespace(**config)
    train_data_loader = None
    dev_data_loader = None
    tet_data_loader = None
    train_data = SeqlabelingDataset(config.path_train, config)
    train_data.analysis_corpus()
    data_config = train_data.config
    collate_fn_sl = SeqLabelingDataCollator(data_config)
    train_data_loader = DataLoader(num_workers=config.num_workers,
                             batch_size=config.batch_size,
                             shuffle=config.flag_shuffle,
                             collate_fn=collate_fn_sl,
                             dataset=train_data,
                             )
    tokenizer = collate_fn_sl.tokenizer

    if config.path_dev and os.path.exists(config.path_dev):
        dev_data = SeqlabelingDataset(config.path_dev, config)
        collate_fn_sl = SeqLabelingDataCollator(data_config)
        dev_data_loader = DataLoader(num_workers=config.num_workers,
                                 batch_size=config.batch_size,
                                 shuffle=config.flag_shuffle,
                                 collate_fn=collate_fn_sl,
                                 dataset=dev_data,
                                 )
    if config.path_tet and os.path.exists(config.path_tet):
        tet_data = SeqlabelingDataset(config.path_tet, config)
        collate_fn_sl = SeqLabelingDataCollator(data_config)
        tet_data_loader = DataLoader(num_workers=config.num_workers,
                                 batch_size=config.batch_size,
                                 shuffle=config.flag_shuffle,
                                 collate_fn=collate_fn_sl,
                                 dataset=tet_data,
                                 )
    return train_data_loader, dev_data_loader, tet_data_loader, tokenizer, data_config
def tet_dataset_and_dataconllator():
    """   册数数据集和预处理函数   """
    path_data_dir = "D:/workspace/pythonMyCode/Pytorch-NLU/pytorch_nlu/corpus/sequence_labeling"
    path_conll_rmrb = os.path.join(path_data_dir, "ner_china_people_daily_1998_conll", "train.conll")
    path_span_rmrb = os.path.join(path_data_dir, "ner_china_people_daily_1998_span", "dev.span")
    config = {"max_len_limit": 512,
              "max_len": 128,

              "xy_keys_predict": ["text", "label"],
              "keys": ["text", "label"],
              "row_sep": " ",

              "padding_side": "LEFT",
              "loss_type": "MARGIN_LOSS",
              "corpus_type": "DATA-SPAN",
              "task_type": "SL-CRF",
              "model_type": "BERT",
              "sl_ctype": "BIO",


              "pretrained_model_name_or_path": "E:/DATA/bert-model/00_pytorch/LLM/hfl_chinese-macbert-base",
              "additional_special_tokens": ["<macropodus>", "<macadam>"],
              "flag_dynamic_encode": True,
              "flag_tokenizer_char": True,
              "flag_shuffle": True,
              "flag_train": True,

              "l2i_conll": {},
              "l2i": {},
              "i2l": {},
              }

    for path in [path_conll_rmrb, path_span_rmrb]:
        print(path_conll_rmrb)
        sd = SeqlabelingDataset(path, config)
        sd.analysis_corpus()
        res = sd.__getitem__(2)
        print(res)
        print(sd.__len__())
        myz = 0

        train_loader = DataLoader(dataset=sd,
                                  batch_size=16,
                                  shuffle=True,
                                  num_workers=0,
                                  collate_fn=SeqLabelingDataCollator(sd.config))
        for t in train_loader:
            print(t)
            myz = 0


if __name__ == '__main__':
    yz = 0

    ### 测试训练集地数据生成
    tet_dataset_and_dataconllator()

