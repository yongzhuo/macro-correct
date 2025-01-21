# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/1/18 21:34
# @author  : Mo
# @function: deal corpus(data preprocess), label转成list形式, 或者span的形式


from collections import Counter
from argparse import Namespace
import logging as logger
import traceback
import argparse
import random
import json
import os

from macro_correct.pytorch_textcorrection.tcConfig import PRETRAINED_MODEL_CLASSES
from macro_correct.pytorch_textcorrection.tcTools import load_json
from macro_correct.pytorch_textcorrection.tcTqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import torch


__all__ = [
    "TextCorrectionDataCollator",
    "TextCorrectionDataset",
    "CreateDataLoader",
    "create_data_loader",
]


class TextCorrectionDataset(Dataset):
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

    def read_corpus_from_json(self, path, encoding="utf-8", len_rate=1, keys=["text_1", "text_2", "ids"]):
        """
        从定制化的标准json文件中读取初始语料, read corpus from json
        config:
            path_json: str, path of corpus
            encoding: str, file encoding type, eg. "utf-8", "gbk"
            len_rate: float, 0-1, eg. 0.5
            keys: list, selected key of json, eg. ["text_1", "text_2", "ids"]
        Returns:
            (xs, ys): tuple
        """
        len_xs = []
        len_ys = []
        count = 0
        datas = load_json(path, encoding=encoding)
        for line_json in datas:
            count += 1
            #  最初想定义可配置化, 但是后期实验较多, 还是设置成一般形式, 可自己定义
            x = line_json.get(keys[0], "")
            len_xs.append(len(x))
            wrong_ids = line_json.get(keys[2], [])
            len_ys.append(len(wrong_ids))
        # 没有验证集的情况
        len_rel = int(count * len_rate) if 0 < len_rate < 1 else count
        datas = datas[:len_rel+1]
        # 如果自动则选择覆盖0.95的长度, 即max_len为None, -1的情形
        len_xs.sort()
        len_ys.sort()
        rate_len_xs_list = [0.00, 0.30, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
        rate_len_ys_list = [0.00, 0.30, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
        len_data = len(len_xs) - 1
        len_xs_rate_list = []
        len_ys_rate_list = []
        self.logger.info("path_data: {}".format(path))
        for _, ent in enumerate(rate_len_xs_list):
            len_xs_num = int((len_data+1) * ent) - 1
            len_xs_rate = len_xs[min(max(len_xs_num, 0), len_data)]  # 最大为len_data, 最小为len_xs
            len_xs_rate_list.append(len_xs_rate)
            self.logger.info("len_xs_max_{}: {}".format(ent, len_xs_rate))
        for _, enst in enumerate(rate_len_ys_list):
            len_ys_num = int((len_data+1) * enst) - 1
            len_ys_rate = len_xs[min(max(len_ys_num, 0), len_data)]  # 最大为len_data, 最小为len_ys
            len_ys_rate_list.append(len_ys_rate)
            self.logger.info("len_ys_max_{}: {}".format(enst, len_ys_rate))
        if not self.config.max_len or self.config.max_len == -1:  # 自动化覆盖95%的文本长度
            self.max_len = min(len_xs_rate_list[-2], self.max_len_limit)
        elif self.config.max_len == 0:  # 即max_len为0则是强制获取语料中的最大文本长度
            self.max_len = min(len_xs_rate_list[-1], self.max_len_limit)
        else:
            self.max_len = min(self.max_len, self.max_len_limit)
        return datas

    def read_corpus_from_texts(self, texts, keys=["text_1", "text_2", "ids"]):
        """
        一般预测用(ids/text_2等需要剔除), 从列表texts中获取json, read corpus from texts
        config:
            texts: List<json>, eg. [{"text_1":"12306", "text_1":"12305", "ids": [4]}]
            encoding: str, file encoding type, eg. "utf-8", "gbk"
            keys: list, selected key of json, eg. ["text_1", "text_2", "ids"]
        Returns:
            (xs, ys): tuple
        """
        data_new = []
        count = 0
        for line_json in texts:
            count += 1
            # if count > 32:
            #     break
            if not line_json:
                continue
            line_json[keys[1]] = ""
            line_json[keys[2]] = []
            data_new.append(line_json)
        return data_new

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
            raise ValueError("error, function read_corpus_from_one error! " + "path is not right, path: " + path)
        # if path.endswith(".json"):
        #     data = self.read_corpus_from_json(path=path, keys=self.config.keys)
        # else:
        #     raise ValueError("error, function read_corpus_from_json error! " + "file type is not right!")
        try:
            data = self.read_corpus_from_json(path=path, keys=self.config.keys)
        except Exception as e:
            self.logger.info(traceback.print_exc())
            raise ValueError("error, function read_corpus_from_json error! " + "file type is not right!(must be json)")
        return data

    def read_dict(self, data):
        """   推理时候传入List<dict>, 如[{"ids": [], "text_1": "你是谁", "text_2": "你是谁"}]   """
        data_new = []
        for data_i in data:
            data_i[self.config.xy_keys_predict[1]] = ""
            data_i[self.config.xy_keys_predict[2]] = []
            data_new.append(data_i)
        self.data = data_new

    def __getitem__(self, id):
        return tuple(self.data[id][k] for k in self.config.xy_keys_predict)

    def __len__(self):
        return len(self.data)


class TextCorrectionDataCollator:
    def __init__(self, config):
        self.additional_special_tokens = config.additional_special_tokens
        self.tokenizer = self.load_tokenizer(config)
        self.special_tokens = [self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.sep_token,
                               self.tokenizer.unk_token] + self.additional_special_tokens
        self.max_len = config.max_len
        self.config = config

    def preprocess_common(self, data_iter, max_len=512):
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
        batch_det = []
        if self.config.flag_dynamic_encode:  # 使用动态文本长度编码
            max_len_bs = max([len(d[0]) for d in data_iter])
            max_len = min(max_len, max_len_bs+2)
        # self.logger.info("max_len:  ", max_len)
        count = 0
        for di in data_iter:
            count += 1
            x, y, wrong_ids = di
            if not y:
                y = x
            tokens_x = self.tokenizer.tokenize(x)
            tokens_y = self.tokenizer.tokenize(y)
            input_id = self.tokenizer.convert_tokens_to_ids(tokens_x)
            output_id = self.tokenizer.convert_tokens_to_ids(tokens_y)
            token_type_id = [0] * max_len
            det_id = [0] * max_len
            for w in wrong_ids:
                if w + 1 > max_len - 1:
                    # print(di)
                    break
                else:
                    det_id[w] = 1
            # padding到最大文本长度
            pad_len = max_len - len(input_id) - 2
            pad_input_id = 0
            pad_out_id = 0
            if pad_len >= 0:
                if self.config.padding_side.upper() == "LEFT":
                    ### cls_id 必须在 pos_id=0, 否则0开头 torch-crf 报错
                    input_id = [self.tokenizer.cls_token_id] + [pad_input_id] * pad_len + input_id + [self.tokenizer.sep_token_id]
                    attention_mask_id = [0] * (1 + pad_len) + [1] * (max_len - pad_len - 1)
                    output_id = [self.tokenizer.cls_token_id] + [pad_out_id] * pad_len + output_id + [self.tokenizer.sep_token_id]
                else:
                    input_id = [self.tokenizer.cls_token_id] + input_id + [self.tokenizer.sep_token_id] + [pad_input_id] * pad_len
                    attention_mask_id = [1] * (max_len - pad_len) + [0] * (pad_len)
                    output_id = [self.tokenizer.cls_token_id] + output_id + [self.tokenizer.sep_token_id] + [pad_out_id] * pad_len
            else:
                input_id = [self.tokenizer.cls_token_id] + input_id[:max_len - 2] + [self.tokenizer.sep_token_id]
                output_id = [self.tokenizer.cls_token_id] + output_id[:max_len - 2] + [self.tokenizer.sep_token_id]
                attention_mask_id = [1] * max_len

            batch_attention_mask.append(attention_mask_id)
            batch_token_type.append(token_type_id)
            batch_input.append(input_id)
            batch_label.append(output_id)
            batch_det.append(det_id)
            batch_text.append(x)  # for eval and pred
        # tensor
        tensor_attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
        tensor_token_type = torch.tensor(batch_token_type, dtype=torch.long)
        tensor_label = torch.tensor(batch_label, dtype=torch.long)
        tensor_input = torch.tensor(batch_input, dtype=torch.long)
        tensor_det = torch.tensor(batch_det, dtype=torch.long)
        return tensor_input, tensor_attention_mask, tensor_token_type, tensor_label, tensor_det, batch_text

    def load_tokenizer(self, config):
        """
        加载标记器, load tokenizer, char-to-char
        config:
            config: dict, enum of parms
        Returns:
            tokenizer: class
        """
        class PretrainedTokenizer(PRETRAINED_MODEL_CLASSES[config.model_type][1]):
            """ char2char, 剔除BPE; 避免自带的tokenize删除空白、或者是其他特殊字符的情况
            paper: C-LLM: Learn to Check Chinese Spelling Errors Character by Character
            url: https://arxiv.org/pdf/2406.16536
            """
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
        data_prepro = self.preprocess_common(data_iter=data, max_len=self.max_len)
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
        train_data = TextCorrectionDataset(config.path_train, config, logger=self.logger)
        data_config = train_data.config
        collate_fn_sl = TextCorrectionDataCollator(data_config)
        train_data_loader = DataLoader(num_workers=config.num_workers,
                                       batch_size=config.batch_size,
                                       shuffle=config.flag_shuffle,
                                       collate_fn=collate_fn_sl,
                                       dataset=train_data,
                                       )
        tokenizer = collate_fn_sl.tokenizer

        if config.path_dev and os.path.exists(config.path_dev):
            dev_data = TextCorrectionDataset(config.path_dev, config, logger=self.logger)
            collate_fn_sl = TextCorrectionDataCollator(data_config)
            dev_data_loader = DataLoader(num_workers=config.num_workers,
                                         batch_size=config.batch_size,
                                         shuffle=config.flag_shuffle,
                                         collate_fn=collate_fn_sl,
                                         dataset=dev_data,
                                         )
        if config.path_tet and os.path.exists(config.path_tet):
            tet_data = TextCorrectionDataset(config.path_tet, config, logger=self.logger)
            collate_fn_sl = TextCorrectionDataCollator(data_config)
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
    train_data = TextCorrectionDataset(config.path_train, config)
    data_config = train_data.config
    collate_fn_sl = TextCorrectionDataCollator(data_config)
    train_data_loader = DataLoader(num_workers=config.num_workers,
                             batch_size=config.batch_size,
                             shuffle=config.flag_shuffle,
                             collate_fn=collate_fn_sl,
                             dataset=train_data,
                             )
    tokenizer = collate_fn_sl.tokenizer

    if config.path_dev and os.path.exists(config.path_dev):
        dev_data = TextCorrectionDataset(config.path_dev, config)
        collate_fn_sl = TextCorrectionDataCollator(data_config)
        dev_data_loader = DataLoader(num_workers=config.num_workers,
                                 batch_size=config.batch_size,
                                 shuffle=config.flag_shuffle,
                                 collate_fn=collate_fn_sl,
                                 dataset=dev_data,
                                 )
    if config.path_tet and os.path.exists(config.path_tet):
        tet_data = TextCorrectionDataset(config.path_tet, config)
        collate_fn_sl = TextCorrectionDataCollator(data_config)
        tet_data_loader = DataLoader(num_workers=config.num_workers,
                                 batch_size=config.batch_size,
                                 shuffle=config.flag_shuffle,
                                 collate_fn=collate_fn_sl,
                                 dataset=tet_data,
                                 )
    return train_data_loader, dev_data_loader, tet_data_loader, tokenizer, data_config
def tet_dataset_and_dataconllator():
    """   册数数据集和预处理函数   """
    path_data_dir = "E:\\DATA\\city_shenzhen_2\\LLM_pretrain\\shibing624_CSC"
    path_train = os.path.join(path_data_dir, "dev.json")
    path_dev = os.path.join(path_data_dir, "test.json")

    config = {"max_len_limit": 512,
              "max_len": 128,

              "xy_keys_predict": ["original_text", "correct_text", "wrong_ids"],
              "keys": ["original_text", "correct_text", "wrong_ids"],
              "row_sep": " ",

              "padding_side": "RIGHT",
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

              # "l2i_conll": {},
              # "l2i": {},
              # "i2l": {},
              }

    for path in [path_train, path_dev]:
        print(path_train)
        sd = TextCorrectionDataset(path, config)
        res = sd.__getitem__(2)
        print(res)
        print(sd.__len__())
        myz = 0

        train_loader = DataLoader(dataset=sd,
                                  batch_size=16,
                                  shuffle=True,
                                  num_workers=0,
                                  collate_fn=TextCorrectionDataCollator(sd.config))
        for t in train_loader:
            print(t)
            myz = 0


if __name__ == '__main__':
    myz = 0

    ### 测试训练集
    tet_dataset_and_dataconllator()

