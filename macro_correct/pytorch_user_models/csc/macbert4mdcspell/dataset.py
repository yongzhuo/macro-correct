# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/1/18 21:34
# @author  : Mo
# @function: Dataset Read
# @paper   : [Chinese Spelling Correction as Rephrasing Language Model](https://arxiv.org/abs/2308.08796).
# @code    : most code copy from https://github.com/Claude-Liu/ReLM
# @code    : mertics code copy from https://github.com/wangwang110/CSC, small modfix


import traceback
import random
import json
import sys
import os
import re

from tqdm import tqdm
import opencc


converter = opencc.OpenCC("t2s.json")
# converter = opencc.OpenCC("tw2sp.json")
# converter = opencc.OpenCC("tw2s.json")
context = converter.convert("汉字")  # 漢字
print(context)


def sent_mertic_det(all_srcs, all_pres, all_trgs, logger, flag_eval="common"):
    """
    句子级别检测指标：所有位置都检测对才算对
    :param all_pres:
    :param all_trgs:
    :return:
    """

    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    change_num = 0
    for src, tgt_pred, tgt in zip(all_srcs, all_pres, all_trgs):
        src_tgt_tag = [1 if s == t else 0 for s, t in zip(list(src), list(tgt))]
        src_tgt_pred_tag = [1 if s == t else 0 for s, t in zip(list(src), list(tgt_pred))]

        if src != tgt_pred:
            change_num += 1

        # 负样本
        if src == tgt:
            # 预测也为负
            if src == tgt_pred:
                TN += 1
                # print('right')
            # 预测为正
            else:
                FP += 1
                # print('wrong')
        # 正样本
        else:
            # 预测也为正
            if src_tgt_tag == src_tgt_pred_tag:
                TP += 1
                # print('right')
            # 预测为负
            else:
                FN += 1
                # print('wrong')
        total_num += 1
    acc = (TP + TN) / total_num
    # precision = TP / (TP + FP) if TP > 0 else 0.0
    # precision = TP / change_num if TP > 0 else 0.0
    # 官方评测以及pycorrect计算p值方法，分母忽略了原句被修改，但是没完全检测对的情况，因此计算出的指标偏高
    if flag_eval == "strict":
        precision = TP / change_num if TP > 0 else 0.0
    else:
        precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    # logger.info(
    #     f'Sentence Level detection: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}')
    # print(
    #     f'Sentence Level detection: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}')
    return acc, precision, recall, f1


def sent_mertic_cor(all_srcs, all_pres, all_trgs, logger, flag_eval="common"):
    """
    句子级别纠正指标：所有位置纠正对才算对
    :param all_pres:
    :param all_trgs:
    :return:
    """

    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    total_num = 0
    change_num = 0
    for src, tgt_pred, tgt in zip(all_srcs, all_pres, all_trgs):

        if src != tgt_pred:
            change_num += 1

        # 负样本
        if src == tgt:
            # 预测也为负
            if tgt == tgt_pred:
                TN += 1
            # 预测为正
            else:
                FP += 1
                # print('wrong')
        # 正样本
        else:
            # 预测也为正
            if tgt == tgt_pred:
                TP += 1
            # 预测为负
            else:
                FN += 1
        total_num += 1
    acc = (TP + TN) / total_num
    # precision = TP / (TP + FP) if TP > 0 else 0.0
    # 官方评测以及pycorrect计算p值方法，分母忽略了原句被修改，但是没改对的情况，因此计算出的指标偏高
    # precision = TP / change_num if TP > 0 else 0.0
    if flag_eval == "strict":
        precision = TP / change_num if TP > 0 else 0.0
    else:
        precision = TP / (TP + FP) if TP > 0 else 0.0
    recall = TP / (TP + FN) if TP > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    # logger.info(
    #     f'Sentence Level correction: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}')
    #
    # print(
    #     f'Sentence Level correction: acc:{acc:.4f}, precision:{precision:.4f}, recall:{recall:.4f}, f1:{f1:.4f}')
    return acc, precision, recall, f1


def compute_sentence_level_prf_paper(results, logger):
    """
    自定义的句级prf，设定需要纠错为正样本，无需纠错为负样本
    :param results:
    :return:
    """

    all_srcs, all_pres, all_trgs = [], [], []
    for item in results:
        src, tgt, pred = item
        all_srcs.append(src)
        all_trgs.append(tgt)
        all_pres.append(pred)
    logger.info("#"*128)
    logger.info("\npaper-mertics\n")
    sent_mertic_det(all_srcs, all_pres, all_trgs, logger)
    sent_mertic_cor(all_srcs, all_pres, all_trgs, logger)
    logger.info("#"*128)


PUN_EN2ZH_DICT = {",": "，", ";": "；", "!": "！", "?": "？", ":": "：",
                  "(": "（", ")": "）", "_": "—"}
PUN_ZH2EN_DICT = {'，': ',', '；': ';', '！': '!', '？': '?', '：': ':', '（': '(', '）': ')',
                      '—': '-', '～': '~', '＜': '<', '＞': '>',
                      "‘": "'", "’": "'", '”': '"', '“': '"',
                      }
PUN_BERT_DICT = {"“":'"', "”":'"', "‘":'"', "’":'"', "—": "_", "——": "__"}
def transfor_english_symbol_to_chinese(text, kv_dict=PUN_EN2ZH_DICT):
        """   将英文标点符号转化为中文标点符号, 位数不能变防止pos_id变化   """
        for k, v in kv_dict.items():  # 英文替换
            text = text.replace(k, v)
        if text and text[-1] == ".":  # 最后一个字符是英文.
            text = text[:-1] + "。"

        if text and "\"" in text:  # 双引号
            index_list = [i.start() for i in re.finditer("\"", text)]
            if index_list:
                for idx, index in enumerate(index_list):
                    symbol = "“" if idx % 2 == 0 else "”"
                    text = text[:index] + symbol + text[index + 1:]

        if text and "'" in text:  # 单引号
            index_list = [i.start() for i in re.finditer("'", text)]
            if index_list:
                for idx, index in enumerate(index_list):
                    symbol = "‘" if idx % 2 == 0 else "’"
                    text = text[:index] + symbol + text[index + 1:]
        return text
def transfor_chinese_symbol_to_english(text, kv_dict=PUN_ZH2EN_DICT):
    """   将英文标点符号转化为中文标点符号, 位数不能变防止pos_id变化   """
    for k, v in kv_dict.items():  # 英文替换
        text = text.replace(k, v)
    return text
def transfor_bert_unk_pun_to_know(text, kv_dict=PUN_BERT_DICT):
    """   将英文标点符号转化为中文标点符号, 位数不能变防止pos_id变化   """
    for k, v in kv_dict.items():  # 英文替换
        text = text.replace(k, v)
    return text
def tradition_to_simple(text):
    """  台湾繁体到大陆简体  """
    return converter.convert(text)
def string_q2b(ustring):
    """把字符串全角转半角"""
    return "".join([q2b(uchar) for uchar in ustring])
def q2b(uchar):
    """全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)


def save_json(jsons, json_path, mode="w", indent=4, encoding="utf-8"):
    """
        保存json
    Args:
        path[String]:, path of file of save, eg. "corpus/xuexiqiangguo.lib"
        jsons[Json]: json of input data, eg. [{"桂林": 132}]
        indent[int]: pretty-printed with that indent level, eg. 4
    Returns:
        None
    """
    with open(json_path, mode=mode, encoding=encoding) as fj:
        fj.write(json.dumps(jsons, ensure_ascii=False, indent=indent))
    fj.close()


def txt_write(lines, path, mode="w", encoding="utf-8"):
    """
    Write Line of list to file
    Args:
        lines: lines of list<str> which need save
        path: path of save file, such as "txt"
        model: type of write, such as "w", "a+"
        encoding: type of encoding, such as "utf-8", "gbk"
    """

    try:
        file = open(path, mode, encoding=encoding)
        file.writelines(lines)
        file.close()
    except Exception as e:
        print(traceback.print_exc())


def txt_read(path, encoding="utf-8", errors=None):
    """
        读取txt文件，默认utf8格式, 不能有空行
    Args:
        path[String]: path of file of read, eg. "corpus/xuexiqiangguo.txt"
        encode_type[String]: data encode type of file, eg. "utf-8", "gbk"
        errors[String]: specifies how encoding errors handled, eg. "ignore", strict
    Returns:
        lines[List]: output lines
    """
    lines = []
    try:
        file = open(path, "r", encoding=encoding, errors=errors)
        lines = file.readlines()
        file.close()
    except Exception as e:
        print(traceback.print_exc())
    finally:
        return lines


def load_json(path, encoding="utf-8"):
    """   加载json文件   """
    with open(path, "r", encoding=encoding) as fj:
        model_json = json.load(fj)
        fj.close()
    return model_json


class InputExample(object):
    def __init__(self, guid, src, trg):
        self.guid = guid
        self.src = src
        self.trg = trg


class DataSetProcessor:
    """Processor for the ECSpell data set."""
    def __init__(self, path_train=None, path_dev=None, path_tet=None, task_name=None):
        self.task_name = task_name
        self.path_train = path_train
        self.path_dev = path_dev
        self.path_tet = path_tet

    def get_train_examples(self):
        datas = self.load_data(self.path_train)
        random.shuffle(datas)
        random.shuffle(datas)
        random.shuffle(datas)
        return self._create_examples(datas, "train")

    def get_dev_examples(self):
        datas = self.load_data(self.path_dev)
        return self._create_examples(datas, "dev")

    def get_test_examples(self):
        datas = self.load_data(self.path_tet)
        return self._create_examples(datas, "test")

    def load_data(self, path):
        """   读取多个文件的 json 或者 jsonl数据"""
        if type(path) == str:
            path = [path]
        datas = []
        for p in path:
            try:
                datas_i = load_json(p)
            except Exception as e:
                print(traceback.print_exc())
                data_list = txt_read(p)
                datas_i = []
                for d in data_list:
                    d_dict = json.loads(d.strip())
                    datas_i.append(d_dict)
            datas.extend(datas_i)
        return datas

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for i, line in tqdm(enumerate(lines), desc="train_preprocess"):
            if "original_text" in line:
                src = line.get("original_text", "")
                trg = line.get("correct_text", "")
            else:
                src = line.get("source", "")
                trg = line.get("target", "")
            ### 数据预处理
            src = string_q2b(src)
            trg = string_q2b(trg)
            src = tradition_to_simple(src)
            trg = tradition_to_simple(trg)
            src = transfor_english_symbol_to_chinese(src)
            trg = transfor_english_symbol_to_chinese(trg)
            src = transfor_bert_unk_pun_to_know(src)
            trg = transfor_bert_unk_pun_to_know(trg)
            src = list(src)
            trg = list(trg)
            guid = "%s-%s" % (set_type, i)
            if len(src) == len(trg):
                examples.append(InputExample(guid=guid, src=src, trg=trg))
        return examples

    @staticmethod
    def _create_predicts(lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            if "original_text" in line:
                src = line.get("original_text", "")
            else:
                src = line.get("source", "")
            src = string_q2b(src)
            src = tradition_to_simple(src)
            src = transfor_english_symbol_to_chinese(src)
            src = transfor_bert_unk_pun_to_know(src)
            src = list(src)
            guid = "%s-%s" % (set_type, i)
            examples.append(InputExample(guid=guid, src=src, trg=None))
        return examples


if __name__ == '__main__':
    myz = 0
    model_wxp = DataSetProcessor()
    res = model_wxp.get_train_examples()
    print(res)
