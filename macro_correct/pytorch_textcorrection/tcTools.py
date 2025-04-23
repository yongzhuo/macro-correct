# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/11/17 21:35
# @author  : Mo
# @function: utils of Pytorch-TextCorrect


from logging.handlers import RotatingFileHandler
from collections import Counter, OrderedDict
from typing import Union, Dict, List, Any
import traceback
import operator
import difflib
import logging
import pickle
import time
import json
import copy
import re
import os

import numpy as np
import opencc


__all_ = ["mertics_precision_recall_fscore_support",
          "mertics_multilabel_confusion_matrix",
          "mertics_report_v1",
          "mertics_report",
          "sklearn_confusion_matrix",
          "sklearn_kfold",
          "char_mertic_det_cor",
          "sent_mertic_det",
          "sent_mertic_cor",
          "chinese_extract_extend",
          "flag_total_chinese",
          "flag_total_number",
          "get_current_time",
          "delete_file",
          "get_logger"
          "save_json",
          "load_json",
          "txt_write",
          "txt_read",
          "dic_sort",
          "del_dir",
          "sigmoid",
          "softmax",
          "transfor_english_symbol_to_chinese",
          "transfor_chinese_symbol_to_english",
          "transfor_bert_unk_pun_to_know",
          "tradition_to_simple",
          "string_q2b",
          "string_b2q",
          "q2b",
          "b2q",

          "cut_sent_by_stay",
          "count_flag_zh",
          "cut_sent_len",
          "flag_no_chinese_number_alphabet_signal",
          "flag_chinese_char",
          "flag_number_other",
          "flag_alphabet",
          "flag_number",
          "flag_alphabet_string",
          "flag_chinese_string",
          ]


converter = opencc.OpenCC("t2s.json")
# converter = opencc.OpenCC("tw2sp.json")
# converter = opencc.OpenCC("tw2s.json")
context = converter.convert("汉字")  # 漢字
# print(context)
PUN_EN2ZH_DICT = {",": "，", ";": "；", "!": "！", "?": "？", ":": "：",
                  "(": "（", ")": "）", "_": "—"}
PUN_ZH2EN_DICT = {'，': ',', '；': ';', '！': '!', '？': '?', '：': ':', '（': '(', '）': ')',
                      '—': '-', '～': '~', '＜': '<', '＞': '>',
                      "‘": "'", "’": "'", '”': '"', '“': '"',
                      }
PUN_BERT_DICT = {"“":'"', "”":'"', "‘":'"', "’":'"', "—": "_", "——": "__"}


def mertics_report_v1(y_true: List, y_pred: List, digits: int=4, eps: float=1e-9, use_acc: bool=True, use_draw: bool=True):
    """ 老版本模型评估指标, 只支持list<str>格式输入的y_true/y_pred
    mertics report, 打印评估指标, 不支持onehot的形式
    Args:
        y_true:  list, 真实样本标签, 可以是单个数字标签, 或者是string格式
        y_pred:  list, 预测样本标签
        digits:  int,  最后保留位数
        eps:     float, 设置一个eps防止为0
        use_acc: bool, 是否使用acc
        use_draw: bool, 是否返回report
        
    Returns:    tuple<float>, 多个返回值
        label_metrics: dict, 包括单个类别/micro/macro/weighted的 prf
        report:        str,  评估指标打印格式, 同sklearn
        error_idxs:    list<int>, 预测错误标签的位置
    """
    from collections import Counter, defaultdict
    counter_y_true = dict(Counter(y_true))
    len_counter_y_true = len(counter_y_true)
    len_y_true = len(y_true)
    label_error_dict = {k:[] for k in counter_y_true}

    def draw_mertics(metrics, digits=4, use_acc=True):
        """
        draw mertics, 绘画评估指标
        Args:
            metrics: dict, eg. {"游戏:132, "旅游":32}
            digits: int, eg. 4
            use_acc: bool, add acc or not, eg.True
        Returns:
            report: str
        """
        # 打印头部
        labels = list(metrics.keys())
        target_names = [lab for lab in labels]  # [u"%s" % l for l in labels]
        name_width = max(len(str(cn)) for cn in target_names)
        width = max(name_width, digits)
        headers = list(metrics[labels[0]].keys())
        if not use_acc: headers.remove("acc")
        head_fmt = u"{:>{width}} " + u" {:>9}" * len(headers)
        report = "\n\n" + head_fmt.format("", *headers, width=width)
        report += "\n\n"
        # 具体label的评估指标
        count = 0
        for lab in labels:
            if lab in ["macro", "micro", "weighted"] and count == 0:
                report += "\n"
                count += 1
            row_fmt = u"{:>{width}} " + u" {:>9.{digits}f}" * (len(headers) - 1) + u" {:>9}\n"
            if use_acc:
                [p, r, f1, a, s] = [metrics[lab][hd] for hd in headers]
                report += row_fmt.format(lab, p, r, f1, a, s, width=width, digits=digits)
            else:
                [p, r, f1, s] = [metrics[lab][hd] for hd in headers]
                report += row_fmt.format(lab, p, r, f1, s, width=width, digits=digits)
        report += "\n"
        return report

    def calculate_fpr(label_dict):
        """
        calculate fpr, 单次计算单次评估指标
        Args:
            label_dict: dict, eg. {"tp": 3, "fp": 1, "tn": 7, "fn": 1}
        Returns:
            report: tuple<float>
        """
        precision = label_dict["tp"] / (label_dict["tp"] + label_dict["fp"] + eps)
        recall = label_dict["tp"] / (label_dict["tp"] + label_dict["fn"] + eps)
        f1 = (2 * precision * recall) / (precision + recall + eps)
        return round(precision, digits), round(recall, digits), round(f1, digits)

    # 首先统计各个类别下的t-f-t-n
    label_dict = {i: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for i in sorted(set(y_true))}
    for idx in range(len(y_true)):
        correct = y_true[idx]
        predict = y_pred[idx]
        for k, v in label_dict.items():
            if correct != predict:  # 预测错误
                if k == correct:    # 当前label=false, False Negative, 被判定为负样本, 但事实上是正样本.
                    label_dict[correct]["fn"] += 1
                elif k == predict:  # 当前predict=false, False Positive, 被判定为正样本, 但事实上是负样本.
                    label_dict[k]["fp"] += 1
                else:               # 其他label=false, True Negative, 被判定为负样本, 事实上也是负样本.
                    label_dict[k]["tn"] += 1
            else:                  # 预测正确
                if k == correct:   # 当前label=true, True Positive, 被判定为正样本, 事实上也是正样本.
                    label_dict[correct]["tp"] += 1
                else:              # 其他label=true, True Negative, 被判定为负样本, 事实上也是负样本.
                    label_dict[k]["tn"] += 1
        if correct != predict: label_error_dict[correct].append(idx)
    # 然后计算各个类别/特殊的prf
    weighted_mertics = {"precision": 0, "recall": 0, "f1": 0, "acc": 0}
    macro_mertics = {"precision": 0, "recall": 0, "f1": 0, "acc": 0}
    micro_fpr = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    label_len_dict = {k: len(v) for k,v in label_error_dict.items()}  # 类别下的错误样本数
    label_metrics = {}
    for k, v in label_dict.items():
        # label-mertics
        label_precision, label_recall, label_f1 = calculate_fpr(v)
        label_acc = 1 - (label_len_dict[k] / counter_y_true[k])
        label_acc = round(label_acc, digits)
        label_metrics[k] = {"precision": label_precision, "recall": label_recall,"f1": label_f1,
                            "acc": label_acc, "support": counter_y_true[k]}
        # macro-fpr
        for mf in micro_fpr.keys():  # tp/fp/tn/fn
            micro_fpr[mf] += v[mf]
        for wm in weighted_mertics.keys():
            weighted_mertics[wm] += label_metrics[k][wm] * counter_y_true[k]
        # macro-mertics
        macro_mertics["precision"] += label_precision
        macro_mertics["recall"] += label_recall
        macro_mertics["f1"] += label_f1
        macro_mertics["acc"] += label_acc
    label_metrics["macro"] = {k:round(v/len_counter_y_true, digits) for k,v in macro_mertics.items()}
    label_metrics["macro"]["support"] = len_y_true
    # micro-mertics
    micro_precision, micro_recall, micro_f1 = calculate_fpr(micro_fpr)
    label_metrics["micro"] = {"precision": round(micro_precision, digits), "recall": round(micro_recall, digits), "f1": round(micro_f1, digits),
                              "acc": round(1-(sum(label_len_dict.values())/len_y_true), digits), "support": len_y_true}
    # weighted-mertics
    label_metrics["weighted"] = {k:round(v/len_y_true, digits) for k,v in weighted_mertics.items()}
    label_metrics["weighted"]["support"] = len_y_true
    # 绘制打印
    report = None
    if use_draw:
        report = draw_mertics(label_metrics, digits=4, use_acc=use_acc)
    return label_metrics, report, label_error_dict


def mertics_report_v2(y_true: Any, y_pred: Any, rounded: int=4, beta: float=1.0, target_names=None):
    """ sklearn版本模型评估指标, 只支持np.array格式输入的y_true/y_pred, 输入必须为onehot
    mertics report, 打印评估指标, 不支持onehot的形式
    Args:
        y_true:  np.array, 真实样本标签, 可以是单个数字标签, 或者是string格式
        y_pred:  np.array, 预测样本标签
        digits:  int,  最后保留位数
        eps:     float, 设置一个eps防止为0
        use_acc: bool, 是否使用acc
        use_draw: bool, 是否返回report
        
    Returns:    tuple<float>, 多个返回值
        label_metrics: dict, 包括单个类别/micro/macro/weighted的 prf
        report:        str,  评估指标打印格式, 同sklearn
        error_idxs:    list<int>, 预测错误标签的位置
    """
    from sklearn.metrics import precision_recall_fscore_support
    result_t = None
    averages = [None, "micro", "macro", "weighted"]
    for average in averages:
        res = precision_recall_fscore_support(y_true, y_pred,
                                              labels=None,
                                              average=average,
                                              sample_weight=None,)
                                              # zero_division=0)
        if average == None:
            result_t = np.array([r for r in res])
        else:
            result_average = np.array([np.array([r]) for r in res])
            result_t = np.hstack((result_t, result_average))
    averages.pop(0)
    # report_dict
    # result_t = result.T
    headers = ["precision", "recall", "f1-score", "support"]
    if not target_names:
        target_names = ["label_{}".format(i) for i in range(len(result_t) - 3)] + [avg + "_avg" for avg in averages]
    else:
        target_names = target_names + [avg + "_avg" for avg in averages]
    rows = zip(target_names, result_t[:, 0], result_t[:, 1], result_t[:, 2], result_t[:, 3])
    report_dict = {label[0]: label[1:] for label in rows}
    for label, scores in report_dict.items():
        report_dict[label] = dict(zip(headers, [round(i, rounded) for i in scores]))
    # report_fmt
    longest_last_line_heading = 'weighted_avg'
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(longest_last_line_heading), rounded)
    head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
    report_hmt = "\n" + head_fmt.format('', *headers, width=width)
    report_hmt += '\n\n'
    row_fmt = '{:>{width}s} ' + ' {:>9.{rounded}f}' * 3 + ' {:>9}\n'
    rows = zip(target_names, result_t[:, 0], result_t[:, 1], result_t[:, 2], np.array(result_t[:, 3], dtype=np.int32))
    count = 0
    for row in rows:
        if count == len(target_names) - 3:
            report_hmt += "\n"
        report_hmt += row_fmt.format(*row, width=width, rounded=rounded)
        count += 1
    report_hmt += '\n'
    # return result_t, report_dict, report_hmt
    return report_dict, report_hmt


def mertics_report(y_true: Any, y_pred: Any, rounded: int=4, beta: float=1.0, target_names=None):
    """
    mertics report, 打印评估指标, 不支持string, onehot的形式
    Args:
        y_true :  list, 真实样本标签, 可以是单个数字标签, 或者是string格式
        y_pred :  list, 预测样本标签
        rounded:  int,  最后保留位数
        beta   :  float, 计算f-score时候的precision的权重
        target_names:  list,  类别标签

    Returns: tuple<float>, 多个返回值
        report_dict:   dict, 包括单个类别/micro/macro/weighted的 prf
        report     :   str,  评估指标打印格式, 同sklearn
    """
    if type(y_true) == list:
        y_true = np.array(y_true)
    if type(y_pred) == list:
        y_pred = np.array(y_pred)
    mcm = mertics_multilabel_confusion_matrix(y_true, y_pred)
    result_t, report_dict, report_hmt = mertics_precision_recall_fscore_support(mcm,
                     beta=beta, rounded=rounded, target_names=target_names)
    return report_dict, report_hmt


def mertics_precision_recall_fscore_support(mcm: Any, beta=1.0, rounded=2, target_names=None):
    """ precision-recall-fscore-support
    计算每个类的精度、召回率、F度量值和支持度。   
    精度是比率``tp/（tp+fp）`，其中``tp``是真阳性和``fp``假阳性的数量。精度是直观地说，分类器不将样本标记为阳性的能力。
    召回率是``tp/（tp+fn）``的比率，其中``tp``是真阳性和``fn``假阴性的数量。召回是直观地分析分类器查找所有正样本的能力。
    F-beta得分可解释为精确性和召回率的加权调和平均值，其中F-beta分数达到最佳值为1，最差分数为0。F-beta评分权重的召回率比精确度高出一倍。
    平均，“微”、“宏”、“加权”或“样本”之一。
    Args:
        mcm     :  {ndarray} of shape (n_outputs, 2, 2), 多标签混淆矩阵
        beta    :  float, weighted of precision when calculate f1-score, like 1.0
        ###averages:  list<str>, average of averages, like ["macro", "micro", "weighted"], ["macro"]
    Returns:    
        result  : {Dict<ndarray>}, 返回结果, prf
    """

    def _prf_calculate(tp_sum, true_sum, pred_sum, beta2):
        """calculate precision, recall and f_score"""
        precision = _prf_divide(tp_sum, pred_sum)
        recall = _prf_divide(tp_sum, true_sum)
        denom = beta2 * precision + recall
        denom[denom == 0.] = 1  # avoid division by 0
        f_score = (1 + beta2) * precision * recall / denom
        return precision, recall, f_score

    def _prf_divide(numerator, denominator):
        """Performs division and handles divide-by-zero.
        On zero-division, sets the corresponding result elements equal to 0 or 1 (according to ``zero_division``).
        Args:
            numerator   :  {array-like, sparse matrix} of shape (labels, 4), 分子
            denominator :  {array-like, sparse matrix} of shape (labels, 4), 分母
        Returns:    
            result : {ndarray} of shape (n_outputs, 2, 2), 多标签混淆矩阵
        """
        mask = denominator == 0.0
        denominator[mask] = 1  # avoid infs/nans
        result = numerator / denominator
        return result

    averages = ["micro", "macro", "weighted"]
    tp_sum = mcm[:, 1, 1]
    pred_sum = tp_sum + mcm[:, 0, 1]
    true_sum = tp_sum + mcm[:, 1, 0]
    beta2 = beta ** 2
    # 计算precision, recall, f_score
    precision, recall, f_score = _prf_calculate(tp_sum, true_sum, pred_sum, beta2)
    result = np.array([precision, recall, f_score, true_sum])
    # average
    for average in averages:
        if "micro" == average:
            true_sum_micro = np.array([true_sum.sum()])
            pred_sum_micro = np.array([pred_sum.sum()])
            tp_sum_micro = np.array([tp_sum.sum()])
            precision_micro, recall_micro, f_score_micro = _prf_calculate(tp_sum_micro, true_sum_micro,
                                                                          pred_sum_micro,
                                                                          beta2)
            # result[average] = np.array([np.array(precision_micro), np.array(recall_micro),
            #                             np.array(f_score_micro), np.array(true_sum_micro)])
            result_average = np.array([np.array(precision_micro), np.array(recall_micro),
                                        np.array(f_score_micro), np.array(true_sum_micro)])
            result = np.hstack((result, result_average))
        else:
            if "weighted" == average:
                weights = true_sum
            else:
                weights = None
            precision_average = np.average(precision, weights=weights)
            recall_average = np.average(recall, weights=weights)
            f_score_average = np.average(f_score, weights=weights)
            # result[average] = [np.array(precision_average), np.array(recall_average), np.array(f_score_average),
            #                    np.array(true_sum.sum())]
            result_average = np.array([np.array([precision_average]), np.array([recall_average]),
                                       np.array([f_score_average]), np.array([true_sum.sum()])])
            result = np.hstack((result, result_average))
    # report_dict
    result_t = result.T
    headers = ["precision", "recall", "f1-score", "support"]
    if not target_names:
        target_names = ["label_{}".format(i) for i in range(len(result_t) - 3)] + [avg + "_avg" for avg in averages]
    else:
        target_names = target_names + [avg + "_avg" for avg in averages]
    rows = zip(target_names, result_t[:, 0], result_t[:, 1], result_t[:, 2], result_t[:, 3])
    report_dict = {label[0]: label[1:] for label in rows}
    for label, scores in report_dict.items():
        report_dict[label] = dict(zip(headers, [round(i.item(), rounded) for i in scores]))

    # report_fmt
    longest_last_line_heading = 'weighted_avg'
    name_width = max(len(cn) for cn in target_names)
    width = max(name_width, len(longest_last_line_heading), rounded)
    head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
    report_hmt = "\n" + head_fmt.format('', *headers, width=width)
    report_hmt += '\n\n'
    row_fmt = '{:>{width}s} ' + ' {:>9.{rounded}f}' * 3 + ' {:>9}\n'
    rows = zip(target_names, result_t[:, 0], result_t[:, 1], result_t[:, 2], np.array(result_t[:, 3], dtype=np.int32))
    count = 0
    for row in rows:
        if count == len(target_names) - 3:
            report_hmt += "\n"
        report_hmt += row_fmt.format(*row, width=width, rounded=rounded)
        count += 1
    report_hmt += '\n'
    return result_t, report_dict, report_hmt


def mertics_multilabel_confusion_matrix(y_true: Any, y_pred: Any):
    """ multilabel-confusion-matrix
    计算每个类或样本的混淆矩阵，按类计算（默认）多标签，
    用于评估分类准确性的混淆矩阵，以及输出每个类别或样本的混淆矩阵。        
    Args:
        y_true :  {array-like, sparse matrix} of shape (n_samples, n_outputs), 真实样本标签, like [[1,0,1], [1,1,1]]
        y_pred :  {array-like, sparse matrix} of shape (n_samples, n_outputs), 预测样本标签, like [[1,0,0], [0,0,1]]
    Returns:    
        multi_confusion : {ndarray} of shape (n_outputs, 2, 2), 多标签混淆矩阵
    """
    if type(y_true) == list:
        y_true = np.array(y_true)
    if type(y_true) == list:
        y_pred = np.array(y_pred)
    # 计算tp, tn, fp, fn等超参数
    ## todo, 将np.array的类型改为压缩的csr-matrix, coo-matrix等
    true_and_pred = y_true * y_pred  # 每个元素相乘, 0,1, 可替换为压缩矩阵"csr"的形式
    tp_sum = np.sum(true_and_pred, axis=0)
    pred_sum = np.sum(y_pred, axis=0)
    true_sum = np.sum(y_true, axis=0)
    fp = pred_sum - tp_sum
    fn = true_sum - tp_sum
    tp = tp_sum
    tn = y_true.shape[0] - tp - fp - fn
    return np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)


def sklearn_confusion_matrix(y_true: List, y_pred: List):
    """ sklearn, 先multi-class, 然后multi-label
    calculate confusion_matrix, 计算混淆矩阵
    Args:
        y_true: list, label of really true, eg. ["TRUE", "FALSE"]
        y_pred: list, label of model predict, eg. ["TRUE", "FALSE"]
    Returns:
        metrics(confusion)
    """
    from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
    try:
        confusion = confusion_matrix(y_true, y_pred)
    except Exception as e:
        logging.info(str(e))
        confusion = multilabel_confusion_matrix(y_true, y_pred)
    return confusion


def sklearn_kfold(xys: np.array, n_splits: int=5, shuffle: bool=False, random_state: int=None):
    """
    StratifiedKFold of sklearn, 留一K折交叉验证数据获取
    Args:
        xys: np.array, corpus
        n_splits: int, number of folds. must be at least 2.
        shuffle: boolean, Whether to shuffle each stratification of the data before splitting into batches.
        random_state: int, random-state instance or none
    Returns:
        xys_kfold: List<Tuple>
    """
    from sklearn.model_selection import StratifiedKFold
    skfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    X, Y = xys
    xys_kfold = []
    for train_idx, dev_idx in skfold.split(X, Y):
        xys_train = X[train_idx], Y[train_idx]
        xys_dev = X[dev_idx], Y[dev_idx]
        xys_kfold.append((xys_train, xys_dev))
    return xys_kfold


def sent_mertic_det(all_srcs, all_pres, all_trgs, logger=None, flag_eval="common"):
    """
    句子级别检测指标：所有位置都检测对才算对
    code-from: https://github.com/wangwang110/CSC
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


def sent_mertic_cor(all_srcs, all_pres, all_trgs, logger=None, flag_eval="common"):
    """
    句子级别纠正指标：所有位置纠正对才算对
    code-from: https://github.com/wangwang110/CSC
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


def char_mertic_det_cor(all_srcs, all_pres, all_trgs, logger=None):
    """
    copy from https://github.com/sunnyqiny/Confusionset-guided-Pointer-Networks-for-Chinese-Spelling-Check/blob/master/utils/evaluation_metrics.py
    """
    TP = 0
    FP = 0
    FN = 0
    all_predict_true_index = []
    all_gold_index = []
    # for item in results:
    # for _ in range(len(src)):
        # src, tgt, predict = item
    for src, tgt, predict in zip(all_srcs, all_trgs, all_pres):
        gold_index = []
        each_true_index = []
        for i in range(len(list(src))):
            if src[i] == tgt[i]:
                continue
            else:
                gold_index.append(i)
        all_gold_index.append(gold_index)
        predict_index = []
        for i in range(len(list(src))):
            if src[i] == predict[i]:
                continue
            else:
                predict_index.append(i)

        for i in predict_index:
            if i in gold_index:
                TP += 1
                each_true_index.append(i)
            else:
                FP += 1
        for i in gold_index:
            if i in predict_index:
                continue
            else:
                FN += 1
        all_predict_true_index.append(each_true_index)

    # For the detection Precision, Recall and F1
    detection_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    detection_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    if detection_precision + detection_recall == 0:
        detection_f1 = 0
    else:
        detection_f1 = 2 * (detection_precision * detection_recall) / (detection_precision + detection_recall)
    # logger.info(
    #     "The detection result is precision={}, recall={} and F1={}".format(detection_precision, detection_recall,
    #                                                                        detection_f1))

    TP = 0
    FP = 0
    FN = 0

    for i in range(len(all_predict_true_index)):
        # we only detect those correctly detected location, which is a different from the common metrics since
        # we want to see the precision improve by using the confusionset
        if len(all_predict_true_index[i]) > 0:
            predict_words = []
            for j in all_predict_true_index[i]:
                predict_words.append(all_pres[i][j])
                if all_trgs[i][j] == all_pres[i][j]:  # 0,24  # 0-23/0-23
                    TP += 1
                else:
                    FP += 1
            for j in all_gold_index[i]:
                if all_trgs[i][j] in predict_words:
                    continue
                else:
                    FN += 1
    # src, tgt, predict
    # For the correction Precision, Recall and F1
    correction_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    correction_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    if correction_precision + correction_recall == 0:
        correction_f1 = 0
    else:
        correction_f1 = 2 * (correction_precision * correction_recall) / (correction_precision + correction_recall)
    # logger.info("The correction result is precision={}, recall={} and F1={}".format(correction_precision,
    #                                                                                 correction_recall,
    #                                                                                 correction_f1))
    return detection_precision, detection_recall, detection_f1, \
           correction_precision, correction_recall, correction_f1


def save_json(lines: Union[List, Dict], path: str, encoding: str = "utf-8", indent: int = 4):
    """
    Write Line of List<json> to file
    Args:
        lines: lines of list[str] which need save
        path: path of save file, such as "json.txt"
        encoding: type of encoding, such as "utf-8", "gbk"
    """

    with open(path, "w", encoding=encoding) as fj:
        fj.write(json.dumps(lines, ensure_ascii=False, indent=indent))
    fj.close()


def txt_write(lines: List[str], path: str, model: str = "w", encoding: str = "utf-8"):
    """
    Write Line of list to file
    Args:
        lines: lines of list<str> which need save
        path: path of save file, such as "txt"
        model: type of write, such as "w", "a+"
        encoding: type of encoding, such as "utf-8", "gbk"
    """

    try:
        file = open(path, model, encoding=encoding)
        file.writelines(lines)
        file.close()
    except Exception as e:
        logging.info(str(e))


def get_logger(log_dir, back_count=32, logger_name="macro_correct_csc_token"):
    """
    get_current_time from time
    Args:
        log_dir: str, log dir of path
        back_count: int, max file-name
        logger_name: str
    Returns:
        logger: class
    """

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    # 日志文件名,为启动时的日期
    log_file_name = time.strftime("{}-%Y-%m-%d".format(logger_name), time.localtime(time.time())) + ".log"
    log_name_day = os.path.join(log_dir, log_file_name)
    logger_level = logging.INFO
    # 全局日志格式
    logging.basicConfig(format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s", level=logger_level)
    # 定义一个日志记录器
    logger = logging.getLogger("macro_correct_csc_token")
    # 文件输出, 定义一个RotatingFileHandler，最多备份32个日志文件，每个日志文件最大32K
    fHandler = RotatingFileHandler(log_name_day, maxBytes=back_count * 1024 * 1024, backupCount=back_count, encoding="utf-8")
    fHandler.setLevel(logger_level)
    logger.addHandler(fHandler)
    # 控制台输出
    console = logging.StreamHandler()
    console.setLevel(logger_level)
    logger.addHandler(console)
    return logger


def load_json(path: str, encoding: str="utf-8") -> Union[List, Any]:
    """
    Read Line of List<json> form file
    Args:
        path: path of save file, such as "txt"
        encoding: type of encoding, such as "utf-8", "gbk"
    Returns:
        model_json: dict of word2vec, eg. [{"大漠帝国":132}]
    """
    with open(path, "r", encoding=encoding) as fj:
        model_json = json.load(fj)
        fj.close()
    return model_json


def txt_read(path: str, encoding: str = "utf-8") -> List[str]:
    """
    Read Line of list form file
    Args:
        path: path of save file, such as "txt"
        encoding: type of encoding, such as "utf-8", "gbk"
    Returns:
        dict of word2vec, eg. {"macadam":[...]}
    """

    lines = []
    try:
        file = open(path, "r", encoding=encoding)
        lines = file.readlines()
        file.close()
    except Exception as e:
        logging.info(str(e))
    finally:
        return lines


def save_pickle(data, path):
    """ 存储 """
    with open(path, "wb") as fp:
        pickle.dump(data, fp, protocol=4)


def load_pickle(path):
    """ 加载 """
    with open(path, "rb") as fp:
        data = pickle.load(fp)
    return data


def get_current_time(time_form: str="%Y%m%d%H%M%S"):
    """
    get_current_time from time
    Args:
        time_form: str, string form of return time
    Returns:
        time_current: str
    """
    time_now = int(time.time())
    time_local = time.localtime(time_now)
    time_current = time.strftime(time_form, time_local)
    return time_current


def chinese_extract_extend(text: str) -> str:
    """
      只提取出中文、字母和数字
    :param text: str, input of sentence
    :return: str
    """
    chinese_extract = "".join(re.findall(u"([\u4e00-\u9fa5A-Za-z0-9@. ])", text))
    return chinese_extract


def flag_total_chinese(text: str) -> bool:
    """
    judge is total chinese or not, 判断是不是全是中文
    Args:
        text: str, eg. "macadam, 碎石路"
    Returns:
        bool, True or False
    """
    for word in text:
        if not "\u4e00" <= word <= "\u9fa5":
            return False
    return True


def flag_total_number(text: str) -> bool:
    """
    judge is total chinese or not, 判断是不是全是数字
    Args:
        text: str, eg. "macadam, 碎石路"
    Returns:
        bool, True or False
    """
    for word in text:
        if word not in "0123456789.%":
            return False
    return True


def dic_sort(dic: dict)-> OrderedDict:
    """
    sort dict by values, 给字典排序(依据值大小)
    Args:
        dic: dict, eg. {"游戏:132, "旅游":32}
    Returns:
        OrderedDict
    """
    in_dict_sort = sorted(dic.items(), key=lambda x:x[1], reverse=True)
    return OrderedDict(in_dict_sort)


def del_dir(path_dir: str):
    """
    Delete model files in the directory, eg. h5/json/pb 
    Args:
        path_dir: path of directory, where many files in the directory
    Returns:
        None
    """
    for i in os.listdir(path_dir):
        # 取文件或者目录的绝对路径
        path_children = os.path.join(path_dir, i)
        if os.path.isfile(path_children):
            if path_children.endswith(".h5") or path_children.endswith(".json") or path_children.endswith(".pb"):
                os.remove(path_children)
        else:  # 递归, 删除目录下的所有文件
            del_dir(path_children)


def sigmoid(x):
    """
    sigmoid 
    Args:
        x: np.array
    Returns:
        s: np.array
    """
    s = 1 / (1 + np.exp(-x))
    return s


def softmax(x):
    """
    softmax 
    Args:
        x: np.array
    Returns:
        s: np.array
    """
    s = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return s


def cut_sent_by_stay(text, return_length=True):
    """  分句但是保存原标点符号  """
    text_sp = re.split(r"！”|？”|。”|……”|”！|”？|”。|”……|》。|）。|！|？|。|…|\!|\?", text)
    conn_symbol = "！？。…”!?》）\n"
    text_length_s = []
    text_cut = []
    len_text = len(text) - 1
    # signal_symbol = "—”>；？…）‘《’（·》“~，、！。：<"
    len_global = 0
    for idx, text_sp_i in enumerate(text_sp):
        text_cut_idx = text_sp[idx]
        len_global_before = copy.deepcopy(len_global)
        len_global += len(text_sp_i)
        while True:
            if len_global <= len_text and text[len_global] in conn_symbol:
                text_cut_idx += text[len_global]
            else:
                # len_global += 1
                if text_cut_idx:
                    text_length_s.append([len_global_before, len_global])
                    text_cut.append(text_cut_idx)
                break
            len_global += 1
    if return_length:
        return text_cut, text_length_s
    return text_cut
def count_flag_zh(text):
    """   统计包含中文的部分  """
    count = 0
    for text_i in text:
        if flag_chinese_string(text_i):
            count += 1
    return count
def cut_sent_len(text):
    """  分句, 包括句子长度  """
    re_sen = re.compile('[!?。？！…]')
    text_sp = re_sen.split(text)
    text_length_s = []
    pos_end = 0
    for idx, text_i in enumerate(text_sp):
        text_length_s.append([pos_end, pos_end+len(text_i)])
        pos_end += len(text_i) + 1
    return text_sp, text_length_s


def flag_no_chinese_number_alphabet_signal(uchar):
    """判断是否非汉字，数字和英文字符，其他数字字符"""
    return not (flag_chinese_char(uchar) or flag_number(uchar) or flag_alphabet(uchar) or flag_number_other(uchar))
def flag_chinese_char(uchar):
    """判断一个unicode是否是汉字"""
    return '\u4e00' <= uchar <= '\u9fa5'
def flag_number_other(uchar):
    """判断数字符号"""
    return True if uchar in "%.+/" else False
def flag_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    return '\u0041' <= uchar <= '\u005a' or '\u0061' <= uchar <= '\u007a'
def flag_number(uchar):
    """判断一个unicode是否是数字"""
    return '\u0030' <= uchar <= '\u0039'


def flag_alphabet_string(string):
    """判断是否全部为英文字母"""
    return all(flag_alphabet(c) for c in string)
def flag_chinese_string(string):
    """判断是否全为汉字"""
    return all(flag_chinese_char(c) for c in string)


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
def string_b2q(ustring):
    """把字符串半角转全角"""
    return "".join([b2q(uchar) for uchar in ustring])
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
def b2q(uchar):
    """半角转全角"""
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e:  # 不是半角字符就返回原来的字符
        return uchar
    if inside_code == 0x0020:  # 除了空格其他的全角半角的公式为:半角=全角-0xfee0
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return chr(inside_code)


def get_errors_for_diff_length(corrected_text, origin_text):
    """Get errors between corrected text and origin text
    code from:  https://github.com/shibing624/pycorrector
    """
    new_corrected_text = ""
    errors = []
    i, j = 0, 0
    unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '擤', '\t', '玕', '']

    while i < len(origin_text) and j < len(corrected_text):
        if origin_text[i] in unk_tokens:
            new_corrected_text += origin_text[i]
            i += 1
        elif corrected_text[j] in unk_tokens:
            new_corrected_text += corrected_text[j]
            j += 1
        # Deal with Chinese characters
        elif flag_total_chinese(origin_text[i]) and flag_total_chinese(corrected_text[j]):
            # If the two characters are the same, then the two pointers move forward together
            if origin_text[i] == corrected_text[j]:
                new_corrected_text += corrected_text[j]
                i += 1
                j += 1
            else:
                # Check for insertion errors
                if j + 1 < len(corrected_text) and origin_text[i] == corrected_text[j + 1]:
                    errors.append(('', corrected_text[j], j))
                    new_corrected_text += corrected_text[j]
                    j += 1
                # Check for deletion errors
                elif i + 1 < len(origin_text) and origin_text[i + 1] == corrected_text[j]:
                    errors.append((origin_text[i], '', i))
                    i += 1
                # Check for replacement errors
                else:
                    errors.append((origin_text[i], corrected_text[j], i))
                    new_corrected_text += corrected_text[j]
                    i += 1
                    j += 1
        else:
            new_corrected_text += origin_text[i]
            if origin_text[i] == corrected_text[j]:
                j += 1
            i += 1
    errors = sorted(errors, key=operator.itemgetter(2))
    return new_corrected_text, errors


def get_errors_for_same_length(corrected_text, origin_text, unk_tokens=[]):
    """Get new corrected text and errors between corrected text and origin text
    code from:  https://github.com/shibing624/pycorrector
    """
    errors = []
    unk_tokens = unk_tokens or [' ', '“', '”', '‘', '’', '琊', '\n', '…', '擤', '\t', '玕', '', '，']

    for i, ori_char in enumerate(origin_text):
        if i >= len(corrected_text):
            continue
        if ori_char in unk_tokens:
            # deal with unk word
            corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
            continue
        if ori_char != corrected_text[i]:
            if not flag_total_chinese(ori_char):
                # pass not chinese char
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                continue
            if not flag_total_chinese(corrected_text[i]):
                corrected_text = corrected_text[:i] + corrected_text[i + 1:]
                continue
            errors.append([ori_char, corrected_text[i], i])
    errors = sorted(errors, key=operator.itemgetter(2))
    return corrected_text, errors


def get_errors_for_difflib_offical(corrected_text, origin_text):
    """官方版本; Get errors between corrected text and origin text"""
    errors = []
    unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '擤', '\t', '玕', '']

    s = difflib.SequenceMatcher(None, origin_text, corrected_text)
    new_corrected_text = ""
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        print(tag, i1, i2, j1, j2)
        if tag == 'replace':
            for i, j in zip(range(i1, i2), range(j1, j2)):
                if origin_text[i] not in unk_tokens and corrected_text[j] not in unk_tokens:
                    errors.append([origin_text[i], corrected_text[j], i])
                new_corrected_text += corrected_text[j]
        elif tag == 'delete':
            for i in range(i1, i2):
                if origin_text[i] not in unk_tokens:
                    errors.append([origin_text[i], '', i])
            # 不拼接被删除的字符
        elif tag == 'insert':
            for j in range(j1, j2):
                if corrected_text[j] not in unk_tokens:
                    errors.append(['', corrected_text[j], j])
                new_corrected_text += corrected_text[j]
        elif tag == 'equal':
            new_corrected_text += origin_text[i1:i2]

    errors = sorted(errors, key=lambda x: x[2])
    return new_corrected_text, errors


def get_errors_for_difflib(corrected_text, origin_text, flag_same=True):
    """Get errors between corrected text and origin text,
    flag_same: same length of corrected_text and origin_text. while flag_same=True, no delete or insert
    code from: https://github.com/shibing624/pycorrector
    依旧不完全, 如奸缔莉月 -> 如果莉蜜月
    """
    errors = []
    unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '擤', '\t', '玕', '']

    s = difflib.SequenceMatcher(None, origin_text, corrected_text)
    new_corrected_text = ""
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        print(new_corrected_text)
        print(tag, i1, i2, j1, j2)

        if tag == 'replace':
            for i, j in zip(range(i1, i2), range(j1, j2)):
                if origin_text[i] not in unk_tokens and corrected_text[j] not in unk_tokens:
                    errors.append([origin_text[i], corrected_text[j], i])
                new_corrected_text += corrected_text[j]
        elif tag == 'delete' and flag_same:
            for i in range(i1, i2):
                if origin_text[i] not in unk_tokens:
                    errors.append([origin_text[i], '', i])
                new_corrected_text += origin_text[i]
        elif tag == 'insert' and flag_same:
            for j in range(j1, j2):
                if corrected_text[j] not in unk_tokens:
                    errors.append(['', corrected_text[j], j])
                new_corrected_text += corrected_text[j]
        elif tag == 'delete' and not flag_same:
            for i in range(i1, i2):
                if origin_text[i] not in unk_tokens:
                    errors.append([origin_text[i], '', i])
            #不拼接被删除的字符, 如果flag_same
        elif tag == 'insert' and not flag_same:
            for j in range(j1, j2):
                if corrected_text[j] not in unk_tokens:
                    errors.append(['', corrected_text[j], j])
                new_corrected_text += corrected_text[j]
        elif tag == 'equal':
            new_corrected_text += origin_text[i1:i2]

    errors = sorted(errors, key=lambda x: x[2])
    return new_corrected_text, errors


def get_errors(corrected_text, origin_text):
    """Get errors between corrected text and origin text
    code from: https://github.com/shibing624/pycorrector
    """
    errors = []
    unk_tokens = [' ', '“', '”', '‘', '’', '琊', '\n', '…', '擤', '\t', '玕', '']

    s = difflib.SequenceMatcher(None, origin_text, corrected_text)
    new_corrected_text = ""
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'replace':
            for i, j in zip(range(i1, i2), range(j1, j2)):
                if origin_text[i] not in unk_tokens and corrected_text[j] not in unk_tokens:
                    errors.append([origin_text[i], corrected_text[j], i])
                new_corrected_text += corrected_text[j]
        elif tag == 'delete':
            for i in range(i1, i2):
                if origin_text[i] not in unk_tokens:
                    errors.append([origin_text[i], '', i])
            # 不拼接被删除的字符
        elif tag == 'insert':
            for j in range(j1, j2):
                if corrected_text[j] not in unk_tokens:
                    errors.append(['', corrected_text[j], j])
                new_corrected_text += corrected_text[j]
        elif tag == 'equal':
            new_corrected_text += origin_text[i1:i2]

    errors = sorted(errors, key=lambda x: x[2])
    return new_corrected_text, errors


if __name__ == '__main__':
    # y_new = '我也便加珍惜这个季节，希望能够如果莉蜜月春天的时光，多做一些有意义的事情，让自己的生活更加丰富多彩。'
    # y_org = '我也便加珍惜这个委节，希望能够如奸缔莉月春天的时光，多做一些有意义的事情，让自己的生活更加丰富多彩1。'
    y_org = '如奸缔莉月'
    y_new = '如果莉蜜月'
    # y_new_op, errors = get_errors_for_difflib(y_new, y_org)
    # y_new_op, errors = get_errors_for_difflib_offical(y_new, y_org)
    y_new_op, errors = get_errors(y_new, y_org)
    print(y_org)
    print(y_new)
    print(y_new_op)
    print(errors)


    """
    y_org:良好得行业生态不应只让消费者炼就"火眼金睛"，自辨真伪，而是应该主动提供看的见摸的着得透明服务。
    y_new:良好的行业生态不应只让消费者炼就"火眼金睛"，自辨真伪，而是应该主动提供看得见摸得着的透明服务。
    y_new_op, errors = get_errors(y_new, y_org)
    print(errors)
    [('得', '的', 2), ('的', '得', 37), ('', '得', 40), ('', '着', 41), ('着', '', 41), ('得', '', 42)]
    """


    # unit test
    y_true = ["O", "I", "O", "B"]
    y_pred = ["I", "I", "O", "B"]

    metrics, report, error_idx = mertics_report_v1(y_true, y_pred)
    print(report)
    confusion_matrix = sklearn_confusion_matrix(y_true, y_pred)
    print(confusion_matrix)

    y_true = [[1,1,1,1], [0,0,0,1], [0,1,0,0]]
    y_pred = [[0,1,1,0], [1,0,0,1], [0,1,1,0]]
    confusion_matrix = mertics_multilabel_confusion_matrix(y_true, y_pred)
    print(confusion_matrix)

    print(get_current_time())

    # mertics
    metrics, report = mertics_report(y_true, y_pred)
    print(report)


    logger = get_logger("./log")
    logger.info("12306")
