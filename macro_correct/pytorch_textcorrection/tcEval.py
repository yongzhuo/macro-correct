# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/12/21 17:35
# @author  : Mo
# @function: 推理预测


import traceback
import logging
import random
import json
import copy
import time
import re
import os
# os.environ["USE_TORCH"] = "1"
import sys
import os

path_sys = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(path_sys)
print(path_sys)

from tcTqdm import tqdm
import opencc


converter = opencc.OpenCC("t2s.json")
# converter = opencc.OpenCC("tw2sp.json")
# converter = opencc.OpenCC("tw2s.json")
context = converter.convert("汉字")  # 漢字
print(context)
PUN_EN2ZH_DICT = {",": "，", ";": "；", "!": "！", "?": "？", ":": "：",
                  "(": "（", ")": "）", "_": "—"}
PUN_ZH2EN_DICT = {'，': ',', '；': ';', '！': '!', '？': '?', '：': ':', '（': '(', '）': ')',
                      '—': '-', '～': '~', '＜': '<', '＞': '>',
                      "‘": "'", "’": "'", '”': '"', '“': '"',
                      }
PUN_BERT_DICT = {
    "“":'"', "”":'"', "‘":'"', "’":'"', "—": "_", "——": "__"}


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


def txt_write(lines, path, model="w", encoding="utf-8"):
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


def save_json(lines, path, encoding="utf-8", indent=4):
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


def load_json(path, encoding="utf-8"):
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


def txt_read(path, encoding="utf-8"):
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


def eval_std_list(path_tet=None, model_name_or_path=None, threshold=0.0):
    """   推理。预测等   """
    from macro_correct.pytorch_textcorrection.tcPredict import TextCorrectPredict
    model_macbert = TextCorrectPredict(model_name_or_path)
    texts = [{"original_text": "真麻烦你了。希望你们好好的跳无"},
             {"original_text": "少先队员因该为老人让坐"},
             {"original_text": "机七学习是人工智能领遇最能体现智能的一个分知"},
             {"original_text": "一只小鱼船浮在平净的河面上"},
             {"original_text": "我的家乡是有明的渔米之乡"},
             {"original_text": "chuxī到了,兰兰和妈妈一起包jiǎo子,兰兰包的jiǎo子十分可爱,\n今天,兰兰和妈妈dù过了一个&快乐的chúxī。"},
             ]

    res = model_macbert.predict_batch(texts, flag_logits=False, threshold=0.01)
    # model_macbert.office.save_model_state("./macbert4mdcspell_extend")
    # print(res)
    for r in res:
        print(r)

    # 真麻烦你了。希望你们好好地跳舞
    texts = [
        '真麻烦你了。希望你们好好的跳无',
        '少先队员因该为老人让坐',
        '还有广告业是只要桌子前面坐者工作未必产生出来好的成果。',
        '寒假的作业里有一个实验—巧制冰糖,根据要求,我按照上面的题示做起了实验',
    ]
    params = {
        # "max_length": 128,
        # "batch_size": 32,
        # "threshold": 0.75,
        # "threshold": 0.6,
        # "threshold": 0.5,
        # "threshold": 0.4,
        # "threshold": 0.3,
        # "threshold": 0.2,
        # "threshold": 0.1,
        "threshold": threshold,
        # "silent": False
    }

    texts_predict = model_macbert.predict_batch(texts, **params)
    print(texts_predict)
    print("#" * 128)

    datas = load_json(path_tet)
    file_name = os.path.split(path_tet)[-1]

    wrong_ids_es = []
    analysis_es = []
    srcs = []
    tgts = []
    for line in datas:
        if "original_text" in line:
            src = line.get("original_text", "")
            tgt = line.get("correct_text", "")
        else:
            src = line.get("source", "")
            tgt = line.get("target", "")
        # if len(srcs) < 100:
        #     print(src, tgt)
        # wrong_ids = line.get("wrong_ids", "")
        # analysis = "".join([f"'{src[s]}'改为'{tgt[s]}'" for s in wrong_ids])
        # analysis = ""
        try:
            src = string_q2b(src)  # 全角转半角
            tgt = string_q2b(tgt)  # 全角转半角
            src = tradition_to_simple(src)
            tgt = tradition_to_simple(tgt)
            src = transfor_english_symbol_to_chinese(src)
            tgt = transfor_english_symbol_to_chinese(tgt)
            # src = transfor_chinese_symbol_to_english(src)
            # tgt = transfor_chinese_symbol_to_english(tgt)
            src = transfor_bert_unk_pun_to_know(src)
            tgt = transfor_bert_unk_pun_to_know(tgt)
            myz = 0
        except Exception as e:
            print(traceback.print_exc())
            continue

        # wrong_ids_es.append(wrong_ids)
        # analysis_es.append(analysis)
        srcs.append(src[:model_macbert.config.max_len-2])
        tgts.append(tgt[:model_macbert.config.max_len-2])


    # print(len(srcs))
    # max_len = len(srcs) + 1  # 256
    # # max_len = 256
    # print(model_name_or_path)
    # res = pred_csc_sentence_by_batch(model_macbert.predict, srcs[:max_len], tgts[:max_len],
    #                                  analysis_es[:max_len], wrong_ids_es[:max_len],
    #                                  params)
    # print(res)
    # print(model_name_or_path)
    # print(path_tet)

    max_len = len(srcs) + 1
    srcs = srcs[:max_len]
    tgts = tgts[:max_len]

    src_batch = []
    preds = []
    for s in tqdm(srcs, desc=file_name):
        src_batch.append(s)
        if len(src_batch) == 32:
            res_i = model_macbert.predict_batch(src_batch, **params)
            preds.extend([r.get("target", "") for r in res_i])
            src_batch = []
    if src_batch:
        res_i = model_macbert.predict_batch(src_batch, **params)
        preds.extend([r.get("target", "") for r in res_i])
        src_batch = []
    print("#" * 128)
    common_det_acc, common_det_precision, common_det_recall, common_det_f1 = sent_mertic_det(
        srcs, preds, tgts)
    common_cor_acc, common_cor_precision, common_cor_recall, common_cor_f1 = sent_mertic_cor(
        srcs, preds, tgts)

    common_det_mertics = f'common Sentence Level detection: acc:{common_det_acc:.4f}, precision:{common_det_precision:.4f}, recall:{common_det_recall:.4f}, f1:{common_det_f1:.4f}'
    common_cor_mertics = f'common Sentence Level correction: acc:{common_cor_acc:.4f}, precision:{common_cor_precision:.4f}, recall:{common_cor_recall:.4f}, f1:{common_cor_f1:.4f}'
    print("#" * 128)
    print(f'flag_eval: common')
    print(common_det_mertics)
    print(common_cor_mertics)
    print("#" * 128)

    strict_det_acc, strict_det_precision, strict_det_recall, strict_det_f1 = sent_mertic_det(
        srcs, preds, tgts, flag_eval="strict")
    strict_cor_acc, strict_cor_precision, strict_cor_recall, strict_cor_f1 = sent_mertic_cor(
        srcs, preds, tgts, flag_eval="strict")

    strict_det_mertics = f'strict Sentence Level detection: acc:{strict_det_acc:.4f}, precision:{strict_det_precision:.4f}, recall:{strict_det_recall:.4f}, f1:{strict_det_f1:.4f}'
    strict_cor_mertics = f'strict Sentence Level correction: acc:{strict_cor_acc:.4f}, precision:{strict_cor_precision:.4f}, recall:{strict_cor_recall:.4f}, f1:{strict_cor_f1:.4f}'
    print("#" * 128)
    print(f'flag_eval: strict')
    print(strict_det_mertics)
    print(strict_cor_mertics)
    print("#" * 128)

    sent_mertics = {
        "common_det_acc": common_det_acc,
        "common_det_precision": common_det_precision,
        "common_det_recall": common_det_recall,
        "common_det_f1": common_det_f1,

        "common_cor_acc": common_cor_acc,
        "common_cor_precision": common_cor_precision,
        "common_cor_recall": common_cor_recall,
        "common_cor_f1": common_cor_f1,

        "strict_det_acc": strict_det_acc,
        "strict_det_precision": strict_det_precision,
        "strict_det_recall": strict_det_recall,
        "strict_det_f1": strict_det_f1,

        "strict_cor_acc": strict_cor_acc,
        "strict_cor_precision": strict_cor_precision,
        "strict_cor_recall": strict_cor_recall,
        "strict_cor_f1": strict_cor_f1,
    }

    detection_precision, detection_recall, detection_f1, \
    correction_precision, correction_recall, correction_f1 = char_mertic_det_cor(srcs, preds, tgts)
    token_result = {"det_precision": detection_precision, "det_recall": detection_recall, "det_f1": detection_f1,
                    "cor_precision": correction_precision, "cor_recall": correction_recall, "cor_f1": correction_f1,
                    }

    token_det_mertics = f'common Token Level correction: precision:{detection_precision:.4f}, recall:{detection_recall:.4f}, f1:{detection_f1:.4f}'
    token_cor_mertics = f'common TOken Level correction: precision:{correction_precision:.4f}, recall:{correction_recall:.4f}, f1:{correction_f1:.4f}'
    print("#" * 128)
    print(f'flag_eval: token')
    print(token_det_mertics)
    print(token_cor_mertics)
    print("#" * 128)

    result_mertics = {"task": os.path.split(path_tet)[-1], "sent": sent_mertics, "token": token_result}

    # ### 测评校验打印
    # count = 0
    # for s, p, t in zip(srcs, preds, tgts):
    #     if p != t:
    #         print("s:", s)
    #         print("t:", t)
    #         print("p:", p)
    #         print("#"*128)
    #         count += 1
    # print("count: ", count)

    return common_det_mertics, common_cor_mertics, strict_det_mertics, strict_cor_mertics, \
           token_det_mertics, token_cor_mertics, result_mertics



if __name__ == '__main__':
    yz = 0

    ### 全量测试
    ### 模型权重
    # path_model_dir = "E:/DATA/bert-model/00_pytorch/MacBERT-chinese_finetuned_correction/csc.config"
    # path_model_dir = "../output/text_correction/macbert4csc_v1/csc.config"  # own-correct
    # path_model_dir = "../output/text_correction/macbert4csc_v2/csc.config"  # own-correct
    # path_model_dir = "../output/text_correction/macbert4mdcspell_v1/csc.config"  # own-correct
    # path_model_dir = "../output/text_correction/bert4csc_v1/csc.config"  # own-correct
    path_model_dir = "../output/text_correction/macbert4mdcspell_extend/csc.config"  # own-correct
    threshold = 0.75
    # threshold = 0.0

    ### 数据集(只用了.train训练(大概1000w数据集), dev/test都没有参与训练)
    path_corpus_dir = os.path.join(path_sys, "macro_correct", "corpus", "text_correction")
    path_tet1 = os.path.join(path_corpus_dir, "public/gen_de3.json")
    path_tet2 = os.path.join(path_corpus_dir, "public/lemon_v2.tet.json")
    path_tet3 = os.path.join(path_corpus_dir, "public/acc_rmrb.tet.json")
    path_tet4 = os.path.join(path_corpus_dir, "public/acc_xxqg.tet.json")

    path_tet5 = os.path.join(path_corpus_dir, "public/gen_passage.tet.json")
    path_tet6 = os.path.join(path_corpus_dir, "public/textproof.tet.json")
    path_tet7 = os.path.join(path_corpus_dir, "public/gen_xxqg.tet.json")

    path_tet8 = os.path.join(path_corpus_dir, "public/faspell.dev.json")
    path_tet9 = os.path.join(path_corpus_dir, "public/lomo_tet.json")
    path_tet10 = os.path.join(path_corpus_dir, "public/mcsc_tet_5k.json")
    path_tet11 = os.path.join(path_corpus_dir, "public/ecspell.dev.json")
    path_tet12 = os.path.join(path_corpus_dir, "public/sighan2013.dev.json")
    path_tet13 = os.path.join(path_corpus_dir, "public/sighan2014.dev.json")
    path_tet14 = os.path.join(path_corpus_dir, "public/sighan2015.dev.json")
    path_tet15 = os.path.join(path_corpus_dir, "public/mcsc_tet.json")



    path_tet_list = [path_tet1, path_tet2, path_tet3, path_tet4, path_tet5,
                     path_tet6, path_tet7, path_tet8, path_tet9, path_tet10,
                     path_tet11, path_tet12, path_tet13, path_tet14, # path_tet15,
                     ]

    for path in path_tet_list:
        if os.path.exists(path):
            datas = load_json(path)
            print(os.path.split(path)[-1])
            print(len(datas))

    string_mertics_list = [path_model_dir + "\n"]
    result_mertics_total = {}
    for path_tet_i in tqdm(path_tet_list, desc="path_tet_list"):
        common_det_mertics, common_cor_mertics, strict_det_mertics, strict_cor_mertics, \
        token_det_mertics, token_cor_mertics, result_mertics \
            = eval_std_list(path_tet=path_tet_i, model_name_or_path=path_model_dir, threshold=threshold)
        file_name = os.path.split(path_tet_i)[-1]
        print(path_tet_i)
        print(file_name)
        string_mertics_list.append("path_tet_i: " + os.path.split(path_tet_i)[-1] + "\n")
        string_mertics_list.append(common_det_mertics + "\n")
        string_mertics_list.append(common_cor_mertics + "\n")
        string_mertics_list.append(strict_det_mertics + "\n")
        string_mertics_list.append(strict_cor_mertics + "\n")
        string_mertics_list.append(token_det_mertics + "\n")
        string_mertics_list.append(token_cor_mertics + "\n")
        string_mertics_list.append("#" * 128 + "\n")
        result_mertics_total[file_name] = result_mertics

    # model_name = path_model_dir.split("/")[-2]
    # txt_write(string_mertics_list, model_name + "_pred_mertics.txt")
    # save_json(result_mertics_total, model_name + "_result_mertics_total.json")
    path_model_dir_save = os.path.split(path_model_dir)[0]

    # txt_write(string_mertics_list, os.path.join(path_model_dir_save, "eval_pred_mertics.txt"))
    # save_json(result_mertics_total, os.path.join(path_model_dir_save, "eval_result_mertics_total.json"))
    txt_write(string_mertics_list, os.path.join(path_model_dir_save, "eval_std.pred_mertics.txt"))
    save_json(result_mertics_total, os.path.join(path_model_dir_save, "eval_std.pred_mertics.json"))

    yz = 0


"""
gen_de3.json
5545
lemon_v2.tet.json
1053
acc_rmrb.tet.json
4636
acc_xxqg.tet.json
5000
gen_passage.tet.json
10000
csc_TextProofreadingCompetition.tet.json
1447
gen_xxqg.tet.json
5000
faspell.dev.json
1000
lomo_tet.json
5000
mcsc_tet.5000.json
5000
ecspell.dev.json
1500
sighan2013.dev.json
1000
sighan2014.dev.json
1062
sighan2015.dev.json
1100
mcsc_tet.json
19605
"""


