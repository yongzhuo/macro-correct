# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/12/29 23:57
# @author  : Mo
# @function: predict_csc_punct_zh, 标点符号纠错, csc


from __future__ import absolute_import, division, print_function
import logging as logger
import traceback
import platform
import time
import sys
import os
import gc
import re
path_sys = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path_sys)
if platform.system().lower() == "windows":
    print(path_sys)
from macro_correct.pytorch_sequencelabeling.slTools import transfor_english_symbol_to_chinese, string_q2b
from macro_correct.pytorch_sequencelabeling.slTools import flag_total_chinese, count_flag_zh, load_json
from macro_correct.pytorch_sequencelabeling.slTools import cut_sent_len, cut_sent_by_stay
from macro_correct.task.punct.predict_ner_punct import PunctPredict


threshold_pun_freq = {
    "，": 0.9, "。": 0.9, "？": 0.9, "！": 0.9, "；": 0.85, "：": 0.85,
    "》": 0.85, "《": 0.85, "（": 0.85, "）": 0.85, "“": 0.85, "”": 0.85,
    "‘": 0.85, "’": 0.85, "“，": 0.85, "“。": 0.85, "“！": 0.85, "“；": 0.85
}  # 可以每个标点给个阈值


def download_model_from_huggface_with_url(repo_id="Macropodus/bert4sl_punct_zh_public", hf_endpoint="https://hf-mirror.com"):
    """   下载模型等数据文件, 从huggface下载, 可指定repo_id/url   """
    os.environ["HF_ENDPOINT"] = hf_endpoint or os.environ.get("HF_ENDPOINT", hf_endpoint)
    from huggingface_hub import snapshot_download
    # logger.basicConfig(level=logger.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    os.environ["PATH_MACRO_CORRECT_MODEL"] = os.environ.get("PATH_MACRO_CORRECT_MODEL",
                                                           os.path.join(path_sys, "macro_correct"))
    local_dir = os.path.join(os.environ["PATH_MACRO_CORRECT_MODEL"], "output", "sequence_labeling", repo_id.split("/")[-1])
    logger.info("PATH_MACRO_CORRECT_MODEL IS " + str(os.environ["PATH_MACRO_CORRECT_MODEL"]))
    logger.info("HF_ENDPOINT IS " + str(os.environ["HF_ENDPOINT"]))
    logger.info("download bert4sl_punct_zh_public from huggface start, please wait a few minute!")
    cache_dir = local_dir + "/cache"
    snapshot_download(cache_dir=cache_dir,
                      local_dir=local_dir,
                      repo_id=repo_id,
                      local_dir_use_symlinks=False,  # 不转为缓存乱码的形式
                      resume_download=False,
                      force_download=True,
                      # allow_patterns=[],
                      # ignore_patterns=[],
                      )
    logger.info("download bert4sl_punct_zh_public from huggface, end!")


def download_model_from_huggface(repo_id="Macropodus/bert4sl_punct_zh_public"):
    """   下载模型等数据文件, 从huggface   """
    try:
        download_model_from_huggface_with_url(hf_endpoint="https://hf-mirror.com", repo_id=repo_id)
    except Exception as e:
        logger.info(traceback.print_exc())
        download_model_from_huggface_with_url(hf_endpoint="https://huggingface.co/models", repo_id=repo_id)


class MacroCSC4Punct:
    def __init__(self, logger=logger):
        self.logger = logger
        self.path_config = os.path.join(path_sys, "macro_correct/output/sequence_labeling/bert4sl_punct_zh_public/sl.config")
        self.check_or_download_hf_model()
        self.load_trained_model()

    def check_or_download_hf_model(self, path_config=None):
        """  从hf国内镜像加载数据   """
        path_config = path_config or self.path_config
        path_model = os.path.join(os.path.split(path_config)[0], "pytorch_model.bin")
        if os.path.exists(path_config) and os.path.exists(path_model):
            pass
        else:
            # dowload model from hf
            download_model_from_huggface(repo_id="Macropodus/bert4sl_punct_zh_public")

    def load_trained_model(self, path_config=None):
        """   模型初始化加载权重   """
        path_config = path_config or self.path_config
        self.model_csc = PunctPredict(path_config)

    def func_csc_punct_long(self, content, threshold=0.55, max_len=128, batch_size=16, rounded=4,
                            limit_num_errors=4, limit_len_char=3, threshold_zh=0.5,
                            threshold_pun_freq=threshold_pun_freq,
                            flag_cut=False, flag_prob=True,
                            **kwargs):
        """   对句子进行文本纠错, 标点符号
        args:
            content: str, text of sentence, eg. "你是谁？"
            flag_cut: bool, whether cut sentence or not, eg.True or False.
            flag_pool_out: bool, output list or not, list is for pool, eg.True or False.
        returns:
            res: List<dict>, eg.[{xxx}]
        """
        params = {
            # "flag_prob": flag_prob,  # 是否返回纠错token处的概率
            # "threshold": threshold,  # token阈值过滤
            "batch_size": batch_size,  # 批大小
            "max_len": max_len,  # 自定义的长度, 如果截断了, 则截断部分不参与纠错, 后续直接一模一样的补回来
            "rounded": rounded,  # 保存4位小数
        }
        # time_start = time.time()
        # limit_num_errors = 4  # 一句话最多的错别字, 多的就剔除
        # limit_len_char = 4  # 一句话的最小字符数
        # threshold_pun = 0.55  # 通用标点阈值
        # threshold_zh = 0.5  # 句子阈值, 中文字符占比的最低值
        # rounded = 4
        output = []
        try:
            if flag_cut:  # 切长句子, 按标点符号切分
                texts, texts_length = cut_sent_by_stay(content, return_length=True)  # 保留原来的标点符号
                texts_filter = []
                texts_map_org = []
                for idx, text in enumerate(texts):
                    ### 先过滤短文本, 中文字符>0.7的文本
                    if text and len(text) > limit_len_char \
                            and count_flag_zh(text) / len(text) >= threshold_zh:
                        text = string_q2b(text)
                        text = transfor_english_symbol_to_chinese(text)
                        texts_filter.append(text)
                        texts_map_org.append(idx)
            else:
                texts_filter = [content]
                texts_map_org = [0]
                texts_length = [[0, len(content)]]

            if texts_filter:
                ### 推理
                texts_filter_pun = self.model_csc.predict_batch(texts_filter, **params)
                # print(texts_filter_pun)
                ### 修改/新增标点符号, 以非标点字符为基准, 先筛选需要【修改/新增标点符号】的, 然后再统一对应标点符号的位置
                for jdx, texts_filter_pun_j in enumerate(texts_filter_pun):
                    label_org = texts_filter_pun_j.get("label_org", [])
                    label_pred = texts_filter_pun_j.get("label", [])
                    label_org_csc = {str(ag.get("pos", [])[0]): [ag.get("type", ""),
                                        ag.get("pun", "")] for ag in label_org}
                    target = texts_filter_pun_j.get("target", [])
                    # 原始句子的位置index
                    jdx_map_idx = texts_map_org[jdx]
                    pos_jdx = texts_length[jdx_map_idx]
                    line_correct = {"index": jdx_map_idx, "source": texts_filter[jdx],
                                    "target": target, "score": 0.0, "errors": []}
                    csc_no_symbol = []
                    for kdx, label_pred_kdx in enumerate(label_pred):
                        pred_pos = label_pred_kdx.get("pos", [])[0]
                        pred_score = label_pred_kdx.get("score", 0)
                        pred_type = label_pred_kdx.get("type", "")
                        pred_pun = label_pred_kdx.get("pun", "")
                        pred_ent = label_pred_kdx.get("ent", "")
                        threshold_real = threshold_pun_freq.get(pred_pun, threshold)  # 修改后前的符号的设置阈值
                        if str(pred_pos) in label_org_csc:
                            org_type_pun = label_org_csc.get(str(pred_pos), [])
                            org_type = org_type_pun[0]
                            org_pun = org_type_pun[1]
                            # 修改后前的符号的设置阈值
                            threshold_real = max(threshold_pun_freq.get(org_pun, threshold), threshold_real)
                            if org_type != pred_type and pred_score > threshold_real:
                                ### 获取原始句子标点符号的位置需要新增的部分, 因为这是以"剔除标点的原文句子"为基础的
                                pos_current_add = 0
                                for lab_o in label_org:
                                    lab_o_pos = lab_o.get("pos", [0, 0])[0]
                                    lab_o_pun = lab_o.get("pun", "")
                                    if lab_o_pos < pred_pos:
                                        pos_current_add += len(lab_o_pun)
                                    else:
                                        break
                                pred_pos += pos_current_add
                                # 错误标点, 正确标点, 当前句子位置, 全文所在位置, 概率
                                if flag_prob:
                                    line_error = [org_pun, pred_pun, pred_pos, pos_jdx[0] + pred_pos, pred_score]
                                else:
                                    line_error = [org_pun, pred_pun, pred_pos, pos_jdx[0] + pred_pos]
                                ### "标点符号错误,建议'”。'修改为'。'"
                                ### todo, 高到低剔除, 原因是后边的标点训练样本太少, 不太准
                                if len(org_pun) > 1 and len(pred_pun) < 2:
                                    continue
                                csc_no_symbol.append(line_error)
                        else:
                            if pred_score > threshold_real:
                                ### 获取原始句子标点符号的位置需要新增的部分, 因为这是以"剔除标点的原文句子"为基础的
                                pos_current_add = 0
                                for lab_o in label_org:
                                    lab_o_pos = lab_o.get("pos", [0, 0])[0]
                                    lab_o_pun = lab_o.get("pun", "")
                                    if lab_o_pos < pred_pos:
                                        pos_current_add += len(lab_o_pun)
                                    else:
                                        break
                                pred_pos += pos_current_add
                                # 错误标点, 正确标点, 当前句子位置, 全文所在位置, 概率
                                if flag_prob:
                                    line_error = ["", pred_pun, pred_pos, pos_jdx[0] + pred_pos, pred_score]
                                else:
                                    line_error = ["", pred_pun, pred_pos, pos_jdx[0] + pred_pos]
                                csc_no_symbol.append(line_error)
                    if csc_no_symbol and len(csc_no_symbol) < limit_num_errors:
                        if flag_prob:
                            line_correct["score"] = round(sum([c[-1] for c in csc_no_symbol]
                                                              ) / len(csc_no_symbol), rounded)
                        line_correct["errors"] = csc_no_symbol
                        output.append(line_correct)
        except Exception as e:
            output = []
            self.logger.info("fail of func_csc_punct_long")
            self.logger.info(traceback.format_exc())
            # print("fail of sentence_correct_char_and_word")
            # print(traceback.format_exc())
        # 再次校验不饿能超过threshold
        # time_end = time.time()
        # time_cost = round(time_end - time_start, 6)
        # self.logger.info("time_cost of func_csc_punct_long: " + str(time_cost))
        return output

    def func_csc_punct_batch(self, texts, threshold=0.55, max_len=128, batch_size=16, rounded=4,
                            limit_num_errors=32, limit_len_char=3, threshold_zh=0.5,
                            threshold_pun_freq=threshold_pun_freq,
                            flag_cut=False, flag_prob=True,
                            **kwargs):
        """   对句子进行文本纠错, 标点符号
        args:
            content: str, text of sentence, eg. "你是谁？"
        returns:
            res: List<dict>, eg.[{xxx}]
        """
        params = {
            # "flag_prob": flag_prob,  # 是否返回纠错token处的概率
            # "threshold": threshold,  # token阈值过滤
            "batch_size": batch_size,  # 批大小
            # "flag_cut": flag_cut,  # 是否切分
            "max_len": max_len,  # 自定义的长度, 如果截断了, 则截断部分不参与纠错, 后续直接一模一样的补回来
            "rounded": rounded,  # 保存4位小数
        }
        # time_start = time.time()
        # threshold_pun_freq = {"，": 0.9, "。": 0.9, "？": 0.9, "！": 0.9, "；": 0.85, "：": 0.85,
        #                       "》": 0.85, "《": 0.85, "（": 0.85, "）": 0.85, "“": 0.85, "”": 0.85,
        #                       "‘": 0.85, "’": 0.85, "“，": 0.85, "“。": 0.85, "“！": 0.85, "“；": 0.85
        #                       }  # 可以每个标点给个阈值
        # limit_num_errors = 4  # 一句话最多的错别字, 多的就剔除
        # threshold = 0.55
        # rounded = 4
        output = []
        try:
            ### 推理
            texts_filter_pun = self.model_csc.predict_batch(texts, **params)
            # print(texts_filter_pun)
            ### 修改/新增标点符号, 以非标点字符为基准, 先筛选需要【修改/新增标点符号】的, 然后再统一对应标点符号的位置
            for jdx, texts_filter_pun_j in enumerate(texts_filter_pun):
                label_org = texts_filter_pun_j.get("label_org", [])
                label_pred = texts_filter_pun_j.get("label", [])
                label_org_csc = {str(ag.get("pos", [])[0]): [ag.get("type", ""),
                                    ag.get("pun", "")] for ag in label_org}
                target = texts_filter_pun_j.get("target", [])
                if flag_prob:
                    line_correct = {"index": jdx, "source": texts[jdx], "target": target, "score": 0.95, "errors": []}
                else:
                    line_correct = {"index": jdx, "source": texts[jdx], "target": target, "errors": []}
                csc_no_symbol = []
                for kdx, label_pred_kdx in enumerate(label_pred):
                    pred_pos = label_pred_kdx.get("pos", [])[0]
                    pred_score = label_pred_kdx.get("score", 0)
                    pred_type = label_pred_kdx.get("type", "")
                    pred_pun = label_pred_kdx.get("pun", "")
                    pred_ent = label_pred_kdx.get("ent", "")
                    threshold_real = threshold_pun_freq.get(pred_pun, threshold)  # 修改后前的符号的设置阈值
                    if str(pred_pos) in label_org_csc:
                        org_type_pun = label_org_csc.get(str(pred_pos), [])
                        org_type = org_type_pun[0]
                        org_pun = org_type_pun[1]
                        # 修改后前的符号的设置阈值
                        threshold_real = max(threshold_pun_freq.get(org_pun, threshold), threshold_real)
                        if org_type != pred_type and pred_score > threshold_real:
                            ### 获取原始句子标点符号的位置需要新增的部分, 因为这是以"剔除标点的原文句子"为基础的
                            pos_current_add = 0
                            for lab_o in label_org:
                                lab_o_pos = lab_o.get("pos", [0, 0])[0]
                                lab_o_pun = lab_o.get("pun", "")
                                if lab_o_pos < pred_pos:
                                    pos_current_add += len(lab_o_pun)
                                else:
                                    break
                            pred_pos += pos_current_add
                            if flag_prob:
                                line_error = [org_pun, pred_pun, pred_pos, pred_score]
                            else:
                                line_error = [org_pun, pred_pun, pred_pos]
                            ### todo, 高到低剔除, 原因是后边的标点训练样本太少, 不太准
                            if len(org_pun) > 1 and len(pred_pun) < 2:
                                continue
                            csc_no_symbol.append(line_error)
                    else:
                        if pred_score > threshold_real:
                            ### 获取原始句子标点符号的位置需要新增的部分, 因为这是以"剔除标点的原文句子"为基础的
                            pos_current_add = 0
                            for lab_o in label_org:
                                lab_o_pos = lab_o.get("pos", [0, 0])[0]
                                lab_o_pun = lab_o.get("pun", "")
                                if lab_o_pos < pred_pos:
                                    pos_current_add += len(lab_o_pun)
                                else:
                                    break
                            pred_pos += pos_current_add
                            if flag_prob:
                                line_error = ["", pred_pun, pred_pos, pred_score]
                            else:
                                line_error = ["", pred_pun, pred_pos]
                            csc_no_symbol.append(line_error)
                if csc_no_symbol and len(csc_no_symbol) < limit_num_errors:
                    if flag_prob:
                        line_correct["score"] = round(sum([c[-1] for c in csc_no_symbol]
                                                          ) / len(csc_no_symbol), rounded)
                    line_correct["errors"] = csc_no_symbol
                    output.append(line_correct)
        except Exception as e:
            output = []
            self.logger.info("fail of func_csc_punct_batch")
            self.logger.info(traceback.format_exc())
            # print("fail of sentence_correct_char_and_word")
            # print(traceback.format_exc())
        # 再次校验不饿能超过threshold
        # time_end = time.time()
        # time_cost = round(time_end - time_start, 6)
        # self.logger.info("time_cost of func_csc_punct_batch: " + str(time_cost))
        return output


if __name__ == '__main__':
    yz = 0

    MODEL_CSC_PUNCT = MacroCSC4Punct()
    func_csc_punct_batch = MODEL_CSC_PUNCT.func_csc_punct_batch
    func_csc_punct_long = MODEL_CSC_PUNCT.func_csc_punct_long

    ### sample
    content = "山不在高，有仙则名。水不在深，有龙则灵。斯是陋室，惟吾德馨。苔痕上阶绿，草色入帘青。" \
              "谈笑有鸿儒，往来无白丁。可以调素琴，阅金经。无丝竹之乱耳，无案牍之劳形。" \
              "南阳诸葛庐，西蜀子云亭。孔子云：何陋之有？"

    texts = ["山不在高有仙则名。",
             "水不在深，有龙则灵",
             "斯是陋室惟吾德馨",
             "苔痕上阶绿草,色入帘青。"
             ]
    res = func_csc_punct_batch(texts)
    for res_i in res:
        print(res_i)
    print("#" * 128)


    params = {
        "limit_num_errors": 4,  # 一句话最多的错别字, 多的就剔除
        "limit_len_char": 4,   # 一句话的最小字符数
        "threshold_zh": 0.5,  # 句子阈值, 中文字符占比的最低值
        "threshold": 0.55,  # token阈值过滤
        "batch_size": 32,  # 批大小
        "max_len": 128,  # 自定义的长度, 如果截断了, 则截断部分不参与纠错, 后续直接一模一样的补回来
        "rounded": 4,  # 保存4位小数
        "flag_prob": True,  # 是否返回纠错token处的概率
    }
    res = func_csc_punct_batch(texts, **params)
    for res_i in res:
        print(res_i)
    print("#"*128)


    content = "山不在高有仙则名。水不在深有龙则灵。"
    time_start = time.time()
    res = func_csc_punct_long(content)
    print(res)
    print("#"*128)


    res = func_csc_punct_long(content, flag_cut=False)
    print(res)
    print("#"*128)


    while True:
        try:
            print("请输入:")
            question = input()
            res = func_csc_punct_long(question)
            print(res)
        except Exception as e:
            print(traceback.print_exc())

