# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/12/29 23:57
# @author  : Mo
# @function: predict_csc_token_zh, 简体中文拼写纠错, csc


from __future__ import absolute_import, division, print_function
import logging as logger
import traceback
import platform
import time
import sys
import os
import gc
import re
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path_root)
if platform.system().lower() == "windows":
    print(path_root)

from macro_correct.pytorch_textcorrection.tcTools import transfor_bert_unk_pun_to_know, tradition_to_simple
from macro_correct.pytorch_textcorrection.tcTools import transfor_english_symbol_to_chinese, string_q2b
from macro_correct.pytorch_textcorrection.tcTools import cut_sent_by_stay, count_flag_zh
from macro_correct.pytorch_textcorrection.tcTools import string_q2b, get_logger
from macro_correct.task.correct.predict_mlm_csc import CscPredict


def download_model_from_huggface_with_url(repo_id="Macropodus/macbert4mdcspell_v1",
                                          hf_endpoint="https://hf-mirror.com",
                                          logger=logger):
    """
        下载模型等数据文件, 从huggface下载, 可指定repo_id/url
    """
    os.environ["HF_ENDPOINT"] = hf_endpoint or os.environ.get("HF_ENDPOINT", hf_endpoint)
    from huggingface_hub import snapshot_download
    # logger.basicConfig(level=logger.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    os.environ["PATH_MACRO_CORRECT_MODEL"] = os.environ.get("PATH_MACRO_CORRECT_MODEL",
                                                           os.path.join(path_root, "macro_correct"))
    local_dir = os.path.join(os.environ["PATH_MACRO_CORRECT_MODEL"], "output", "text_correction", repo_id.split("/")[-1])
    logger.info("PATH_MACRO_CORRECT_MODEL IS " + str(os.environ["PATH_MACRO_CORRECT_MODEL"]))
    logger.info("HF_ENDPOINT IS " + str(os.environ["HF_ENDPOINT"]))
    logger.info("download " + repo_id + " from huggface start, please wait a few minute!")
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
    logger.info("download " + repo_id + " from huggface, end!")


def download_model_from_huggface(repo_id="Macropodus/macbert4mdcspell_v1", logger=logger):
    """
        下载模型等数据文件, 从huggface
    """
    try:
        download_model_from_huggface_with_url(hf_endpoint="https://hf-mirror.com", repo_id=repo_id, logger=logger)
    except Exception as e:
        logger.info(traceback.print_exc())
        download_model_from_huggface_with_url(hf_endpoint="https://huggingface.co/models", repo_id=repo_id, logger=logger)


class MacroCSC4Token:
    def __init__(self, path_config=None, logger=logger):
        self.logger = logger
        self.path_config = path_config or os.path.join(path_root, "macro_correct/output/text_correction/macbert4mdcspell_v1/csc.config")
        self.check_or_download_hf_model(path_config=path_config)
        self.load_trained_model()

    def check_or_download_hf_model(self, path_config=None):
        """  从hf国内镜像加载数据   """
        ### 如果传入的是其他模型的repo_id
        if path_config and path_config.startswith("Macropodus/"):
            # dowload model from hf
            path_config_select = os.path.join(path_root, "macro_correct/output/text_correction",
                                              path_config.split("/")[-1], "csc.config")
            path_model_select = os.path.join(path_root, "macro_correct/output/text_correction",
                                              path_config.split("/")[-1], "pytorch_model.bin")
            if os.path.exists(path_config_select) and os.path.exists(path_model_select):
                pass
            else:
                self.logger.info("download_model_from_huggface " + path_config + " start!")
                download_model_from_huggface(repo_id=path_config, logger=self.logger)
                self.logger.info("download_model_from_huggface " + path_config + " success!")
            ### 更改模型路径
            self.model_csc.model.office.config.model_save_path = os.path.split(path_model_select)[0]
            self.path_config = path_config_select
        else:
            ### 默认加载Macropodus/macbert4mdcspell_v1
            path_config = path_config or self.path_config
            path_model = os.path.join(os.path.split(path_config)[0], "pytorch_model.bin")
            if os.path.exists(path_config) and os.path.exists(path_model):
                pass
            else:
                # dowload model from hf
                repo_id = "Macropodus/macbert4mdcspell_v1"
                self.logger.info("download_model_from_huggface " + repo_id + " start!")
                download_model_from_huggface(repo_id=repo_id, logger=self.logger)
                self.logger.info("download_model_from_huggface " + repo_id + " success!")
        return True

    def load_trained_model(self, path_config=None):
        """   模型初始化加载权重   """
        path_config = path_config or self.path_config
        self.model_csc = CscPredict(path_config)

    def func_csc_token_long(self, content, threshold=0.6, max_len=128, batch_size=16, rounded=4,
                            num_rethink=0, flag_confusion=False, flag_prob=True, **kwargs):
        """   对句子进行文本纠错, 字词级别   """
        # time_start = time.time()
        params = {
            "flag_confusion": flag_confusion,  # 是否使用默认的混淆词典
            "flag_prob": flag_prob,  # 是否返回纠错token处的概率
            "num_rethink": num_rethink,  # 多次预测, think-twice
            "batch_size": batch_size,  # 批大小
            "threshold": threshold,  # token阈值过滤
            "max_len": max_len,  # 自定义的长度, 如果截断了, 则截断部分不参与纠错, 后续直接一模一样的补回来
            "rounded": rounded,  # 保存4位小数
        }
        limit_nums_errors = 10  # 一句话最多的错别字, 多的就剔除
        limit_length_char = 3  # 一句话的最小字符数, 不大于就不选中
        threshold_zh = 0.5  # 中文字符占比的最低值, 不大于就不选中
        output = []
        try:
            texts, texts_length = cut_sent_by_stay(content, return_length=True)
            texts_filter = []
            texts_map_org = []
            for idx, text in enumerate(texts):
                ### 先过滤短文本, 中文字符>0.7的文本
                if text and len(text) > limit_length_char \
                        and count_flag_zh(text)/len(text) >= threshold_zh:
                    text = transfor_english_symbol_to_chinese(text)
                    text = string_q2b(text)
                    text = tradition_to_simple(text)
                    texts_filter.append(text)
                    texts_map_org.append(idx)
            if texts_filter:
                ### 字词错误纠正
                texts_filter_csc = self.model_csc.predict_batch_score(texts_filter, **params)
                for jdx, texts_filter_csc_j in enumerate(texts_filter_csc):
                    errors = texts_filter_csc_j.get("errors", {})
                    target = texts_filter_csc_j.get("target", "")
                    ### 错误字词
                    if errors and len(errors) <= limit_nums_errors:
                        jdx_map_idx = texts_map_org[jdx]  # 原始句子的位置index
                        pos_jdx = texts_length[jdx_map_idx]
                        line_correct = {"index": jdx_map_idx, "source": texts_filter[jdx], "target": target, "errors": []}
                        for k, errors_i in enumerate(errors):
                            pos_start = pos_jdx[0] + errors_i[2]
                            token_wrong = errors_i[0]
                            token_true = errors_i[1]
                            score = errors_i[-1]
                            if score < params.get("threshold", threshold):
                                continue
                            line_error = [token_wrong, token_true, pos_start, score]
                            ### 如果是连续的错误就拼接在一起(待确认)
                            if k > 0 and errors[k][2] - errors[k-1][2] == 1:
                                if line_correct["errors"]:
                                    score_avg = (line_correct["errors"][-1][-1] + score) / 2
                                    line_correct["errors"][-1][-1] = round(score_avg, rounded)
                                    line_correct["errors"][-1][1] += token_true
                                    line_correct["errors"][-1][0] += token_wrong
                                else:
                                    line_correct["errors"].append(line_error)
                            else:
                                line_correct["errors"].append(line_error)
                        if line_correct.get("errors", []):
                            output.append(line_correct)
        except Exception as e:
            output = []
            self.logger.info("fail of func_csc_token_long")
            self.logger.info(traceback.format_exc())
        # 再次校验不饿能超过threshold
        # time_end = time.time()
        # time_cost = round(time_end - time_start, rounded)
        # self.logger.info(time_cost)
        return output

    def func_csc_token_batch(self, texts, threshold=0.6, max_len=128, batch_size=16, rounded=4,
                            num_rethink=0, flag_confusion=False, flag_prob=True, **kwargs):
        """   对句子进行文本纠错, 字词级别   """
        # time_start = time.time()
        params = {
            "flag_confusion": flag_confusion,  # 是否使用默认的混淆词典
            "flag_prob": flag_prob,  # 是否返回纠错token处的概率
            "num_rethink": num_rethink,  # 多次预测, think-twice
            "batch_size": batch_size,  # 批大小
            "threshold": threshold,  # token阈值过滤
            "max_len": max_len,  # 自定义的长度, 如果截断了, 则截断部分不参与纠错, 后续直接一模一样的补回来
            "rounded": rounded,  # 保存4位小数
        }
        limit_nums_errors = 5  # 一句话最多的错别字, 多的就剔除
        output = []
        try:
            ### 字词错误纠正
            texts_filter_csc = self.model_csc.predict_batch_score(texts, **params)
            for jdx, texts_filter_csc_j in enumerate(texts_filter_csc):
                errors = texts_filter_csc_j.get("errors", {})
                target = texts_filter_csc_j.get("target", "")
                line_correct = {"index": jdx, "source": texts[jdx], "target": target, "errors": []}
                ### 错误字词
                if errors and len(errors) <= limit_nums_errors:
                    for k, errors_i in enumerate(errors):
                        pos_start = errors_i[2]
                        token_wrong = errors_i[0]
                        token_true = errors_i[1]
                        score = errors_i[-1]
                        if score < params.get("threshold", 0.0):
                            continue
                        line_error = [token_wrong, token_true, pos_start, score]
                        ### 如果是连续的错误就拼接在一起(待确认)
                        if k > 0 and errors[k][2] - errors[k-1][2] == 1:
                            if line_correct["errors"]:
                                score_avg = (line_correct["errors"][-1][-1] + score) / 2
                                line_correct["errors"][-1][-1] = round(score_avg, rounded)
                                line_correct["errors"][-1][1] += token_true
                                line_correct["errors"][-1][0] += token_wrong
                            else:
                                line_correct["errors"].append(line_error)
                        else:
                            line_correct["errors"].append(line_error)
                output.append(line_correct)
        except Exception as e:
            output = []
            self.logger.info("fail of func_csc_token_batch")
            self.logger.info(traceback.format_exc())
        # 再次校验不饿能超过threshold
        # time_end = time.time()
        # time_cost = round(time_end - time_start, rounded)
        # self.logger.info(time_cost)
        return output


if __name__ == '__main__':
    yz = 0

    """
    ### 加载和使用其他模型权重
    import os
    os.environ["MACRO_CORRECT_FLAG_CSC_TOKEN"] = "1"
    from macro_correct import MODEL_CSC_TOKEN
    repo_id = "Macropodus/macbert4csc_v2"
    MODEL_CSC_TOKEN.__init__(path_config=repo_id)
    res = MODEL_CSC_TOKEN.func_csc_token_batch(texts, threshold=0.0)
    print("#####   " + repo_id)
    for res_i in res:
        print(res_i)
    print("#" * 128)
    """

    ### init model
    logger = get_logger("./")
    MODEL_CSC_TOKEN = MacroCSC4Token(logger=logger)
    predict_batch = MODEL_CSC_TOKEN.model_csc.predict_batch  # 基础方法预测, 没有后处理
    func_csc_token_long = MODEL_CSC_TOKEN.func_csc_token_long  # 处理单一篇的文本
    func_csc_token_batch = MODEL_CSC_TOKEN.func_csc_token_batch  # 批处理小于max_len的句子

    ### sample
    texts = ["一个分知,陌光回聚,莪受打去,祢爱带馀",
             "馀额还有100w",
             '放在陌光下',
             '真麻烦你了。希望你们好好的跳无',
             '少先队员因该为老人让坐',
             '机七学习是人工智能领遇最能体现智能的一个分知',
             '一只小鱼船浮在平净的河面上',
             '我每天六天半起床。',
             '教帅是一个高尚的职业。',
             '街上青一色地全是小汽车。',
             '不属于拼写检查错误',
             '明天陪你们出站',
             '明天排你们出战',
             '刚刚出道的明星通过各种渠道不断的刷脸，提高在众多明星中的知名度。',
             ]
    content = "。".join(texts)
    res = func_csc_token_batch(texts, threshold=0.0)
    for res_i in res:
        print(res_i)
    print("#" * 128)

    time_start = time.time()
    res = func_csc_token_long(content, threshold=0.0)
    print(res)
    print("#" * 128)
    while True:
        try:
            print("请输入:")
            question = input()
            res = func_csc_token_long(question, threshold=0.0)
            print(res)
        except Exception as e:
            print(traceback.print_exc())

