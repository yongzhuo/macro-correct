# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/7/25 19:30
# @author  : Mo
# @function: predict model, 预测模块


# 适配linux
import logging as logger
import traceback
import platform
import copy
import sys
import os

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(path_root)
if platform.system().lower() == "windows":
    print(path_root)
# os.environ["CUDA_VISIBLE_DEVICES"] = model_config.get("CUDA_VISIBLE_DEVICES", "0")
from macro_correct.pytorch_sequencelabeling.slData import SeqLabelingDataCollator
from macro_correct.pytorch_sequencelabeling.slData import SeqlabelingDataset
from macro_correct.pytorch_sequencelabeling.slTools import get_logger
from macro_correct.pytorch_sequencelabeling.slTools import load_json
from macro_correct.pytorch_sequencelabeling.slTools import txt_read
from macro_correct.pytorch_sequencelabeling.slOffice import Office

from torch.utils.data import DataLoader
from argparse import Namespace


class SequenceLabelingPredict:
    def __init__(self, path_config, logger=logger, flag_load_model=True, CUDA_VISIBLE_DEVICES="0"):
        """ 初始化 """
        self.CUDA_VISIBLE_DEVICES = CUDA_VISIBLE_DEVICES
        self.load_config(path_config)
        self.logger = logger
        if flag_load_model:
            self.load_model()

    def load_config(self, path_config):
        """ 加载超参数  """
        config = load_json(path_config)
        self.config = Namespace(**config)
        # 将模型目录替换为实际目录, 方便移动的时候不影响; bert_tokenizer/bert_config等都从模型里边读取
        model_save_path_real = os.path.split(path_config)[0]
        self.config.CUDA_VISIBLE_DEVICES = self.CUDA_VISIBLE_DEVICES  # os.environ.get("CUDA_VISIBLE_DEVICES", "-1")
        self.config.pretrained_model_name_or_path = model_save_path_real
        self.config.model_save_path = model_save_path_real
        # path_log_dir = os.path.join(model_save_path_real, "log")
        # self.logger = get_logger(path_log_dir)
        self.l2i, self.i2l = self.config.l2i, self.config.i2l
        # 数据预处理类
        self.sl_dataset = SeqlabelingDataset(path="", config=self.config, flag_shuffle=False)
        self.sl_collator = SeqLabelingDataCollator(config=self.config)

    def load_model(self):
        """ 加载模型  """
        self.office = Office(config=self.config, tokenizer=self.sl_collator.tokenizer, logger=self.logger)
        try:
            # self.office.load_model()
            self.office.load_model_state()
        except Exception as e:
            self.logger.info(traceback.print_exc())
            self.logger.info("self.office.load_model_state() fail!")
            # self.office.load_model_state()
            self.office.load_model()
            self.logger.info("self.office.load_model() success!")

    def process(self, texts, max_len=None, batch_size=None, **kwargs):
        """ 数据预处理, process """
        # token 转 idx, 训练集/验证集
        self.sl_dataset.read_dict(texts)
        if max_len:  # 动态变化max_len
            self.sl_collator.max_len = min(max_len, self.sl_collator.config.max_len)
        else:
            self.sl_collator.max_len = self.sl_collator.config.max_len
        if not batch_size:
            batch_size_dy = self.config.batch_size
        else:
            batch_size_dy = batch_size

        pred_data_loader = DataLoader(num_workers=self.config.num_workers,
                                      collate_fn=self.sl_collator,
                                      batch_size=batch_size_dy,
                                      dataset=self.sl_dataset,
                                      # pin_memory=True
                                      )
        return pred_data_loader

    def predict(self, texts, max_len=128, batch_size=32, **kwargs):
        """ 预测 """
        dataset = self.process(texts, max_len=max_len, batch_size=batch_size, **kwargs)
        res = self.office.predict(dataset)
        return res


if __name__ == "__main__":
    yz = 0
    from macro_correct.pytorch_sequencelabeling.slTools import transfor_english_symbol_to_chinese
    from macro_correct.pytorch_sequencelabeling.slTools import tradition_to_simple
    from macro_correct.pytorch_sequencelabeling.slTools import string_q2b

    # path_config = "../output/sequence_labeling/model_ner_rmrb_ERNIE_lr-5e-05_bs-32_epoch-12_v1/sl.config"
    path_config = "../output/sequence_labeling/bert4sl_punct_zh_public/sl.config"
    idx2pun = load_json(os.path.join(os.path.split(path_config)[0], "idx2pun.json"))
    pun2idx = {v: k for k, v in idx2pun.items()}

    tcp = SequenceLabelingPredict(path_config)
    texts = [{"text": "平乐县，古称昭州，隶属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，南接昭平，西北毗邻阳朔，北连恭城，总面积1919.34平方公里。"},
             {"text": "平乐县主要旅游景点有榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等，平乐县为漓江分界点，平乐以北称漓江，以南称桂江，是著名的大桂林旅游区之一。"},
             {"text": "桂林山水甲天下, 阳朔山水甲桂林"},
             {"text": "山不在高，有仙则名。水不在深，有龙则灵。斯是陋室，惟吾德馨。"}
             ]
    texts_new = []
    for line in texts:
        text = line.get("text", "")
        text = string_q2b(text)
        text = tradition_to_simple(text)
        text = transfor_english_symbol_to_chinese(text)
        text_new_i = ""
        for t in text:
            if t not in pun2idx:
                text_new_i += t
        text_new_i = "#" + text_new_i
        line["text"] = text_new_i
        texts_new.append(line)

    res = tcp.predict(texts_new)
    for idx, r in enumerate(res):
        len_punct = 0
        print(r)
        print(texts[idx].get("text", ""))
        org_text = r.get("text", "")
        org_label = r.get("label", [])
        for kdx, label_j in enumerate(org_label):
            label_j_type = label_j.get("type", "")
            label_j_pos = label_j.get("pos", [0, 0])[0]
            punct_current = idx2pun.get(label_j_type, "#")
            pos_currect = label_j_pos + len_punct + 1
            org_text = org_text[:pos_currect] + punct_current + org_text[pos_currect:]
            len_punct += len(punct_current)
        print(org_text)

    while True:
        print("请输入:")
        text = input()
        text = string_q2b(text)
        text = tradition_to_simple(text)
        text = transfor_english_symbol_to_chinese(text)
        text_new_i = ""
        for t in text:
            if t not in pun2idx:
                text_new_i += t
        text_new_i = "#" + text_new_i
        res = tcp.predict([{"text": text_new_i}])
        print(res)
        print(text)
        org_text = res[0].get("text", "")
        org_label = res[0].get("label", [])
        len_punct = 0
        for kdx, label_j in enumerate(org_label):
            label_j_type = label_j.get("type", "")
            label_j_pos = label_j.get("pos", [0, 0])[0]
            punct_current = idx2pun.get(label_j_type, "#")
            pos_currect = label_j_pos + len_punct + 1
            org_text = org_text[:pos_currect] + punct_current + org_text[pos_currect:]
            len_punct += len(punct_current)
        print(org_text)
        print(len_punct)


"""
windows下num_workers设置为0
self = reduction.pickle.load(from_parent) EOFError: Ran out of input


17223 35000
0.4920857142857143
"""

