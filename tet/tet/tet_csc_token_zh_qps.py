# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/29 21:41
# @author  : Mo
# @function: tet qps of csc of token, 测评QPS


import time
import os
os.environ["MACRO_CORRECT_FLAG_CSC_TOKEN"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tqdm import tqdm

from macro_correct.pytorch_textcorrection.tcTools import cut_sent_by_stay
from macro_correct import correct_basic
from macro_correct import correct_long
from macro_correct import correct
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


### 默认纠错(list输入)
text_list = ["一个分知,陌光回聚,莪受打去,祢爱带馀",
             "馀额还有100w",
             "放在陌光下",
             "真麻烦你了。希望你们好好的跳无",
             "少先队员因该为老人让坐",
             "机七学习是人工智能领遇最能体现智能的一个分知",
             "一只小鱼船浮在平净的河面上",
             "一颗火流星画过北京上空。",
             "一颗火流星刬过北京上空。",
             ]


text_csc = correct(text_list, flag_confusion=True)
print("默认纠错(list输入):")
for res_i in text_csc:
    print(res_i)
print("#" * 128)

bad_errors = [
    ("纠心的场景", "揪心的场景"),
    ("废铜乱铁", "废铜烂铁"),
    ("有的枝干以死", "有的枝干已死"),
    ("打盹的功夫", "打盹儿的工夫"),
    ("星星和我作伴", "星星和我做伴"),
    ("所有的慷概", "所有的慷慨"),
    ("在生命的红荒中", "在生命的洪荒中"),
    ("恒古不变", "亘古不变"),
    ("心率不齐", "心律不齐"),
    ("汉烟叶子", "旱烟叶子"),
    ("闪烁光讪", "闪烁光芒"),
    ("喝斥说：", "呵斥说："),
    ("捣腾", "倒腾"),
    ("三角架", "三脚架"),
    ("如坐摇蓝般摇摆", "如坐摇篮般摇摆"),
    ("落地即溶的雪", "落地即融的雪"),
    ("天腑身痛哭。地垂首流泪", "天俯身痛哭。地垂首流泪"),
    ("膈应人", "硌硬人"),
    ("陪着笑脸", "赔着笑脸"),
    ("摇得像拔浪鼓似的", "摇得像拨浪鼓似的"),
    ("一个劲地", "一个劲儿地"),
    ("胆颤心惊", "胆战心惊"),
    ("不断蠕动的嘴", "不断嚅动的嘴"),
    ("往锅里填了一瓢水", "往锅里添了一瓢水"),
    ("趾高气昂", "趾高气扬"),
    ("急冲冲", "急匆匆"),
    ("九一八事件", "九一八事变"),
]

text_all = "".join([b[0] for b in bad_errors])
text_one = text_all[:128]


print("qps-start")
time_start = time.time()
### 默认纠错(list输入, 参数配置)
params = {
    "threshold": 0.75,  # token阈值过滤
    "batch_size": 32,  # 批大小
    "max_len": 128,  # 自定义的长度, 如果截断了, 则截断部分不参与纠错, 后续直接一模一样的补回来
    "rounded": 4,  # 保存4位小数
    "limit_nums_errors": 5,  # 一句话最多的错别字, 多的就剔除(全不纠错)
    "num_rethink": 0,  # 多次预测, think-twice
    "flag_confusion": True,  # 是否使用默认的混淆词典
    "flag_prob": True,  # 是否返回纠错token处的概率
    "flag_cut": False,  # 是否切分, 大于126长度的句子(128去掉了CLS/SEP)
}
texts_batch = []
step = 3200
for i in tqdm(range(step), desc="qps"):
    texts_batch.append(str(i) + text_one)
    if i > 0 and i % params.get("batch_size") == 0:
        # text_csc = correct(texts_batch, **params)
        text_csc = correct_basic(texts_batch, **params)
        texts_batch = []
time_end = time.time()
time_cost = time_end - time_start
print("time_cost：")
print(time_cost)
print("step/time_cost：")
print(step/time_cost)



"""
gpu-rtx2060, no-confusion; no-prob; bs=32
qps: 100%|██████████| 3200/3200 [00:32<00:00, 98.22it/s]
time_cost：
32.581300497055054
step/time_cost：
98.21584624251695

gpu-rtx2060, confusion; prob; bs=64
qps: 100%|██████████| 3200/3200 [00:32<00:00, 96.98it/s]
time_cost：
32.999271631240845
step/time_cost：
96.97183731081257

cpu-i7-10700, confusion; prob; bs=32
qps: 100%|██████████| 3200/3200 [05:47<00:00,  9.21it/s]
time_cost：
347.2709403038025
step/time_cost：
9.214707102185253
"""
