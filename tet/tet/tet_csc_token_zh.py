# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/29 21:41
# @author  : Mo
# @function: tet csc of token, 简体中文拼写纠错(支持各领域纠错, 泛化性强, 包括古文)


import traceback
import time
import os
os.environ["MACRO_CORRECT_FLAG_CSC_TOKEN"] = "1"

from macro_correct.pytorch_textcorrection.tcTools import cut_sent_by_stay
from macro_correct import correct_basic
from macro_correct import correct_long
from macro_correct import correct


### 默认纠错(list输入)
text_list = ["一个分知,陌光回聚,莪受打去,祢爱带馀",
             "馀额还有100w",
             "放在陌光下",
             "真麻烦你了。希望你们好好的跳无",
             "少先队员因该为老人让坐",
             "机七学习是人工智能领遇最能体现智能的一个分知",
             "一只小鱼船浮在平净的河面上",
             "这一条次,我选择了一条与往常不同的路线。"
             ]


text_csc = correct(text_list, flag_confusion=True,)
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
text_csc = correct([b[0] for b in bad_errors], flag_confusion=True)
print("默认纠错(list输入):")
for idx, res_i in enumerate(text_csc):
    if res_i["target"] == bad_errors[idx][1]:
        category = True
    else:
        category = False
    print(category, bad_errors[idx], res_i)
print("#" * 128)



### 默认纠错(list输入, 参数配置)
params = {
    "threshold": 0.01,  # token阈值过滤
    "batch_size": 32,  # 批大小
    "max_len": 128,  # 自定义的长度, 如果截断了, 则截断部分不参与纠错, 后续直接一模一样的补回来
    "rounded": 4,  # 保存4位小数
    "flag_confusion": True,  # 是否使用默认的混淆词典
    "flag_prob": True,  # 是否返回纠错token处的概率
}
text_csc = correct(text_list, **params)
print("默认纠错(list输入, 参数配置):")
for res_i in text_csc:
    print(res_i)
print("#" * 128)


### 基础纠错(不包括后处理和过滤)
text_list = ["真麻烦你了。希望你们好好的跳无",
             "少先队员因该为老人让坐",
             "机七学习是人工智能领遇最能体现智能的一个分知",
             "一只小鱼船浮在平净的河面上"
             ]
text_csc = correct_basic(text_list)
print("基础纠错(不包括后处理和过滤):")
for res_i in text_csc:
    print(res_i)
print("#" * 128)


### 长句纠错(默认切句)
text = "真麻烦你了。希望你们好好的跳无。少先队员因该为老人让坐。机七学习是人工智能领遇最能体现智能的一个分知。一只小鱼船浮在平净的河面上。"
text_csc = correct_long(text)
print("长句纠错(默认切句):")
for res_i in text_csc:
    print(res_i)
print("#" * 128)


# while 1:
#     try:
#         print("请输入：")
#         text = input()
#         time_start = time.time()
#         res = correct_long(text)
#         time_end = time.time()
#         print(res)
#         print(time_end - time_start)
#     except Exception as e:
#         print(traceback.print_exc())


"""

默认纠错(list输入):
{'index': 1, 'source': '馀额还有100w', 'target': '余额还有100w', 'errors': [['馀', '余', 0, 1.0]]}
{'index': 2, 'source': '放在陌光下', 'target': '放在阳光下', 'errors': [['陌', '阳', 2, 1.0]]}
{'index': 3, 'source': '真麻烦你了。希望你们好好的跳无', 'target': '真麻烦你了。希望你们好好地跳舞', 'errors': [['的', '地', 12, 0.6584], ['无', '舞', 14, 1.0]]}
{'index': 4, 'source': '少先队员因该为老人让坐', 'target': '少先队员应该为老人让坐', 'errors': [['因', '应', 4, 0.995]]}
{'index': 5, 'source': '机七学习是人工智能领遇最能体现智能的一个分知', 'target': '机器学习是人工智能领域最能体现智能的一个分支', 'errors': [['七', '器', 1, 0.9998], ['遇', '域', 10, 0.9999], ['知', '支', 21, 1.0]]}
{'index': 6, 'source': '一只小鱼船浮在平净的河面上', 'target': '一只小鱼船浮在平静的河面上', 'errors': [['净', '静', 8, 0.9961]]}
################################################################################################################################
默认纠错(list输入, 参数配置):
{'index': 1, 'source': '馀额还有100w', 'target': '余额还有100w', 'errors': [['馀', '余', 0, 1.0]]}
{'index': 2, 'source': '放在陌光下', 'target': '放在阳光下', 'errors': [['陌', '阳', 2, 1.0]]}
{'index': 3, 'source': '真麻烦你了。希望你们好好的跳无', 'target': '真麻烦你了。希望你们好好地跳舞', 'errors': [['的', '地', 12, 0.6584], ['无', '舞', 14, 1.0]]}
{'index': 4, 'source': '少先队员因该为老人让坐', 'target': '少先队员应该为老人让坐', 'errors': [['因', '应', 4, 0.995]]}
{'index': 5, 'source': '机七学习是人工智能领遇最能体现智能的一个分知', 'target': '机器学习是人工智能领域最能体现智能的一个分支', 'errors': [['七', '器', 1, 0.9998], ['遇', '域', 10, 0.9999], ['知', '支', 21, 1.0]]}
{'index': 6, 'source': '一只小鱼船浮在平净的河面上', 'target': '一只小鱼船浮在平静的河面上', 'errors': [['净', '静', 8, 0.9961]]}
################################################################################################################################
基础纠错(不包括后处理和过滤):
{'source': '真麻烦你了。希望你们好好的跳无', 'target': '真麻烦你了。希望你们好好地跳舞', 'errors': [['的', '地', 12], ['无', '舞', 14]]}
{'source': '少先队员因该为老人让坐', 'target': '少先队员应该为老人让坐', 'errors': [['因', '应', 4]]}
{'source': '机七学习是人工智能领遇最能体现智能的一个分知', 'target': '机器学习是人工智能领域最能体现智能的一个分支', 'errors': [['七', '器', 1], ['遇', '域', 10], ['知', '支', 21]]}
{'source': '一只小鱼船浮在平净的河面上', 'target': '一只小鱼船浮在平静的河面上', 'errors': [['净', '静', 8]]}
################################################################################################################################
长句纠错(默认切句):
{'index': 1, 'source': '希望你们好好的跳无。', 'target': '希望你们好好的跳舞。', 'errors': [['无', '舞', 14, 1.0]]}
{'index': 2, 'source': '少先队员因该为老人让坐。', 'target': '少先队员应该为老人让坐。', 'errors': [['因', '应', 20, 0.9924]]}
{'index': 3, 'source': '机七学习是人工智能领遇最能体现智能的一个分知。', 'target': '机器学习是人工智能领域最能体现智能的一个分支。', 'errors': [['七', '器', 29, 1.0], ['遇', '域', 38, 1.0], ['知', '支', 49, 1.0]]}
{'index': 4, 'source': '一只小鱼船浮在平净的河面上。', 'target': '一只小鱼船浮在平静的河面上。', 'errors': [['净', '静', 59, 0.9981]]}
################################################################################################################################
"""

while 1:
    try:
        print("请输入：")
        text = input()
        print(correct([text]))
    except Exception as e:
        print(str(e))

