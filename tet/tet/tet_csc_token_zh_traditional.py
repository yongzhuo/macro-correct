# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/29 21:41
# @author  : Mo
# @function: tet csc of token, 繁体中文拼写纠错(支持各领域纠错, 泛化性强, 包括古文)


import os
os.environ["MACRO_CORRECT_FLAG_CSC_TOKEN"] = "1"

from macro_correct.pytorch_textcorrection.tcTools import cut_sent_by_stay
from macro_correct import correct_tradition


### 默认纠错(list输入)
text_list = ["一個分知,陌光回聚,莪受打去,禰愛帶餘",
            "余額還有100w",
            "放在陌光下",
            "真麻煩你了。希望你們好好的跳舞",
            "少先隊員因該爲老人讓坐",
            "機七學習是人工智能領遇最能體現智能的一個分知",
            "一只小魚船浮在平淨的河面上",
            "這一條次,我選擇了一條與往常不同的路線。",
            "春節發貨部"
             ]
### 默认纠错(list输入, 参数配置)
params = {
    "flag_confusion": True,  # 是否使用默认的混淆词典
    "flag_prob": True,  # 是否返回纠错token处的概率
    "flag_cut": True,  # 是否切分句子, 长句, False会只处理前max_len长度的文本; True会按照标点切分(在超出就按照maxlen切分)
    "limit_nums_errors": 8,  # 一句话最多的错别字, 多的就剔除(全不纠错)
    "num_rethink": 0,  # 多次预测, think-twice
    "batch_size": 32,  # 批大小
    "threshold": 0.01,  # token阈值过滤
    "max_len": 128,  # 自定义的长度, 如果截断了, 则截断部分不参与纠错, 后续直接一模一样的补回来
    "rounded": 4,  # 保存4位小数
}
text_csc = correct_tradition(text_list, **params)
print("默认纠错(list输入, 参数配置):")
for res_i in text_csc:
    print(res_i)
print("#" * 128)

"""
默认纠错(list输入, 参数配置):
{'index': 0, 'source': '一個分知,陌光回聚,莪受打去,禰愛帶餘', 'target': '一個分支，陽光回聚，莪受打擊，禰愛帶餘', 'errors': [['知', '支', 3, 1.0], ['陌', '陽', 5, 1.0], ['去', '擊', 13, 1.0]]}
{'index': 1, 'source': '余額還有100w', 'target': '餘額還有100w', 'errors': [['余', '餘', 0, 1]]}
{'index': 2, 'source': '放在陌光下', 'target': '放在陽光下', 'errors': [['陌', '陽', 2, 1.0]]}
{'index': 3, 'source': '真麻煩你了。希望你們好好的跳舞', 'target': '真麻煩你了。希望你們好好地跳舞', 'errors': [['的', '地', 12, 0.8873]]}
{'index': 4, 'source': '少先隊員因該爲老人讓坐', 'target': '少先隊員應該爲老人讓坐', 'errors': [['因', '應', 4, 0.9933]]}
{'index': 5, 'source': '機七學習是人工智能領遇最能體現智能的一個分知', 'target': '機器學習是人工智能領域最能體現智能的一個分支', 'errors': [['七', '器', 1, 0.9999], ['遇', '域', 10, 1.0], ['知', '支', 21, 1.0]]}
{'index': 6, 'source': '一只小魚船浮在平淨的河面上', 'target': '一隻小魚船浮在平靜的河面上', 'errors': [['只', '隻', 1, 1], ['淨', '靜', 8, 0.9849]]}
{'index': 7, 'source': '這一條次,我選擇了一條與往常不同的路線。', 'target': '這一條次，我選擇了一條與往常不同的路線。', 'errors': []}
{'index': 8, 'source': '春節發貨部', 'target': '春節發貨不', 'errors': [['部', '不', 4, 0.3544]]}
################################################################################################################################
"""

while 1:
    try:
        print("请输入：")
        text = input()
        print(correct_tradition([text]))
    except Exception as e:
        print(str(e))

