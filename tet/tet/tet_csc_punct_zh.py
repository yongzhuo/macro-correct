# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/29 21:41
# @author  : Mo
# @function: tet csc of punct, 标点符号纠错(支持新增/修改, 不支持删除)


import os
os.environ["MACRO_CORRECT_FLAG_CSC_PUNCT"] = "1"

from macro_correct.pytorch_sequencelabeling.slTools import cut_sent_by_stay
from macro_correct import correct_punct_basic
from macro_correct import correct_punct_long
from macro_correct import correct_punct


### 1.默认标点纠错(list输入)
text_list = ["山不在高有仙则名。",
             "水不在深，有龙则灵",
             "斯是陋室惟吾德馨",
             "苔痕上阶绿草,色入帘青。"
             ]
text_csc = correct_punct(text_list)
print("默认标点纠错(list输入):")
for res_i in text_csc:
    print(res_i)
print("#" * 128)


### 2.默认标点纠错(list输入, 参数配置详情)
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
text_csc = correct_punct(text_list, **params)
print("默认标点纠错(list输入, 参数配置详情):")
for res_i in text_csc:
    print(res_i)
print("#"*128)


### 3.基础标点预测(短句, 可用于从0标注标点符号)
text = ["山不在高有仙则名水不在深有龙则灵",
        "斯是陋室惟吾德馨",
        "苔痕上阶绿草色入帘青"
        ]
text_csc = correct_punct_basic(text)
print("基础标点预测(短句, 可用于从0标注标点符号):")
for res_i in text_csc:
    print(res_i)
print("#" * 128)


### 4.默认切长句(string输入)
text = "予谓菊，花之隐逸者也牡丹，花之富贵者也莲，花之君子者也。噫！菊之爱陶后鲜有闻莲之爱同予者何人？牡丹之爱，宜乎众矣。"
text_csc = correct_punct_long(text)
print("默认切长句(string输入):")
for res_i in text_csc:
    print(res_i)
print("#" * 128)


### 5.可选不切句(不切句对长文本不友好)
text = "山不在高有仙则名。水不在深有龙则灵。"
text_csc = correct_punct_long(text, flag_cut=False)
print("可选不切句(不切句对长文本不友好):")
for res_i in text_csc:
    print(res_i)
print("#" * 128)


"""
默认标点纠错(list输入):
{'index': 0, 'source': '山不在高有仙则名。', 'target': '山不在高，有仙则名。', 'score': 0.9917, 'errors': [['', '，', 4, 0.9917]]}
{'index': 1, 'source': '水不在深，有龙则灵', 'target': '水不在深，有龙则灵。', 'score': 0.9995, 'errors': [['', '。', 9, 0.9995]]}
{'index': 2, 'source': '斯是陋室惟吾德馨', 'target': '斯是陋室，惟吾德馨。', 'score': 0.9999, 'errors': [['', '，', 4, 0.9999], ['', '。', 8, 0.9998]]}
{'index': 3, 'source': '苔痕上阶绿草,色入帘青。', 'target': '苔痕上阶绿，草色入帘青。', 'score': 0.9998, 'errors': [['', '，', 5, 0.9998]]}
################################################################################################################################
默认标点纠错(list输入, 参数配置详情):
{'index': 0, 'source': '山不在高有仙则名。', 'target': '山不在高，有仙则名。', 'score': 0.9917, 'errors': [['', '，', 4, 0.9917]]}
{'index': 1, 'source': '水不在深，有龙则灵', 'target': '水不在深，有龙则灵。', 'score': 0.9995, 'errors': [['', '。', 9, 0.9995]]}
{'index': 2, 'source': '斯是陋室惟吾德馨', 'target': '斯是陋室，惟吾德馨。', 'score': 0.9999, 'errors': [['', '，', 4, 0.9999], ['', '。', 8, 0.9998]]}
{'index': 3, 'source': '苔痕上阶绿草,色入帘青。', 'target': '苔痕上阶绿，草色入帘青。', 'score': 0.9998, 'errors': [['', '，', 5, 0.9998]]}
################################################################################################################################
基础标点预测(短句, 可用于从0标注标点符号):
{'source': '山不在高有仙则名水不在深有龙则灵', 'target': '山不在高，有仙则名；水不在深，有龙则灵。', 'text': '#山不在高有仙则名水不在深有龙则灵', 'label': [{'type': '0', 'ent': '高', 'pos': [4, 4], 'score': 0.9981, 'pun': '，'}, {'type': '3', 'ent': '名', 'pos': [8, 8], 'score': 0.9998, 'pun': '；'}, {'type': '0', 'ent': '深', 'pos': [12, 12], 'score': 0.999, 'pun': '，'}, {'type': '1', 'ent': '灵', 'pos': [16, 16], 'score': 0.9996, 'pun': '。'}], 'label_org': []}
{'source': '斯是陋室惟吾德馨', 'target': '斯是陋室，惟吾德馨。', 'text': '#斯是陋室惟吾德馨', 'label': [{'type': '0', 'ent': '室', 'pos': [4, 4], 'score': 0.9999, 'pun': '，'}, {'type': '1', 'ent': '馨', 'pos': [8, 8], 'score': 0.9998, 'pun': '。'}], 'label_org': []}
{'source': '苔痕上阶绿草色入帘青', 'target': '苔痕上阶绿，草色入帘青。', 'text': '#苔痕上阶绿草色入帘青', 'label': [{'type': '0', 'ent': '绿', 'pos': [5, 5], 'score': 0.9998, 'pun': '，'}, {'type': '1', 'ent': '青', 'pos': [10, 10], 'score': 0.9999, 'pun': '。'}], 'label_org': []}
################################################################################################################################
默认切长句(string输入):
{'index': 0, 'source': '予谓菊，花之隐逸者也牡丹，花之富贵者也莲，花之君子者也。', 'target': '予谓菊，花之隐逸者也；牡丹，花之富贵者也；莲花之君子者也。', 'score': 0.9999, 'errors': [['', '；', 10, 10, 0.9999], ['', '；', 19, 19, 0.9999]]}
{'index': 2, 'source': '菊之爱陶后鲜有闻莲之爱同予者何人？', 'target': '菊之爱，陶后鲜有闻；莲之爱，同予者何人？', 'score': 0.9842, 'errors': [['', '，', 3, 33, 0.9996], ['', '；', 8, 38, 0.9541], ['', '，', 11, 41, 0.9989]]}
################################################################################################################################
可选不切句(不切句对长文本不友好):
{'index': 0, 'source': '山不在高有仙则名。水不在深有龙则灵。', 'target': '山不在高，有仙则名；水不在深，有龙则灵。', 'score': 0.999, 'errors': [['', '，', 4, 4, 0.9981], ['。', '；', 8, 8, 0.9998], ['', '，', 13, 13, 0.999]]}
################################################################################################################################
"""

