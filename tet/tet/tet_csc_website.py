# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/29 21:41
# @author  : Mo
# @function: tet csc of token, 简体中文拼写纠错(支持各领域纠错, 泛化性强, 包括古文)


import os
os.environ["MACRO_CORRECT_FLAG_CSC_TOKEN"] = "1"

from macro_correct import correct
### 默认纠错(list输入)
text_list = ["真麻烦你了。希望你们好好的跳无",
             "少先队员因该为老人让坐",
             "机七学习是人工智能领遇最能体现智能的一个分知",
             "一只小鱼船浮在平净的河面上"
             ]
text_csc = correct(text_list)
print("默认纠错(list输入):")
for res_i in text_csc:
    print(res_i)
print("#" * 128)

"""
默认纠错(list输入):
{'index': 0, 'source': '真麻烦你了。希望你们好好的跳无', 'target': '真麻烦你了。希望你们好好地跳舞', 'errors': [['的', '地', 12, 0.6584], ['无', '舞', 14, 1.0]]}
{'index': 1, 'source': '少先队员因该为老人让坐', 'target': '少先队员应该为老人让坐', 'errors': [['因', '应', 4, 0.995]]}
{'index': 2, 'source': '机七学习是人工智能领遇最能体现智能的一个分知', 'target': '机器学习是人工智能领域最能体现智能的一个分支', 'errors': [['七', '器', 1, 0.9998], ['遇', '域', 10, 0.9999], ['知', '支', 21, 1.0]]}
{'index': 3, 'source': '一只小鱼船浮在平净的河面上', 'target': '一只小鱼船浮在平静的河面上', 'errors': [['净', '静', 8, 0.9961]]}
"""

