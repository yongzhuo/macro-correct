<p align="center">
    <img src="tet/images/csc_logo.png" width="480">
</p>

# [macro-correct](https://github.com/yongzhuo/macro-correct) 
[![PyPI](https://img.shields.io/pypi/v/macro-correct)](https://pypi.org/project/macro-correct/)
[![Build Status](https://travis-ci.com/yongzhuo/macro-correct.svg?branch=master)](https://travis-ci.com/yongzhuo/macro-correct)
[![PyPI_downloads](https://img.shields.io/pypi/dm/macro-correct)](https://pypi.org/project/macro-correct/)
[![Stars](https://img.shields.io/github/stars/yongzhuo/macro-correct?style=social)](https://github.com/yongzhuo/macro-correct/stargazers)
[![Forks](https://img.shields.io/github/forks/yongzhuo/macro-correct.svg?style=social)](https://github.com/yongzhuo/macro-correct/network/members)
[![Join the chat at https://gitter.im/yongzhuo/macro-correct](https://badges.gitter.im/yongzhuo/macro-correct.svg)](https://gitter.im/yongzhuo/macro-correct?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
>>> macro-correct, 文本纠错工具包(Text Correct), 支持中文拼写纠错/标点符号纠错(CSC, Chinese Spelling Correct / Check), CSC支持各领域数据(包括古文), 模型在大规模、各领域的、现代/当代语料上训练而得, 泛化性强.

>>> macro-correct是一个只依赖pytorch、transformers、numpy、opencc的文本纠错(CSC, 中文拼写纠错; Punct, 中文标点纠错)工具包，专注于中文文本纠错的极简自然语言处理工具包。
使用大部分市面上的开源数据集构建生成的混淆集,使用人民日报语料&学习强国语料等生成1000万+训练数据集来训练模型;
支持MDCSpell、Macbert、ReLM、SoftBERT、BertCRF等多种经典模型;
支持中文拼写纠错、中文标点符号纠错、中文语法纠错(待续)、独立的检测模型/识别模型(待续);
具有依赖轻量、代码简洁、注释详细、调试清晰、配置灵活、拓展方便、适配NLP等特性。


## 目录
* [安装](#安装)
* [调用](#调用)
* [测评](#测评)
* [日志](#日志)
* [参考](#参考)
* [论文](#论文)
* [Cite](#Cite)


# 安装 
```bash
pip install macro-correct

# 清华镜像源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple macro-correct

# 如果不行, 则不带依赖安装, 之后缺什么包再补充什么
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple macro-correct --no-dependencies
```


# 调用
  更多样例sample详情见/tet目录
  - 使用example详见/tet/tet目录, 中文拼写纠错代码为tet_csc_token_zh.py, 中文标点符号纠错代码为tet_csc_punct_zh.py, CSC也可以直接用tet_csc_flag_transformers.py
  - 训练代码详见/tet/train目录, 可配置本地预训练模型地址和各种参数等;

## 2.调用-文本纠错
### 2.1 CSC 使用 macro-bert
```python
# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/29 21:41
# @author  : Mo
# @function: 文本纠错, 使用macro-correct


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
```

### 2.2 CSC 使用 transformers
```bash
# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/29 21:41
# @author  : Mo
# @function: transformers直接加载bert类模型测试


import traceback
import time
import sys
import os
os.environ["USE_TORCH"] = "1"
from transformers import BertConfig, BertTokenizer, BertForMaskedLM
import torch

# pretrained_model_name_or_path = "shibing624/macbert4csc-base-chinese"
pretrained_model_name_or_path = "Macropodus/macbert4mdcspell_v1"
# pretrained_model_name_or_path = "Macropodus/macbert4csc_v1"
# pretrained_model_name_or_path = "Macropodus/macbert4csc_v2"
# pretrained_model_name_or_path = "Macropodus/bert4csc_v1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_len = 128

print("load model, please wait a few minute!")
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
bert_config = BertConfig.from_pretrained(pretrained_model_name_or_path)
model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path)
model.to(device)
print("load model success!")

texts = [
    "机七学习是人工智能领遇最能体现智能的一个分知",
    "我是练习时长两念半的鸽仁练习生蔡徐坤",
    "真麻烦你了。希望你们好好的跳无",
    "他法语说的很好，的语也不错",
    "遇到一位很棒的奴生跟我疗天",
    "我们为这个目标努力不解",
]
len_mid = min(max_len, max([len(t)+2 for t in texts]))

with torch.no_grad():
    outputs = model(**tokenizer(texts, padding=True, max_length=len_mid,
                                return_tensors="pt").to(device))

def get_errors(source, target):
    """   极简方法获取 errors   """
    len_min = min(len(source), len(target))
    errors = []
    for idx in range(len_min):
        if source[idx] != target[idx]:
            errors.append([source[idx], target[idx], idx])
    return errors

result = []
for probs, source in zip(outputs.logits, texts):
    ids = torch.argmax(probs, dim=-1)
    tokens_space = tokenizer.decode(ids[1:-1], skip_special_tokens=False)
    text_new = tokens_space.replace(" ", "")
    target = text_new[:len(source)]
    errors = get_errors(source, target)
    print(source, " => ", target, errors)
    result.append([target, errors])
print(result)
"""
机七学习是人工智能领遇最能体现智能的一个分知  =>  机器学习是人工智能领域最能体现智能的一个分支 [['七', '器', 1], ['遇', '域', 10], ['知', '支', 21]]
我是练习时长两念半的鸽仁练习生蔡徐坤  =>  我是练习时长两年半的个人练习生蔡徐坤 [['念', '年', 7], ['鸽', '个', 10], ['仁', '人', 11]]
真麻烦你了。希望你们好好的跳无  =>  真麻烦你了。希望你们好好地跳舞 [['的', '地', 12], ['无', '舞', 14]]
他法语说的很好，的语也不错  =>  他法语说得很好，德语也不错 [['的', '得', 4], ['的', '德', 8]]
遇到一位很棒的奴生跟我疗天  =>  遇到一位很棒的女生跟我聊天 [['奴', '女', 7], ['疗', '聊', 11]]
我们为这个目标努力不解  =>  我们为这个目标努力不懈 [['解', '懈', 10]]
"""
```

## 3.调用-标点纠错
```python
import os
os.environ["MACRO_CORRECT_FLAG_CSC_TOKEN"] = "1"
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

"""
默认标点纠错(list输入):
{'index': 0, 'source': '山不在高有仙则名。', 'target': '山不在高，有仙则名。', 'score': 0.9917, 'errors': [['', '，', 4, 0.9917]]}
{'index': 1, 'source': '水不在深，有龙则灵', 'target': '水不在深，有龙则灵。', 'score': 0.9995, 'errors': [['', '。', 9, 0.9995]]}
{'index': 2, 'source': '斯是陋室惟吾德馨', 'target': '斯是陋室，惟吾德馨。', 'score': 0.9999, 'errors': [['', '，', 4, 0.9999], ['', '。', 8, 0.9998]]}
{'index': 3, 'source': '苔痕上阶绿草,色入帘青。', 'target': '苔痕上阶绿，草色入帘青。', 'score': 0.9998, 'errors': [['', '，', 5, 0.9998]]}
"""
```


# 测评
## 说明
* 所有训练数据均来自公网或开源数据, 训练数据为1千万左右, 混淆词典较大;
* 所有测试数据均来自公网或开源数据, 测评数据地址为[Macropodus/csc_eval_public](https://huggingface.co/datasets/Macropodus/csc_eval_public);
* 测评代码主要为[tcEval.py](https://github.com/yongzhuo/macro-correct/macro_correct/pytorch_textcorrection/tcEval.py); 其中[qwen25_1-5b_pycorrector]()的测评代码在目录[eval](https://github.com/yongzhuo/macro-correct/tet/eval)
* 评估标准：过纠率(过度纠错, 即高质量正确句子的错误纠正); 句子级宽松标准的准确率/精确率/召回率/F1(同[shibing624/pycorrector](https://github.com/shibing624/pycorrector)); 句子级严格标准的准确率/精确率/召回率/F1(同[wangwang110/CSC](https://github.com/wangwang110/CSC)); 字符级别的准确率/精确率/召回率/F1(错别字);
* qwen25_1-5b_pycorrector权重地址在[shibing624/chinese-text-correction-1.5b](https://huggingface.co/shibing624/chinese-text-correction-1.5b)
* macbert4csc_pycorrector权重地址在[shibing624/macbert4csc-base-chinese](https://huggingface.co/shibing624/macbert4csc-base-chinese);
* macbert4mdcspell_v1权重地址在[Macropodus/macbert4mdcspell_v1](https://huggingface.co/Macropodus/macbert4mdcspell_v1);
* macbert4csc_v2权重地址在[Macropodus/macbert4csc_v2](https://huggingface.co/Macropodus/macbert4csc_v2);
* macbert4csc_v1权重地址在[Macropodus/macbert4csc_v1](https://huggingface.co/Macropodus/macbert4csc_v1);
* bert4csc_v1权重地址在[Macropodus/bert4csc_v1](https://huggingface.co/Macropodus/bert4csc_v1);

## 3.1 测评数据
``` 
1.gen_de3.json(5545): '的地得'纠错, 由人民日报/学习强国/chinese-poetry等高质量数据人工生成;
2.lemon_v2.tet.json(1053): relm论文提出的数据, 多领域拼写纠错数据集(7个领域), ; 包括game(GAM), encyclopedia (ENC), contract (COT), medical care(MEC), car (CAR), novel (NOV), and news (NEW)等领域;
3.acc_rmrb.tet.json(4636): 来自NER-199801(人民日报高质量语料);
4.acc_xxqg.tet.json(5000): 来自学习强国网站的高质量语料;
5.gen_passage.tet.json(10000): 源数据为qwen生成的好词好句, 由几乎所有的开源数据汇总的混淆词典生成;
6.textproof.tet.json(1447): NLP竞赛数据, TextProofreadingCompetition;
7.gen_xxqg.tet.json(5000): 源数据为学习强国网站的高质量语料, 由几乎所有的开源数据汇总的混淆词典生成;
8.faspell.dev.json(1000): 视频字幕通过OCR后获取的数据集; 来自爱奇艺的论文faspell;
9.lomo_tet.json(5000): 主要为音似中文拼写纠错数据集; 来自腾讯; 人工标注的数据集CSCD-NS;
10.mcsc_tet.5000.json(5000): 医学拼写纠错; 来自腾讯医典APP的真实历史日志; 注意论文说该数据集只关注医学实体的纠错, 常用字等的纠错并不关注;
11.ecspell.dev.json(1500): 来自ECSpell论文, 包括(law/med/gov)等三个领域;
12.sighan2013.dev.json(1000): 来自sighan13会议;
13.sighan2014.dev.json(1062): 来自sighan14会议;
14.sighan2015.dev.json(1100): 来自sighan15会议;
```

## 3.2 测评再说明
```
1.数据预处理, 测评数据都经过 全角转半角,繁简转化,标点符号标准化等操作;
2.指标带common的极为宽松指标, 同开源项目pycorrector的评估指标;
3.指标带strict的极为严格指标, 同开源项目[wangwang110/CSC](https://github.com/wangwang110/CSC);
4.macbert4mdcspell_v1模型为训练使用mdcspell架构+bert的mlm-loss, 但是推理的时候只用bert-mlm;
5.acc_rmrb/acc_xxqg数据集没有错误, 用于评估模型的误纠率(过度纠错);
6.qwen25_1-5b_pycorrector的模型为shibing624/chinese-text-correction-1.5b, 其训练数据包括了lemon_v2/mcsc_tet/ecspell的验证集和测试集, 其他的bert类模型的训练不包括验证集和测试集;
```

## 3.3 测评结果
### 3.3.1 F1(common_cor_f1)
| model/common_cor_f1| avg| gen_de3| lemon_v2| gen_passage| text_proof| gen_xxqg| faspell| lomo_tet| mcsc_tet| ecspell| sighan2013| sighan2014| sighan2015 |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| macbert4csc_pycorrector| 45.8| 42.44| 42.89| 31.49| 46.31| 26.06| 32.7| 44.83| 27.93| 55.51| 70.89| 61.72| 66.81 |
| bert4csc_v1| 62.28| 93.73| 61.99| 44.79| 68.0| 35.03| 48.28| 61.8| 64.41| 79.11| 77.66| 51.01| 61.54 |
| macbert4csc_v1| 68.55| 96.67| 65.63| 48.4| 75.65| 38.43| 51.76| 70.11| 80.63| 85.55| 81.38| 57.63| 70.7 |
| macbert4csc_v2| 68.6| 96.74| 66.02| 48.26| 75.78| 38.84| 51.91| 70.17| 80.71| 85.61| 80.97| 58.22| 69.95 |
| macbert4mdcspell_v1| 71.1| 96.42| 70.06| 52.55| 79.61| 43.37| 53.85| 70.9| 82.38| 87.46| 84.2| 61.08| 71.32 |
| qwen25_1-5b_pycorrector| 45.11| 27.29| 89.48| 14.61| 83.9| 13.84| 18.2| 36.71| 96.29| 88.2| 36.41| 15.64| 20.73 |

### 3.3.2 acc(common_cor_acc)
| model/common_cor_acc| avg| gen_de3| lemon_v2| gen_passage| text_proof| gen_xxqg| faspell| lomo_tet| mcsc_tet| ecspell| sighan2013| sighan2014| sighan2015 |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| macbert4csc_pycorrector| 48.26| 26.96| 28.68| 34.16| 55.29| 28.38| 22.2| 60.96| 57.16| 67.73| 55.9| 68.93| 72.73 |
| bert4csc_v1| 60.76| 88.21| 45.96| 43.13| 68.97| 35.0| 34.0| 65.86| 73.26| 81.8| 64.5| 61.11| 67.27 |
| macbert4csc_v1| 65.34| 93.56| 49.76| 44.98| 74.64| 36.1| 37.0| 73.0| 83.6| 86.87| 69.2| 62.62| 72.73 |
| macbert4csc_v2| 65.22| 93.69| 50.14| 44.92| 74.64| 36.26| 37.0| 72.72| 83.66| 86.93| 68.5| 62.43| 71.73 |
| macbert4mdcspell_v1| 67.15| 93.09| 54.8| 47.71| 78.09| 39.52| 38.8| 71.92| 84.78| 88.27| 73.2| 63.28| 72.36 |
| qwen25_1-5b_pycorrector| 46.09| 15.82| 81.29| 22.96| 82.17| 19.04| 12.8| 50.2| 96.4| 89.13| 22.8| 27.87| 32.55 |

### 3.3.3 acc(acc_true, thr=0.75)
| model/acc                | avg| acc_rmrb| acc_xxqg |
|:-------------------------|:-----------------|:-----------------|:-----------------|
| macbert4csc_pycorrector  | 99.24| 99.22| 99.26 |
| bert4csc_v1          | 98.71| 98.36| 99.06 |
| macbert4csc_v1           | 97.72| 96.72| 98.72 |
| macbert4csc_v2           | 97.89| 96.98| 98.8 |
| macbert4mdcspell_v1      | 97.75| 96.51| 98.98 |
| qwen25_1-5b_pycorrector  | 82.0| 77.14| 86.86 |

### 3.3.4 结论(Conclusion)
```
1.macbert4csc_v1/macbert4csc_v2/macbert4mdcspell_v1等模型使用多种领域数据训练, 比较均衡, 也适合作为第一步的预训练模型, 可用于专有领域数据的继续微调;
2.比较macbert4csc_pycorrector/bertbase4csc_v1/macbert4csc_v2/macbert4mdcspell_v1, 观察表2.3, 可以发现训练数据越多, 准确率提升的同时, 误纠率也会稍微高一些;
3.MFT(Mask-Correct)依旧有效, 不过对于数据量足够的情形提升不明显, 可能也是误纠率升高的一个重要原因;
4.训练数据中也存在文言文数据, 训练好的模型也支持文言文纠错;
5.训练好的模型对"地得的"等高频错误具有较高的识别率和纠错率;
```


# 日志
```
1. v20240129, 完成csc_punct模块;
2. v20241001, 完成csc_token模块;
3. v20250117, 完成csc_eval模块；
```


# 参考
This library is inspired by and references following frameworks and papers.

* Chinese-text-correction-papers: [nghuyong/Chinese-text-correction-papers](https://github.com/nghuyong/Chinese-text-correction-papers)
* pycorrector: [shibing624/pycorrector](https://github.com/shibing624/pycorrector)
* CTCResources: [destwang/CTCResources](https://github.com/destwang/CTCResources)
* CSC: [wangwang110/CSC](https://github.com/wangwang110/CSC)
* char-similar: [yongzhuo/char-similar](https://github.com/yongzhuo/char-similar)
* MDCSpell: [iioSnail/MDCSpell_pytorch](https://github.com/iioSnail/MDCSpell_pytorch)
* CSCD-NS: [nghuyong/cscd-ns](https://github.com/nghuyong/cscd-ns)
* lemon: [gingasan/lemon](https://github.com/gingasan/lemon)
* ReLM: [Claude-Liu/ReLM](https://github.com/Claude-Liu/ReLM)


# 论文
## 中文拼写纠错(CSC, Chinese Spelling Correction)
* 共收录34篇论文, 写了一个简短的综述. 详见[README.csc_survey.md](https://github.com/yongzhuo/macro-correct/blob/master/README.csc_survey.md)


# Cite
For citing this work, you can refer to the present GitHub project. For example, with BibTeX:
```
@software{macro-correct,
    url = {https://github.com/yongzhuo/macro-correct},
    author = {Yongzhuo Mo},
    title = {macro-correct},
    year = {2025}

```


