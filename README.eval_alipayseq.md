# CSC测评(macro-correct)
## 一、测评(Test)
### 1.1 测评数据来源
``` 所有训练数据均来自公网或开源数据, 训练数据为1千万左右, 混淆词典较大;
1.alipayseq.tet.json: 阿里搜索真实数据, 论文:《Towards Better Chinese Spelling Check for Search Engines: A New Dataset and Strong Baseline》;
```
### 1.2 测评数据预处理
```
测评数据都经过 全角转半角,繁简转化,标点符号标准化等操作;
```

### 1.3 其他说明
```
1.指标带common的极为宽松指标, 同开源项目pycorrector的评估指标;
2.指标带strict的极为严格指标, 同开源项目[wangwang110/CSC](https://github.com/wangwang110/CSC);
3.macbert4mdcspell_v1模型为训练使用mdcspell架构+bert的mlm-loss, 但是推理的时候只用bert-mlm;
4.acc_rmrb/acc_xxqg数据集没有错误, 用于评估模型的误纠率(过度纠错);
5.qwen25_1-5b_pycorrector的模型为shibing624/chinese-text-correction-1.5b, 其训练数据包括了lemon_v2/mcsc_tet/ecspell的验证集和测试集, 其他的bert类模型的训练不包括验证集和测试集;
```


## 二、重要指标
### 2.1 F1(common_cor_f1)
| model/common_cor_f1| alipayseq |
|:-----------------|:----------|
| macbert4csc_pycorrector| 15.36     |
| bert4csc_v1| 42.23     | 
| macbert4csc_v1| 48.45     |
| macbert4csc_v2| 45.60     | 
| macbert4mdcspell_v1| 48.97     | 

### 2.2 acc(common_cor_acc)
| model/common_cor_acc    | alipayseq |
|:------------------------|:----------|
| macbert4csc_pycorrector | 13.74     |
| bert4csc_v1             | 41.61     | 
| macbert4csc_v1          | 48.51     |
| macbert4csc_v2          | 46.70     | 
| macbert4mdcspell_v1     | 51.90     | 

