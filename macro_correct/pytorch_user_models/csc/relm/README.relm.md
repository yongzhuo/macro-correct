# Chinese Spelling Correction as Rephrasing Language Model

This is the repo for AAAI 2024 paper [Chinese Spelling Correction as Rephrasing Language Model](https://arxiv.org/abs/2308.08796).



## ReLM

Rephrasing Language Model (ReLM) is trained to rephrase the entire sentence by infilling additional slots, instead of character-to-character tagging. 
ReLM significantly enhances the generalizability of CSC models and refreshes the new state-of-the-art results across fine-tuned and zeroshot CSC benchmarks.  We also evaluate the transferability of ReLM in multi-task settings. Our analysis shows that ReLM effectively exploits and retains the pre-trained knowledge within PLMs, while tagging models do not.

You can find *ReLM* pre-trained model which is trained on 34 million monolingual data from [the repo of Rethinking Masked Language Modeling for Chinese Spelling Correction](https://github.com/gingasan/lemon).



## data
### data_style_1
file of json, List<dict> data-type, eg. macro-correct/macro_correct/corpus/text_correction/espell/csc_espell_med.train
```
[
{
"source": "累计债权余额不超过公司净资产的百分之四十",
"target": "累计债券余额不超过公司净资产的百分之四十"
},
{
"source": "法院适用特别程序审理案件，陪审员不参加案件的合议庭",
"target": "法院适用特别程序审理案件，陪审员不参加案件的合议庭"
},
{
"source": "下列属于证券交易内幕信息的之情人的是?",
"target": "下列属于证券交易内幕信息的知情人的是?"
}
]
```
### data_style_2
file of json, List<dict> data-type, same with pycorrector, eg. macro-correct/macro_correct/corpus/text_correction/sighan/sighan2015.train.json
```
[
{
"original_text": "但是我不能去参加，因为我有一点事情阿！",
"correct_text": "但是我不能去参加，因为我有一点事情啊！",
"wrong_ids": [
17
]
},
{
"original_text": "听起来是一份很好的公司。又意思又很多钱。",
"correct_text": "听起来是一份很好的公司。有意思又很多钱。",
"wrong_ids": [
    12
]
},
{
"original_text": "敬祝身体建慷。",
"correct_text": "敬祝身体健康。",
"wrong_ids": [
    4,
    5
]
}
]
```



## useage
### train
```bash
python train_yield.py
or
python train.py
```
### pred
```bash
python predict.py
```

### eval
```bash
python eval_product.py
or
python eval_std.py
```


## Experiments
### **ReLM on ECSpell**
```bash
std-offical
Sentence Level correction: acc:    , precision:0.8990, recall:0.9450, f1:0.9120

prompt=10
Sentence Level detection: acc:0.9160, precision:0.8700, recall:0.9526, f1:0.9094
Sentence Level correction: acc:0.9120, precision:0.8633, recall:0.9453, f1:0.9024

prompt=1
Sentence Level detection: acc:0.9280, precision:0.8925, recall:0.9396, f1:0.9154
Sentence Level correction: acc:0.9240, precision:0.8853, recall:0.9321, f1:0.9081

prompt=0; weights=relm
Sentence Level detection: acc:0.9520, precision:0.9158, recall:0.9804, f1:0.9470
Sentence Level correction: acc:0.9480, precision:0.9084, recall:0.9725, f1:0.9394
```


### **ReLM on Sighan15**
```bash
shibing624/CSC
test.json 和 dev.json 为 SIGHAN数据集, 包括SIGHAN13 14 15,来自 官方csc.html ,文件大小: 339kb,4千条;
train.json 为 Wang271k数据集,包括 Wang271k ,来自 Automatic-Corpus-Generation dimmywang提供 ,文件大小: 93MB,27万条;

prompt=0; weights=relm; corpus=wang271k-org; lr=3e-5; bs=32*4; loss=focal_loss; scheduler=linear; mask_mode=noerror; mask_rate=0.3; det_rate=0.3; 
dev.json
Sentence Level detection: acc:0.8463, precision:0.9056, recall:0.8452, f1:0.8744
Sentence Level correction: acc:0.7822, precision:0.8361, recall:0.7803, f1:0.8072
test.json
Sentence Level detection: acc:0.8045, precision:0.7312, recall:0.7164, f1:0.7237
Sentence Level correction: acc:0.7718, precision:0.6635, recall:0.6501, f1:0.6567


prompt=0; weights=relm; corpus=wang271k-clean; lr=3e-5; bs=8; loss=focal_loss; scheduler=linear; mask_mode=noerror; mask_rate=0.3; det_rate=0.3; 
dev.json
Sentence Level detection: acc:0.8773, precision:0.9184, recall:0.8763, f1:0.8969
Sentence Level correction: acc:0.8301, precision:0.8682, recall:0.8284, f1:0.8478

```


## Modfix
```
1.remove prompt of bi-lstm(to be simple concise and efficient);
2.move <CLS + source + SEP + target> to <CLS + source + SEP + target + SEP>;
```

## log
### 2023-11-19
推理时候要和训练时候使用一样的csc.config, 防止报错


## 个人感受
```
1.该模型泛化性好, 适合小样本数据集, 使用少量领域数据就能取得不错的效果;
2.该模型性能不行(最大文本长度翻倍), 指标也不太好, 如wank271k效果就不太行;
```

