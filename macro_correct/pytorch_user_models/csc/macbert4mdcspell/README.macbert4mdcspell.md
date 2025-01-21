# MDCSpell: A Multi-task Detector-Corrector Framework for Chinese Spelling Correction

MDCSpell: A Multi-task Detector-Corrector Framework for Chinese Spelling Correction

https://aclanthology.org/2022.findings-acl.98/

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
### **ReLM on ecspell-law**
```bash
Experiments-mdcspell-ecspell-law---macbert---lr=3e5---batchsize=8*16---MFT_mask_rate=0.15---det_rate=0.15---epoch=5k---losstype=focal_loss---scheduler=linear
Sentence Level detection: acc:0.8700, precision:0.8035, recall:0.8980, f1:0.8481
Sentence Level correction: acc:0.8600, precision:0.7860, recall:0.8784, f1:0.8296(bert-mft==0.76)

Experiments-mdcspell-ecspell-law---macbert---lr=3e5---batchsize=16*4---MFT_mask_rate=0.15---det_rate=0.15---epoch=5k---losstype=focal_loss---scheduler=linear
Sentence Level detection: acc:0.8960, precision:0.8750, recall:0.8235, f1:0.8485
Sentence Level correction: acc:0.8920, precision:0.8667, recall:0.8157, f1:0.8404

Experiments-mdcspell-ecspell-law---macbert---lr=3e5---batchsize=16*8---MFT_mask_rate=0.15---det_rate=0.15---epoch=5k---losstype=focal_loss---scheduler=linear
Sentence Level detection: acc:0.8820, precision:0.8231, recall:0.8941, f1:0.8571
Sentence Level correction: acc:0.8780, precision:0.8159, recall:0.8863, f1:0.8496

Experiments-mdcspell-ecspell-law---macbert---lr=3e5---batchsize=32*4---MFT_mask_rate=0.15---det_rate=0.15---epoch=5k---losstype=focal_loss---scheduler=linear
Sentence Level detection: acc:0.8860, precision:0.8284, recall:0.8901, f1:0.8582
Sentence Level correction: acc:0.8820, precision:0.8211, recall:0.8823, f1:0.8506
```


### **ReLM on Sighan15**
```bash
mdcspell-std-offical-sighan15
Sentence Level 
MDCSpell-det	80.8	80.6	80.7	
MDCSpell-cor    78.4	78.2	78.3

shibing624/CSC
test.json 和 dev.json 为 SIGHAN数据集, 包括SIGHAN13 14 15,来自 官方csc.html ,文件大小: 339kb,4千条;
train.json 为 Wang271k数据集,包括 Wang271k ,来自 Automatic-Corpus-Generation dimmywang提供 ,文件大小: 93MB,27万条;


graph=mdcspell; prompt=0; weights=macbert; corpus=wang271k_clean_std; lr=3e-5; bs=32*4; epoch=10; loss=focal_loss; scheduler=linear; mask_mode=noerror; mask_rate=0.15; det_rate=0.15; 
################################################################################################################################
dev.json
flag_eval: common
Sentence Level detection: acc:0.8934, precision:0.9989, recall:0.8927, f1:0.9428
Sentence Level correction: acc:0.8410, precision:0.9988, recall:0.8396, f1:0.9123
flag_eval: strict
Sentence Level detection: acc:0.8934, precision:0.9261, recall:0.8927, f1:0.9091
Sentence Level correction: acc:0.8410, precision:0.8710, recall:0.8396, f1:0.8550
################################################################################################################################
test.json
flag_eval: common
Sentence Level detection: acc:0.8245, precision:0.8454, recall:0.7840, f1:0.8135
Sentence Level correction: acc:0.8036, precision:0.8379, recall:0.7412, f1:0.7866
flag_eval: strict
Sentence Level detection: acc:0.8245, precision:0.7234, recall:0.7840, f1:0.7525
Sentence Level correction: acc:0.8036, precision:0.6838, recall:0.7412, f1:0.7113


graph=mdcspell; prompt=0; weights=macbert; corpus=wang271k_org_std; lr=3e-5; bs=32*4; epoch=10; loss=focal_loss; scheduler=linear; mask_mode=noerror; mask_rate=0.15; det_rate=0.15; 
################################################################################################################################
dev.json
flag_eval: common
Sentence Level detection: acc:0.8925, precision:0.9993, recall:0.8919, f1:0.9425
Sentence Level correction: acc:0.8397, precision:0.9993, recall:0.8384, f1:0.9118
flag_eval: strict
Sentence Level detection: acc:0.8925, precision:0.9248, recall:0.8919, f1:0.9081
Sentence Level correction: acc:0.8397, precision:0.8694, recall:0.8384, f1:0.8536
################################################################################################################################
test.json
flag_eval: common
Sentence Level detection: acc:0.8118, precision:0.8347, recall:0.7716, f1:0.8019
Sentence Level correction: acc:0.7927, precision:0.8274, recall:0.7330, f1:0.7773
flag_eval: strict
Sentence Level detection: acc:0.8118, precision:0.7102, recall:0.7716, f1:0.7396
Sentence Level correction: acc:0.7927, precision:0.6746, recall:0.7330, f1:0.7026
```



## Modfix
```
1.add MFT
```

## log
### 2023-11-19
1.推理时候要和训练时候使用一样的csc.config, 防止报错;
2.这个只在relm中使用, mlm-loss还是要计算一个句子的, trg_ids[(src_ids == trg_ids)] = -100;

## 个人感受
```
1.该模型将检测和纠错交互, 使得CSC任务的指标好不少;
2.因为新增了网络架构, 并用于推理, 使得模型泛化性不强(需要在大规模语料上微调, 否则不太适合);
```

