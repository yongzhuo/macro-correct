# 训练
## 数据
### 数据格式一
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
### 数据格式二
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


## 使用pytorch_user_models/csc里的模型训练
详见以下readme.md
macbert4mdcspell: 详见[README.macbert4mdcspell.md](https://github.com/yongzhuo/macro-correct/blob/master/macro_correct/pytorch_user_models/csc/macbert4mdcspell/README.macbert4mdcspell.md)
macbert4csc: 详见[README.Macbert4CSC.md](https://github.com/yongzhuo/macro-correct/blob/master/macro_correct/pytorch_user_models/csc/macbert4csc/README.Macbert4CSC.md)
relm: 详见[README.relm.md](https://github.com/yongzhuo/macro-correct/blob/master/macro_correct/pytorch_user_models/csc/relm/README.relm.md)


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
