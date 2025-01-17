# bert4sl_punct_zh_public
## 时间(time)
2024.6

## 训练数据构成(dataset)
使用高质量语料过滤而成, 收集高质量语料, 并使用PPL过滤等;
 - [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)
 - [chinese-poetry/huajianji](https://github.com/chinese-poetry/huajianji)
 - [garychowcmu/daizhigev20](https://github.com/garychowcmu/daizhigev20)
 - [yangjianxin1/Firefly](https://github.com/yangjianxin1/Firefly)
 - [学习强国428万数据](https://huggingface.co/datasets/Macropodus/xuexiqiangguo_428w); 国内源[Macropodus/xuexiqiangguo_428w](https://hf-mirror.com/datasets/Macropodus/xuexiqiangguo_428w) 
 - [xi_talk40万](https://huggingface.co/datasets/Papersnake/xi_talk); 国内源[Papersnake/xi_talk](https://hf-mirror.com/datasets/Papersnake/xi_talk)
 - [qwen-7b生成的100万好句]
 - [人民日报语料2000万]

## 训练说明
每种标点的最大句子数为10万, 总计500万训练句子, 训练3epoch;

