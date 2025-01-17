# csc_eval_public
## 一、测评数据说明
### 1.1 测评数据来源
``` 所有训练数据均来自公网或开源数据
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
### 1.2 测评数据预处理
```
测评数据都经过 全角转半角,繁简转化,标点符号标准化等操作;
```
