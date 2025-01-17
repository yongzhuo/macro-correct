# 中文拼写纠错(CSC, Chinese Spelling Correct)一：综述、概述、简介
## 一、综述(概述)

>> 中文拼写纠错, CSC(Chinese Spell Correction, 有时候也叫Chinese Spell Checking), 是一种对中文文本中存在的拼写错误进行检测和纠正的技术，是自然语言处理NLP下的一项基础子任务。
狭义的拼写纠错一般只涉及字词的修改，不涉及新增和删除，也不涉及语法语义。

>> 需要指出的是，不同于英文等字母文字，汉语独特的普通话发音、象形文字字形、成句不留空格的书写系统等，使得中文在文本纠错尤其是中文拼写纠错上有着天然的独特性和挑战性。
这不，即便是大模型LLM时代的2024，“传统”的端到端的类BERT模型依旧是中文拼写错误任务的中流砥柱，大语言模型依旧无法替代这类细粒度的基础NLP任务。


![csc_sample.png](./test/images/csc_sample.png "csc样例")

ReLM, C-LLM, MacBERT4CSC, MaskCorrect, BERT-MFT, ECOPO, SoftMask-BERT, Bi-DCSpell, ChineseBERT, SpellBERT, SCOPE, DCN,
FASPell, Hybrid(Wang271k), CSCD-IME(+NS), MCSCSet, ChunkCSC, ThinkTwice


## 二、中文经典拼写纠错模型汇总(综述/概述)
### 2.1 引入拼音字形信息
引入拼音字形信息(Embedding层/中间层/输出层-Loss层)
```
Embedding层引入: ChineseBERT, SpellBERT, ReaLiSe, ECSPell, PLOME;
Middle中间层引入: PHMOSpell, SpellGCN, SCOPE;
输出层/Loss层引入: BERTCrsGad, PTCSpell, SpellGCN, LEAD, DCN;
```

### 2.2 检测错误和纠错一体化融合
```
SoftMask-BERT, Bi-DCSpell, MDCSpell, DCSpell, DR-CSC, Coin
```

### 2.3 prompt对齐预训练
```
ReLM, DORM, EGCM
```
### 2.4 训练+微调方式的调整
```
MaskCorrect, BERT-MFT, CL, ECOPO, SDCL, CRASpell, TwoWays, MLM-Phonetics
```
### 2.5 前处理+后处理
前处理+后处理, 主要是混淆集过滤或增强数据
```
FASPell, Hybrid(Wang271k), CSCD-IME(+NS), MCSCSet, Refining,
EdaCSC, RERIC(KNN-CSC), ChunkCSC, ThinkTwice
```
### 2.6 基于大模型
```
C-LLM, Eval-GCSC
```

## 三、经典CSC模型汇总分类(主要是BERT类)
### 3.1 引入拼音字形信息

![csc_embedding.png](./test/images/csc_embedding.png)

#### a1.引入拼音字形信息(embedding层)
```
将拼音、字形信息融入embedding, 可重新训练语言训练模型(如), 不过我比较喜欢简洁简单的, 这种方案没有采用;
1.ChineseBERT: 香浓科技, 重新训练了一个预训练模型(字符字形拼音嵌入融合), embedding(char + glyph + pinyin) -->> fusion-embedding + pos-embedding;
2.SpellBERT: 复旦大学, 重新训练了一个4-Layer的BERT模型, Embedding层引入拼音声母/韵母/拆字/结构信息(通过GCN); 同时训练任务也包含了正确句子的拼音声母/韵母/拆字等;
3.ReaLiSe: 腾讯, 基于hfl/chinese-roberta-wwm-ext, 新加入了Graphic-Encode(单字的图片ResNet)编码, Phonetic-Encoder(拼音GRU)编码, 不同的是单字字符+拼音两个编码的相关性通过训练OCR任务实现联动; 多embedding融合的时候使用了4个Linear类似LSTM遗忘门的形式;
4.ECSPell: 苏州大学, 架构为: 拼音(pinyin-CNN)-字形(glyph-CNN)-语义(BERT) -->> concat(embed1, embed2, embed3) + 2-layer-transformer, 新加了一个glyph classification task; 新数据集ECSpell(law/med/gov);
5.PLOME: 腾讯, 重新训练一个预训练模型(100w页wiki-zh + 300w页新闻共 1.6亿个句子), embedding(glyph + pinyin)拼音字形使用GRU生成; 预训练使用混淆集而不是[MASK](60%拼音混淆集/15%字形/15%原字符/10%随机); 训练微调中有拼音(如feng)的ner任务; t推理时候使用联合概率(char*pun)。
```

#### a2.引入拼音字形信息(中间层)
```
新加入一个分支融合拼音、字形信息, 或者是与原transformer层交互;
1.SpellGCN: 阿里, 新加了一个分支(图卷积网络), 基于混淆集的拼音node-embedding+字形embedding, 然后在拼音+字形间引入attention, 用于det分类(即错误检测);
2.SCOPE: 中国科学技术大学, one-encoder-two-decoder, 新加一个拼音预测分支(包括声母/韵母/声调); CIC推理(多次迭代,每次2个[最左和最右],迭代3次,每次都改变的恢复原状); 此外基于混淆词典的预训练也能提1%(15%的mask中,80%混淆词典,10%随机,10%不变);
3.PHMOSpell: 复旦大学, 拼音/字形/bert三个分支, 最后输出交互; 拼音使用TTS语音模型Tacotron2; 字形使用类似VGG19识别文字(类似NER); 最后输出融合使用bert-out*拼音out(0.6) + bert-out*字形out + bert-out(0.4);
```

#### a3.引入拼音字形信息(输出层/LOSS层)
```
重点是在输出层, 或者是Loss层等交互;
1.DCN: 讯飞, 在RoBERTa后新加了一个拼音编码层Pinyin-Encode与预训练模型输出交互(Attention); 只有MLM-Loss(没有det);
2.SpellGCN: 阿里, 新加了一个分支(图卷积网络), 基于混淆集的拼音node-embedding + 字形embedding, 然后在拼音+字形间引入attention, 用于分类(即错误检测);
3.LEAD: 清华大学, 拼音/字形/释义的对比学习Loss, 即拉近target句子与拼音/字形/语义相似句子的距离, 样本选择(拼音/字形正样本为替换拼音/字形相似的句子, 反样本不一样的随机); 语义样本(正样本选择最相似的-会有多个的情况, 其他随机); 负样本数为8; mlm:infoNCE=1/0.1(训练)-1/0.25(微调)
4.PTCSpell: 上海大学, electra检测, BERT纠错(拼音/字形), 为了处理训练微调的gap问题, 使用混淆集替换后进行预训练(预训练1.10%替换混淆集, 2.4%选中正确的字符其他mask为[Z](现实错误率为3.3%)); 为了避免过度矫正问题, 把正确句子识别为[Z]参与loss计算(有点类似于识别了);
5.BERTCrsGad: 平安银行, 引入一个GlobalAttention层, 用于强化拼音字形的权重(即混淆词典的词向量c-topk * bert-encode后的V, topk取3最好); 此外还加了一个det模块(同macbert4csc, rate取0.1); 
```


### 3.2 检测错误和纠错一体化融合

![csc_method.png](./test/images/csc_method.png)

```
1.SoftMask-BERT: 字节跳动, 提出的检测识别一体化模型, 使用bert-embedding+BiGRU用于检测(有det-loss), 然后使用det-label进行遮挡embedding+原始embedding; 以目前的视角来看, BiGRU用来检测可能效果和泛化性不佳; 检测和识别只使用embedding交互可能也太浅了;
3.Bi-DCSpell: 北京理工大学, 识别模型和检测模型相互交互(Cross-Attention, 2层效果最佳, 检测模块交互程度α=0.4, 纠错模块交互程度β=0.3, 检测损失的比率rate=0.2), 有点类似阅读理解的双流注意力机制, 指标提升了一点, 单句子准确率提升还好; 
3.MDCSpell: 阿里, 识别模型和检测模型相互交互(Pipeline, detect -->> correct), 检测指导纠错模块; det使用2层较好(bert-init), det-loss的λ为0.15;
4.DCSpell: 华中科技大学, 检测纠错pipeline, 先用electra进行det, 然对错误位置进行[mask]的MLM预测等, 最后使用混淆集过滤; 这让我想到NER的pipeline当时也达到了stoa;
5.DR-CSC: 清华大学, 提出的检测识别一体化模型, 三个loss并行, 即: MLM; Detection(检测错误位置, bert_embedding+2GRU); Reasoning(两个并行分支, 检测错误位置 + 类型拼音/字形, bert+BertPredictionHeadTransform(low-mid)) -->> Search(混淆集过滤cor_prob, 非混淆集的词典id置0)
6.Coin: 武汉大学, 提出的检测识别一体化模型(ReLM的改进版), 重点在于electra检测模型分为高recall-高precision两种情况, 高recall结果mask为target的输入, 高precision的结果embedding与source的embedding融合;
```

### 3.3 prompt对齐预训练

![csc_prompt.png](./test/images/csc_prompt.png)

```
1.ReLM: 上海交大, 提出的一种把CSC任务转化为RLM(Rephrasing Language Model)的模型; prompt构建方式为 source + target(MASK) -->> source + target; 注意他们重新再预训练了句对任务的BERT;
2.DORM: 中山大学, 原句(拼接原句的拼音): source + char-pinyin -->> target + target; 对比学习用kl散度; 融合了bert-mlm-loss/kl-loss/拼接source-mlm-loss/拼接-拼音-mlm-loss; 
3.EGCM: 北京大学, Seq2Seq结构, encode加入GAM(错别字的掩码, Guidance Attention Mask), 训练阶段还会mask原始句子的片段(混淆词典用于对比学习, char/所有混淆词-包括char自己), 推理时候放一个N*N的对角MASK的生成用于检测det(原char不在top3的mask); 然后由小到大预测可能出错的token(即前面检测到的位置);
```

### 3.4 训练+微调方式的调整

![csc_ft.png](./test/images/csc_ft.png)

```
1.MaskCorrect: 浙江大学, [mask]正确的char能提点(15-25%); 其中sighan/wang271k数据集等是20%最佳, ASR数据集是15%最佳;(PS, 这么看的话, 数据量大最好是mask20%, 数据量少就mask25%)
2.BERT-MFT: 上海交大, 0.2比率的[mask]一个大高质量语料能获取无偏结果(随机mask非错误字符char), 泛化迁移能力更强, random-mask也比confusion-mask效果好, 这个tricks能提点3-8%; 
3.CL: 清华大学, 课程学习, 课程学习由易到难, 句子向量由不相似到相似, 错别字数有少到多, 难度分为10块, 效果最好;
4.ECOPO: 清华大学(深圳), 对比学习, 新增对比函数对抗高频char(优先取不常见char), 即目标token的概率减去topk均值的概率最小化; cpo损失:原始损失=1:1; 可能对LLM有害, 且有负值, 难训练;
5.SDCL: 上海AiLab, 引入对比学习, 字的对比学习(词典+其他字符-), 句子的对比学习(即便是错别字也相似?); 想法有点意思, 但也仅仅是想法, 代码和实验细节等;
6.CRASpell: 腾讯, 使用PointNet(copy+gen机制) + 噪声模型对比(kl-loss, 加入混淆词); 噪声选择什么(只在一个错字时加入, 70%音似/15%形似/15%随机), 噪声加入距离(5以内); cBERT问题过度纠错没那么严重;
7.TwoWays: 复旦大学, 预训练+对抗学习, 预训练语料收集于Weibo和Wikipedia(共930万句子), 句子中的char进行25%替换(90%混淆集, 10%随机); 对抗有点类似于数据增强, 对于那些容易出错的(真实logits-错字logits高的), 多加入混淆词典训练;
8.MLM-Phonetics: 百度, char-embed先det, 结果perr * pinyin-embed + (1-perr)*char-embed形成组合fuse-embed; 预训练的时候改变句子中20%的字(40%[mask]/30%拼音/30%混淆词典), 输入token引入一个拼音[mask](不带声调, 如de); 最后loss也使用perr来动态平衡det-loss/cor-loss; 地得的等相同拼音的不好处理;
```

### 3.5 前处理+后处理(混淆集的使用, 数据增强);

![csc_post.webp](./test/images/csc_post.webp)

```
1.FASPell: 百度, 爱奇艺构建的CSC训练集, 由视频字幕通过OCR后获取的数据集; 训练集为3575条, 测试集为1000条; 使用BERT的MLM, 同时融合混淆集解码(CSD, Confidence-Similarity Decoder), 具体为0.8*字符概率 + 0.2 * 相似度(拼音字形);
2.Hybrid(Wang271k): 腾讯, 构建训练音似加形似CSC数据集的方法, OCR+ASR; 初始语料为5w人民日报语料经过谷歌的Tesseract进行OCR经过高斯模糊的文本数据构建4w+中文拼写纠错数据(字形); 14w演讲数据AIShell经过Kaidi语音工具将语音数据生成文本数据的方式构建4w+拼写纠错数据集(拼音); D-ocr:D:asr数据配比为4:6效果最佳;
3.CSCD-IME(+NS): 腾讯, 构建的音似CSC数据集方法, 谷歌拼音输入法取得一个句子特定位置的topk/混淆拼音, 使用LCSTS摘要数据集通过谷歌拼音输入法构建的2M数据集可用于预训练; 
4.MCSCSet: 腾讯, 来自腾讯医典APP的真实历史日志; 医学拼写纠错; 原始数据由90w数据清洗后构建的20w数据, 有5种错误, 除了字音字形, 还有多字/少字/语序混乱; 这里的20w数据只专注于拼写纠错。注意论文说该数据集只关注医学实体的纠错, 常用字等的纠错并不关注。
5.Refining: 北京大学, 提出是的过滤训练数据集中噪声的方法, 900w句中国新闻数据随机替换混淆词典训练模型1过滤噪声(1.source/pred字符的logits大于0.9是噪声(单个句子中的如'他'-'她'), 混淆词典org&pred字符余弦相似度大于0.8是多答案(如'要50元'-'收50元')); 此外置信度取1e-2效果最佳; f1在9M数据时候最佳, FPR在1w数据时候最佳;
6.EdaCSC: 武汉理工大学, 数据增强, 切分为短句(Split sentences)/多个错误减少一个或多个(Reduce Typos), 莫名地感觉这也能中么;
7.RERIC(KNN-CSC): 北大, 基于KNN-LM对数据集制作3-gram的向量, 推理时候匹配训练集的向量数据库, 重排序, 概率为λPknn + (1-λ)Plm;
8.ChunkCSC: 阿里, 块Chunk解码, 即解码的时候使用beam-search, 只考虑一个编辑距离(1次替换), 候选集考虑拼音/字形/语义的混淆集;
9.ThinkTwice: 电子科技大学, 新加了一个后处理SVM分类器防止误测, 使用字符概率, 候选字符位置(rank), 字符相似度, 句子相似度等信息训练;
10.DISC: 阿里, 后处理概率计算引入拼音字形等的相似度(1.1), 0.7sim-pinyin + 0.3glyph; 拼音为1-编辑距离除以拼音长度和, 字形为四角编码(相同码的个数/4), 结构四角码(结构标志+偏旁部首, 1-编辑距离/SFC的和), 拆字编辑距离(1-编辑距离/拆字序列的和), 拆字最长公共子串(LCS/两个句子中最长的序列);
```

### 3.6 大模型;

![csc_LLM.png](./test/images/csc_LLM.png)

```
1.C-LLM: 清华大学, 修改并初始化tokenizer使得大模型编码char而不是word(剪切word); 继续预训练大模型, 然后SFT, 效果在全领域上还可以, 感觉更多是的大模型规模以及训练数据量带来的;
```

### 3.7 理论
```
1.IDPI: 复旦大学, 提出了一种CSC原理猜测, BERT模型等包含多少拼音/字形的信息; bert-embeddig训练的二分类实验, 音近(57%, 称,程)，形近(75%, 称,尔); 此外还有一个泛化的指标CCCR(即只测不在训练集中的字的准确率, 对比全局的, 查看泛化性);
2.Eval-GCSC: 清华大学, 提出大模型测评CSC任务的新基准(非同音/不等长); chatgpt更适合纠错语义问题/ngram问题/常识纠错; 但也大概率会改变语义/误纠/增删; 其他问题:不常见的单词/知识错误、两个连续的拼写错误、有多个合理的拼写错误更正。
```

### 汇总
```
简单有效的方法: pinyin/glyph, MFT, Rethink, Det/Cor union, Confusion; 
```

## 四、应用-场景
```
主要纠错：
    普通字词、人名、成语、古诗、机构、地点、日期、术语等。
具体来说：
    字词纠错包含音近字、形近字、数字、词语、成语、古诗等内容的纠错；
    标点纠错包含中英文标点混用、成对标点符号缺失、标点冗余等内容的纠错；
    专名纠错包含专有名词、固定短语等内容的纠错(地名/人名/机构/领导人职称/政治术语)；
    地址纠错包含地址别字、地址搭配、地址缺失纠错；
```


## 五、论文时间线(历史概述)
我们可以按照时间和技术特征等给已知的经典中文拼写纠错算法CSC分类。
按照NLP历史发展脉络来看, 可以分为基于规则类; 传统机器学习类; 深度学习类(特指CNN, LSTM时代); BERT类; 大模型类;

### 5.1 基于规则
```
基于规则的论文比较久远了, 目前用得比较多的就混淆词典. 
如前缀树+混淆词典; AC自动机+混淆词典;
```

### 5.2 传统机器学习
```
尽管传统机器学习兴起于20世纪90年代, 尽管CNN在2012年的计算机视觉领域崛起, 
但无论是sighan2013, 还是sighan2014, sighan2015, CSC学术界的主流技术还是传统机器学习;
一般为特征(N-Gram语言模型/分词/词性/依存句法)+SVM/CRF/ME/LR等的技术方案, 分为2个步骤(2个模型)的pipeline(检测+纠错), 也可以理解为候选集+重排模式;
比如CSC经典项目pycorrector默认的方法, 就是基于Kenlm统计语言模型工具训练了中文NGram语言模型，结合规则方法、混淆集可以纠正中文拼写错误，方法速度快，扩展性强，效果一般.
SRILM是由美国斯坦福国际研究院（SRI International）开发的，最初版本的开发始于1990年代中期，具体公开发布的时间可能是在1990年代末到2000年代初。
KenLM是由卡内基梅隆大学（Carnegie Mellon University）的Kenneth Heafield开发的，首次发布是在2009年左右，随着后续的研究和改进，KenLM逐渐成为了效率较高的语言模型工具之一。
```

### 5.3 深度学习-CNN/LSTM时代
```
深度学习最先在CV领域开始崛起, 标志是2012年的AlexNet, 此后, Inception、VGG、ResNet、DenseNet等经典CV骨干网络如雨后春笋般涌现;
但CNN并没有如在CV领域那样, 统治NLP, 如TextCNN依旧打不过SVM; 在我的印象里, Bi-LSTM才是那几年统治NLP的那个;
这方面的论文我看的不多, 那时候也没有关注CSC任务, 不过以现在的视角看, TextRCNN和BiLSTM-CRF可用于错字检测, Bi-LSTM和Seq2seq可用于错字纠正;
```

### 5.4 深度学习-BERT时代
```
BERT时代, google的预训练语言模型BERT以令人震惊的效果横扫了NLP榜单, 各种NLP任务子任务甚至没有了存在的必要;
NLP四大子任务之三, 分类/序列标注/句子关系判断等的工业化落地纷纷取得突破, CSC任务更是被统治至今, 依旧难以被撼动, 即便是LLM;
基于BERT类预训练模型的CSC论文就多了, 大体可以分为 a.引入拼音字形信息(embedding层/中间层/尾部/loss);
b.检测错误和纠错一体化融合; c.混淆集的使用; d.prompt对齐预训练; e.前处理+后处理; f.训练+微调方式的调整;
```

### 5.5 大模型-LLM时代
```
大模型时代, chatgpt更擅长创造性的生成, 相对更适合中文语法纠错CGEC一些. 不过也有问题, chatgpt偏润色而不是最小改错. 
至于这种要求严格中文拼写纠错CSC, chatgpt表现不佳, 即便是论文微调过, 也远不如BERT类模型;
结合最新的论文, 可能的原因有:
1. token因素, 大模型的字典压缩率太高, 编码更偏向于词和N-Gram, 天然对CSC等任务不利;
2. 位置信息影响, 大模型网络层更深, 对pos-embedding的绝对位置破坏越大, 位置信息更弱;
3. 预训练任务影响, 更多数据训练得更久, 训练语料中可能充斥了错别字的语料; 而且大模型的语义能力和泛化能力更强, 使得错误句子和正确句子的语义相似度特别高, 二者向量几乎一致, 使得模型难以区分;
```


## 六、背景-历史
CSC是NLP下的一个子任务，其发展历史必然遵循NLP发展的一般规律。
- a.起源时代【实体书时代】(?~1946)
```
  起源时代。自语言文字诞生的那一刻开始，其规范性要求和教育性功能就使得拼写纠错成为可能。
  在汉字发展过程中，早期的拼写纠错主要是通过官方的文字规范来进行。
  在传统的教育体系中，教师在教授学生汉字书写时，会进行一对一的拼写纠错。
```
- b.计算机时代(1946-1989)
```
  计算机时代。随着计算机硬件技术的发展，人工智能符号主义学派大兴，代表成果有启发式算法、专家系统、知识工程。
  其规则体系等至今依旧实用（如正则文法），规则一般能解决80%的问题。基于混淆词典的文本纠错也是专家系统和知识工程的一种应用。
```
- c.互联网时代(1989-2012)
```
  互联网时代。随着个人电脑的普及，互联网开始兴起，与之对应的是概率主义学派，即传统的经典机器学习。
  代表成果有线性回归、逻辑回归、决策树、K近邻算法、支持向量机、朴素贝叶斯、K均值聚类、主成分分析、条件随机场、随机森林、梯度提升算法。
  这一时期的特点是模型简单，需要人工提取特征，并依赖于统计学原理来实现模式识别。基础NLP任务得到突破。
  基于NLP基础子任务(N-GRAM语言模型、中文分词、词性分析、依存句法)，中文拼写纠错系统形成了'错误检测->候选召回->候选排序'的多模型Pipeline。
```
- d.移动互联网时代(2012-2020)
```
  移动互联网时代。随着3G和手机的普及，移动互联网开始崛起，与之以统发展而来的，还有深度学习。
  代表成果有AlexNet、ResNet、Word2Vec、Bi-LSTM、GAN、DeepFM、Transformer、BERT、SimCLR等。
  这一时期的特点是模型参数量大层次多、无需手动构建特征、端到端学习、需要比较多的标注数据、泛化性强。
  这一时期，除了精细化的拼写纠错，语法纠错等也取得突破，传统机器学习的多模型Pipeline也发展为只需要一个模型的端到端系统。
```
- e.大模型时代(2021-至今)
```
 大模型时代。随着深度学习的发展，手工标注越来越力不从心。技术的发展逐渐转向无监督学习和对比学习，利用凝结整个人类知识的互联网海量数据，大数据量训练超级大模型。
 代表成果有prompt-tuning、GPT3、InstructGPT、ChatGPT、Claude、Gmini、Llama、AlpacaGPT、LoRA等。
 这一时期的特点是模型参数量无比巨大层次无比多、Transformer架构开始一统NLP/NLU、Prompt工程、RAG增强、Agent代理等。
 这一时期，偏生成式任务、需要句意语义的任务基本被LLM攻陷，有时候甚至狂热到抬高至第四次工业革命的程度。
```


## 七、论文paper

 - 2024-Refining: [Refining Corpora from a Model Calibration Perspective for Chinese](https://arxiv.org/abs/2407.15498)
 - 2024-ReLM: [Chinese Spelling Correction as Rephrasing Language Model](https://arxiv.org/abs/2308.08796)
 - 2024-DICS: [DISC: Plug-and-Play Decoding Intervention with Similarity of Characters for Chinese Spelling Check](https://arxiv.org/abs/2412.12863)

 - 2023-Bi-DCSpell: [A Bi-directional Detector-Corrector Interactive Framework for Chinese Spelling Check]()
 - 2023-BERT-MFT: [Rethinking Masked Language Modeling for Chinese Spelling Correction](https://arxiv.org/abs/2305.17721)
 - 2023-PTCSpell: [PTCSpell: Pre-trained Corrector Based on Character Shape and Pinyin for Chinese Spelling Correction](https://arxiv.org/abs/2212.04068)
 - 2023-DR-CSC: [A Frustratingly Easy Plug-and-Play Detection-and-Reasoning Module for Chinese](https://aclanthology.org/2023.findings-emnlp.771)
 - 2023-DROM: [Disentangled Phonetic Representation for Chinese Spelling Correction](https://arxiv.org/abs/2305.14783)
 - 2023-EGCM: [An Error-Guided Correction Model for Chinese Spelling Error Correction](https://arxiv.org/abs/2301.06323)
 - 2023-IGPI: [Investigating Glyph-Phonetic Information for Chinese Spell Checking: What Works and What’s Next?](https://arxiv.org/abs/2212.04068)
 - 2023-CL: [Contextual Similarity is More Valuable than Character Similarity-An Empirical Study for Chinese Spell Checking]()

 - 2022-CRASpell: [CRASpell: A Contextual Typo Robust Approach to Improve Chinese Spelling Correction](https://aclanthology.org/2022.findings-acl.237)
 - 2022-MDCSpell: [MDCSpell: A Multi-task Detector-Corrector Framework for Chinese Spelling Correction](https://aclanthology.org/2022.findings-acl.98)
 - 2022-SCOPE: [Improving Chinese Spelling Check by Character Pronunciation Prediction: The Effects of Adaptivity and Granularity](https://arxiv.org/abs/2210.10996)
 - 2022-ECOPO: [The Past Mistake is the Future Wisdom: Error-driven Contrastive Probability Optimization for Chinese Spell Checking](https://arxiv.org/abs/2203.00991)

 - 2021-MLMPhonetics: [Correcting Chinese Spelling Errors with Phonetic Pre-training](https://aclanthology.org/2021.findings-acl.198)
 - 2021-ChineseBERT: [ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information](https://aclanthology.org/2021.acl-long.161/)
 - 2021-BERTCrsGad: [Global Attention Decoder for Chinese Spelling Error Correction](https://aclanthology.org/2021.findings-acl.122)
 - 2021-ThinkTwice: [Think Twice: A Post-Processing Approach for the Chinese Spelling Error Correction](https://www.mdpi.com/2076-3417/11/13/5832)
 - 2021-PHMOSpell: [PHMOSpell: Phonological and Morphological Knowledge Guided Chinese Spelling Chec](https://aclanthology.org/2021.acl-long.464)
 - 2021-SpellBERT: [SpellBERT: A Lightweight Pretrained Model for Chinese Spelling Check](https://aclanthology.org/2021.emnlp-main.287)
 - 2021-TwoWays: [Exploration and Exploitation: Two Ways to Improve Chinese Spelling Correction Models](https://aclanthology.org/2021.acl-short.56)
 - 2021-ReaLiSe: [Read, Listen, and See: Leveraging Multimodal Information Helps Chinese Spell Checking](https://arxiv.org/abs/2105.12306)
 - 2021-DCSpell: [DCSpell: A Detector-Corrector Framework for Chinese Spelling Error Correction](https://dl.acm.org/doi/10.1145/3404835.3463050)
 - 2021-PLOME: [PLOME: Pre-training with Misspelled Knowledge for Chinese Spelling Correction](https://aclanthology.org/2021.acl-long.233)
 - 2021-DCN: [Dynamic Connected Networks for Chinese Spelling Check](https://aclanthology.org/2021.findings-acl.216/)

 - 2020-SoftMaskBERT: [Spelling Error Correction with Soft-Masked BERT](https://arxiv.org/abs/2005.07421)
 - 2020-SpellGCN: [SpellGCN：Incorporating Phonological and Visual Similarities into Language Models for Chinese Spelling Check](https://arxiv.org/abs/2004.14166)
 - 2020-ChunkCSC: [Chunk-based Chinese Spelling Check with Global Optimization](https://aclanthology.org/2020.findings-emnlp.184)

 - 2019-FASPell: [FASPell: A Fast, Adaptable, Simple, Powerful Chinese Spell Checker Based On DAE-Decoder Paradigm](https://aclanthology.org/D19-5522)
 - 2018-Hybrid: [A Hybrid Approach to Automatic Corpus Generation for Chinese Spelling Checking](https://aclanthology.org/D18-1273)

 - 2015-Sighan15: [Introduction to SIGHAN 2015 Bake-off for Chinese Spelling Check](https://aclanthology.org/W15-3106/)
 - 2014-Sighan14: [Overview of SIGHAN 2014 Bake-off for Chinese Spelling Check](https://aclanthology.org/W14-6820/)
 - 2013-Sighan13: [Chinese Spelling Check Evaluation at SIGHAN Bake-off 2013](https://aclanthology.org/W13-4406/)

## 八、参考
 - [nghuyong/Chinese-text-correction-papers](https://github.com/nghuyong/Chinese-text-correction-papers)
 - [destwang/CTCResources](https://github.com/destwang/CTCResources)
 - [wangwang110/CSC](https://github.com/wangwang110/CSC)
 - [shibing624/pycorrector](https://github.com/shibing624/pycorrector)

## 九、数据
### 9.1 csc_punct_zh
 - [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)
 - [chinese-poetry/huajianji](https://github.com/chinese-poetry/huajianji)
 - [garychowcmu/daizhigev20](https://github.com/garychowcmu/daizhigev20)
 - [yangjianxin1/Firefly](https://github.com/yangjianxin1/Firefly)
 - [qwen-7b生成的100万好词好句]
 - [人民日报2000万数据]
 - [学习强国400万数据]

### 9.2 csc_char_zh
 - [chinese-poetry/chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)
 - [chinese-poetry/huajianji](https://github.com/chinese-poetry/huajianji)
 - [garychowcmu/daizhigev20](https://github.com/garychowcmu/daizhigev20)
 - [yangjianxin1/Firefly](https://github.com/yangjianxin1/Firefly)
 - [qwen-7b生成的100万好词好句]
 - [人民日报2000万数据]
 - [学习强国400万数据]

诧异! 发现中文CSC居然没什么综述, 这里概述一下, 希望对你有所帮助!

