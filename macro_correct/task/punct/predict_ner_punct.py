# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/1/18 21:34
# @author  : Mo
# @function: using sl of punct, 序列标注---标点符号


# 适配linux
import logging as logger
import traceback
import platform
import json
import copy
import sys
import os

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(path_root)
if platform.system().lower() == "windows":
    print(path_root)

# os.environ["CUDA_VISIBLE_DEVICES"] = model_config.get("CUDA_VISIBLE_DEVICES", "0")
from macro_correct.pytorch_sequencelabeling.slTools import tradition_to_simple, string_b2q, string_q2b
from macro_correct.pytorch_sequencelabeling.slTools import flag_no_chinese_number_alphabet_signal
from macro_correct.pytorch_sequencelabeling.slTools import transfor_english_symbol_to_chinese
from macro_correct.pytorch_sequencelabeling.slTools import transfor_chinese_symbol_to_english
from macro_correct.pytorch_sequencelabeling.slPredict import SequenceLabelingPredict
from macro_correct.pytorch_sequencelabeling.slTools import PUN_ZH_PAIR
from macro_correct.pytorch_sequencelabeling.slTools import string_q2b
from macro_correct.pytorch_sequencelabeling.slTools import get_logger
from macro_correct.pytorch_sequencelabeling.slTools import load_json
from macro_correct.pytorch_sequencelabeling.slTools import txt_read


class PunctPredict(SequenceLabelingPredict):
    def __init__(self, path_config):
        super(PunctPredict, self).__init__(path_config, logger=logger, flag_load_model=True, CUDA_VISIBLE_DEVICES="0")
        self.idx2pun = load_json(os.path.join(os.path.split(path_config)[0], "idx2pun.json"))
        self.pun2idx = {v: k for k, v in self.idx2pun.items()}
        self.pun_str = "".join(list(set(list("".join(list(self.pun2idx.keys()))))))  ### 所有的标点符号

    def postprocess(self, texts, texts_del=[]):
        """   后处理, 构建生成后的带标点符号的句子   """
        texts_new = []
        for idx, te in enumerate(texts):
            len_punct = 0
            text = te.get("text", "")
            label = te.get("label", [])
            text_org = copy.deepcopy(text)
            label_new = []
            for kdx, label_j in enumerate(label):
                label_j_type = label_j.get("type", "")
                label_j_pos = label_j.get("pos", [0, 0])[0]
                punct_current = self.idx2pun.get(label_j_type, "#")
                pos_currect = label_j_pos + len_punct + 1
                text = text[:pos_currect] + punct_current + text[pos_currect:]
                len_punct += len(punct_current)
                label_j["pun"] = punct_current
                label_new.append(label_j)
            target = self.check_double([text])[0]  #[1:]
            # te["label"] = label_new
            te_new_dict = {"source": "", "target": target[1:],
                           "text": text_org, "label": label_new,
                           "label_org": {}}
            if texts_del:
                label_org = texts_del[idx].get("label_org")
                source = texts_del[idx].get("text_org")
                te_new_dict["label_org"] = label_org
                te_new_dict["source"] = source
            texts_new.append(te_new_dict)
        return texts_new

    def create_punct(self, text):
        """   正常文本转标点符号NER任务需要的数据   """
        symbol_list = []
        symbol_str = ""
        d_new = "#"
        ents = []
        for idx, t in enumerate(text):
            if flag_no_chinese_number_alphabet_signal(t):
                if t in self.pun_str:
                    symbol_str += t
                    if idx == len(text) - 1:  # 最后一个符号是特殊符号
                        if self.pun2idx.get(symbol_str, ""):  # 标点符号 必须 存在已知字典
                            ents_i = {"type": self.pun2idx.get(symbol_str, ""),
                                      "ent": d_new[-1],
                                      "pos": [len(d_new) - 1, len(d_new) - 1],
                                      "pun": symbol_str,
                                      }
                            ents.append(ents_i)
                            symbol_list.append(symbol_str)
                            symbol_str = ""
                        else:
                            symbol_str = ""
                            break
                else:
                    continue
            else:
                if symbol_str:  # 特殊符号
                    if symbol_str not in self.pun2idx:
                        symbol_str = ""
                        continue
                    else:
                        if self.pun2idx.get(symbol_str, ""):
                            ents_i = {"type": self.pun2idx.get(symbol_str, ""),
                                      "ent": d_new[-1],
                                      "pos": [len(d_new) - 1, len(d_new) - 1],
                                      "pun": symbol_str,
                                      }
                            ents.append(ents_i)
                            symbol_list.append(symbol_str)
                        symbol_str = ""
                d_new += t
        line_dict = {"text_org": text, "label_org": ents, "text_new": d_new}
        return line_dict

    def delete_punct(self, texts):
        """   标准化(全角/繁简/英文符号) + 删除已有的标点符号   """
        texts_new = []
        for line in texts:
            text = line.get("text", "")
            text = string_q2b(text)
            text = tradition_to_simple(text)
            text = transfor_english_symbol_to_chinese(text)
            text_org_punct_dict = self.create_punct(text)
            text_new_i = ""
            for tdx, t in enumerate(text):
                if t not in self.pun2idx:
                    text_new_i += t
            text_new_i = "#" + text_new_i
            line["text"] = text_new_i
            line["text_org"] = text_org_punct_dict["text_org"]
            line["label_org"] = text_org_punct_dict["label_org"]
            texts_new.append(line)
            # texts_in.append({"text": text_new_i, "label": []})
        return texts_new

    def check_double(self, texts):
        """   校验成对标点符号, 如果不成对, 都剔除   """
        ### todo, 更精确的删除, 固定到某一对标点符号(如果有多对的话)
        texts_new = []
        for text in texts:
            for pun_zh_pair_i in PUN_ZH_PAIR:
                pun_1 = pun_zh_pair_i[0]
                pun_2 = pun_zh_pair_i[1]
                if text.count(pun_1) != text.count(pun_2):
                    text = text.replace(pun_1, "").replace(pun_2, "")
            texts_new.append(text)
        return texts_new

    def predict_batch(self, texts, batch_size=32, max_len=128, rounded=4, **kwargs):
        """
        推理, 传入 List
        args:
            texts: str, path of ner-corpus
            flag_del: bool, clean text and delete existing characters, eg.True or False.
            flag_post: bool, postprocess is generate the inference text(contain zh symbol), eg.True or False.
        returns:
            res: List<dict>, eg.[{'type': '0', 'ent': '亭', 'pos': [74, 74], 'score': 0.9943}]
        """
        if texts and type(texts[0]) == str:
            texts = [{"text": text} for text in texts]
        ### 删除标点
        texts_del = self.delete_punct(texts)
        text_in = [{"text": t.get("text")} for t in texts_del]
        ### token编码
        dataset = self.process(text_in, max_len=max_len, batch_size=batch_size)
        ### 推理
        res = self.office.predict(dataset, rounded=rounded, **kwargs)
        ### 后处理
        res_post = self.postprocess(res, texts_del)
        return res_post

    def predict(self, text, **kwargs):
        """
        推理, 单个预测
        """
        res_post = self.predict_batch([text], **kwargs)
        return res_post[0] if res_post else []



def tet_eval():
    """   测试验证   """

    from macro_correct.pytorch_sequencelabeling.slTqdm import tqdm
    path_config = "../../output/sequence_labeling/bert4sl_punct_zh_public/sl.config"
    path_tet = os.path.join(path_root, "macro_correct", "corpus", "sequence_labeling", "chinese_symbol", "chinese_symbol.dev.span")
    idx2pun = load_json(os.path.join(os.path.split(path_config)[0], "idx2pun.json"))
    pun2idx = {v: k for k, v in idx2pun.items()}

    tcp = PunctPredict(path_config)
    texts = ["平乐县，古称昭州，隶属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，南接昭平，西北毗邻阳朔，北连恭城，总面积1919.34平方公里。",
         "平乐县主要旅游景点有榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等，平乐县为漓江分界点，平乐以北称漓江，以南称桂江，是著名的大桂林旅游区之一。",
         "桂林山水甲天下,阳朔山水甲桂林",
         "水陆草木之花，可爱者甚蕃。晋陶渊明独爱菊；自李唐来，世人盛爱牡丹；予独爱莲之出淤泥而不染，濯清涟而不妖，中通外直，不蔓不枝，香远益清，亭亭净植，可远观而不可亵玩焉。",
         "山不在高，有仙则名。水不在深，有龙则灵。斯是陋室，惟吾德馨。苔痕上阶绿，草色入帘青。谈笑有鸿儒，往来无白丁。可以调素琴，阅金经。无丝竹之乱耳，无案牍之劳形。南阳诸葛庐，西蜀子云亭。孔子云：何陋之有？",
        "《<123>》12445《<123>》",
        "（《五四（《五四（《五四",
             ]

    res = tcp.predict_batch(copy.deepcopy(texts))
    for idx, r in enumerate(res):
        print(r)
        print(texts[idx])
        print(r.get("predict"))

    data_tet = txt_read(path_tet)
    data_tet_json_i = []
    batch_size = 32
    count_acc = 0
    for d in tqdm(data_tet, desc="data"):
        d_json = json.loads(d.strip())
        # ### 原始的输入剔除“#”
        # if ".span" in path_tet and d_json["text"].startswith("#"):
        #     d_json["text"] = d_json["text"][1:]
        data_tet_json_i.append(d_json)
        if len(data_tet_json_i) == batch_size:
            texts = copy.deepcopy(data_tet_json_i)
            ### 初始的输入不要带"#"开头
            for idx, t in enumerate(texts):
                if ".span" in path_tet and texts[idx]["text"].startswith("#"):
                    texts[idx]["text"] = texts[idx]["text"][1:]
            res = tcp.predict(texts)
            data_tet_json_i = tcp.postprocess(data_tet_json_i)
            for y_true, y_pred in zip(data_tet_json_i, res):
                text_pred = y_pred.get("target", "")
                text_true = y_true.get("target", "")
                if text_pred == text_true:
                    count_acc += 1
                else:
                    print(text_true)
                    print(text_pred)
            data_tet_json_i = []
    print("acc:")
    print(count_acc, len(data_tet))
    print(count_acc / len(data_tet))
    """
        4371 9932
        0.4400926298832058
        
        flag_dynamic_encode==true
        4416 9932
        0.44462343938783727
    """


if __name__ == '__main__':
    yz = 0
    # ### 测试集
    # tet_eval()

    ### predict推理
    # path_config = "../output/sequence_labeling/model_ner_rmrb_ERNIE_lr-5e-05_bs-32_epoch-12/sl.config"
    path_config = "../../output/sequence_labeling/bert4sl_punct_zh_public/sl.config"
    idx2pun = load_json(os.path.join(os.path.split(path_config)[0], "idx2pun.json"))
    pun2idx = {v: k for k, v in idx2pun.items()}

    tcp = PunctPredict(path_config)
    texts = [
        "《<123>》12445《<123>》",
        "（《五四（《五四（《五四",

        "（《红楼梦》）是一部好小说（《红楼梦》）",
        "平乐县，古称昭州，隶属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，南接昭平，西北毗邻阳朔，北连恭城，总面积1919.34平方公里。",
         "平乐县主要旅游景点有榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等，平乐县为漓江分界点，平乐以北称漓江，以南称桂江，是著名的大桂林旅游区之一。",
         "桂林山水甲天下,阳朔山水甲桂林",
         "水陆草木之花，可爱者甚蕃。晋陶渊明独爱菊；自李唐来，世人盛爱牡丹；予独爱莲之出淤泥而不染，濯清涟而不妖，中通外直，不蔓不枝，香远益清，亭亭净植，可远观而不可亵玩焉。",
         "山不在高，有仙则名。水不在深，有龙则灵。斯是陋室，惟吾德馨。苔痕上阶绿，草色入帘青。谈笑有鸿儒，往来无白丁。可以调素琴，阅金经。无丝竹之乱耳，无案牍之劳形。南阳诸葛庐，西蜀子云亭。孔子云：何陋之有？",

        "文中指出了战前的政治准备——取信于民，叙述了利于转入反攻的阵地——长勺，叙述了利于开始反攻的时机——彼竭我盈之时，叙述了追击开始的时机——辙乱旗靡之时。",
        "醉心阅读使我得到了报偿——从小学三年级开始学写作文起，我便常常跃居全班之冠，而阅读也大大扩展了我的想象力。",
        "你的生日——四月十八日——每年我总记得。",
        "她的坚强，她的意志的纯洁，她的律己之严，她的客观，她的公正不阿的判断——所有这一切都难得地集中在一个人身上。",
        "——凡此种种，都可以说某些歌剧中缺乏革命浪漫主义的具体表现。",
        "你买这本吧——这本比那本好。",
        "鲁大海，你现在没有资格跟我说话——矿上已经把你开除了。",
        "灯光，不管是哪个人的家的灯光，都可以给行人——甚至像我这样的一个异乡人——指路。",
        "在时钟的冷冰冰的计时声中——您仔细听听罢——有一种无所不知而又对所知的东西感到厌倦的意味。",
        "他既不关心他的军队，也不喜欢去看戏，也不喜欢乘着马车去游乐园 ——除非是为了去显耀一下他的新衣服。",
        "我这么一直坚持奋发读书，也想借此唤起弟妹们热爱生活的希望——无论环境多么困难。",
        "这一切都像是在提醒这位声学家，不能用任何简单的方式对待一个人——一个有活力、有思想、有感情的人。",
        "三只五只的白鸥轻轻地掠过，翅膀扑着波浪——一点一点躁怒起来的波浪。",
        "画得真好。——你为什么这样勇敢，不怕他？",
        "我偷偷睁眼看了看女医生，见她皱着眉头，脸色很紧张地说：“现在还不能判断，叫她冷静一会儿再说。大家都去学习去，——提壶开水来。",
        "让他一个人留在房里总共不过两分钟，等我们再进去的时候，便发现他在安乐椅上安静地睡着了——但已经永远地睡着了。",
        "在这一刻满屋子人的心都是相同的，都有一样东西，这就是——对死者的纪念。",
        "在几千公里的铁路上，在几百公里的公路上，我从车窗望出去，我的眼睛在到处寻觅——森林。",
        "他纳闷了——究竟是计算的什么方法失误，还是运用不到家？",
        "别看他们闹得这么凶，可是他们是兔子的尾巴——长不了。",
        "赵庄的人们这时都说开了，有的说：“把田村家得罪上来，咱们也没有取上利。阉猪割耳朵——两头受罪。",
        "“可慌哩！比什么也慌，比过新年，娶新——也没有见他这么慌过。”",
        "“可怜的妈妈，”箍桶匠说，“你不知道我多爱你。——还有你，我的儿！”",
        "这时，我忽然记起哪本杂志上的访问记——“哦！您，您就是——”",
        "知道最大的忠孝，是去实现前辈的瞩望，于是，他又去攀登——",
        "别看他们闹得这么凶，可是他们是兔子的尾巴——长不了。",
    ]

    res = tcp.predict_batch(copy.deepcopy(texts))
    for idx, r in enumerate(res):
        print(r)
        print(texts[idx])
        print(r.get("target"))

    while True:
        print("请输入:")
        question = input()
        res = tcp.predict_batch([{"text": question}])
        print(question)
        print(res[0].get("target"))

"""
sl.config中可配置, 如何实际输入 len(texts) > batch_size(32), 则需要使用"flag_dynamic_encode": true,

逻辑为: 
  序列标注用于标点符号纠错, 相比BERT-MLM, SequenceLabeling(SL)能够处理增删问题;
  同时能精调不同的标点符号,比如说逗号/句号等概率能够设置大一点;
北京市（Beijing），简称“京”，古称燕京、北平，是中华人民共和国首都、直辖市、国家中心城市、超大城市， [185]国务院批复确定的中国政治中心、文化中心、国际交往中心、科技创新中心， [1]中国历史文化名城和古都之一，世界一线城市。 [3] [142] [188]截至2023年10月，北京市下辖16个区，总面积16410.54平方千米。 [82] [193] [195]2023年末，北京市常住人口2185.8万人。
"""


