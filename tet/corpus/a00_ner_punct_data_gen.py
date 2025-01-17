# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/2/8 16:37
# @author  : Mo
# @function: 标点符号训练数据生成


from collections import OrderedDict, Counter, defaultdict
from difflib import SequenceMatcher
import logging as logger
import traceback
import random
import copy
import json
import re
import os
import gc

from pypinyin import pinyin
from tqdm import tqdm
import pandas as pd
import pypinyin

pair_symbol = [("“", "”"), ("‘", "’"), ("《", "》"), ("（", "）"),
               ("〈", "〉"), ("〈", "〉"), ("［", "]"), ("【", "】"),
               ("〔", "〕"), ("「", "」"), ("『", "』")]


def check_pair_symbol(text):
    """   判断某些标点符号是否成对   """
    for ps in pair_symbol:
        if text.count(ps[0]) != text.count(ps[1]):
            return False
    return True


pun_dict = {"?": "？", "!":"！",",": "，", ";":"；", ":": "：", "’":"'", "‘":"'", '“':'"', '”':'"'}
pun_1 = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟 〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
pun_2 = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
puns = pun_1 + pun_2
def delete_last_punctuation(text):
    """   -删除句末的标点符号-   """
    while len(text) > 0 and text[-1] in puns:
        text = text[:-1]
    return text
def transfor_english_symbol_to_chinese_v1(text):
    """   替换英文标点符号为中文  """
    text_new = ""
    for t in text:
        if t in pun_dict:
            t_rep = t.replace(t, pun_dict.get(t))
            text_new += t_rep
        else:
            text_new += t
    return text_new


def is_chinese_char(uchar):
    """判断一个unicode是否是汉字"""
    return '\u4e00' <= uchar <= '\u9fa5'


def is_chinese_string(string):
    """判断是否全为汉字"""
    return all(is_chinese_char(c) for c in string)


def is_number(uchar):
    """判断一个unicode是否是数字"""
    return '\u0030' <= uchar <= '\u0039'


def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    return '\u0041' <= uchar <= '\u005a' or '\u0061' <= uchar <= '\u007a'


def is_alphabet_string(string):
    """判断是否全部为英文字母"""
    return all(is_alphabet(c) for c in string)


def is_alphabet_number_string(string):
    """判断全是数字和英文字符"""
    return all((is_alphabet(c) or is_number(c)) for c in string)

def is_number_other(uchar):
    """判断数字符号"""
    return True if uchar in "%.+/" else False

def is_other(uchar):
    """判断是否非汉字，数字和英文字符，其他数字字符"""
    return not (is_chinese_char(uchar) or is_number(uchar) or is_alphabet(uchar) or is_number_other(uchar))



def B2Q(uchar):
    """半角转全角"""
    inside_code = ord(uchar)
    if inside_code < 0x0020 or inside_code > 0x7e:  # 不是半角字符就返回原来的字符
        return uchar
    if inside_code == 0x0020:  # 除了空格其他的全角半角的公式为:半角=全角-0xfee0
        inside_code = 0x3000
    else:
        inside_code += 0xfee0
    return chr(inside_code)


def Q2B(uchar):
    """全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)


def stringQ2B(ustring):
    """把字符串全角转半角"""
    return "".join([Q2B(uchar) for uchar in ustring])
def stringB2Q(ustring):
    """   把字符串半角转全角   """
    return "".join([B2Q(uchar) for uchar in ustring])

def uniform(ustring):
    """格式化字符串，完成全角转半角，大写转小写的工作"""
    return stringQ2B(ustring).lower()


def get_homophones_by_char(input_char):
    """
    根据汉字取同音字
    :param input_char:
    :return:
    """
    result = []
    # CJK统一汉字区的范围是0x4E00-0x9FA5,也就是我们经常提到的20902个汉字
    for i in range(0x4e00, 0x9fa6):
        if pinyin([chr(i)], style=pypinyin.NORMAL)[0][0] == pinyin(input_char, style=pypinyin.NORMAL)[0][0]:
            result.append(chr(i))
    return result
def get_heterotony_by_char(input_char):
    """
    根据汉字取同音字
    :param input_char:
    :return:
    """
    result = []
    # CJK统一汉字区的范围是0x4E00-0x9FA5,也就是我们经常提到的20902个汉字
    for i in range(0x4e00, 0x9fa6):
        if pinyin([chr(i)], style=pypinyin.TONE2)[0][0] == pinyin(input_char, style=pypinyin.TONE2)[0][0]:
            result.append(chr(i))
    return result
def get_homophones_by_pinyin(input_pinyin):
    """
    根据拼音取同音字
    :param input_pinyin:
    :return:
    """
    result = []
    # CJK统一汉字区的范围是0x4E00-0x9FA5,也就是我们经常提到的20902个汉字
    for i in range(0x4e00, 0x9fa6):
        if pinyin([chr(i)], style=pypinyin.TONE2)[0][0] == input_pinyin:
            # TONE2: 中zho1ng
            result.append(chr(i))
    return result


def txt_write(lines, path, mode="w", encode_type="utf-8"):
    """
        txt【list】写入文件
    Args:
        lines[List]: input data to write, eg. ["桂林山水甲天下\\n"]
        path[String]: path of file of read, eg. "corpus/xuexiqiangguo.txt"
        mode[String]: write mode of file, eg. "w", "a+", "wb"
        encode_type[String]: data encode type of file, eg. "utf-8", "gbk"
    Returns:
        lines[List]: output lines
    """
    try:
        file = open(path, mode, encoding=encode_type)
        file.writelines(lines)
        file.close()
    except Exception as e:
        logger.info(str(e))
def txt_read(path, encode_type="utf-8", errors=None):
    """
        读取txt文件，默认utf8格式, 不能有空行
    Args:
        path[String]: path of file of read, eg. "corpus/xuexiqiangguo.txt"
        encode_type[String]: data encode type of file, eg. "utf-8", "gbk"
        errors[String]: specifies how encoding errors handled, eg. "ignore", strict
    Returns:
        lines[List]: output lines
    """
    lines = []
    try:
        file = open(path, "r", encoding=encode_type, errors=errors)
        lines = file.readlines()
        file.close()
    except Exception as e:
        logger.info(str(e))
    finally:
        return lines
def save_xlsx(path_json, path_xlsx="nkpmath.xlsx"):
    """   json转化为xlsx的excel文件   """
    kg_list = load_json(path_json)
    res_xlsx = {}
    for kg_i in kg_list:
        for jdx, kg_i_j in enumerate(kg_i):
            jdx_str = str(jdx)
            if jdx_str in res_xlsx:
                res_xlsx[jdx_str].append(kg_i_j)
            else:
                res_xlsx[jdx_str] = [kg_i_j]
    pdr = pd.DataFrame(res_xlsx)
    with pd.ExcelWriter(path_xlsx, engine="xlsxwriter",
            options={"strings_to_urls": False}) as writer:
        pdr.to_excel(writer)
def save_json(jsons, json_path, indent=4):
    """
        保存json
    Args:
        path[String]:, path of file of save, eg. "corpus/xuexiqiangguo.lib"
        jsons[Json]: json of input data, eg. [{"桂林": 132}]
        indent[int]: pretty-printed with that indent level, eg. 4
    Returns:
        None
    """
    with open(json_path, "w", encoding="utf-8") as fj:
        fj.write(json.dumps(jsons, ensure_ascii=False, indent=indent))
    fj.close()
def load_json(path, parse_int=None):
    """
        加载json
    Args:
        path_file[String]:, path of file of save, eg. "corpus/xuexiqiangguo.lib"
        parse_int[Boolean]: equivalent to int(num_str), eg. True or False
    Returns:
        data[Any]
    """
    with open(path, mode="r", encoding="utf-8") as fj:
        model_json = json.load(fj, parse_int=parse_int)
    return model_json
def get_all_dirs_files(path_dir):
    """
        递归获取某个目录下的所有文件(所有层, 包括子目录)
    Args:
        path_dir[String]:, path of dir, eg. "/home/data"
    Returns:
        data[List]: data of input, eg. ["2020_01_08.txt"]
    """
    path_files = []
    for root, dirs, files in os.walk(path_dir):  # 分别代表根目录、文件夹、文件
        for file in files:  # 遍历文件
            file_path = os.path.join(root, file)  # 获取文件绝对路径
            path_files.append(file_path)  # 将文件路径添加进列表
    files = list(set(path_files))
    files.sort()
    return files
def find_diff_pos(sent1, sent2):
    """
    判断两个病句的不同之处, 返回insert/delete/replace, difflib-SequenceMatcher
    args:
        sent1: str, sentence of org, eg. "春天来了，越来越来暖和了。"
        sent2: str, sentence of fix, eg. "春天来了，天气越来越暖和了。"
    return:
        diff_pos_s: List<Tuple>, tag and position, eg. ""
    """
    matcher = SequenceMatcher(None, sent1, sent2)
    diff_pos_s = []
    for tag, idx_1, idx_2, jdx_1, jdx_2 in matcher.get_opcodes():
        if tag != "equal":
            line_tuple = [tag, sent1[idx_1:idx_2],
                          sent2[jdx_1: jdx_2], [idx_1, idx_2]]
            diff_pos_s.append(line_tuple)
    return diff_pos_s
def cut_sentence(text):
    """  分句(文本摘要)  """
    # re_sen = re.compile('[:;!?。：；？！\n\r]') #.不加是因为不确定.是小数还是英文句号(中文省略号......)
    # re_sen = re.compile('[!?。？！\n\r]')
    # re_sen = re.compile('[,，"“”、<>《》{}【】:;!?。：；？！\n\r]') #.不加是因为不确定.是小数还是英文句号(中文省略号......)
    re_sen = re.compile('[!?。？！\n\r…]')
    sentences = re_sen.split(text)
    return sentences


pun_1 = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟 〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
pun_2 = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
puns = pun_1 + pun_2
def delete_last_punctuation(text):
    """   -删除句末的标点符号-   """
    while len(text) > 0 and text[-1] in puns:
        text = text[:-1]
    return text
import opencc
converter = opencc.OpenCC('t2s.json')
# converter = opencc.OpenCC('tw2sp.json')
# converter = opencc.OpenCC('tw2s.json')
# converter = opencc.OpenCC('hk2s.json')
context = converter.convert('楽趣')  # 漢字, 楽---日本新体字
print(context)
def tradition_to_simple(text):
    """  台湾繁体到大陆简体  """
    return converter.convert(text)



PUN_EN2ZH_DICT = {",": "，", ";": "；", "!": "！", "?": "？", ":": "：",
                  "(": "（", ")": "）", "_": "—"}
def transfor_english_symbol_to_chinese(text, kv_dict=PUN_EN2ZH_DICT):
    """   将英文标点符号转化为中文标点符号, 位数不能变防止pos_id变化   """
    for k, v in kv_dict.items():  # 英文替换
        text = text.replace(k, v)
    if text and text[-1] == ".":   # 最后一个字符是英文.
        text = text[:-1] + "。"

    if text and "\"" in text:   # 双引号
        index_list = [i.start() for i in re.finditer("\"", text)]
        if index_list:
            for idx, index in enumerate(index_list):
                symbol = "“" if idx % 2 == 0 else "”"
                text = text[:index] + symbol + text[index + 1:]

    if text and "'" in text:   # 单引号
        index_list = [i.start() for i in re.finditer("'", text)]
        if index_list:
            for idx, index in enumerate(index_list):
                symbol = "‘" if idx % 2 == 0 else "’"
                text = text[:index] + symbol + text[index + 1:]
    return text
def enhance_csc_punct_from_renmingribao_by_punct_dict_v1(LIMIT=100000):
    """    从人民日报进行数据增强, 字形类数据, 所有的混淆词典, 取3500常用字容易混淆的
    """
    idx2pun_dict = load_json("idx2pun.json")
    pun2idx_dict = {v:k for k, v in idx2pun_dict.items()}

    files = get_all_dirs_files("sentence_people_daily_news")
    files = [file for file in files if "_punct_enhance" not in file and "json_enhance" not in file]

    enhance_prior_dict = {k: [] for k, v in pun2idx_dict.items()}

    for file in files[::-1]:
        data_sents = load_json(file)
        for sent in tqdm(data_sents, desc=file):
            try:
                sent = stringQ2B(sent)   # 全角转半角
            except Exception as e:
                print(traceback.print_exc())
                continue

            if " " not in sent and "　" not in sent and " " not in sent:
                if len(sent) < 4:
                    continue
                sent = transfor_english_symbol_to_chinese(sent)
                if "“" in sent and "”" not in sent:
                    continue
                if "“" not in sent and "”" in sent:
                    continue
                if "‘" in sent and "’" not in sent:
                    continue
                if "‘" not in sent and "’" in sent:
                    continue
                if "《" in sent and "》" not in sent:
                    continue
                if "《" not in sent and "》" in sent:
                    continue
                if "（" in sent and "）" not in sent:
                    continue
                if "（" not in sent and "）" in sent:
                    continue
                if "~~" in sent or sent.endswith("~") or "、？" in sent:
                    continue
                # "穆尔希德·汗外长强调",
                if is_chinese_char(sent[-1]):
                    continue
                if sent.startswith("——") or sent.endswith("——")\
                        or sent.endswith("）") or sent.startswith("（") \
                        or sent.endswith("：") or sent.endswith("：“…"):
                    continue
                if sent.startswith("▲") or sent.startswith("(") \
                        or sent.startswith(")") or sent.startswith("”，") \
                        or sent.startswith("”") or sent.startswith("’"):
                    continue
                if "”“" in sent or "说，" in sent or "表示，" in sent:
                    continue
                # ”我国思想理论界也有一句名言：“一具体就深刻”。
                if sent.count("“") != sent.count("”"):
                    continue
                if sent.count("‘") != sent.count("’"):
                    continue
                if sent.count("《") != sent.count("》"):
                    continue
                if sent.count("（") != sent.count("）"):
                    continue
                if sent.count("(") != sent.count(")"):
                    continue
                for k, v in pun2idx_dict.items():
                    if k in sent and len(enhance_prior_dict.get(k)) < LIMIT:
                        enhance_prior_dict[k].append(sent)
                        break
        for k, v in enhance_prior_dict.items():
            print((k, len(v)))
        save_json(enhance_prior_dict, file + "_punct_enhance.json")
    ee = 0
"""   
('，', 100000)
('。', 100000)
('、', 100000)
('；', 100000)
('？', 100000)
('！', 100000)
('：', 100000)
('“', 100000)
('”', 100000)
('‘', 56690)
('’', 0)
('—', 100000)
('…', 100000)
('·', 100000)
('~', 365)
('《', 100000)
('》', 100000)
('（', 100000)
('）', 100000)
('<', 54)
('>', 223)
('：“', 82424)
('。”', 0)
('！”', 0)
('？”', 0)
('，“', 100000)
('”，', 100000)
('”。', 100000)
('”！', 5336)
('”？', 11052)
('……', 0)
('——', 100000)
('》。', 37769)
('”“', 0)
('》，', 70616)
('，《', 18741)
('”、', 100000)
('、“', 42982)
('……”', 0)
('》《', 1161)
('”；', 18858)
('”：', 8791)
('”、“', 0)
('”，“', 22566)
('”——', 1797)
('“‘', 0)
('’”', 0)
('，‘', 0)
('；“', 3893)
('》、《', 21571)
('”…', 892)
('》、', 2848)
('》…', 105)
('），', 58017)
('）。', 37558)
('）、', 10931)
('——“', 1918)
('”（', 6445)
"""
def select_enhance_csc_punct_from_renmingribao_by_punct_dict_v1(LIMLT=20000):
    """   获取数据, 不超过50w   """
    # path = "sentence_people_daily_news/sentence_people_daily_news.all.json_punct_enhance.json"
    path = "sentence_people_daily_news/sentence_people_daily_news.all.json_punct_enhance.json.v2"
    data_dict = load_json(path)

    data = []
    for k, v in data_dict.items():
        if len(v) >= 50000:
            vi = v[:int(LIMLT/2)]
        else:
            vi = v[:LIMLT]
        data.extend(vi)
    data = list(set(data))
    print(len(data))
    # save_json(data, "sl_v2_punct_label_2w.limit")
    save_json(data, "sl_v2_punct_label_5w.limit")
    """  466329  """
def select_enhance_csc_punct_from_renmingribao_by_punct_dict_v2(LIMLT=20000):
    """   获取数据, 不超过50w   """
    # path = "sentence_people_daily_news/sentence_people_daily_news.all.json_punct_enhance.json"
    path = "sentence_people_daily_news/sentence_people_daily_news.all.json_punct_enhance.json.v2"
    data_dict = load_json(path)
    start_list = ["”“", "（", "）", "：", "●", "△", "【", "”（", "》",
                  "3.", "37、", "◆", "”（", "”", "’", "·"]
    end_list = ["）", "》", ">", "“", "”", "‘", "’", ]
    mid_list = [("‘", "“"), ("‘", "”"), ("<", "《"), (">", "》")]
    # end:）》>
    # ‘ in not ”“


    data = []
    data_dict_1 = {}
    data_dict_2 = {}
    data_dict_3 = {}
    data_dict_4 = {}
    data_dict_5 = {}
    for k, v_org in data_dict.items():
        v = []
        for vi in v_org:
            flag = True
            for s in start_list:
                if vi.startswith(s):
                    flag = False
                    break
            for e in end_list:
                if vi.endswith(e):
                    flag = False
                    break
            for m in mid_list:
                if m[0] in vi and m[1] not in vi:
                    flag = False
                    break
            if flag:
                v.append(vi)

        # 2000; 0,1,2,7,8
        # if k in ["，", "。", "、", "“", "”"]:
        #     data_dict_1[k] = v
        # ## 500; 15, 16, 17, 18, "26","27","49"
        # elif k in ["《", "》", "（", "）", "”，","”。","》、《"]:
        #     data_dict_2[k] = v
        # # 200; 25,32,35,40,43,55
        # elif k in ["，“", "，《", "）、", "”，“", "》。", "”；"]:
        #     data_dict_3[k] = v
        # # 3,4,5,6, 9,10
        # elif k in ["；", "？", "！", "：", "‘", "’"]:
        #     data_dict_4[k] = v
        # else:
        #     data_dict_5[k] = v
        if "（" in k or "）" in k:
            data += v[-1000:]
        else:
            print("#" * 128)
            print(k)
            print(len(v))
            v = [vi for vi in v if "（" not in vi and "）" not in vi]
            print(len(v))

        if k in ["，", "。", "、", "“", "”"]:
            # data += v[-LIMLT:]
            continue
        ## 500; 15, 16, 17, 18, "26","27","49"
        elif k in ["《", "》", "（", "）", "”，","”。","》、《"]:
            if len(v) > LIMLT:
                data += v[-LIMLT:]
            else:
                data += v
        # 200; 25,32,35,40,43,55
        elif k in ["，“", "，《", "）、", "”，“", "》。", "”；"]:
            if len(v) > LIMLT:
                data += v[-LIMLT:]
            else:
                data += v
        # 3,4,5,6, 9,10
        elif k in ["；", "？", "！", "：", "‘", "’"]:
            if len(v) > LIMLT:
                data += v[-LIMLT:]
            else:
                data += v
        else:
            if len(v) > LIMLT:
                data += v[-LIMLT:]
            else:
                data += v
        # if len(v) >= 50000:
        #     vi = v[:int(LIMLT/2)]
        # else:
        #     vi = v[:LIMLT]
        # data.extend(vi)
    data = list(set(data))
    print(len(data))
    # save_json(data, "sl_v2_punct_label_2w.limit")
    # save_json(data, "sl_v2_punct_label_5w.limit_1w.v2")
    save_json(data, "sl_v2_punct_label_5w.limit_1w_filter.v2")
    """  323059  """


def collect_none_chinese(text):
    """   返回非CJK的字符集   """
    symbol_list = []
    symbol_str = ""
    for t in text:
        # if not is_chinese_char(t):
        if is_other(t):
            symbol_str += t
        else:
            if symbol_str:
                symbol_list.append(symbol_str)
                symbol_str = ""
    return symbol_list
def collect_pun_of_none_chinese(text):
    """   统计非中文字符  """
    text = stringB2Q(text)
    symbols = collect_none_chinese(text)
    return symbols
def a00_counter_none_chinese_of_all_quality_dataset_step1():
    """   统计非中文   """
    path_dir = "a00_quality_sent\\rmrb_2000w"
    files = get_all_dirs_files(path_dir)

    path_huanjianci_14k = "a00_quality_sent\\huajianci_poetry_14k\\v2_pun_huajianji.json.ppl.select"
    path_good_100w = "a00_quality_sent\\qwen_gen_passage_by_log_100w\\good_sents.jsonsplitsent_and_duplite.json.ppl.select"
    path_good_4k = "a00_quality_sent\\qwen_gen_passage_by_order_4k\\good_sents_order.jsonsplitsent_and_duplite.json.ppl.select"
    path_good_12w = "a00_quality_sent\\qwen_gen_sent_by_log_12w\\good_list_create_train.json.ppl.select"
    path_poetry_140w = "a00_quality_sent\\chinese_poetry_140w\\chinese_poetry.json.ppl.select"
    path_wenyanwen_82w = "\a00_quality_sent\\wenyanwen_nopun_yinhao_82w\\v2_wenyanwen_nopun.json.ppl.select"
    path_irefly_train_1_1m_378w = "a00_quality_sent\\firefly_train_1_1m_378w\\firefly-train-1.1M.jsonl.json.ppl.select"
    path_sft_alpaca_gpt4_data_zh_14w = "a00_quality_sent\\sft_alpaca_gpt4_data_zh_14w/alpaca_gpt4_data_zh.json.ppl.select"
    paths = [path_huanjianci_14k,
             path_good_100w,
             path_good_4k,
             path_good_12w,
             path_poetry_140w,
             path_wenyanwen_82w,
             path_irefly_train_1_1m_378w,
             path_sft_alpaca_gpt4_data_zh_14w,
             ] + files
    punct_dict = {}
    for path in tqdm(paths, desc="datas-1"):
        punct_list = []
        datas_i = load_json(path)
        datas_i = list(set(datas_i))
        file_name = os.path.split(path)[-1]
        try:
            for d in tqdm(datas_i, desc=file_name):
                d_symbols = collect_pun_of_none_chinese(d)
                punct_list.extend(d_symbols)
        except Exception as e:
            print(traceback.print_exc())
        punct_count_dict = dict(Counter(punct_list))
        punct_dict[path] = punct_count_dict

    path = "a00_quality_sent\\xuexiqiangguo_400w\\xuexiqiangguo_ppl_select.json"
    datas_dict = load_json(path)
    datas_i = []
    for k, v in datas_dict.items():
        datas_i.extend(v)
    datas_i = list(set(datas_i))
    file_name = os.path.split(path)[-1]
    punct_list = []
    for d in tqdm(datas_i, desc=file_name):
        d_symbols = collect_pun_of_none_chinese(d)
        punct_list.extend(d_symbols)
    punct_count_dict = dict(Counter(punct_list))
    punct_dict[path] = punct_count_dict


    path = "a00_quality_sent\\a01_lomo_lcsts_ime_2m_3w\\lomo_lcsts-ime-2m.2m.json"
    datas_i = load_json(path)
    datas_i = [d.get("correct_text") for d in datas_i]
    datas_i = list(set(datas_i))
    file_name = os.path.split(path)[-1]
    punct_list = []
    for d in tqdm(datas_i, desc=file_name):
        d_symbols = collect_pun_of_none_chinese(d)
        punct_list.extend(d_symbols)
    punct_count_dict = dict(Counter(punct_list))
    punct_dict[path] = punct_count_dict

    path = "a00_quality_sent\\a02_wang271k_20w\\train.json.handle_clean.ppl.select"
    datas_i = load_json(path)
    datas_i = [d.get("correct_text") for d in datas_i]
    datas_i = list(set(datas_i))
    file_name = os.path.split(path)[-1]
    punct_list = []
    for d in tqdm(datas_i, desc=file_name):
        d_symbols = collect_pun_of_none_chinese(d)
        punct_list.extend(d_symbols)
    punct_count_dict = dict(Counter(punct_list))
    punct_dict[path] = punct_count_dict


    save_json(punct_dict, "a00_punct/punct_dict.dict")
def a01_counter_none_chinese_of_all_quality_dataset_step2():
    """   统计整理标点符号, 高频数据   """
    str_dig = "0123456789abcdefghijklmnopqrstuvwxyz<>《》"
    str_dig_b = stringB2Q(str_dig) + stringB2Q(str_dig.upper())
    char_list = list(str_dig_b) + ["\n", "\t", "\r"]
    char_list = list(set(char_list))
    print(char_list)
    path = "a00_punct/punct_dict.dict"
    punct_dict = load_json(path)
    # key_all = []
    # for k, v in punct_dict.items():
    #     v_list = list(v.keys())
    #     key_all.extend(v_list)
    # key_all_set = list(set(key_all))
    pun_counter_dict = {"none": 0}
    for k, v in punct_dict.items():
        for vk, vv in v.items():
            for kkk in char_list:
                vk = vk.replace(kkk, "")
            if vk:
                if vk not in pun_counter_dict:
                    pun_counter_dict[vk] = vv
                else:
                    pun_counter_dict[vk] += vv
            else:
                pun_counter_dict["none"] += vv
    pun_counter_dict_sort = dict(sorted(pun_counter_dict.items(), key=lambda x:x[-1], reverse=True))
    save_json(pun_counter_dict_sort, "a00_punct/pun_counter_dict_sort.json")


def enhance_csc_punct_from_quality_dataset_by_punct_dict_wenyanwen_v2(LIMIT=100000):
    """    从高质量语料中获取标点符号标注数据,
    文言文, 140w, 全选不计算了
    """
    path = "D:/workspace/code_own/idx2pun.v2.json"
    idx2pun_dict = load_json(path)
    pun2idx_dict = {v: k for k, v in idx2pun_dict.items()}
    path_dir = "./a00_quality_sent/"
    path_1 = path_dir + "chinese_poetry_140w" + "/chinese_poetry.json.ppl.select"
    data_1 = load_json(path_1)
    random.shuffle(data_1)
    v_list = data_1
    category = "wenyanwen"

    enhance_prior_dict = {k: [] for k, v in pun2idx_dict.items()}
    for _ in range(1):
        # for file in files:
        #     data_sents = load_json(file)
        #     for sent in tqdm(data_sents, desc=file):
        for sent in tqdm(v_list, desc="v_list"):
            try:
                # sent = stringQ2B(sent)   # 全角转半角
                sent = stringB2Q(sent)  # 半角转全角
            except Exception as e:
                print(traceback.print_exc())
                continue

            if " " not in sent and "　" not in sent and " " not in sent:
                if len(sent) < 4:
                    continue
                sent = transfor_english_symbol_to_chinese(sent)
                if "“" in sent and "”" not in sent:
                    continue
                if "“" not in sent and "”" in sent:
                    continue
                if "‘" in sent and "’" not in sent:
                    continue
                if "‘" not in sent and "’" in sent:
                    continue
                if "《" in sent and "》" not in sent:
                    continue
                if "《" not in sent and "》" in sent:
                    continue
                if "（" in sent and "）" not in sent:
                    continue
                if "（" not in sent and "）" in sent:
                    continue
                if "~~" in sent or sent.endswith("~") or "、？" in sent:
                    continue
                # "穆尔希德·汗外长强调",
                if is_chinese_char(sent[-1]):
                    continue
                if sent.startswith("——") or sent.endswith("——") \
                        or sent.endswith("）") or sent.startswith("（") \
                        or sent.endswith("：") or sent.endswith("：“…"):
                    continue
                if sent.startswith("▲") or sent.startswith("(") \
                        or sent.startswith(")") or sent.startswith("”，") \
                        or sent.startswith("”") or sent.startswith("’"):
                    continue
                if "”“" in sent or "说，" in sent or "表示，" in sent:
                    continue
                # ”我国思想理论界也有一句名言：“一具体就深刻”。
                if sent.count("“") != sent.count("”"):
                    continue
                if sent.count("‘") != sent.count("’"):
                    continue
                if sent.count("《") != sent.count("》"):
                    continue
                if sent.count("（") != sent.count("）"):
                    continue
                if sent.count("(") != sent.count(")"):
                    continue
                for k, v in pun2idx_dict.items():
                    if k in sent and len(enhance_prior_dict.get(k)) < LIMIT:
                        enhance_prior_dict[k].append(sent)
                        break
        for k, v in enhance_prior_dict.items():
            print((k, len(v)))
        save_json(enhance_prior_dict, "a00_punct/punct_enhance_{}.json".format(category))
    """
    v_list: 100%|██████████| 4283494/4283494 [01:40<00:00, 42587.05it/s]
('，', 100000)
('。', 100000)
('、', 100000)
('；', 100000)
('？', 36394)
('！', 17017)
('：', 100000)
('“', 100000)
('”', 100000)
('‘', 22736)
('’', 0)
('—', 11466)
('…', 25691)
('·', 10843)
('~', 0)
('《', 83800)
('》', 0)
('（', 47047)
('）', 0)
('＜', 3)
('＞', 37)
('：“', 792)
('。”', 5601)
('！”', 0)
('？”', 0)
('，“', 33871)
('”，', 81407)
('”。', 55341)
('”！', 0)
('”？', 0)
('……', 0)
('——', 0)
('》。', 0)
('”“', 0)
('》，', 0)
('，《', 0)
('”、', 5706)
('、“', 4462)
('……”', 0)
('》《', 0)
('”；', 295)
('”：', 19)
('”、“', 0)
('”，“', 0)
('”——', 0)
('“‘', 0)
('’”', 0)
('，‘', 0)
('；“', 53)
('》、《', 0)
('”…', 0)
('》、', 0)
('》…', 0)
('），', 0)
('）。', 0)
('）、', 0)
('——“', 0)
('”（', 0)
('》（', 0)
('～', 2340)
('：《', 0)
('”）', 0)
('：（', 0)
('）》', 0)
('「', 0)
('」', 0)
('）：', 0)
('－－', 19)
('）》，', 0)
('》），', 0)
('）；', 0)
('，（', 0)
('；（', 0)
('《“', 0)
('》）', 0)
('”；“', 0)
('〈', 1)
('〉', 0)
('》；', 0)
('：“‘', 0)
('）“', 0)
('）”', 0)
('》：', 0)
('”：“', 0)
('［', 38)
('］', 0)
('—”', 0)
('；《', 0)
('）》（', 0)
('’、', 0)
('》，《', 0)
('（“', 0)
('’、‘', 0)
('”（《', 0)
('，“‘', 0)
('”》', 0)
('〔', 31)
('〕', 0)
('）、“', 0)
('、‘', 0)
('：‘', 0)
('（《', 0)
('？“', 0)
('」，', 0)
('《〈', 0)
('》，“', 0)
('。“', 0)
('！“', 0)
('：「', 0)
('」）', 0)
('：“《', 0)
('“《', 0)
('）（', 0)
('『', 0)
('』', 0)
('“（', 0)
('）、《', 0)
('），“', 0)
('）》《', 0)
('、（', 0)
    """

    ee = 0

def enhance_csc_punct_from_quality_dataset_by_punct_dict_rmrb_v4(LIMIT=100000):
    """    从高质量语料中获取标点符号标注数据
    文言文, 140w, 全选不计算了
    """
    path = "D:/workspace/code_own/a00_punct/idx2pun.v2.json"
    idx2pun_dict = load_json(path)
    pun2idx_dict = {v: k for k, v in idx2pun_dict.items()}

    path_dir = "D:\\workspace\\code_own\\sentence_people_daily_news"
    files = get_all_dirs_files(path_dir)
    files = [file for file in files if file.endswith(".json.ppl.select")]
    print(files)
    category = "rmrb"

    enhance_prior_dict = {k: [] for k, v in pun2idx_dict.items()}
    for file in tqdm(files[::-1], desc="files"):
        print(file)
        file_name = os.path.split(file)[-1]
        data_sents = load_json(file)
        random.shuffle(data_sents)
        random.shuffle(data_sents)
        random.shuffle(data_sents)
        for sent in tqdm(data_sents, desc=file_name):

            sent = str(sent)
            sent = stringQ2B(sent)  # 半角
            sent = transfor_english_symbol_to_chinese(sent)
            sent = tradition_to_simple(sent)
            if not check_pair_symbol(sent):
                continue
        # for sent in tqdm(v_list, desc="v_list"):
        #     try:
        #         # sent = stringQ2B(sent)   # 全角转半角
        #         sent = stringB2Q(sent)  # 半角转全角
        #     except Exception as e:
        #         print(traceback.print_exc())
        #         continue

            if " " not in sent and "　" not in sent and " " not in sent:
                if len(sent) < 4:
                    continue
                sent = transfor_english_symbol_to_chinese(sent)
                if "“" in sent and "”" not in sent:
                    continue
                if "“" not in sent and "”" in sent:
                    continue
                if "‘" in sent and "’" not in sent:
                    continue
                if "‘" not in sent and "’" in sent:
                    continue
                if "《" in sent and "》" not in sent:
                    continue
                if "《" not in sent and "》" in sent:
                    continue
                if "（" in sent and "）" not in sent:
                    continue
                if "（" not in sent and "）" in sent:
                    continue
                if "~~" in sent or sent.endswith("~") or "、？" in sent:
                    continue
                # "穆尔希德·汗外长强调",
                if is_chinese_char(sent[-1]):
                    continue
                if sent.startswith("——") or sent.endswith("——") \
                        or sent.endswith("）") or sent.startswith("（") \
                        or sent.endswith("：") or sent.endswith("：“…"):
                    continue
                if sent.startswith("▲") or sent.startswith("(") \
                        or sent.startswith(")") or sent.startswith("”，") \
                        or sent.startswith("”") or sent.startswith("’"):
                    continue
                if "”“" in sent or "说，" in sent or "表示，" in sent:
                    continue
                # ”我国思想理论界也有一句名言：“一具体就深刻”。
                if sent.count("“") != sent.count("”"):
                    continue
                if sent.count("‘") != sent.count("’"):
                    continue
                if sent.count("《") != sent.count("》"):
                    continue
                if sent.count("（") != sent.count("）"):
                    continue
                if sent.count("(") != sent.count(")"):
                    continue
                for k, v in pun2idx_dict.items():
                    if k in sent and len(enhance_prior_dict.get(k)) < LIMIT:
                        enhance_prior_dict[k].append(sent)
                        break
        for k, v in enhance_prior_dict.items():
            print((k, len(v)))
        # save_json(enhance_prior_dict, file + "_punct_enhance.json")
        # save_json(enhance_prior_dict, "xuexiqiangguo_punct_enhance.json")
    save_json(enhance_prior_dict, "./punct_enhance_{}.json".format(category))
    ee = 0
    """
('，', 100000)
('。', 100000)
('、', 100000)
('；', 100000)
('？', 100000)
('！', 100000)
('：', 100000)
('“', 100000)
('”', 100000)
('‘', 40193)
('’', 0)
('—', 100000)
('…', 100000)
('·', 100000)
('~', 207)
('《', 100000)
('》', 100000)
('（', 100000)
('）', 100000)
('＜', 0)
('＞', 0)
('：“', 100000)
('。”', 100000)
('！”', 76610)
('？”', 30139)
('，“', 100000)
('”，', 100000)
('”。', 100000)
('”！', 4069)
('”？', 7238)
('……', 16434)
('——', 100000)
('》。', 21590)
('”“', 0)
('》，', 40644)
('，《', 11405)
('”、', 93516)
('、“', 23577)
('……”', 0)
('》《', 665)
('”；', 14963)
('”：', 7018)
('”、“', 0)
('”，“', 20693)
('”——', 176)
('“‘', 0)
('’”', 0)
('，‘', 0)
('；“', 1855)
('》、《', 12138)
('”…', 1)
('》、', 1508)
('》…', 0)
('），', 25988)
('）。', 16198)
('）、', 3430)
('——“', 199)
('”（', 2735)
('》（', 357)
('～', 0)
('：《', 661)
('”）', 385)
('：（', 5048)
('）》', 31)
('「', 0)
('」', 0)
('）：', 419)
('－－', 0)
('）》，', 0)
('》），', 0)
('）；', 875)
('，（', 534)
('；（', 192)
('《“', 302)
('》）', 31)
('”；“', 0)
('〈', 301)
('〉', 0)
('》；', 404)
('：“‘', 0)
('）“', 102)
('）”', 457)
('》：', 59)
('”：“', 0)
('［', 0)
('］', 0)
('—”', 6)
('；《', 163)
('）》（', 0)
('’、', 0)
('》，《', 0)
('（“', 49)
('’、‘', 0)
('”（《', 0)
('，“‘', 0)
('”》', 163)
('〔', 739)
('〕', 0)
('）、“', 0)
('、‘', 0)
('：‘', 0)
('（《', 9)
('？“', 0)
('」，', 0)
('《〈', 0)
('》，“', 0)
('。“', 0)
('！“', 0)
('：「', 0)
('」）', 0)
('：“《', 90)
('“《', 108)
('）（', 22)
('『', 0)
('』', 0)
('“（', 181)
('）、《', 0)
('），“', 0)
('）》《', 0)
('、（', 48)
('’”。', 0)
('！”。', 0)
('”……', 0)
('’！”', 0)
('’。”', 0)
('？？', 143)
('？？？', 0)
('！！', 50)
('！！！', 0)
('！？', 296)
('？！', 1959)
('？——', 0)
('？”。', 0)
('——？', 1)
('——!', 0)
('/', 29134)
('-', 1413)
files: 100%|██████████| 8/8 [28:48<00:00, 216.03s/it]
2702846

    """
def select_enhance_csc_punct_from_renmingribao_by_punct_dict_v4(LIMLT=30000):
    """   获取数据, 不超过50w   """
    path = "a00_good_essay_qwen_7b_all_clean_v2_for_punct/punct_enhance_rmrb.json"

    data_dict = load_json(path)
    start_list = ["”“", "（", "）", "：", "●", "△", "【", "”（", "》",
                  "3.", "37、", "◆", "”（", "”", "’", "·"]
    end_list =["）", "》", ">", "“", "‘", "’", ]  #  ["）", "》", ">", "“", "”", "‘", "’", ]
    mid_list = [("‘", "“"), ("‘", "”"), ("<", "《"), (">", "》")]
    # end:）》>
    # ‘ in not ”“

    data = []
    data_dict_1 = {}
    data_dict_2 = {}
    data_dict_3 = {}
    data_dict_4 = {}
    data_dict_5 = {}
    for k, v_org in data_dict.items():
        v = []
        for vi in v_org:
            flag = True
            for s in start_list:
                if vi.startswith(s):
                    flag = False
                    break
            for e in end_list:
                if vi.endswith(e):
                    flag = False
                    break
            for m in mid_list:
                if m[0] in vi and m[1] not in vi:
                    flag = False
                    break
            if flag:
                v.append(vi)
        # v = v[:50000]
        random.shuffle(v)
        random.shuffle(v)
        random.shuffle(v)

        # 2000; 0,1,2,7,8
        # if k in ["，", "。", "、", "“", "”"]:
        #     data_dict_1[k] = v
        # ## 500; 15, 16, 17, 18, "26","27","49"
        # elif k in ["《", "》", "（", "）", "”，","”。","》、《"]:
        #     data_dict_2[k] = v
        # # 200; 25,32,35,40,43,55
        # elif k in ["，“", "，《", "）、", "”，“", "》。", "”；"]:
        #     data_dict_3[k] = v
        # # 3,4,5,6, 9,10
        # elif k in ["；", "？", "！", "：", "‘", "’"]:
        #     data_dict_4[k] = v
        # else:
        #     data_dict_5[k] = v
        # if "（" in k or "）" in k:
        #     data += v[-1000:]
        # else:
        #     print("#" * 128)
        #     print(k)
        #     print(len(v))
        #     v = [vi for vi in v if "（" not in vi and "）" not in vi]
        #     print(len(v))



        # if k in ["，", "。", "、", "“", "”"]:
        #     data += v[-LIMLT:]
        #     continue
        # ## 500; 15, 16, 17, 18, "26","27","49"
        # elif k in ["《", "》", "（", "）", "”，","”。","》、《"]:
        #     if len(v) > LIMLT:
        #         data += v[-LIMLT:]
        #     else:
        #         data += v
        # # 200; 25,32,35,40,43,55
        # elif k in ["，“", "，《", "）、", "”，“", "》。", "”；"]:
        #     if len(v) > LIMLT:
        #         data += v[-LIMLT:]
        #     else:
        #         data += v
        # # 3,4,5,6, 9,10
        # elif k in ["；", "？", "！", "：", "‘", "’"]:
        #     if len(v) > LIMLT:
        #         data += v[-LIMLT:]
        #     else:
        #         data += v
        # else:
        #     if len(v) > LIMLT:
        #         data += v[-LIMLT:]
        #     else:
        #         data += v



        # if len(v) >= 50000:
        #     vi = v[:int(LIMLT/2)]
        # else:
        #     vi = v[:LIMLT]
        # data.extend(vi)
        data.extend(v[:LIMLT])
    data = list(set(data))
    print(len(data))
    data.sort()
    # save_json(data, "sl_v2_punct_label_2w.limit")
    # save_json(data, "sl_v2_punct_label_5w.limit_1w.v2")
    save_json(data, path+".limlt")
    """
    rmrb 
    2570716:100000
    1456241: 50000---
    xxqg
    1406885
    """
def enhance_csc_punct_from_quality_dataset_by_punct_dict_xuexiqiangguo_v3(LIMIT=100000):
    """    从高质量语料中获取标点符号标注数据, xuexiqiangguo
    """
    path = "D:/workspace/code_own/a00_punct/idx2pun.v2.json"
    idx2pun_dict = load_json(path)
    pun2idx_dict = {v:k for k, v in idx2pun_dict.items()}
    # files = get_all_dirs_files("sentence_people_daily_news")
    # files = [file for file in files if "_punct_enhance" not in file and "json_enhance" not in file]


    path = "./xuexiqiangguo_ppl_select_list.json"
    category = "xuexiqiangguo"
    v_list = load_json(path)
    random.shuffle(v_list)
    random.shuffle(v_list)
    random.shuffle(v_list)


    enhance_prior_dict = {k: [] for k, v in pun2idx_dict.items()}
    for _ in range(1):
    # for file in files:
    #     data_sents = load_json(file)
    #     for sent in tqdm(data_sents, desc=file):
        for sent in tqdm(v_list, desc="v_list"):
            # try:
            #     # sent = stringQ2B(sent)   # 全角转半角
            #     sent = stringB2Q(sent)  # 半角转全角
            # except Exception as e:
            #     print(traceback.print_exc())
            #     continue

            if " " not in sent and "　" not in sent and " " not in sent:
                if len(sent) < 4:
                    continue
                sent = transfor_english_symbol_to_chinese(sent)
                if "“" in sent and "”" not in sent:
                    continue
                if "“" not in sent and "”" in sent:
                    continue
                if "‘" in sent and "’" not in sent:
                    continue
                if "‘" not in sent and "’" in sent:
                    continue
                if "《" in sent and "》" not in sent:
                    continue
                if "《" not in sent and "》" in sent:
                    continue
                if "（" in sent and "）" not in sent:
                    continue
                if "（" not in sent and "）" in sent:
                    continue
                if "~~" in sent or sent.endswith("~") or "、？" in sent:
                    continue
                # "穆尔希德·汗外长强调",
                if is_chinese_char(sent[-1]):
                    continue
                if sent.startswith("——") or sent.endswith("——")\
                        or sent.endswith("）") or sent.startswith("（") \
                        or sent.endswith("：") or sent.endswith("：“…"):
                    continue
                if sent.startswith("▲") or sent.startswith("(") \
                        or sent.startswith(")") or sent.startswith("”，") \
                        or sent.startswith("”") or sent.startswith("’"):
                    continue
                if "”“" in sent or "说，" in sent or "表示，" in sent:
                    continue
                # ”我国思想理论界也有一句名言：“一具体就深刻”。
                if sent.count("“") != sent.count("”"):
                    continue
                if sent.count("‘") != sent.count("’"):
                    continue
                if sent.count("《") != sent.count("》"):
                    continue
                if sent.count("（") != sent.count("）"):
                    continue
                if sent.count("(") != sent.count(")"):
                    continue
                for k, v in pun2idx_dict.items():
                    if k in sent and len(enhance_prior_dict.get(k)) < LIMIT:
                        enhance_prior_dict[k].append(sent)
                        break
        len_all = 0
        for k, v in enhance_prior_dict.items():
            print((k, len(v)))
            len_all += len(v)
        print(len_all)
        # save_json(enhance_prior_dict, file + "_punct_enhance.json")
        # save_json(enhance_prior_dict, "xuexiqiangguo_punct_enhance.json")
        save_json(enhance_prior_dict, "./punct_enhance_{}.json".format(category))

    """
    v_list: 100%|██████████| 4283494/4283494 [01:40<00:00, 42587.05it/s]
('，', 100000)
('。', 100000)
('、', 100000)
('；', 100000)
('？', 36394)
('！', 17017)
('：', 100000)
('“', 100000)
('”', 100000)
('‘', 22736)
('’', 0)
('—', 11466)
('…', 25691)
('·', 10843)
('~', 0)
('《', 83800)
('》', 0)
('（', 47047)
('）', 0)
('＜', 3)
('＞', 37)
('：“', 792)
('。”', 5601)
('！”', 0)
('？”', 0)
('，“', 33871)
('”，', 81407)
('”。', 55341)
('”！', 0)
('”？', 0)
('……', 0)
('——', 0)
('》。', 0)
('”“', 0)
('》，', 0)
('，《', 0)
('”、', 5706)
('、“', 4462)
('……”', 0)
('》《', 0)
('”；', 295)
('”：', 19)
('”、“', 0)
('”，“', 0)
('”——', 0)
('“‘', 0)
('’”', 0)
('，‘', 0)
('；“', 53)
('》、《', 0)
('”…', 0)
('》、', 0)
('》…', 0)
('），', 0)
('）。', 0)
('）、', 0)
('——“', 0)
('”（', 0)
('》（', 0)
('～', 2340)
('：《', 0)
('”）', 0)
('：（', 0)
('）》', 0)
('「', 0)
('」', 0)
('）：', 0)
('－－', 19)
('）》，', 0)
('》），', 0)
('）；', 0)
('，（', 0)
('；（', 0)
('《“', 0)
('》）', 0)
('”；“', 0)
('〈', 1)
('〉', 0)
('》；', 0)
('：“‘', 0)
('）“', 0)
('）”', 0)
('》：', 0)
('”：“', 0)
('［', 38)
('］', 0)
('—”', 0)
('；《', 0)
('）》（', 0)
('’、', 0)
('》，《', 0)
('（“', 0)
('’、‘', 0)
('”（《', 0)
('，“‘', 0)
('”》', 0)
('〔', 31)
('〕', 0)
('）、“', 0)
('、‘', 0)
('：‘', 0)
('（《', 0)
('？“', 0)
('」，', 0)
('《〈', 0)
('》，“', 0)
('。“', 0)
('！“', 0)
('：「', 0)
('」）', 0)
('：“《', 0)
('“《', 0)
('）（', 0)
('『', 0)
('』', 0)
('“（', 0)
('）、《', 0)
('），“', 0)
('）》《', 0)
('、（', 0)
    """

    ee = 0
def select_enhance_csc_punct_from_xuexiqiangguo_by_punct_dict_v4(LIMLT=70000):
    """   获取数据, 不超过50w   """
    path = "./punct_enhance_xuexiqiangguo.json"

    data_dict = load_json(path)
    start_list = ["”“", "（", "）", "：", "●", "△", "【", "”（", "》",
                  "3.", "37、", "◆", "”（", "”", "’", "·"]
    end_list =["）", "》", ">", "“", "‘", "’", ]  #  ["）", "》", ">", "“", "”", "‘", "’", ]
    mid_list = [("‘", "“"), ("‘", "”"), ("<", "《"), (">", "》")]
    # end:）》>
    # ‘ in not ”“

    data = []
    data_dict_1 = {}
    data_dict_2 = {}
    data_dict_3 = {}
    data_dict_4 = {}
    data_dict_5 = {}
    for k, v_org in data_dict.items():
        v = []
        for vi in v_org:
            flag = True
            for s in start_list:
                if vi.startswith(s):
                    flag = False
                    break
            for e in end_list:
                if vi.endswith(e):
                    flag = False
                    break
            for m in mid_list:
                if m[0] in vi and m[1] not in vi:
                    flag = False
                    break
            if flag:
                v.append(vi)
        # v = v[:50000]
        random.shuffle(v)
        random.shuffle(v)
        random.shuffle(v)

        # 2000; 0,1,2,7,8
        # if k in ["，", "。", "、", "“", "”"]:
        #     data_dict_1[k] = v
        # ## 500; 15, 16, 17, 18, "26","27","49"
        # elif k in ["《", "》", "（", "）", "”，","”。","》、《"]:
        #     data_dict_2[k] = v
        # # 200; 25,32,35,40,43,55
        # elif k in ["，“", "，《", "）、", "”，“", "》。", "”；"]:
        #     data_dict_3[k] = v
        # # 3,4,5,6, 9,10
        # elif k in ["；", "？", "！", "：", "‘", "’"]:
        #     data_dict_4[k] = v
        # else:
        #     data_dict_5[k] = v
        # if "（" in k or "）" in k:
        #     data += v[-1000:]
        # else:
        #     print("#" * 128)
        #     print(k)
        #     print(len(v))
        #     v = [vi for vi in v if "（" not in vi and "）" not in vi]
        #     print(len(v))



        # if k in ["，", "。", "、", "“", "”"]:
        #     data += v[-LIMLT:]
        #     continue
        # ## 500; 15, 16, 17, 18, "26","27","49"
        # elif k in ["《", "》", "（", "）", "”，","”。","》、《"]:
        #     if len(v) > LIMLT:
        #         data += v[-LIMLT:]
        #     else:
        #         data += v
        # # 200; 25,32,35,40,43,55
        # elif k in ["，“", "，《", "）、", "”，“", "》。", "”；"]:
        #     if len(v) > LIMLT:
        #         data += v[-LIMLT:]
        #     else:
        #         data += v
        # # 3,4,5,6, 9,10
        # elif k in ["；", "？", "！", "：", "‘", "’"]:
        #     if len(v) > LIMLT:
        #         data += v[-LIMLT:]
        #     else:
        #         data += v
        # else:
        #     if len(v) > LIMLT:
        #         data += v[-LIMLT:]
        #     else:
        #         data += v



        # if len(v) >= 50000:
        #     vi = v[:int(LIMLT/2)]
        # else:
        #     vi = v[:LIMLT]
        # data.extend(vi)
        data.extend(v[:LIMLT])
    data = list(set(data))
    print(len(data))
    data.sort()
    # save_json(data, "sl_v2_punct_label_2w.limit")
    # save_json(data, "sl_v2_punct_label_5w.limit_1w.v2")
    save_json(data, path+".limlt")
    """
    rmrb 
    2570716:100000
    1456241: 50000---
    xxqg
    1406885
    """
def enhance_csc_punct_from_quality_dataset_by_punct_dict_lcsts_v3(LIMIT=100000):
    """    从高质量语料中获取标点符号标注数据, xuexiqiangguo
    """
    path = "D:/workspace/code_own/idx2pun.v2.json"
    idx2pun_dict = load_json(path)
    pun2idx_dict = {v:k for k, v in idx2pun_dict.items()}
    # files = get_all_dirs_files("sentence_people_daily_news")
    # files = [file for file in files if "_punct_enhance" not in file and "json_enhance" not in file]


    # category = "xuexiqiangguo"
    path = "a00_quality_sent/a01_lomo_lcsts_ime_2m_3w_ppl/lomo_lcsts-ime-2m.2m.json.1.json.ppl.select"
    category = "lcsts"
    v_list = load_json(path)
    random.shuffle(v_list)
    random.shuffle(v_list)
    random.shuffle(v_list)


    enhance_prior_dict = {k: [] for k, v in pun2idx_dict.items()}
    for _ in range(1):
    # for file in files:
    #     data_sents = load_json(file)
    #     for sent in tqdm(data_sents, desc=file):
        for sent in tqdm(v_list, desc="v_list"):
            # try:
            #     # sent = stringQ2B(sent)   # 全角转半角
            #     sent = stringB2Q(sent)  # 半角转全角
            # except Exception as e:
            #     print(traceback.print_exc())
            #     continue
            if ":" in sent or "：" in sent or " " in sent:
                continue
            if " " not in sent and "　" not in sent and " " not in sent:
                if len(sent) < 4:
                    continue
                sent = transfor_english_symbol_to_chinese(sent)
                if "“" in sent and "”" not in sent:
                    continue
                if "“" not in sent and "”" in sent:
                    continue
                if "‘" in sent and "’" not in sent:
                    continue
                if "‘" not in sent and "’" in sent:
                    continue
                if "《" in sent and "》" not in sent:
                    continue
                if "《" not in sent and "》" in sent:
                    continue
                if "（" in sent and "）" not in sent:
                    continue
                if "（" not in sent and "）" in sent:
                    continue
                if "~~" in sent or sent.endswith("~") or "、？" in sent:
                    continue
                # "穆尔希德·汗外长强调",
                if is_chinese_char(sent[-1]):
                    continue
                if sent.startswith("——") or sent.endswith("——")\
                        or sent.endswith("）") or sent.startswith("（") \
                        or sent.endswith("：") or sent.endswith("：“…"):
                    continue
                if sent.startswith("▲") or sent.startswith("(") \
                        or sent.startswith(")") or sent.startswith("”，") \
                        or sent.startswith("”") or sent.startswith("’"):
                    continue
                if "”“" in sent or "说，" in sent or "表示，" in sent:
                    continue
                # ”我国思想理论界也有一句名言：“一具体就深刻”。
                if sent.count("“") != sent.count("”"):
                    continue
                if sent.count("‘") != sent.count("’"):
                    continue
                if sent.count("《") != sent.count("》"):
                    continue
                if sent.count("（") != sent.count("）"):
                    continue
                if sent.count("(") != sent.count(")"):
                    continue
                for k, v in pun2idx_dict.items():
                    if k in sent and len(enhance_prior_dict.get(k)) < LIMIT:
                        enhance_prior_dict[k].append(sent)
                        break
        len_all = 0
        for k, v in enhance_prior_dict.items():
            print((k, len(v)))
            len_all += len(v)
        print(len_all)
        # save_json(enhance_prior_dict, file + "_punct_enhance.json")
        # save_json(enhance_prior_dict, "xuexiqiangguo_punct_enhance.json")
        save_json(enhance_prior_dict, "./punct_enhance_{}.json".format(category))

    """v_list: 100%|██████████| 1109290/1109290 [00:10<00:00, 109390.58it/s]
('，', 100000)
('。', 100000)
('、', 100000)
('；', 11318)
('？', 78400)
('！', 38631)
('：', 0)
('“', 96313)
('”', 0)
('‘', 870)
('’', 0)
('—', 975)
('…', 10892)
('·', 2540)
('~', 809)
('《', 11601)
('》', 0)
('（', 10295)
('）', 0)
('＜', 0)
('＞', 1)
('：“', 0)
('。”', 0)
('！”', 0)
('？”', 0)
('，“', 0)
('”，', 0)
('”。', 0)
('”！', 0)
('”？', 0)
('……', 0)
('——', 0)
('》。', 0)
('”“', 0)
('》，', 0)
('，《', 0)
('”、', 0)
('、“', 0)
('……”', 0)
('》《', 0)
('”；', 0)
('”：', 0)
('”、“', 0)
('”，“', 0)
('”——', 0)
('“‘', 0)
('’”', 0)
('，‘', 0)
('；“', 0)
('》、《', 0)
('”…', 0)
('》、', 0)
('》…', 0)
('），', 0)
('）。', 0)
('）、', 0)
('——“', 0)
('”（', 0)
('》（', 0)
('～', 465)
('：《', 0)
('”）', 0)
('：（', 0)
('）》', 0)
('「', 467)
('」', 3)
('）：', 0)
('－－', 4)
('）》，', 0)
('》），', 0)
('）；', 0)
('，（', 0)
('；（', 0)
('《“', 0)
('》）', 0)
('”；“', 0)
('〈', 1)
('〉', 0)
('》；', 0)
('：“‘', 0)
('）“', 0)
('）”', 0)
('》：', 0)
('”：“', 0)
('［', 0)
('］', 1)
('—”', 0)
('；《', 0)
('）》（', 0)
('’、', 0)
('》，《', 0)
('（“', 0)
('’、‘', 0)
('”（《', 0)
('，“‘', 0)
('”》', 0)
('〔', 3)
('〕', 0)
('）、“', 0)
('、‘', 0)
('：‘', 0)
('（《', 0)
('？“', 0)
('」，', 0)
('《〈', 0)
('》，“', 0)
('。“', 0)
('！“', 0)
('：「', 0)
('」）', 0)
('：“《', 0)
('“《', 0)
('）（', 0)
('『', 12)
('』', 0)
('“（', 0)
('）、《', 0)
('），“', 0)
('）》《', 0)
('、（', 0)
('’”。', 0)
('！”。', 0)
('”……', 0)
('’！”', 0)
('’。”', 0)
('？？', 0)
('？？？', 0)
('！！', 0)
('！！！', 0)
('！？', 0)
('？！', 0)
('？——', 0)
('？”。', 0)
('——？', 0)
('——!', 0)
('/', 3353)
('-', 5624)
('》？', 0)
('》！', 0)
('）》。', 0)
('”）。', 0)
('》）。', 0)
('》……', 0)
('）”。', 0)
('？！”', 0)
('）。”', 0)
('》。”', 0)
('！？”', 0)
572578
446502
    """

    ee = 0
def select_enhance_csc_punct_from_lcsts_by_punct_dict_v4(LIMLT=70000):
    """   获取数据, 不超过50w   """
    path = "./punct_enhance_lcsts.json"

    data_dict = load_json(path)
    start_list = ["”“", "（", "）", "：", "●", "△", "【", "”（", "》",
                  "3.", "37、", "◆", "”（", "”", "’", "·"]
    end_list =["）", "》", ">", "“", "‘", "’", ]  #  ["）", "》", ">", "“", "”", "‘", "’", ]
    mid_list = [("‘", "“"), ("‘", "”"), ("<", "《"), (">", "》")]
    # end:）》>
    # ‘ in not ”“

    data = []
    data_dict_1 = {}
    data_dict_2 = {}
    data_dict_3 = {}
    data_dict_4 = {}
    data_dict_5 = {}
    for k, v_org in data_dict.items():
        v = []
        for vi in v_org:
            flag = True
            for s in start_list:
                if vi.startswith(s):
                    flag = False
                    break
            for e in end_list:
                if vi.endswith(e):
                    flag = False
                    break
            for m in mid_list:
                if m[0] in vi and m[1] not in vi:
                    flag = False
                    break
            if flag:
                v.append(vi)
        # v = v[:50000]
        random.shuffle(v)
        random.shuffle(v)
        random.shuffle(v)

        # 2000; 0,1,2,7,8
        # if k in ["，", "。", "、", "“", "”"]:
        #     data_dict_1[k] = v
        # ## 500; 15, 16, 17, 18, "26","27","49"
        # elif k in ["《", "》", "（", "）", "”，","”。","》、《"]:
        #     data_dict_2[k] = v
        # # 200; 25,32,35,40,43,55
        # elif k in ["，“", "，《", "）、", "”，“", "》。", "”；"]:
        #     data_dict_3[k] = v
        # # 3,4,5,6, 9,10
        # elif k in ["；", "？", "！", "：", "‘", "’"]:
        #     data_dict_4[k] = v
        # else:
        #     data_dict_5[k] = v
        # if "（" in k or "）" in k:
        #     data += v[-1000:]
        # else:
        #     print("#" * 128)
        #     print(k)
        #     print(len(v))
        #     v = [vi for vi in v if "（" not in vi and "）" not in vi]
        #     print(len(v))



        # if k in ["，", "。", "、", "“", "”"]:
        #     data += v[-LIMLT:]
        #     continue
        # ## 500; 15, 16, 17, 18, "26","27","49"
        # elif k in ["《", "》", "（", "）", "”，","”。","》、《"]:
        #     if len(v) > LIMLT:
        #         data += v[-LIMLT:]
        #     else:
        #         data += v
        # # 200; 25,32,35,40,43,55
        # elif k in ["，“", "，《", "）、", "”，“", "》。", "”；"]:
        #     if len(v) > LIMLT:
        #         data += v[-LIMLT:]
        #     else:
        #         data += v
        # # 3,4,5,6, 9,10
        # elif k in ["；", "？", "！", "：", "‘", "’"]:
        #     if len(v) > LIMLT:
        #         data += v[-LIMLT:]
        #     else:
        #         data += v
        # else:
        #     if len(v) > LIMLT:
        #         data += v[-LIMLT:]
        #     else:
        #         data += v



        # if len(v) >= 50000:
        #     vi = v[:int(LIMLT/2)]
        # else:
        #     vi = v[:LIMLT]
        # data.extend(vi)
        data.extend(v[:LIMLT])
    data = list(set(data))
    print(len(data))
    data.sort()
    # save_json(data, "sl_v2_punct_label_2w.limit")
    # save_json(data, "sl_v2_punct_label_5w.limit_1w.v2")
    save_json(data, path+".limlt")
    """
    lcstc:
    110w
    select:44w
    
    """

def merge_punctuation_composition_single_from_path(path):
    """ 生成span格式的标点符号生成数据集
    文言文
    学生作文
    人民日报
    学习强国
   """
    if type(path) == list:
        data_all = path
    else:
        data_all = load_json(path)

    path_idx2pun_v2 = "D:/workspace/sample_v2_2023/a00_punct/idx2pun.v2.json"
    idx2pun = load_json(path_idx2pun_v2)
    pun2idx = {v: k for k,v in idx2pun.items()}

    # pun2idx = {k: str(idx) for idx, (k, v) in enumerate(pun_dict.items())}
    # idx2pun = {v: k for k,v in pun2idx.items()}
    pun_string = "".join(list(set(list("".join(list(pun2idx.keys()))))))
    # save_json([pun_string], "pun_string.json")
    # save_json(idx2pun, "idx2pun.json")

    symbol_str_dict = {}
    symbols_all = []
    res_list = []
    for d in tqdm(data_all, desc="file"):
        # if "豳居允荒" in d:
        #     ee = 0

        if "实覃实" in d:
            ee = 0
        if "泛泛杨舟绋" in d:
            ee = 0
        if type(d) == dict:
            d = d.get("target", "")
        if not d or " " in d or "\t" in d or "\n" in d or len(d) < 4 or d.startswith("”“") or d.startswith("（") \
                or ("‘" in d and "”" not in d) or len(d) > 125 or ("□" in d) \
                or d.startswith("…") or d.startswith("..."):
                # or (d.startswith("《") or d.startswith("》")) \
            continue
        if "\"" in d:
            index_list = [i.start() for i in re.finditer("\"", d)]
            if index_list and len(index_list)%2==0:
                print(d)
                for idx, index in enumerate(index_list):
                    symbol = "“" if idx % 2 == 0 else "”"
                    d = d[:index] + symbol + d[index + 1:]
                print(d)
            else:
                continue

        if " " in d or "　" in d or " " in d:
            continue

        if "“" in d and "”" not in d:
            continue
        if "“" not in d and "”" in d:
            continue
        if "‘" in d and "’" not in d:
            continue
        if "‘" not in d and "’" in d:
            continue
        if "《" in d and "》" not in d:
            continue
        if "《" not in d and "》" in d:
            continue
        if "（" in d and "）" not in d:
            continue
        if "（" not in d and "）" in d:
            continue
        if "~~" in d or d.endswith("~") or "、？" in d:
            continue
        # "穆尔希德·汗外长强调",
        if is_chinese_char(d[-1]):
            continue
        if d.startswith("——") or d.endswith("——") \
                or d.endswith("）") or d.startswith("（"):
            continue
        if d.startswith("▲"):
            continue

        # try:
        #     d_mid = tradition_to_simple(d)
        #     if len(d_mid) != len(d):
        #         print(d, d_mid)
        #         continue
        #     d = stringQ2B(d_mid)
        # except Exception as e:
        #     print(traceback.print_exc())
        #     print(d)
        #     continue
        # d = d.replace("  ", "").replace("\n", "").replace("\t", "").replace(" ", "") \
        #     .replace(",", "，").replace(";", "；").replace("!", "！") \
        #     .replace("?", "？").replace(":", "：").replace("∶", "：").replace("(", "（") \
        #     .replace(")", "）").replace("......", "……").replace("__", "——")
        if d.count("“") != d.count("”") or d.count("‘") != d.count("’")\
                or d.count("（") != d.count("）") or d.count("<") != d.count(">")\
                or d.count("<") != d.count(">") or d.count("《") != d.count("》"):
            continue

        # d = transfor_english_symbol_to_chinese(d)
        d = stringQ2B(d)  # 半角
        d = transfor_english_symbol_to_chinese(d)
        d = tradition_to_simple(d)
        if not check_pair_symbol(d):
            continue

        symbol_list = []
        symbol_str = ""
        d_new = "#"
        ents = []
        flag = True
        for idx, t in enumerate(d):
            if idx == len(d) - 1:
                myz = 0
            ee = 0
            if is_other(t):
                if t in pun_string:
                    symbol_str += t
                    if idx==len(d)-1:  # 最后一个符号是特殊符号
                        # d_new += t
                        if pun2idx.get(symbol_str, ""):  # 标点符号 必须 存在已知字典
                            ents_i = {"type": pun2idx.get(symbol_str, ""), "ent": d_new[-1], "pos": [len(d_new) - 1, len(d_new) - 1]}
                            ents.append(ents_i)
                            symbol_list.append(symbol_str)
                            symbol_str = ""
                        else:
                            symbols_all.append(symbol_str)
                            if symbol_str not in symbol_str_dict:
                                symbol_str_dict[symbol_str] = 1
                            else:
                                symbol_str_dict[symbol_str] += 1
                            # print(d)
                            # print(symbol_str)
                            symbol_str = ""
                            ee = 0
                            flag = False
                            break
                else:
                    flag = False
                    break
                # else:  # 有特殊字符的都不要，如古体字
                #     flag = False
                #     symbol_str = ""
                #     break
            else:
                if symbol_str:  # 特殊符号
                    if symbol_str not in pun2idx:
                        symbols_all.append(symbol_str)
                        symbol_str = ""
                        break
                    else:
                        if pun2idx.get(symbol_str, ""):
                            ents_i = {"type": pun2idx.get(symbol_str, ""),
                                      "ent": d_new[-1],
                                      "pos": [len(d_new)-1, len(d_new)-1]
                                      }
                            ents.append(ents_i)
                            symbol_list.append(symbol_str)
                        symbol_str = ""
                        # else:
                        #     symbols_all.append(symbol_str)
                        #     print(d)
                        #     print(symbol_str)
                        #     ee = 0
                        #     symbol_str = ""
                        #     flag = False
                        #     break
                d_new += t
            myz = 0

        if d_new and len(d_new) > 3 and flag:
            d_new = stringQ2B(d_new)  # 半角
            line_dict = {"label": ents, "text": d_new}
            res_list.append(json.dumps(line_dict, ensure_ascii=False)+"\n")
    "'我是被称为“世界独生子”的国家一级重点保护野生动物黔金丝猴，还被世界自然保护联盟列为濒危度最高的物种之一。'"
    "！……"
    "？……"
    "，……"
    "………。"
    "”！？"
    "！……"
    "’！”"
    "”？！"
    "”？。"
    "’。”"
    "……”。"
    "”？！"
    "……。"

    "？”。"

    random.shuffle(res_list)
    random.shuffle(res_list)
    random.shuffle(res_list)
    print(len(res_list))
    lens = len(res_list)
    return res_list, lens, symbol_str_dict

    # res_train = res_list[: int(0.9*len(res_list))]
    # res_dev = res_list[int(0.9*len(res_list)): ]

    # save_json(symbols_all_counter_sorted, "symbols_all_counter_sorted.json")
    # txt_write(res_train, "good_list_create_train.train.span")
    # txt_write(res_dev, "good_list_create_train.dev.span")
def merge_punctuation_composition_single_from_path_all_dirs_mdcspell_public():
    """   处理生成所有的标点符号数据集(划分好以后方便组合), 开源的数据集合成的数据   """
    myz = 0
    path_dir = "a00_quality_sent/z00_punct_labeling/"
    path_xxqg = path_dir + "punct_enhance_xuexiqiangguo.json.limlt"
    path_rmrb = path_dir + "punct_enhance_rmrb.json.limlt"
    path_poerty = path_dir + "chinese_poetry.json.ppl.select"
    path_lcstc = path_dir + "punct_enhance_lcsts.json.limlt"

    path_v2_pun_huajianji = path_dir + "v2_pun_huajianji.json.ppl.select"
    path_v2_wenyanwen_nopun = path_dir + "v2_wenyanwen_nopun.json.ppl.select"

    paths = [path_xxqg,
             path_rmrb,
             path_poerty,
             path_v2_pun_huajianji,
             path_v2_wenyanwen_nopun,
             path_lcstc
    ]
    res_train = []
    res_dev = []
    res_tet = []
    len_dev = 5000
    symbol_str_dict_all = {}
    for path in paths:
        data = load_json(path)
        print(path)
        print(len(data))
        data.sort()
        print(data[:5])
        if "rmrb" in path:
            data = data[2:]
        if "v2_pun_huajianji" in path:
            data = data * 10
        # if "schoolpassage_mrc" in path:
        #     data = data * 5
        # if "sql_tb_composition" in path:
        #     data = data * 3
        if "lcsts" in path:
            data = data * 2
        res_list, lens, symbol_str_dict = merge_punctuation_composition_single_from_path(data)
        random.shuffle(res_list)
        # res_list = data
        res_train.extend(res_list[:-2*len_dev])
        res_dev.extend(res_list[-2*len_dev:-len_dev])
        res_tet.extend(res_list[-len_dev:])
        for k, v in symbol_str_dict.items():
            if k not in symbol_str_dict_all:
                symbol_str_dict_all[k] = v
            else:
                symbol_str_dict_all[k] += v
    print(len(res_train))
    symbol_str_dict_all_sorted = sorted(symbol_str_dict_all.items(), key=lambda x: x[-1], reverse=True)
    print(symbol_str_dict_all_sorted)
    txt_write(res_train, "merge_punct_ner.train.jsonl")
    txt_write(res_dev, "merge_punct_ner.dev.jsonl")
    txt_write(res_tet, "merge_punct_ner.tet.jsonl")
    """
1022612
['0-14岁少儿人口的数量比2010年增加了3092万人，比重上升了1.35个百分点。', '0-3岁这段时间可以作为孩子发展的窗口。', '0-6岁是儿童成长的奠基阶段，也是培养和塑造良好性格和品质的关键期。', '0-6岁的儿童眼保健也是纳入了0-6岁儿童健康管理的内容，包括近视在内的屈光不正，还会发现斜视、弱视等等眼健康的问题。', '0.004%，看似微不足道的数字，却是这家企业引领行业的“法宝”。']
file: 100%|██████████| 1022612/1022612 [02:02<00:00, 8360.64it/s]
945666
a00_quality_sent/z00_punct_labeling/punct_enhance_rmrb.json.limlt
1044363
['-最初的梦想-：中国体育代表团在第一天的比赛中取得了开门红，4枚金牌的含金量都特别高，有奥运首金，有男子游泳首金，有破世界纪录的，还有等待十几年之久的金牌。', '-非现实的梦想家-：实现利率市场化，让银行业充分市场化，让四大行和民营资本银行拥有相同的市场地位，并且让经营状况不佳的银行破产，更应该挤掉其不当的高福利高工资，让高收入与高风险并存！', '0.01毫米，只有圆珠笔芯直径的1/50，比头发丝还细，这是目前全球最薄的柔性显示屏。', '0.455元∕千瓦时是燃煤机组脱硫“标杆电价”，是此前上述8家电厂把电卖给电网公司的价格。', '007系列电影推出40周年之际，007的扮演者皮尔斯·布鲁斯南向观众揭示电影特技背后的科技内涵。']
file: 100%|██████████| 1044361/1044361 [02:17<00:00, 7600.52it/s]
1026542
a00_quality_sent/z00_punct_labeling/chinese_poetry.json.ppl.select
1441229
file:   0%|          | 0/1441229 [00:00<?, ?it/s]['一《题石门山》，诗及前后题款年月凡十一行：一《永嘉怀古》，诗及题款凡八行，俱正书，径寸。', '一㸃新萤报秋信，不知何处是菩提。', '一一七穿八穴，明明百匝千重。', '一一不可拣，于君况夙昔。', '一一不是单，三三亦非九。']
file: 100%|██████████| 1441229/1441229 [01:13<00:00, 19498.51it/s]
1430445
a00_quality_sent/z00_punct_labeling/v2_pun_huajianji.json.ppl.select
13886
['一不可也。', '一丘常欲卧，三径苦无资。', '一为迁客去长沙，西望长安不见家。', '一为钓叟一耕佣。', '一之日于貉，取彼狐狸，为公子裘。']
file: 100%|██████████| 138860/138860 [00:07<00:00, 19145.95it/s]
137720
a00_quality_sent/z00_punct_labeling_/v2_wenyanwen_nopun.json.ppl.select
818043
['一、五行：一曰水，二曰火，三曰木，四曰金，五曰土。', '一、保圣躬。', '一、在京恶逆与强盗真犯，虽停刑之年，亦不时处决。', '一、强盗肆行劫杀，按赃拟辟，决不待时。', '一、律称伪造诸衙门印信者斩。']
file: 100%|██████████| 818043/818043 [00:52<00:00, 15545.85it/s]
811180
a00_quality_sent/z00_punct_labeling/punct_enhance_lcsts.json.limlt
446502
['0-1不敌圣洛伦索后，阿根廷老牌劲旅独立队提前一轮降级，这是其建队108年来首次降级。', '0-1不敌马拉加，梅西本场哑火，13次过人7次成功，1脚射门和2次关键传球的数据显得很是尴尬。', '0.5个百分点将释放出4000亿到5000亿元的流动性，量是比较大的。', '0.5元~1元，就值这点儿钱。', '0.5元的菜品，浙江农林大学食堂卖了20年！']
file: 100%|██████████| 893004/893004 [01:23<00:00, 10671.34it/s]
882474
5174027
[('……。', 3832), ('，……', 1073), ('？……', 714), ('……。”', 543), ('？。', 453), ('……”。', 439), ('！……', 360), ('！。', 319), ('」。', 269), ('………', 250), ('”？。', 250), ('。。', 238), ('。”？', 205), ('。……”', 184), ('！……”', 168), ('’？”', 142), ('”……。', 138), ('，……”', 124), ('；……', 123), ('~！', 118), ('）！', 110), ('）……', 99), ('）？', 84), ('“？”', 78), ('，……。', 75), ('。”……', 74), ('···', 72), ('”》。', 70), ('………。', 69), ('」）。', 68), ('！”……', 66), ('…”', 63), ('〕。', 59), ('？……”', 58), ('”？！', 58), ('。………', 58), ('，。', 58), ('）！”', 57), ('？》。', 55), ('“。', 55), ('…………', 53), ('………”', 48), ('、……', 43), ('？”……', 40), ('’……”', 40), ('。…………', 37), ('》”。', 36), ('————？', 34), ('--', 30), ('（）。', 29), ('”，……', 29), ('！》。', 28), ('······', 26), ('，…', 25), ('〉》。', 23), ('———？', 22), ('。”。', 21), ('》？”', 21), ('，………', 21), ('：……', 20), ('！）。', 20), ('---', 20), ('’”……', 19), ('……！', 19), ('；……。', 18), ('。）。', 18), ('…………。', 17), ('………。”', 17), ('……？', 17), ('’。', 16), ('）…', 16), ('”！？', 15), ('………”。', 14), ('~~', 14), ('----', 14), ('？………', 13), ('’”！', 12), ('”。？', 12), ('……………', 12), ('」！', 12), ('）？”', 11), ('，……。”', 11), ('“？”。', 11), ('！………', 11), ('…。', 11), ('’”？', 10), ('…………”', 10), ('》。？', 10), ('、。', 10), ('“？', 10), ('」？', 10), ('！………”', 9), ('。………”', 9), ('）……。', 9), ('；……”', 9), ('，…………', 9), ('-。', 9), ('————。', 9), ('》……。', 8), ('，……”。', 8), ('——。', 8), ('》！”', 7), ('）》）。', 7), ('——！”', 7), ('…………。”', 7), ('……？”', 7), ('）……”', 6), ('）》……', 6), ('……）。', 6), ('“……', 6), ('——”。', 6), ('，………”', 6), ('，？？', 6), ('———————？', 6), ('—————？', 6), ('“！', 6), ('------', 6), ('》”）。', 5), ('……》。', 5), ('“！”', 5), ('！…………', 5), ('？………”', 5), ('……！”', 5), ('。……………', 5), ('…………”。', 5), ('”………', 5), ('？？”', 5), ('…”。', 5), ('』。', 5), ('）”……', 4), ('！”？', 4), ('。……。', 4), ('！！”', 4), ('’……”。', 4), ('），……', 4), ('（？）。', 4), ('“？？”', 4), ('？……”。', 4), ('））。', 4), ('！……。', 4), ('”………。', 4), ('！……”。', 4), ('、…', 4), ('—————。', 4), ('~。', 4), ('-----', 4), ('，！', 4), ('··', 4), ('”-----', 4), ('…？', 4), ('————————。', 4), ('~？', 4), ('——————？', 4), ('”~', 4), ('”）……', 3), ('）》？', 3), ('”》……', 3), ('？！……', 3), ('”？……', 3), ('—。', 3), ('》？。', 3), ('………………', 3), ('。”…………', 3), ('”，……。', 3), ('”》）。', 3), ('”；……', 3), ('：…………', 3), ('。？！', 3), ('（！）”', 3), ('’……。”', 3), ('。？？', 3), ('。”）。', 3), ('？……。', 3), ('”。”', 3), ('”（！）。', 3), ('。………………', 3), ('。”………', 3), ('；。', 3), ('、……。', 3), ('，“？”', 3), ('，？', 3), ('”…………', 3), ('？…………', 3), ('--。', 3), ('〉。', 3), ('”）？', 2), ('’？”。', 2), ('！！！”', 2), ('’）”。', 2), ('……”！', 2), ('？”？', 2), ('（？）”。', 2), ('）；……', 2), ('（！）。', 2), ('；…………', 2), ('’）。”', 2), ('“……”', 2), ('。？', 2), ('’”？。', 2), ('。……”？', 2), ('。…………”', 2), ('……”？', 2), ('”！！', 2), ('………………”。', 2), ('”）！”', 2), ('）”！', 2), ('（？）。”', 2), ('”、“……”', 2), ('！》”', 2), ('？》）。', 2), ('。…………………', 2), ('）：……', 2), ('………？”', 2), ('……”……', 2), ('”，“……”。', 2), ('」）。。', 2), ('—————！', 2), ('！！！”。', 2), ('，”。', 2), ('，”', 2), ('·！', 2), ('…！', 2), ('”······', 2), ('————！', 2), ('——————。', 2), ('·····', 2), ('，---', 2), ('》）···', 2), ('“·”。', 2), ('。”！', 2), ('—…', 2), ('-”', 2), ('，--！', 2), ('‘”。', 2), ('）······', 2), ('——————————。', 2), ('”~！', 2), ('》”', 2), ('~”……', 2), ('）、。', 2), ('）----', 2), ('」…', 2), ('’”）。', 1), ('；”。', 1), ('）》”。', 1), ('）”？', 1), ('”？》。', 1), ('”……！', 1), ('〉》？。', 1), ('》，？', 1), ('）……”。', 1), ('，……”……', 1), ('“。”', 1), ('………！', 1), ('’”。？', 1), ('！………………', 1), ('！》）。', 1), ('”。…………', 1), ('………”！', 1), ('，？？？', 1), ('？……。”', 1), ('——…………', 1), ('…………！”', 1), ('？！）”', 1), ('”）！', 1), ('……）？', 1), ('），……。”', 1), ('》”！', 1), ('”）。”', 1), ('！！。', 1), ('。…………。', 1), ('》。”……', 1), ('。……………………', 1), ('）？！”', 1), ('）”。？', 1), ('？）”。？', 1), ('）。？', 1), ('，……！”', 1), ('…………？', 1), ('）”）。', 1), ('”’……”', 1), ('）……。”', 1), ('”？…………', 1), ('……………。', 1), ('；………', 1), ('’）？”', 1), ('，”……', 1), ('》）”。', 1), ('（？）……', 1), ('”。………', 1), ('？）。', 1), ('），……”', 1), ('）。………', 1), ('、……”', 1), ('………………。', 1), ('。）”', 1), ('—”。', 1), ('’。……”', 1), ('”…………。', 1), ('。？？？？？', 1), ('“-”。', 1), ('》）？', 1), ('~”。', 1), ('！………。', 1), ('“！”。', 1), ('”、…………', 1), ('》……！', 1), ('！）”', 1), ('’？！”', 1), ('……”？。', 1), ('（？）”', 1), ('………！”', 1), ('；……。”', 1), ('！？……', 1), ('？……？', 1), ('——！', 1), ('……………”。', 1), ('。）……', 1), ('》）。”', 1), ('：……。', 1), ('〕）。', 1), ('“”。', 1), ('？）……', 1), ('》）……。', 1), ('——……', 1), ('）。”？', 1), ('”……？', 1), ('—？”', 1), ('…………………”', 1), ('？………………”', 1), ('’！？”', 1), ('？”……。', 1), ('’？……”', 1), ('？！”。', 1), ('。……。”', 1), ('）………', 1), ('———。', 1), ('。……………”', 1), ('！”）。', 1), ('？”！', 1), ('……。”……', 1), ('”，………', 1), ('！……。”', 1), ('—！”', 1), ('…………………', 1), ('”、……。', 1), ('》……”', 1), ('》）……', 1), ('”）”。', 1), ('；…………。', 1), ('？？。', 1), ('）！”。', 1), ('（！？）”。', 1), ('：……”', 1), ('！”……。', 1), ('”！……', 1), ('’？”……', 1), ('——……”', 1), ('（！）”。', 1), ('》？！', 1), ('？！……”', 1), ('………………”', 1), ('（！？）。', 1), ('。”……。', 1), ('”）……。', 1), ('”、“？”', 1), ('”……………。', 1), ('」）？', 1), ('〕？', 1), ('」。）。', 1), ('》」。', 1), ('：？', 1)]
('……。', 3832)
('，……', 1073)
('？……', 714)
('……。”', 543)

"""


def collect_punct_label(text = "你是谁，你的名字叫什么？"):
    """   收集原始的标点符号(未知的)   """
    pun2idx = {}
    symbol_list = []
    symbol_str = ""
    d_new = "#"
    ents = []
    for idx, t in enumerate(text):
        ### 非中文
        if is_other(t):
            if t in pun2idx:
                symbol_str += t
                if idx == len(text) - 1:  # 最后一个符号是特殊符号
                    # d_new += t
                    if pun2idx.get(symbol_str, ""):  # 标点符号 必须 存在已知字典
                        ents_i = {"type": pun2idx.get(symbol_str, ""),
                                  "ent": d_new[-1],
                                  "pos": [len(d_new) - 1, len(d_new) - 1],
                                  "pun": symbol_str
                                  }
                        ents.append(ents_i)
                        symbol_list.append(symbol_str)
                        symbol_str = ""
                    else:
                        # print(d)
                        # print(symbol_str)
                        symbol_str = ""
                        flag = False
                        break
            else:
                flag = False
                break
            # else:  # 有特殊字符的都不要，如古体字
            #     flag = False
            #     symbol_str = ""
            #     break
        else:
            if symbol_str:  # 特殊符号
                if symbol_str not in pun2idx:
                    symbol_str = ""
                    break
                else:
                    if pun2idx.get(symbol_str, ""):
                        ents_i = {"type": pun2idx.get(symbol_str, ""),
                                  "ent": d_new[-1],
                                  "pos": [len(d_new) - 1, len(d_new) - 1],
                                  "pun": symbol_str
                                  }
                        ents.append(ents_i)
                        symbol_list.append(symbol_str)
                    symbol_str = ""
            d_new += t
    d_new = stringQ2B(d_new)  # 半角
    line_dict = {"label": ents, "text": d_new}
    return line_dict

def clear_punct_label(text = "你是谁，你的名字叫什么？"):
    """   构建原始的   """
    idx2pun = {
    "0": "，",
    "1": "。",
    "2": "、",
    "3": "；",
    "4": "？",
    "5": "！",
    "6": "：",
    "7": "“",
    "8": "”",
    "9": "‘",
    "10": "’",
    "11": "—",
    "12": "…",
    "13": "·",
    "14": "~",
    "15": "《",
    "16": "》",
    "17": "（",
    "18": "）",
    "19": "＜",
    "20": "＞",
    "21": "：“",
    "22": "。”",
    "23": "！”",
    "24": "？”",
    "25": "，“",
    "26": "”，",
    "27": "”。",
    "28": "”！",
    "29": "”？",
    "30": "……",
    "31": "——",
    "32": "》。",
    "33": "”“",
    "34": "》，",
    "35": "，《",
    "36": "”、",
    "37": "、“",
    "38": "……”",
    "39": "》《",
    "40": "”；",
    "41": "”：",
    "42": "”、“",
    "43": "”，“",
    "44": "”——",
    "45": "“‘",
    "46": "’”",
    "47": "，‘",
    "48": "；“",
    "49": "》、《",
    "50": "”…",
    "51": "》、",
    "52": "》…",
    "53": "），",
    "54": "）。",
    "55": "）、",
    "56": "——“",
    "57": "”（",
    "58": "》（",
    "59": "～",
    "60": "：《",
    "61": "”）",
    "62": "：（",
    "63": "）》",
    "64": "「",
    "65": "」",
    "66": "）：",
    "67": "－－",
    "68": "）》，",
    "69": "》），",
    "70": "）；",
    "71": "，（",
    "72": "；（",
    "73": "《“",
    "74": "》）",
    "75": "”；“",
    "76": "〈",
    "77": "〉",
    "78": "》；",
    "79": "：“‘",
    "80": "）“",
    "81": "）”",
    "82": "》：",
    "83": "”：“",
    "84": "［",
    "85": "］",
    "86": "—”",
    "87": "；《",
    "88": "）》（",
    "89": "’、",
    "90": "》，《",
    "91": "（“",
    "92": "’、‘",
    "93": "”（《",
    "94": "，“‘",
    "95": "”》",
    "96": "〔",
    "97": "〕",
    "98": "）、“",
    "99": "、‘",
    "100": "：‘",
    "101": "（《",
    "102": "？“",
    "103": "」，",
    "104": "《〈",
    "105": "》，“",
    "106": "。“",
    "107": "！“",
    "108": "；“",
    "109": "：「",
    "110": "」）",
    "111": "：“《",
    "112": "“《",
    "113": "）（",
    "114": "『",
    "115": "』",
    "116": "“（",
    "117": "）、《",
    "118": "），“",
    "119": "）》《",
    "120": "、（"
}
    pun2idx = {v: k for k,v in idx2pun.items()}

    symbol_str = ""
    d_new = "#"
    ents = []
    for idx, t in enumerate(text):
        ### 非中文
        if t in pun2idx:
            symbol_str += t
            if idx == len(text) - 1:  # 最后一个符号是特殊符号
                ents_i = {"type": pun2idx.get(symbol_str, ""),
                          "ent": d_new[-1],
                          "pos": [len(d_new) - 1, len(d_new) - 1],
                          "pun": symbol_str
                          }
                ents.append(ents_i)
                symbol_str = ""
            else:
                ents_i = {"type": pun2idx.get(symbol_str, ""),
                          "ent": d_new[-1],
                          "pos": [len(d_new) - 1, len(d_new) - 1],
                          "pun": symbol_str
                          }
                ents.append(ents_i)
                symbol_str = ""
    d_new = stringQ2B(d_new)  # 半角
    line_dict = {"label": ents, "text": d_new}
    return line_dict

def collect_punct_label_123_old(text):
    """ 生成span格式的标点符号生成数据集
    文言文
    学生作文
    人民日报
    学习强国
   """

    # path_idx2pun_v2 = "D:/workspace/sample_v2_2023/a00_punct/idx2pun.v2.json"
    # idx2pun = load_json(path_idx2pun_v2)

    idx2pun = {
        "0": "，",
        "1": "。",
        "2": "、",
        "3": "；",
        "4": "？",
        "5": "！",
        "6": "：",
        "7": "“",
        "8": "”",
        "9": "‘",
        "10": "’",
        "11": "—",
        "12": "…",
        "13": "·",
        "14": "~",
        "15": "《",
        "16": "》",
        "17": "（",
        "18": "）",
        "19": "＜",
        "20": "＞",
        "21": "：“",
        "22": "。”",
        "23": "！”",
        "24": "？”",
        "25": "，“",
        "26": "”，",
        "27": "”。",
        "28": "”！",
        "29": "”？",
        "30": "……",
        "31": "——",
        "32": "》。",
        "33": "”“",
        "34": "》，",
        "35": "，《",
        "36": "”、",
        "37": "、“",
        "38": "……”",
        "39": "》《",
        "40": "”；",
        "41": "”：",
        "42": "”、“",
        "43": "”，“",
        "44": "”——",
        "45": "“‘",
        "46": "’”",
        "47": "，‘",
        "48": "；“",
        "49": "》、《",
        "50": "”…",
        "51": "》、",
        "52": "》…",
        "53": "），",
        "54": "）。",
        "55": "）、",
        "56": "——“",
        "57": "”（",
        "58": "》（",
        "59": "～",
        "60": "：《",
        "61": "”）",
        "62": "：（",
        "63": "）》",
        "64": "「",
        "65": "」",
        "66": "）：",
        "67": "－－",
        "68": "）》，",
        "69": "》），",
        "70": "）；",
        "71": "，（",
        "72": "；（",
        "73": "《“",
        "74": "》）",
        "75": "”；“",
        "76": "〈",
        "77": "〉",
        "78": "》；",
        "79": "：“‘",
        "80": "）“",
        "81": "）”",
        "82": "》：",
        "83": "”：“",
        "84": "［",
        "85": "］",
        "86": "—”",
        "87": "；《",
        "88": "）》（",
        "89": "’、",
        "90": "》，《",
        "91": "（“",
        "92": "’、‘",
        "93": "”（《",
        "94": "，“‘",
        "95": "”》",
        "96": "〔",
        "97": "〕",
        "98": "）、“",
        "99": "、‘",
        "100": "：‘",
        "101": "（《",
        "102": "？“",
        "103": "」，",
        "104": "《〈",
        "105": "》，“",
        "106": "。“",
        "107": "！“",
        "108": "；“",
        "109": "：「",
        "110": "」）",
        "111": "：“《",
        "112": "“《",
        "113": "）（",
        "114": "『",
        "115": "』",
        "116": "“（",
        "117": "）、《",
        "118": "），“",
        "119": "）》《",
        "120": "、（"
    }

    pun2idx = {v: k for k,v in idx2pun.items()}


    # pun2idx = {k: str(idx) for idx, (k, v) in enumerate(pun_dict.items())}
    # idx2pun = {v: k for k,v in pun2idx.items()}
    pun_string = "".join(list(set(list("".join(list(pun2idx.keys()))))))
    # save_json([pun_string], "pun_string.json")
    # save_json(idx2pun, "idx2pun.json")

    symbol_str_dict = {}
    symbols_all = []
    res_list = []
    for d in tqdm([text], desc="file"):
        # if "豳居允荒" in d:
        # d = transfor_english_symbol_to_chinese(d)
        d = stringQ2B(d)  # 半角
        d = transfor_english_symbol_to_chinese(d)
        d = tradition_to_simple(d)
        symbol_list = []
        symbol_str = ""
        d_new = "#"
        ents = []
        flag = True
        for idx, t in enumerate(d):
            if is_other(t):
                if t in pun_string:
                    symbol_str += t
                    if idx == len(d)-1:  # 最后一个符号是特殊符号
                        if pun2idx.get(symbol_str, ""):  # 标点符号 必须 存在已知字典
                            ents_i = {"type": pun2idx.get(symbol_str, ""),
                                      "ent": d_new[-1],
                                      "pos": [len(d_new) - 1, len(d_new) - 1],
                                      }
                            ents.append(ents_i)
                            symbol_list.append(symbol_str)
                            symbol_str = ""
                        else:
                            symbols_all.append(symbol_str)
                            if symbol_str not in symbol_str_dict:
                                symbol_str_dict[symbol_str] = 1
                            else:
                                symbol_str_dict[symbol_str] += 1
                            symbol_str = ""
                            flag = False
                            break
                else:
                    flag = False
                    break
                # else:  # 有特殊字符的都不要，如古体字
                #     flag = False
                #     symbol_str = ""
                #     break
            else:
                if symbol_str:  # 特殊符号
                    if symbol_str not in pun2idx:
                        symbols_all.append(symbol_str)
                        symbol_str = ""
                        break
                    else:
                        if pun2idx.get(symbol_str, ""):
                            ents_i = {"type": pun2idx.get(symbol_str, ""),
                                      "ent": d_new[-1],
                                      "pos": [len(d_new)-1, len(d_new)-1]
                                      }
                            ents.append(ents_i)
                            symbol_list.append(symbol_str)
                        symbol_str = ""
                d_new += t
            yz = 0

        d_new = stringQ2B(d_new)  # 半角
        line_dict = {"label": ents, "text": d_new}
        print(line_dict)

def collect_punct_label_123(text):
    """ 生成span格式的标点符号生成数据集
    文言文
    学生作文
    人民日报
    学习强国
   """

    # path_idx2pun_v2 = "D:/workspace/sample_v2_2023/a00_punct/idx2pun.v2.json"
    # idx2pun = load_json(path_idx2pun_v2)

    idx2pun = {
        "0": "，",
        "1": "。",
        "2": "、",
        "3": "；",
        "4": "？",
        "5": "！",
        "6": "：",
        "7": "“",
        "8": "”",
        "9": "‘",
        "10": "’",
        "11": "—",
        "12": "…",
        "13": "·",
        "14": "~",
        "15": "《",
        "16": "》",
        "17": "（",
        "18": "）",
        "19": "＜",
        "20": "＞",
        "21": "：“",
        "22": "。”",
        "23": "！”",
        "24": "？”",
        "25": "，“",
        "26": "”，",
        "27": "”。",
        "28": "”！",
        "29": "”？",
        "30": "……",
        "31": "——",
        "32": "》。",
        "33": "”“",
        "34": "》，",
        "35": "，《",
        "36": "”、",
        "37": "、“",
        "38": "……”",
        "39": "》《",
        "40": "”；",
        "41": "”：",
        "42": "”、“",
        "43": "”，“",
        "44": "”——",
        "45": "“‘",
        "46": "’”",
        "47": "，‘",
        "48": "；“",
        "49": "》、《",
        "50": "”…",
        "51": "》、",
        "52": "》…",
        "53": "），",
        "54": "）。",
        "55": "）、",
        "56": "——“",
        "57": "”（",
        "58": "》（",
        "59": "～",
        "60": "：《",
        "61": "”）",
        "62": "：（",
        "63": "）》",
        "64": "「",
        "65": "」",
        "66": "）：",
        "67": "－－",
        "68": "）》，",
        "69": "》），",
        "70": "）；",
        "71": "，（",
        "72": "；（",
        "73": "《“",
        "74": "》）",
        "75": "”；“",
        "76": "〈",
        "77": "〉",
        "78": "》；",
        "79": "：“‘",
        "80": "）“",
        "81": "）”",
        "82": "》：",
        "83": "”：“",
        "84": "［",
        "85": "］",
        "86": "—”",
        "87": "；《",
        "88": "）》（",
        "89": "’、",
        "90": "》，《",
        "91": "（“",
        "92": "’、‘",
        "93": "”（《",
        "94": "，“‘",
        "95": "”》",
        "96": "〔",
        "97": "〕",
        "98": "）、“",
        "99": "、‘",
        "100": "：‘",
        "101": "（《",
        "102": "？“",
        "103": "」，",
        "104": "《〈",
        "105": "》，“",
        "106": "。“",
        "107": "！“",
        "108": "；“",
        "109": "：「",
        "110": "」）",
        "111": "：“《",
        "112": "“《",
        "113": "）（",
        "114": "『",
        "115": "』",
        "116": "“（",
        "117": "）、《",
        "118": "），“",
        "119": "）》《",
        "120": "、（"
    }

    pun2idx = {v: k for k,v in idx2pun.items()}


    # pun2idx = {k: str(idx) for idx, (k, v) in enumerate(pun_dict.items())}
    # idx2pun = {v: k for k,v in pun2idx.items()}
    pun_string = "".join(list(set(list("".join(list(pun2idx.keys()))))))
    # save_json([pun_string], "pun_string.json")
    # save_json(idx2pun, "idx2pun.json")

    symbol_str_dict = {}
    symbols_all = []
    res_list = []
    for d in tqdm([text], desc="file"):
        # if "豳居允荒" in d:
        # d = transfor_english_symbol_to_chinese(d)
        d = stringQ2B(d)  # 半角
        d = transfor_english_symbol_to_chinese(d)
        d = tradition_to_simple(d)
        symbol_list = []
        symbol_str = ""
        d_new = "#"
        ents = []
        flag = True
        for idx, t in enumerate(d):
            if is_other(t):
                if t in pun_string:
                    symbol_str += t
                    if idx == len(d)-1:  # 最后一个符号是特殊符号
                        if pun2idx.get(symbol_str, ""):  # 标点符号 必须 存在已知字典
                            ents_i = {"type": pun2idx.get(symbol_str, ""),
                                      "ent": d_new[-1],
                                      "pos": [len(d_new) - 1, len(d_new) - 1],
                                      }
                            ents.append(ents_i)
                            symbol_list.append(symbol_str)
                            symbol_str = ""
                        else:
                            symbol_str = ""
                            flag = False
                            break
                else:
                    flag = False
                    break
            else:
                if symbol_str:  # 特殊符号
                    if symbol_str not in pun2idx:
                        symbol_str = ""
                        break
                    else:
                        if pun2idx.get(symbol_str, ""):
                            ents_i = {"type": pun2idx.get(symbol_str, ""),
                                      "ent": d_new[-1],
                                      "pos": [len(d_new)-1, len(d_new)-1]
                                      }
                            ents.append(ents_i)
                            symbol_list.append(symbol_str)
                        symbol_str = ""
                d_new += t
            yz = 0

        d_new = stringQ2B(d_new)  # 半角
        line_dict = {"label": ents, "text": d_new}
        print(line_dict)
        return line_dict


if __name__ == '__main__':
    yz = 0
    text = "（《五四（《五四（《五四"
    # text_punct = collect_punct_label(text)
    text_punct = collect_punct_label_123(text)
    print(text_punct)
    yz = 0

    # ### punct数据
    # enhance_csc_punct_from_quality_dataset_by_punct_dict_rmrb_v4()
    # select_enhance_csc_punct_from_renmingribao_by_punct_dict_v4()
    # select_enhance_csc_punct_from_xuexiqiangguo_by_punct_dict_v4()
    # ### 学习强国,  limit
    # enhance_csc_punct_from_quality_dataset_by_punct_dict_xuexiqiangguo_v3()
    # ## lcsts-200w-from-ime, limit
    # enhance_csc_punct_from_quality_dataset_by_punct_dict_lcsts_v3()
    # select_enhance_csc_punct_from_lcsts_by_punct_dict_v4()
    #
    ### 合并数据集punct
    # merge_punctuation_composition_single_from_path_all_dirs_mdcspell_public()
    #
    # ### 统计非中文字符, 并整理高频标点符号, 用于构建标点符号字典等(主要是一些多个标点拼接的数据)
    # a00_counter_none_chinese_of_all_quality_dataset_step1()
    # a01_counter_none_chinese_of_all_quality_dataset_step2()



"""
"0": "，",
"1": "。",
"2": "、",
"3": "；",
"7": "“",
"8": "”",

"4": "？",
"5": "！",
"6": "：",
"9": "‘",
"10": "’",
    

  precision    recall  f1-score   support

0     0.8762    0.8826    0.8794     57563
1     0.9291    0.9703    0.9493     24661
2     0.8334    0.6630    0.7385     10597
3     0.7270    0.3936    0.5107      1245
7     0.7181    0.2528    0.3740      3556
8     0.7369    0.2863    0.4123      2613   
"""


"""
∶
＂
．
’，
～
＜＞
---------------“”
---------------；．
"""


"""
punct: 
huajianci_poetry_14k

"""


