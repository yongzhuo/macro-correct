# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/6/28 14:34
# @author  : Mo
# @function: 清洗和处理古诗文花间词, 用于古文的标点符号处理


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


def is_other(uchar):
    """判断是否非汉字，数字和英文字符"""
    return not (is_chinese_char(uchar) or is_number(uchar) or is_alphabet(uchar))


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


def collect_poerty_huajianji():
    """   获取数据集花间集   """
    path_dir = "E:/DATA/corpus/a01_github/huajianji/data"
    files = get_all_dirs_files(path_dir)
    # files = [file for file in files if ".json" in file and "old" not in file]
    files = [file for file in files if ".json" in file]

    "[1]"
    res = []
    for file in tqdm(files, desc="i"):
        data = load_json(file)
        for d in tqdm(data, desc="j"):
            d_paragraphs = d.get("paragraphs", [])
            notes = d.get("notes", [])
            title = d.get("title", "")
            # if notes and len(notes) > 1:
            #     res.extend(notes)
            #
            # if "《" not in title:
            #     title = "《" + title + "》"
            #     res.append(title)
            if d_paragraphs:
                d_paragraphs_new = [d for d in d_paragraphs if d]
                if d_paragraphs_new:
                    if type(d_paragraphs_new[0]) == str:
                        d_paragraphs_new_string = "".join(d_paragraphs_new).replace("\n", "")\
                            .replace("\t", "").replace(" ", "") \
                            .replace(",", "，").replace(";", "；").replace("!", "！") \
                            .replace("?", "？").replace(":", "：").replace("(", "（") \
                            .replace(")", "）").replace('："', "：”")\
                            .replace('。"', "。”").replace('？"', "？”").replace('!"', "!”")\
                            .replace('「', "“").replace('」', "”").replace("﹑", "、")

                        # re.compile(r"(?i)<math.*?>(?P<math>.+?)</math>")
                        # d_paragraphs_new_string = re.compile(r"（.+?）]").sub("", d_paragraphs_new_string)
                        if "[" in d_paragraphs_new_string:
                            continue
                        if "（" in d_paragraphs_new_string:
                            d_paragraphs_new_string = re.sub(r"（.+?）", "", d_paragraphs_new_string)
                            d_paragraphs_new_string = re.sub(r"【.+?】", "", d_paragraphs_new_string)

                        ee = 0
                        res.append(d_paragraphs_new_string)
                    elif type(d_paragraphs_new[0]) == dict:
                        for d_paragraphs_new_i in d_paragraphs_new:
                            d_paragraphs_new_v = d_paragraphs_new_i.get("paragraphs", [])
                            d_paragraphs_new_string = "".join(d_paragraphs_new_v).replace("\n", "")\
                                .replace("\t", "") \
                                .replace(",", "，").replace(";", "；").replace("!", "！") \
                                .replace("?", "？").replace(":", "：").replace("(", "（") \
                                .replace(")", "）")
                            if "[" in d_paragraphs_new_string:
                                continue
                            # d_paragraphs_new_string = re.compile(r"（.+?）]").sub("", d_paragraphs_new_string)
                            if "（" in d_paragraphs_new_string:
                                d_paragraphs_new_string = re.sub(r"（.+?）", "", d_paragraphs_new_string)
                                d_paragraphs_new_string = re.sub(r"【.+?】", "", d_paragraphs_new_string)

                                ee = 0

                            res.append(d_paragraphs_new_string)
                else:
                    ee = 0
                ee = 0
    print(len(res))
    res = list(set(res))
    print(len(res))
    save_json(res, "v2_pun_huajianji.json")
    ee = 0

def counter_pun_poerty():
    """   统计诗词中的标点符号   """
    # data_pun_huajianji = load_json("pun_huajianji.json")
    data_pun_huajianji = load_json("v2_pun_huajianji.json")
    symbols_all = []
    for d in data_pun_huajianji:
        d = tradition_to_simple(d)
        d = stringQ2B(d)
        d = d.replace("\n", "").replace("\t", "").replace(" ", "")\
            .replace(",", "，").replace(";", "；").replace("!", "！")\
            .replace("?", "？").replace(":", "：").replace("(", "（")\
            .replace(")", "）")
        d_symbols = collect_none_chinese(d)
        if d_symbols:
            symbols_all.extend(d_symbols)
    # from collections import Counter, OrderedDict
    symbols_all_counter = Counter(symbols_all)
    symbols_all_counter_dict = OrderedDict(sorted(symbols_all_counter.items(), key=lambda x:x[-1], reverse=True))
    # save_json(symbols_all_counter_dict, "pun_huajianji_counter.json")
    save_json(symbols_all_counter_dict, "v2_pun_huajianji_counter.json")


if __name__ == '__main__':
    ee = 0

    # text = "征人--指出征或戍边的军人。（唐）苏拯 《古塞下》诗：“血染长城沙，马踏征人骨。”（唐）李益《夜上受降城闻笛》：“不知何处吹芦管，一夜征人尽望乡。”"
    # for t in text:
    #     if not is_chinese_char(t):
    #         print(t)

    ## 转化
    collect_poerty_huajianji()
    ## 统计标点符号
    counter_pun_poerty()
    # ### 统计诗词
    # data_pun_huajianji = load_json("pun_huajianji.json")
    # res = []
    # for d in data_pun_huajianji:
    #     if "。”" in d:
    #         d_sp = d.split("。”")
    #         d_list = [di+"。”" if is_chinese_char(di[-1]) else di for di in d_sp if len(di)>3 ]
    #     else:
    #         d_sp = d.split("。")
    #         d_list = [di+"。" if is_chinese_char(di[-1]) else di for di in d_sp if len(di)>3 ]
    #     res.extend(d_list)
    # save_json(res, "pun_huajianji.pun.json")

    data = {
        "，": 20680,
        "。": 12759,
        "？": 750,
        "；": 502,
        "、": 486,
        "！": 338,
        "：": 182,
        "《": 81,
        "》": 62,
        "）": 79,
        "（": 30,
        "“": 31,
        "”": 31,
        "——": 31,
        "......": 31,

        "：“": 644,
        "。”": 341,
        "？”": 167,
        "！”": 84,
        "。（": 81,

        "：‘": 26,
    }


