# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2024/4/12 10:55
# @author  : Mo
# @function: 清洗SIGHAN和Wang271k数据集中地脏数据


import logging as logger
import traceback
import random
import copy
import json
import re
import os
import gc

from difflib import SequenceMatcher
from tqdm import tqdm
import pandas as pd


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
def save_xlsx_from_json(res_xlsx, path_xlsx="save.xlsx"):
    """   json转化为xlsx的excel文件   """
    pdr = pd.DataFrame(res_xlsx)
    with pd.ExcelWriter(path_xlsx, engine="xlsxwriter",
            # options={"strings_to_urls": False}
                        ) as writer:
        pdr.to_excel(writer)
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

def tet_csc_clean_train_dataset_wang271k():
    """   清洗wang271数据集   """
    import json
    import sys
    import os
    path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(path_root)
    print(path_root)

    path_dir = os.path.join(path_root, "macro_correct/corpus/text_correction/wang271k")
    path_train = path_dir + "/train.json"
    path_dev = path_dir + "/dev.json"
    path_tet = path_dir + "/test.json"
    paths = [path_train, path_dev, path_tet]
    """
        {
            "id":"--",
            "original_text":"国中数学课辅班",
            "wrong_ids":[],
            "correct_text":"国中数学课辅班"
        }
    """
    count_yu_1 = 0
    count_yu_2 = 0
    count_fu_1 = 0
    count_fu_2 = 0
    count_ta_1 = 0
    count_ta_2 = 0
    count_de_1 = 0
    for path in paths:
        if not os.path.exists(path):
            print("path is not ecist: " + path)
            continue
        data_json_list = load_json(path)
        data_json_list_new = []
        for jdx, d_json_org in enumerate(data_json_list):
            d_json = copy.deepcopy(d_json_org)
            original_text = d_json.get("original_text", "")
            correct_text = d_json.get("correct_text", "")
            wrong_ids = d_json.get("wrong_ids", [])
            wrong_ids_new = []
            for wid in wrong_ids:
                char_error = original_text[wid]
                char_true = correct_text[wid]
                # 余 - 馀
                flag_add = True
                if char_error=="余" and char_true=="馀":
                    original_text = original_text[:wid] + char_true + original_text[wid+1:]
                    correct_text = correct_text[:wid] + char_error + correct_text[wid+1:]
                    count_yu_1 += 1
                elif char_true=="馀":
                    correct_text = correct_text[:wid] + "余" + correct_text[wid+1:]
                    count_yu_2 += 1

                # 复-覆
                # 覆-复
                if char_error == "覆" and char_true == "复":
                    original_text = original_text[:wid] + char_true + original_text[wid + 1:]
                    correct_text = correct_text[:wid] + char_error + correct_text[wid + 1:]
                    count_fu_1 += 1
                elif char_true == "覆":
                    # 答疆/回覆/反覆
                    # 覆审
                    if correct_text[wid-1:wid+1] in ["答覆", "回覆", "反覆"]:
                        correct_text = correct_text[:wid] + "复" + correct_text[wid + 1:]
                    elif correct_text[wid:wid+2] in ["覆审"]:
                        correct_text = correct_text[:wid] + "复" + correct_text[wid + 1:]
                    count_fu_2 += 1

                # 功-工
                if char_error == "功" and char_true == "工":
                    "同工/工作/竣工"
                    "工夫/工效"
                    if correct_text[wid:wid + 2] in ["工夫", "工效"]:
                        original_text = original_text[:wid] + "工" + original_text[wid + 1:]
                        correct_text = correct_text[:wid] + "功" + correct_text[wid + 1:]

                # 借-藉
                if char_error == "借" and char_true == "藉":
                    original_text = original_text[:wid] + char_true + original_text[wid + 1:]
                    correct_text = correct_text[:wid] + char_error + correct_text[wid + 1:]
                # 琅-瑯
                if char_error == "琅" and char_true == "瑯":
                    original_text = original_text[:wid] + char_true + original_text[wid + 1:]
                    correct_text = correct_text[:wid] + char_error + correct_text[wid + 1:]

                # 叶门-也门
                if char_error == "也" and char_true == "叶":
                    if correct_text[wid:wid + 2] in ["叶门"]:
                        correct_text = correct_text[:wid] + char_error + correct_text[wid + 1:]
                        correct_text = correct_text.replace("叶门", "也门")   # 多余得叶门
                # 跨越数白百万公里后
                if char_error == "百" and char_true == "白":
                    if correct_text[wid-1:wid + 2] in ["数白百"]:
                        correct_text = correct_text[:wid] + char_error + correct_text[wid + 1:]

                # 震振有词 只改一半
                if char_error == "震" and char_true == "振":
                    if correct_text[wid - 1:wid + 1] in ["震振"]:
                        correct_text = correct_text[:wid-1] + char_true + correct_text[wid-1+1:]
                        wrong_ids_new.append(wid-1)
                # 外部, 经不起-禁不起
                if char_error == "经" and char_true == "禁":
                    if correct_text[wid:wid + 3] in ["禁不住", "禁不起"] and \
                            "她曾经禁不住落泪" not in correct_text and \
                        "大家都禁不住拍手鼓掌" not in correct_text:
                        original_text = original_text[:wid] + char_true + original_text[wid + 1:]
                        correct_text = correct_text[:wid] + char_error + correct_text[wid + 1:]

                # 他-她   不纠
                if char_error == "他" and char_true == "她":
                    correct_text = correct_text[:wid] + char_error + correct_text[wid + 1:]
                    flag_add = False
                    count_ta_1 += 1
                #  她 - 他
                if char_error == "她" and char_true == "他":
                    correct_text = correct_text[:wid] + char_error + correct_text[wid + 1:]
                    flag_add = False
                    count_ta_2 += 1
                # 小-晓  ### 人名不改
                if char_error == "小" and char_true == "晓":
                    correct_text = correct_text[:wid] + char_error + correct_text[wid + 1:]
                    flag_add = False
                # 一-逸  ### 人名不改, train都是人名, 吕逸涛/范逸臣/周逸雄
                if char_error == "一" and char_true == "逸":
                    correct_text = correct_text[:wid] + char_error + correct_text[wid + 1:]
                    flag_add = False
                    # 吕一套被任命为二零一六央视春晚总导演。
                # 佳-家  ### 人名不改, train都是人名, 张家玮 家慧
                if char_error == "佳" and char_true == "家":
                    if correct_text[wid:wid + 2] in ["家玮", "家慧"]:
                        correct_text = correct_text[:wid] + char_error + correct_text[wid + 1:]
                        flag_add = False
                # 得-地  ### 人名不改, 马哈得
                if char_error == "得" and char_true == "地":
                    if correct_text[wid-2:wid+1] in ["马哈地"]:
                        correct_text = correct_text[:wid] + char_error + correct_text[wid + 1:]
                        flag_add = False
                # 红-虹  ### 人名不改, 马哈得
                if char_error == "红" and char_true == "虹":  # 刘虹 虹姐 张虹 秀虹
                    if correct_text[wid-1:wid+1] in ["刘虹", "张虹", "秀虹"] or \
                        correct_text[wid:wid+2] in ["虹姐"]:
                        correct_text = correct_text[:wid] + char_error + correct_text[wid + 1:]
                        flag_add = False
                # 民-明  ### 人名不改, 鲍明 冯大明 许传明 杨爱明
                if char_error == "民" and char_true == "明":
                    if correct_text[wid - 1:wid + 1] in ["鲍明"] or \
                            correct_text[wid-2:wid + 1] in ["冯大明", "许传明", "杨爱明"]:
                        correct_text = correct_text[:wid] + char_error + correct_text[wid + 1:]
                        flag_add = False

                # # 的 - 地
                # # 的 - 得
                # # 它 - 他
                # # 哪 - 那

                # # 改-大小改:  余-馀 覆-复 借-藉 功-工 琅-瑯 震-振 百-白 也-叶 经-禁(经不起-禁不起)
                # # 部分不变(人名)： 小-晓 一-逸 佳-家 得-地(马哈得) 红-虹 民-明
                # # 匹配上但是不改的： 惟-唯 象-像 查-察 立-利 止-只 建-健 他-它 地-的 定-订 带-戴 力-利 成-城 点-店
                # # 匹配上但是不改的： 作-做 得-的 场-厂 身-生 有-由 种-重 理-里


                # # 空白没匹配上： 今-在 年-今 前-目 当-在 目-在 者-是
                # # 外国人名等：其-齐 课-科 博-波
                #
                # if char_error == "民" and char_true == "明":
                #     # 张家玮 家慧  刘虹 虹姐 张虹 秀虹  鲍明 冯大明 许传明 杨爱明
                #     # print(original_text)
                #     # print(correct_text)
                #     # print(correct_text[wid - 1:wid + 1], correct_text[wid:wid+2])
                #     correct_text = correct_text[:wid] + char_error + correct_text[wid + 1:]
                #     count_de_1 += 1
                #     # "是那一段时间"
                #     # 发言人并未说明他住在那家疗养院，也未说明入院原因。
                #     # 但俄国官员却说，如果两千万美元费用未付，蓝斯那儿也去不成，目前双方还僵持在付费日期上。
                #     # 跨越数白百万公里后
                #     # 我们的眼睛也开始禁不起烟尘的刺激。 禁不住琢磨 禁不住 禁不起
                #     # 不禁令人担心
                #     # 她曾经禁不住落泪 大家都禁不住拍手鼓掌
                if flag_add:
                    wrong_ids_new.append(wid)

            d_json["original_text"] = original_text
            d_json["correct_text"] = correct_text
            d_json["wrong_ids"] = wrong_ids_new
            data_json_list_new.append(d_json)
            if wrong_ids_new != wrong_ids:
                print("#"*128)
                print("original_text_or: " + data_json_list[jdx].get("original_text", ""))
                print("correct_text_org: " + data_json_list[jdx].get("correct_text", ""))
                print("correct_text_new: " + correct_text)
                print("wrong_ids_new: ", wrong_ids_new)
                print("wrong_ids: ", wrong_ids)

        save_json(data_json_list_new, os.path.split(path)[-1]+".handle_clean")
    print(count_yu_1)
    print(count_yu_2)
    print(count_fu_1)
    print(count_fu_2)


# test.json 和 dev.json 为 SIGHAN数据集， 包括SIGHAN13 14 15，来自 官方csc.html ，文件大小：339kb，4千条。
# train.json 为 Wang271k数据集，包括 Wang271k ，来自 Automatic-Corpus-Generation dimmywang提供 ，文件大小：93MB，27万条。


if __name__ == '__main__':
    yz = 0

    ### 清洗数据
    tet_csc_clean_train_dataset_wang271k()


"""
余-馀: 替换为馀-余
other - 馀: 替换为余
覆-复: 替换为复-覆
other-覆: # 答疆/回覆/反覆
          # 覆审
他-她:不纠
她-他:不纠
人名不纠: 识别人名并丢弃


抽取人民日报语料中分错的(错得多的都需要补充预料)
者-是
或-还
立-利
震-振
即-既

"""

