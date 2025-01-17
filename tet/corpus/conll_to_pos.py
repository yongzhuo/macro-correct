# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/8/24 23:09
# @author  : Mo
# @function: transform conll to span, 将CONLL格式的数据转化为SPAN格式{pos:[1,3]}


import logging
import os


def get_all_dirs_files(path_dir):
    """
        递归获取某个目录下的所有文件(所有层, 包括子目录)
    :param path_dir: str, like '/home/data'
    :return: list, like ['2020_01_08.txt']
    """
    path_files = []

    def get_path_files(path_dir):
        """
            递归函数, 获取某个目录下的所有文件
        :param path_dir: str, like '/home/data'
        :return: list, like ['2020_01_08.txt']
        """
        for root, dirs, files in os.walk(path_dir):
            for fi in files:  # 递归的终止条件
                path_file = os.path.join(root, fi)
                path_files.append(path_file)
            for di in dirs:  # 语间目录便继续递归
                path_dir = os.path.join(root, di)
                get_path_files(path_dir)

    get_path_files(path_dir)
    path_files = list(set(path_files))
    path_files.sort()
    return path_files
def txt_write(lines, path: str, model: str = "w", encoding: str = "utf-8"):
    """
    Write Line of list to file
    Args:
        lines: lines of list<str> which need save
        path: path of save file, such as "txt"
        model: type of write, such as "w", "a+"
        encoding: type of encoding, such as "utf-8", "gbk"
    """

    try:
        file = open(path, model, encoding=encoding)
        file.writelines(lines)
        file.close()
    except Exception as e:
        logging.info(str(e))
def save_json(lines, path, encoding: str = "utf-8", indent: int = 4):
    """
    Write Line of List<json> to file
    Args:
        lines: lines of list[str] which need save
        path: path of save file, such as "json.txt"
        encoding: type of encoding, such as "utf-8", "gbk"
    """

    with open(path, "w", encoding=encoding) as fj:
        fj.write(json.dumps(lines, ensure_ascii=False, indent=indent))
    fj.close()
def get_pos_from_common(words0, tag1):
    """从common模型的输出中重构标注, 即获取未知信息---position
    common analysis for sequence-labeling
    Args:
        words0: String/List, origin text,  eg. "沪是上海"
        tag1  : List, common-output of labels,  eg. ["S-city", "O", "B-city", "I-city"]
    Returns:
        reault: List, eg. [{"type":"city", "ent":"沪", "pos":[2:4]}]
    """
    res = []
    ws = ""
    start_pos_1 = 0
    end_pos_1 = 0
    sentence = ""
    types = ""
    for i in range(len(tag1)):
        if tag1[i].startswith("S-"):
            ws += words0[i]
            start_pos_1 = i
            end_pos_1 = i
            sentence += words0[i]
            types = tag1[i][2:]
            res.append([ws, start_pos_1, end_pos_1, types])
            ws = ""
            types = ""

        if tag1[i].startswith("B-"):
            if len(ws) > 0:
                res.append([ws, start_pos_1, end_pos_1, types])
                ws = ""
                types = ""
            if len(ws) == 0:
                ws += words0[i]
                start_pos_1 = i
                end_pos_1 = i
                sentence += words0[i]
                types = tag1[i][2:]

        elif tag1[i].startswith("I-"):
            if len(ws) > 0 and types == tag1[i][2:]:
                ws += words0[i]
                sentence += words0[i]
                end_pos_1 = i

            elif len(ws) > 0 and types != tag1[i][2:]:
                res.append([ws, start_pos_1, end_pos_1, types])
                ws = ""
                types = ""

            if len(ws) == 0:
                ws += words0[i]
                start_pos_1 = i
                end_pos_1 = i
                sentence += words0[i]
                types = tag1[i][2:]

        elif tag1[i].startswith("M-"):
            if len(ws) > 0 and types == tag1[i][2:]:
                ws += words0[i]
                sentence += words0[i]
                end_pos_1 = i

            elif len(ws) > 0 and types != tag1[i][2:]:
                res.append([ws, start_pos_1, end_pos_1, types])
                ws = ""
                types = ""

            if len(ws) == 0:
                ws += words0[i]
                start_pos_1 = i
                end_pos_1 = i
                sentence += words0[i]
                types = tag1[i][2:]

        elif tag1[i].startswith('E-'):
            if len(ws) > 0 and types == tag1[i][2:]:
                ws += words0[i]
                sentence += words0[i]
                end_pos_1 = i
                res.append([ws, start_pos_1, end_pos_1, types])
                ws = ""
                types = ""

            if len(ws) > 0 and types != tag1[i][2:]:
                res.append([ws, start_pos_1, end_pos_1, types])
                ws = ""
                ws += words0[i]
                start_pos_1 = i
                end_pos_1 = i
                sentence += words0[i]
                types = tag1[i][2:]
                res.append([ws, start_pos_1, end_pos_1, types])
                ws = ""
                types = ""

        elif tag1[i] == "O":

            if len(ws) > 0:
                res.append([ws, start_pos_1, end_pos_1, types])
                ws = ""
                types = ""

            sentence += words0[i]

        if i == len(tag1) - 1 and len(ws) > 0:
            res.append([ws, start_pos_1, end_pos_1, types])
            ws = ""
            types = ""
    reault = []
    for r in res:
        entity_dict = {}
        entity_dict["type"] = r[3]
        entity_dict["ent"] = r[0]
        entity_dict["pos"] = [r[1], r[2]]
        reault.append(entity_dict)
    return reault
def read_corpus(corpus_path):
    """读取CONLL数据
    read corpus for sequence-labeling
    Args:
        corpus_path: String, path/origin text,  eg. "ner.conll"
    Returns:
        data: List<tuple>, <sent_, tag_>
    """
    data = []
    with open(corpus_path, encoding="utf-8") as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != "\n":
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []
    return data


if __name__ == '__main__':
    import json
    import sys
    import os
    path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.append(path_root)
    print(path_root)
    # path = path_root + "/corpus/sequence_labeling/chinese_symbol/"
    # for t in ["train", "dev", "test"]:
    #     t = t + ".conll"
    #     path_real = path + t
    #     if not os.path.exists(path_real):
    #         print("path is not exist: " + path_real)
    #         continue

    path_dir = path_root + "/macro_correct/corpus/sequence_labeling/chinese_symbol/"
    files = get_all_dirs_files(path_dir)
    for path_real in files:
        if not path_real.endswith(".conll"):
            print("path is not exist: " + path_real)
            continue
        data = read_corpus(path_real)
        res = []
        for d in data:
            label = get_pos_from_common(d[0], d[1])
            line = {"label":label, "text": "".join(d[0])}
            res.append(json.dumps(line, ensure_ascii=False) + "\n")
        txt_write(res, path_real.replace(".conll", ".span"))
        ee = 0
    # transform conll to span, 将CONLL格式的数据转化为SPAN格式{pos:[1,3]}


