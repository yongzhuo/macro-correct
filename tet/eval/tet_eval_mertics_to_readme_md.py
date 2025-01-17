# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/7/8 20:42
# @author  : Mo
# @function: 将eval-mertics结果转化为readme.md


import logging as logger
import difflib
import json
import os


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


def load_json(path: str, encoding: str = "utf-8"):
    """
    Read Line of List<json> form file
    Args:
        path: path of save file, such as "txt"
        encoding: type of encoding, such as "utf-8", "gbk"
    Returns:
        model_json: dict of word2vec, eg. [{"大漠帝国":132}]
    """
    with open(path, "r", encoding=encoding) as fj:
        model_json = json.load(fj)
        fj.close()
    return model_json


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
    res = list(set(path_files))
    res.sort()
    return res


if __name__ == '__main__':
    myz = 0

    path = "../../macro_correct/output/text_correction"
    files = get_all_dirs_files(path)
    # files_filter = [file for file in files if "eval_result_mertics_total.json" in file]
    # files_filter = [file for file in files if "eval_result_mertics_total.json" in file
    #                 and "v1_eval_result_mertics_total.json" not in file]
    files_filter = [file for file in files if "v1_eval_result_mertics_total.json" in file]
    # files_filter = [file for file in files if "v0.75_eval_result_mertics_total.json" in file]

    data_dict_filter = load_json(files_filter[0])
    k1k2_list = []
    for k, v in data_dict_filter.items():
        for vk, vv in v.items():
            if vk in ["sent", "token"]:
                for vv_k in vv:
                    k1k2_list.append([vk, vv_k])

    # k1k2 = ["sent", "common_det_acc"]
    for k1k2 in k1k2_list:
        # md_keys = ["model", "avg"] + [k.split(".")[0]+"_5k" if k=="mcsc_tet.5000.json" else k.split(".")[0] for k in data_dict_filter.keys()]
        md_keys = ["model/"+k1k2[-1], "avg"] + [k.split(".")[0] for k in data_dict_filter.keys() if k not in ["mcsc_tet.json", "acc_rmrb.tet.json", "acc_xxqg.tet.json"]]
        text_1 = "| " + "| ".join(md_keys) + " |"
        text_2 = "|" + ":-----------------|" * len(md_keys)
        print(text_1)
        print(text_2)
        text = f"""\n### {k1k2[-1]}\n{text_1}\n{text_2}"""
        for file in files_filter:
            path_dir = os.path.split(file)[0]
            task_name = os.path.split(path_dir)[-1]
            data_dict = load_json(file)
            data_line = []
            for jdx, (k, v) in enumerate(data_dict.items()):
                ### 测试过度纠错的不计数
                if k in ["mcsc_tet.json", "acc_rmrb.tet.json", "acc_xxqg.tet.json"]:
                    continue
                sent_mertics = v.get(k1k2[0], "")
                common_det_acc = sent_mertics.get(k1k2[1], 0)
                common_det_acc_round = round(common_det_acc*100, 2)
                data_line.append(common_det_acc_round)
                print(common_det_acc_round)
                print(k.replace(".json", ""))
            data_line_all = [task_name, round(sum(data_line) / len(data_line), 2)] + data_line
            data_line_all = [str(d) for d in data_line_all]
            text_line = "| " + "| ".join(data_line_all) + " |"
            text += "\n" + text_line
        text = text.replace("MacBERT-chinese_finetuned_correction", "macbert4csc_shibing624").replace("csc_TextProofreadingCompetition", "text_proof")
        # txt_write([t+"\n" for t in text.split("\n")], "_".join(k1k2) + ".md")
        text_list = [t.strip() + "\n" for t in text.split("\n")]
        txt_write(text_list, "eval_mertics.md", mode="a+")


    md_keys = ["model/acc", "avg"] + ["acc_rmrb", "acc_xxqg"]
    text_1 = "| " + "| ".join(md_keys) + " |"
    text_2 = "|" + ":-----------------|" * len(md_keys)
    text = f"""\n### acc_true\n{text_1}\n{text_2}"""

    for file in files_filter:
        path_dir = os.path.split(file)[0]
        task_name = os.path.split(path_dir)[-1]
        data_dict = load_json(file)
        data_line = []
        for jdx, (k, v) in enumerate(data_dict.items()):
            ### 测试过度纠错的不计数
            if k in ["acc_rmrb.tet.json", "acc_xxqg.tet.json"]:
                sent_mertics = v.get("sent", {})
                common_det_acc = sent_mertics.get("common_cor_acc", 0)
                common_det_acc_round = round(common_det_acc * 100, 2)
                data_line.append(common_det_acc_round)
                print(common_det_acc_round)
                print(k.replace(".json", ""))
        data_line_all = [task_name, round(sum(data_line) / len(data_line), 2)] + data_line
        data_line_all = [str(d) for d in data_line_all]
        text_line = "| " + "| ".join(data_line_all) + " |"
        text += "\n" + text_line
    text = text.replace("MacBERT-chinese_finetuned_correction", "macbert4csc_pycorrector").replace("csc_TextProofreadingCompetition", "text_proof")
    text_list = [t.strip() + "\n" for t in text.split("\n")]
    txt_write(text_list, "eval_mertics.md", mode="a+")



"""
将eval-mertics结果转化为readme.md
"""
