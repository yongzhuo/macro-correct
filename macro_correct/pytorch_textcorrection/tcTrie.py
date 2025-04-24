# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/11/17 21:35
# @author  : Mo
# @function: ConfusionCorrector using trie-tree


import logging as logger
import platform
import json
import copy
import sys
import re
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(path_root)
if platform.system().lower() == "windows":
    print(path_root)


class ConfusionCorrect:
    def __init__(self, path="", path_user="", confusion_dict={}, user_dict={}):
        self.confusion_dict = {}
        ### 默认词典(基础), 全覆盖(path什么都不传的情况, 或者传了任意值)
        if path or (not path and not path_user):
            self.confusion_dict = self.load_confusion_dict(path=path)
        ### 用户混淆词典地址, 更新
        if path_user and os.path.exists(path_user):
            user_dict = self.load_user_dict(path_user)
            self.confusion_dict.update(user_dict)
        ### 默认全量混淆词典, 全覆盖
        if confusion_dict:
            self.confusion_dict = confusion_dict
        ### 用户混淆词典, 更新
        if user_dict:
            self.confusion_dict.update(user_dict)
        # 构建前缀树
        self.trietree = self.create_trie_tree(list(self.confusion_dict.keys()))

    def predict_batch(self, texts, flag_prob=True, **kwargs):
        """
            批量句子纠错
        :param sentences: 句子文本列表
        :param kwargs: 其他参数
        :return: list of {'source': 'src', 'target': 'trg', 'errors': [(error_word, correct_word, position), ...]}
        """
        return [self.predict(s, flag_prob=flag_prob, **kwargs) for s in texts]

    def predict(self, text, flag_prob=True, **kwargs):
        """
        基于混淆集纠错
        :param sentence: str, 待纠错的文本
        :return: dict, {'source': 'src', 'target': 'trg', 'errors': [(error_word, correct_word, position), ...]}
        """
        text_cor = copy.deepcopy(text)
        details = self.trietree.find_keyword(text)
        details_new = []
        for detail in details:
            err, pos_start, pos_end = detail[0], detail[1], detail[-1]
            truth = self.confusion_dict.get(err, "")
            text_cor = text_cor[:pos_start] + truth + text_cor[pos_end:]
            # details_new.append((err, truth, pos_start))
            # 如果混淆词典匹配上了, 就对比替换词典中不同的字符
            for idx, (err_i, truth_i) in enumerate(zip(err, truth)):
                if err_i != truth_i:
                    if flag_prob:
                        details_new.append([err_i, truth_i, pos_start+idx, 1.0])
                    else:
                        details_new.append([err_i, truth_i, pos_start+idx])
        return {"source": text, "target": text_cor, "errors": details_new}

    def load_confusion_dict(self, path):
        """
            加载默认混淆词典, 只加载一个
        :param path:
        :return: dict, {variant: origin}, eg: {"交通先行": "交通限行"}
        """
        path = os.path.join(path_root, "macro_correct", "output", "confusion_dict.json")
        with open(path, "r", encoding="utf-8") as fj:
            confusion_dict = json.load(fj)
            fj.close()
        return confusion_dict

    def load_user_dict(self, path=""):
        """
            加载自定义困惑集, 只加载一个
        :param path:
        :return: dict, {variant: origin}, eg: {"交通先行": "交通限行"}
        """
        with open(path, "r", encoding="utf-8") as fj:
            user_dict = json.load(fj)
            fj.close()
        return user_dict

    def create_trie_tree(self, keywords):
        """
            根据list关键词，初始化trie树
        :param keywords: list, input
        :return: objext, 返回实例化的trie
        """
        trie = TrieTree()
        trie.add_keywords_from_list(keywords)
        return trie


class TrieTree:
    """
        前缀树构建、新增关键词、关键词词语查找等
    """
    def __init__(self):
        self.root = TrieNode()

    def add_keyword_single(self, keyword):
        """
            新增一个关键词
        :param keyword: str,构建的关键词
        :return: None
        """
        node_curr = self.root
        for word in keyword:
            if not node_curr.child.get(word):
                node_next = TrieNode()
                node_curr.child[word] = node_next
            node_curr = node_curr.child[word]
        # 每个关键词词后边，加入end标志位
        if not node_curr.child.get("<END>"):
            node_next = TrieNode()
            node_curr.child["<END>"] = node_next
        node_curr = node_curr.child["<END>"]

    def add_keywords_from_list(self, keywords):
        """
            新增关键词s, 格式为list
        :param keyword: list, 构建的关键词
        :return: None
        """
        for keyword in keywords:
            self.add_keyword_single(keyword)

    def find_keyword(self, text):
        """
            从句子中提取关键词，取得大于2个的，例如有人名"大漠帝国"，那么"大漠帝"也取得
        :param sentence: str, 输入的句子
        :return: list, 提取到的关键词
        """
        if not text:
            return []
        node_curr = self.root # 关键词的第一位， 每次遍历完一个后重新初始化
        name_list = []
        name = ""
        for idx, word in enumerate(text):
            if not node_curr.child.get(word): # 查看有无后缀, 没有前缀
                if name: # 提取到的关键词(也可能是前面的几位)
                    if node_curr.child.get("<END>"): # 取以end结尾的关键词
                        name_list.append((name, idx-len(name), idx))
                    node_curr = self.root  # 重新初始化
                    if self.root.child.get(word):
                        name = word
                        node_curr = node_curr.child[word]
                        if idx == len(text)-1 and node_curr.child.get("<END>"):  # 单字最末尾的情况
                            name_list.append((name, idx-len(name)+1, idx+1))
                    else:
                        name = ""
            else: # 有缀就加到name里边
                name = name + word
                node_curr = node_curr.child[word]
                if idx == len(text)-1:  # 多字最末尾的情况
                    if node_curr.child.get("<END>"):
                        name_list.append((name, idx-len(name)+1, idx+1))
        return name_list


class TrieNode:
    """
        前缀树节点-链表
    """
    def __init__(self):
        self.child = {}


if __name__ == "__main__":
    yz = 0

    confusion_dict = {"馀": "余",
                      "摺": "折",
                      "莪": "我",
                      "祢": "你",
                      "一个分知": "一个分支",
                      "陌光": "阳光",
                      "受打去": "受打击",
                      "回聚": "汇聚",
                      "爱带": "爱戴",
                      "12305": "12306"
                      }
    model = ConfusionCorrect(confusion_dict=confusion_dict)
    sents = ["一个分知,陌光回聚,莪受打去,祢爱带馀", "一个分知,陌光回聚,莪受打去,祢爱带馀爱带"]
    res = model.predict_batch(sents)
    print(res)

    sent = "一个分知,陌光回聚,莪受打去,祢爱带馀"
    res = model.predict(sent)
    print(res)
    print(model.confusion_dict.get("12305"))
    yz = 0


"""
{'source': '一个分知,陌光回聚,莪受打去,祢爱带馀', 'target': '一个分支,阳光汇聚,我受打击,你爱戴余', 'errors': [('知', '支', 3), ('陌', '阳', 5), ('回', '汇', 7), ('莪', '我', 10), ('去', '击', 13), ('祢', '你', 15), ('带', '戴', 17), ('馀', '余', 18)]}
12306
"""


