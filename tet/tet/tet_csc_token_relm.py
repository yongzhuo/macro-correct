# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/29 21:41
# @author  : Mo
# @function: tet csc of punct, 标点符号纠错(支持新增/修改, 不支持删除)


import traceback
import time
import os

from macro_correct.pytorch_user_models.csc.relm.predict import RelmPredict


path_model_dir = "../../macro_correct/output/text_correction/relm_v1"
# path_model_dir = "Macropodus/relm_v1"

model = RelmPredict(path_model_dir, path_model_dir)
texts = [
    {"source": "真麻烦你了。希望你们好好的跳无"},
    {"source": "少先队员因该为老人让坐"},
    {"source": "机七学习是人工智能领遇最能体现智能的一个分知"},
    {"source": "一只小鱼船浮在平净的河面上"},
    {"source": "我的家乡是有明的渔米之乡"},
    {"source": "哪里有上大学，不想念书的道理？"},
    {"source": "从那里，我们可以走到纳福境的新光三钺百货公司逛一逛"},
    {"source": "他主动拉了姑娘的手，心里很高心，嘴上故作生气"},
    {"source": "我这一次写信给你是想跟你安排一下关以我们要见面的。"},
    {"source": "为了减少急遍的生孩子率，需要呼吁适当的生育政策。"},
    {"source": "他主动拉了姑娘的手，心里很高心，嘴上故作生气"},
    {"source": "这种千亿市值的板块龙头，昨天都尽然可以这么杀？"},
    {"source": "关于刑法的解释，下列说法挫误的是"},
    {"source": "购买伟造的增值税专用发票罪"},
    {"source": "下裂选项有关劳动争议仲裁的表述中，正确的有:"},
    {"source": "本票上有关票面金页、收款人名称以及出票日期的记载不得余改，否作该票无效"},
    {"source": "包销分为全额包销和余额包销两种形式"},
    { "source": "如果“教酸他人犯罪”中的“他人”必须达到刑事法定年龄，则教锁15周岁的人盗窃的，不能适用第29条第一款后半段的规定"},
    {"source": "从那里，我们可以走到纳福境的新光三钺百货公司逛一逛"},
    {"source": "哪里有上大学，不想念书的道理？"},
    {"source": "他主动拉了姑娘的手，心里很高心，嘴上故作生气"},
]

res = model.predict(texts)
for r in res:
    print(r)
# model.model.bert.save_pretrained("relm_csc", safe_serialization=False)

while 1:
    try:
        print("请输入：")
        text = input()
        text_in = [{"original_text": text}]
        time_start = time.time()
        res = model.predict(text_in)
        time_end = time.time()
        print(res)
        print(time_end - time_start)
    except Exception as e:
        print(traceback.print_exc())


