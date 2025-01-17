# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/12/21 17:35
# @author  : Mo
# @function: 推理预测


from logging.handlers import RotatingFileHandler
import traceback
import requests
import logging
import pathlib
import random
import base64
import time
import uuid
import json
import copy
import sys
import os

path_sys = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(path_sys)
print(path_sys)

from sanic.response import json
from sanic.log import logger
from sanic import Sanic

from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


checkpoint = "shibing624/chinese-text-correction-1.5b"
device = "cuda"  # for GPU usage or "cpu" for CPU usage

app = Sanic("ecal_csc_eval_csc_shibing624_chinese-text-correction-1_5b")


def get_logger(log_dir: str, back_count: int=32, logger_name: str="ecal_csc"):
    """
    get_current_time from time
    Args:
        log_dir: str, log dir of path
        back_count: int, max file-name
        logger_name: str
    Returns:
        logger: class
    """

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    # 日志文件名,为启动时的日期
    log_file_name = time.strftime("{}-%Y-%m-%d".format(logger_name), time.localtime(time.time())) + ".log"
    log_name_day = os.path.join(log_dir, log_file_name)
    logger_level = logging.INFO
    # log目录地址
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    # 全局日志格式
    logging.basicConfig(format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s", level=logger_level)
    # 定义一个日志记录器
    logger = logging.getLogger("ecal_csc")
    # 文件输出, 定义一个RotatingFileHandler，最多备份32个日志文件，每个日志文件最大32K
    fHandler = RotatingFileHandler(log_name_day, maxBytes=back_count * 1024 * 1024 * 1024, backupCount=back_count, encoding="utf-8")
    fHandler.setLevel(logger_level)
    logger.addHandler(fHandler)
    # 控制台输出
    console = logging.StreamHandler()
    console.setLevel(logger_level)
    logger.addHandler(console)
    return logger


def tet_predict():
    """   测试一个   """
    # pip install transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" # for GPU usage or "cpu" for CPU usage
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    input_content = "文本纠错：\n少先队员因该为老人让坐。"

    messages = [{"role": "user", "content": input_content}]
    input_text=tokenizer.apply_chat_template(messages, tokenize=False)

    print(input_text)

    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=1024, temperature=0, do_sample=False, repetition_penalty=1.08)

    print(tokenizer.decode(outputs[0]))

    while True:
        try:
            print("请输入:")
            question = input()
            #res = func_csc_punct(question)
            messages = [{"role": "user", "content": question}]
            input_text=tokenizer.apply_chat_template(messages, tokenize=False)
            print(input_text)
            inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
            outputs = model.generate(inputs, max_new_tokens=128, temperature=0, do_sample=False, repetition_penalty=1.08)
            print(outputs)
        except Exception as e:
            print(traceback.print_exc())


def csc_predict(query):
    """   推理  """
    input_content = "文本纠错：\n" + query.strip()
    messages = [{"role": "user", "content": input_content}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model_csc.generate(inputs, max_new_tokens=256, temperature=0,
                                 do_sample=False, repetition_penalty=1.08)
    query_correct = tokenizer.decode(outputs[0])
    query_correct_2 = query_correct.replace(input_text, "").replace("<|im_start|>user", "")\
        .replace("<|im_start|>assisstant", "").replace("<|im_start|>", "").replace("<|im_end|>", "")

    # print(input_text)
    # print(tokenizer.decode(outputs[0]))
    # prompt_len = len(inputs.get("input_ids")[0])
    # query_correct = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    # generated_sequence = generated_sequence[prompt_len:]
    # gen_text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
    # gen_text = gen_text.strip()
    return query_correct.strip(), query_correct_2.strip()


@app.route("/nlg/llm/eval_csc", methods=["POST"])
def tet_eval_csc(request):
    """   模型推理等
    """
    if "application/json" in request.content_type:
        data = request.json
    else:
        data = request.form
    try:
        query = data.get("query", "")
        res_1, res_2 = csc_predict(query)
        result = {"code": 200, "data": {"csc_1": res_1, "csc_2": res_2}, "message": "success"}
        # res = csc_predict(query)
        # result = {"code": 200, "data": res, "message": "success"}
    except Exception as e:
        logger.info(traceback.print_exc())
        result = {"code": 500, "data": {}, "message": "fail"}
    return json(result)


tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model_csc = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
input_content = "文本纠错：\n少先队员因该为老人让坐。"
messages = [{"role": "user", "content": input_content}]
input_text=tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model_csc.generate(inputs, max_new_tokens=512, temperature=0, do_sample=False, repetition_penalty=1.08)
print(input_text)
print(tokenizer.decode(outputs[0]))



if __name__ == "__main__":
    ### sanic app

    app.run(host="0.0.0.0",
            port=8036,
            workers=3,
            access_log=True,
            debug=True)

"""
# shell
# nohup python eval_csc_shibing624_chinese-text-correction-1_5b_sanic_start.py > tc.eval_csc_shibing624_chinese-text-correction-1_5b_sanic_start.py.log 2>&1 &
# tail -n 1000  -f tc.eval_csc_shibing624_chinese-text-correction-1_5b_sanic_start.py.log
# |myz|
"""
