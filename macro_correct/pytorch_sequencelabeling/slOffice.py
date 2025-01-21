# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/11/17 21:36
# @author  : Mo
# @function: office of transformers


import logging as logger
import numpy as np
import traceback
import platform
import random
import codecs
import copy
import json
import os

from transformers import AdamW, get_linear_schedule_with_warmup, get_scheduler
from torch.utils.data import DataLoader
import torch

from macro_correct.pytorch_sequencelabeling.slConfig import _SL_MODEL_SOFTMAX, _SL_MODEL_GRID, _SL_MODEL_SPAN, _SL_MODEL_CRF, model_config
from macro_correct.pytorch_sequencelabeling.slTools import padding_1d_2d_3d_4d_dim, chinese_extract_extend, get_logger
from macro_correct.pytorch_sequencelabeling.slTools import get_pos_from_common, get_pos_from_span, sigmoid, softmax
from macro_correct.pytorch_sequencelabeling.slTools import mertics_report_sequence_labeling
from macro_correct.pytorch_sequencelabeling.slData import SeqLabelingDataCollator
from macro_correct.pytorch_sequencelabeling.slData import SeqlabelingDataset
from macro_correct.pytorch_sequencelabeling.slData import CreateDataLoader
from macro_correct.pytorch_sequencelabeling.slTqdm import tqdm, trange
from macro_correct.pytorch_sequencelabeling.slAdversarial import FGM
from macro_correct.pytorch_sequencelabeling.slGraph import Graph


class Office:
    def __init__(self, config, tokenizer, logger=logger):
        """
        初始化主训练器/算法网络架构/数据集, init Trainer
        config:
            config: json, params of graph, eg. {"num_labels":17, "model_type":"BERT"}
            train_corpus: List, train corpus of dataset
            dev_corpus: List, dev corpus of dataset 
            tet_corpus: List, tet corpus of dataset 
        Returns:
            None
        """
        self.tokenizer = tokenizer
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() and config.flag_cuda else "cpu"
        self.set_random_seed(config.seed)  # 初始化随机种子
        self.model = Graph(config, tokenizer, logger).to(self.device)  # 初始化模型网络架构
        self.logger = logger
        self.logger.info("config:")
        self.logger.info(config)
        self.logger.info("config_detail:")
        for config_name in dir(config):
            if not config_name.startswith("_"):  # 排除内部变量
                self.logger.info("{}: {}".format(config_name, getattr(config, config_name)))

    def save_onnx(self, path_dir="", opset_version=10):
        """  存储为ONNX格式  """
        # self.load_model_state()
        ### ONNX---路径
        if not path_dir:
            path_dir = os.path.join(self.config.model_save_path, "onnx")
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        batch_data = [[[1, 2, 3, 4] * 32] * 32, [[1, 0] * 64] * 32, [[0, 1] * 64] * 32]

        # for name, param in self.model.named_parameters():  # 查看可优化的参数有哪些
        #     # if param.requires_grad:
        #         self.logger.info(name)
        # ee = 0
        # batch_data = [bd.to(self.device) for bd in batch_data]  # device
        with torch.no_grad():
            inputs = {"attention_mask": torch.tensor(batch_data[1]).to(self.device),
                      "token_type_ids": torch.tensor(batch_data[2]).to(self.device),
                      "input_ids": torch.tensor(batch_data[0]).to(self.device),
                      }
            output = self.model(**inputs)
            # loss, logits = output[:2]

            input_names = ["input_ids", "attention_mask", "token_type_ids"]
            output_names = ["outputs"]
            torch.onnx.export(self.model,
                              (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
                              os.path.join(path_dir, "tc_model.onnx"),
                              input_names=input_names,
                              output_names=output_names,  ## Be carefule to write this names
                              opset_version=opset_version,  # 9,
                              do_constant_folding=True,
                              flag_external_data_format=True,
                              dynamic_axes={
                                  "input_ids": {0: "batch", 1: "sequence"},
                                  "attention_mask": {0: "batch", 1: "sequence"},
                                  "token_type_ids": {0: "batch", 1: "sequence"},
                                  output_names[0]: {0: "batch"}
                              }
                              )

    def load_model_state(self, path_dir="", epoch=0, step=0):
        """  仅加载模型参数(推荐使用)  """
        try:
            if path_dir:
                path_model = path_dir
            else:
                path_model = os.path.join(self.config.model_save_path, self.config.model_name)
            model_params = torch.load(path_model, map_location=torch.device(self.device))
            model_params = {k.replace("pretrain_model.", "bert."): v for k, v in model_params.items()}
            self.model.load_state_dict(model_params, strict=False)
            self.model.to(self.device)
            self.logger.info("******model loaded success******")
            self.logger.info("self.device: {}".format(self.device))
        except Exception as e:
            self.logger.info(traceback.print_exc())
            self.logger.info(str(e))
            raise Exception("******load model error******")

    def save_model_state(self, path_dir=""):
        """  仅保存模型参数(推荐使用)  """
        if path_dir:
            model_save_path = path_dir
        else:
            model_save_path = self.config.model_save_path
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        # save bert.config
        self.model.pretrained_config.save_pretrained(save_directory=self.config.model_save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(save_directory=self.config.model_save_path)
        # save config
        self.config.flag_train = False  # 存储后的用于预测
        path_config = os.path.join(self.config.model_save_path, self.config.config_name)
        with codecs.open(filename=path_config, mode="w", encoding="utf-8") as fc:
            self.config.flag_train = False
            self.config.num_workers = 0
            json.dump(vars(self.config), fc, indent=4, ensure_ascii=False)
            fc.close()
        # save model
        path_model = os.path.join(self.config.model_save_path, self.config.model_name)
        torch.save(self.model.state_dict(), path_model)
        self.logger.info("******model_save_path is {}******".format(path_model))
        
    def load_model(self, path_dir=""):
        """  加载模型  """
        try:
            if path_dir:
                path_model = path_dir
            else:
                path_model = os.path.join(self.config.model_save_path, self.config.model_name)
            # self.device = "cpu"
            self.model = torch.load(path_model, map_location=torch.device(self.device))  # , map_location=torch.device(self.device))
            # =torch.device('cpu')
            # self.model.to(self.device)
            self.logger.info("******model loaded success******")
        except Exception as e:
            self.logger.info(traceback.print_exc())
            raise Exception("******load model error******")

    def save_model(self, path_dir=""):
        """  存储模型  """
        if path_dir:
            self.config.model_save_path = path_dir
        if not os.path.exists(self.config.model_save_path):
            os.makedirs(self.config.model_save_path)
        # save bert.config
        self.model.pretrained_config.save_pretrained(save_directory=self.config.model_save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(save_directory=self.config.model_save_path)
        # save sl.config
        self.config.flag_train = False  # 存储后的用于预测
        path_config = os.path.join(self.config.model_save_path, self.config.config_name)
        with codecs.open(filename=path_config, mode="w", encoding="utf-8") as fc:
            self.config.flag_train = False
            self.config.num_workers = 0
            json.dump(vars(self.config), fc, indent=4, ensure_ascii=False)
            fc.close()
        # save pytorch_model.bin
        path_model = os.path.join(self.config.model_save_path, self.config.model_name)
        torch.save(self.model, path_model)
        self.logger.info("******model_save_path is {}******".format(path_model))

    def load_tokenizer(self, config):
        """   加载tokenizer   """
        sl_collate_fn =SeqLabelingDataCollator(config=config)
        self.tokenizer = sl_collate_fn.tokenizer
        return self.tokenizer

    def set_random_seed(self, seed):
        """
        初始化随机种子, init seed
        config:
            seed: int, seed of all, eg. 2021
        Returns:
            None
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if self.config.flag_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def predict(self, data_loader, logits_type="softmax", rounded=4, **kwargs):
        """
        预测, pred
        config:
            data_loader: tensor, eg. tensor(1,2,3)
        Returns:
            ys_prob: list<json>
        """
        ys_pred_id = None
        texts = []
        #  预测 batch-size
        self.model.eval()
        for batch_data in data_loader:
            batch_text = batch_data[-1]
            texts.extend(batch_text)
            batch_data = [bd.to(self.device) for bd in batch_data[:-1]]  # device
            headers = ["input_ids", "attention_mask", "token_type_ids", "labels", "labels_start", "labels_end"]  # 必须按照顺序
            inputs = dict(zip(headers[:-3], batch_data))
            with torch.no_grad():
                output = self.model(**inputs)
                loss, logits, logits_org = output[:3]
            # 获取numpy格式label/logits
            logits_org_numpy = logits_org.detach().cpu().numpy()
            logits_numpy = logits.detach().cpu().numpy()
            # 后处理, 转化为类似概率的形式
            if logits_type.upper() == "SIGMOID" and self.config.task_type.upper() != _SL_MODEL_CRF:
                logits_org_numpy = sigmoid(logits_org_numpy)
                logits_numpy = softmax(logits_numpy)
            elif logits_type.upper() == "SOFTMAX" and self.config.task_type.upper() != _SL_MODEL_CRF:
                logits_org_numpy = softmax(logits_org_numpy)
                logits_numpy = softmax(logits_numpy)
            # 追加, extend
            if ys_pred_id is not None:
                ys_prob_id = np.append(ys_prob_id, logits_org_numpy, axis=0)
                ys_pred_id = np.append(ys_pred_id, logits_numpy, axis=0)
            else:
                ys_prob_id = logits_org_numpy
                ys_pred_id = logits_numpy
        res_myz_pred = []
        # 最后输出, top-1
        if self.config.task_type.upper() in [_SL_MODEL_CRF]:
            ys_pred_id_argmax = ys_pred_id.tolist()
            ys_prob_id = softmax(ys_prob_id)
            # ys_prob_id = sigmoid(ys_prob_id)
            ys_prob_id = ys_prob_id.tolist()
            # count = 0
            for idx, (x, z) in enumerate(zip(ys_pred_id_argmax, texts[:len(ys_pred_id_argmax)])):
                len_text = len(z)
                if self.config.padding_side.upper() == "LEFT":  # 以实际的len_text来剔除左右padding
                    x1 = [self.config.i2l[str(int(xi))] for xi in x[1:-1]][-len_text:]
                else:
                    x1 = [self.config.i2l[str(int(xi))] for xi in x[1:-1]][:len_text]
                pos_pred = get_pos_from_common(z, x1)
                for jdx, pos_i in enumerate(pos_pred):
                    pos_i_ij = pos_i.get("pos", [])
                    # score = sum([max(ys_prob_id[idx][ijk]) for ijk in range(pos_i_ij[0], pos_i_ij[1] + 1)]) / (pos_i_ij[1]+1-pos_i_ij[0])
                    score = ys_prob_id[idx][pos_i_ij[0]][x[pos_i_ij[0]]]
                    score = round(float(score), rounded)
                    pos_pred[jdx]["pos"] = [pos_i_ij[0], pos_i_ij[1]]
                    pos_pred[jdx]["score"] = score
                res_myz_pred += [{"label": pos_pred, "text": z}]
                # if count < 5:
                #     self.logger.info("y_pred:{}".format(x1))
                #     self.logger.info("pos_pred:{}".format(pos_pred))
                # count += 1
        elif self.config.task_type.upper() in [_SL_MODEL_SOFTMAX]:
            ys_pred_id_argmax = np.argmax(ys_pred_id, axis=-1)
            ys_pred_id_argmax = ys_pred_id_argmax.tolist()
            ys_pred_id = ys_pred_id.tolist()
            # count = 0
            for idx, (x, z) in enumerate(zip(ys_pred_id_argmax, texts[:len(ys_pred_id_argmax)])):
                # len_text = len(z)
                # x1 = [self.config.i2l[str(int(xi))] for xi in x][:len_text]
                len_text = len(z)
                if self.config.padding_side.upper() == "LEFT":  # 以实际的len_text来剔除左右padding
                    x1 = [self.config.i2l[str(int(xi))] for xi in x[1:-1]][-len_text:]
                else:
                    x1 = [self.config.i2l[str(int(xi))] for xi in x[1:-1]][:len_text]
                pos_pred = get_pos_from_common(z, x1)
                for jdx, pos_i in enumerate(pos_pred):
                    pos_i_ij = pos_i.get("pos", [])
                    score = sum([max(ys_pred_id[idx][ijk]) for ijk in range(pos_i_ij[0], pos_i_ij[1] + 1)]
                                ) / (pos_i_ij[1] + 1 - pos_i_ij[0])
                    score = round(float(score), rounded)
                    pos_pred[jdx]["pos"] = [pos_i_ij[0], pos_i_ij[1]]
                    pos_pred[jdx]["score"] = score
                res_myz_pred += [{"label": pos_pred, "text": z}]
                # if count < 5:
                #     self.logger.info("y_pred:{}".format(x1))
                #     self.logger.info("pos_pred:{}".format(pos_pred))
                # count += 1
        elif self.config.task_type.upper() in [_SL_MODEL_SPAN]:
            # count = 0
            for idx, (logits, text_i) in enumerate(zip(ys_pred_id, texts[:len(ys_pred_id)])):
                len_text = len(text_i)
                len_logits = logits.shape[0]
                logits_start = logits[:int(len_logits / 2)][1:-1][:len_text]
                logits_end = logits[int(len_logits / 2):][1:-1][:len_text]
                pos_logits = get_pos_from_span(logits_start.tolist(), logits_end.tolist(), self.config.i2l)
                pos_true, pos_pred = [], []
                for ps_i in pos_logits:
                    pos_start = ps_i[1]
                    pos_end = ps_i[2]
                    score = ps_i[3]
                    score = round(float(score), rounded)
                    pos_pred.append({"type": ps_i[0], "pos": [pos_start, pos_end],
                                     "ent": text_i[pos_start:pos_end + 1], "score": score})
                res_myz_pred += [{"label": pos_pred, "text": text_i}]
                # if count < 5:
                #     self.logger.info("pos_pred:{}".format(pos_pred))
                # count += 1
        elif self.config.task_type.upper() in [_SL_MODEL_GRID]:
            # count = 0
            for idx, (logits, text_i) in enumerate(zip(ys_pred_id, texts[:len(ys_pred_id)])):
                # for logits, text_i in zip(ys_pred_id, texts[:len(ys_pred_id)]):
                pos_pred = []
                logits = np.array(logits)
                # logits[:, [0, -1]] -= np.inf
                # logits[:, :, [0, -1]] -= np.inf
                len_text = len(text_i)
                if self.config.padding_side.upper() == "LEFT":
                    logits = logits[:, -len_text - 1:-1, :]
                    logits = logits[:, :, -len_text - 1:-1]
                else:
                    logits = logits[:, 1:len_text + 1, :]
                    logits = logits[:, :, 1:len_text + 1]
                for pos_type, pos_start, pos_end in zip(*np.where(logits > self.config.grid_pointer_threshold)):
                    # pos_start, pos_end = pos_start - 1, pos_end - 1
                    pos_type = self.config.i2l.get(str(int(pos_type)), "")
                    if pos_type != "O":
                        score = sum([ys_pred_id[idx][ijk] for ijk in range(pos_start, pos_end + 1)]
                                    ) / (pos_end + 1 - pos_start)
                        score = round(float(score), rounded)
                        line = {"type": pos_type, "pos": [pos_start, pos_end],
                                "ent": text_i[pos_start:pos_end + 1], "score": score}
                        pos_pred.append(line)
                res_myz_pred += [{"label": pos_pred, "text": text_i}]
                # if count < 5:
                #     self.logger.info("pos_pred:{}".format(pos_pred))
                # count += 1
        return res_myz_pred

    def evaluate(self, data_loader, flag_load_model=False):
        """
        验证, evalate
        config:
            data_loader: class, dataset-loader of dev or eval, eg. train_data_loader
            flag_load_model: bool, reload newest model or not, default=False, eg. true, false
        Returns:
            result: json
        """
        ### 加载效果最好的模型(训练完模型的最后阶段用)
        if flag_load_model:
            if self.config.flag_save_model_state:
                self.load_model_state()
            else:
                self.load_model()
        model_graph_config_flag_train = copy.deepcopy(self.model.graph_config.flag_train)
        self.model.graph_config.flag_train = False  # flag_train选择导致输入输出不一致
        ys_pred_id, ys_true_id = None, None
        eval_loss = 0.0
        eval_steps = 0
        # 验证
        texts = []
        for batch_data in tqdm(data_loader, desc="evaluate"):
            batch_text = batch_data[-1]
            texts.extend(batch_text)
            batch_data = [bd.to(self.device) for bd in batch_data[:-1]]  # device
            # warning: 必须按照该顺序zip
            # SPAN:  tensor_input, tensor_attention_mask, tensor_token_type, tensor_start, tensor_end
            # CRF-SOFTMAX-GRID:  tensor_input, tensor_attention_mask, tensor_token_type, tensor_label
            headers = ["input_ids", "attention_mask", "token_type_ids", "labels", "labels_start", "labels_end"]  # 必须按照顺序
            if self.config.task_type.upper() in [_SL_MODEL_SOFTMAX, _SL_MODEL_CRF, _SL_MODEL_GRID]:  # CRF or softmax
                inputs = dict(zip(headers[:-2], batch_data))
            elif self.config.task_type.upper() in [_SL_MODEL_SPAN]:  # SPAN
                inputs = dict(zip(headers[:-3] + headers[-2:], batch_data))
            else:
                raise ValueError("invalid data_loader, length of batch_data must be 4 or 6!")
            with torch.no_grad():
                output = self.model(**inputs)
                loss, logits = output[:2]
                eval_loss += loss.item()
                eval_steps += 1
            # 获取numpy格式label/logits
            if self.config.task_type.upper() in [_SL_MODEL_SPAN]:
                labels_start = inputs.get("labels_start").detach().cpu().numpy()
                labels_end = inputs.get("labels_end").detach().cpu().numpy()
                logits_numpy = logits.detach().cpu().numpy()
                inputs_numpy = np.concatenate([labels_start, labels_end], axis=1)
                # self.logger.info(self.config.max_len)
                # self.logger.info(labels_start.shape)
                # self.logger.info(labels_end.shape)
                ### 动态encode的时候, 左右padding, 用于适配np.append, np.append需要相同的维度
                if self.config.flag_dynamic_encode:
                    pad_len = self.config.max_len - labels_start.shape[1]  # <32, 128>, eval需要padding到最大长度, 便于比较
                    if pad_len > 0:
                        pad_side = self.config.padding_side
                        labels_start = padding_1d_2d_3d_4d_dim(labels_start, pad_len=pad_len, pad_side=pad_side, mode="constant", value=0)
                        labels_end = padding_1d_2d_3d_4d_dim(labels_end, pad_len=pad_len, pad_side=pad_side, mode="constant", value=0)
                        inputs_numpy = np.concatenate([labels_start, labels_end], axis=1)
                        len_logits = logits.shape[1]
                        logits_start = logits_numpy[:, int(len_logits / 2):, :]
                        logits_end = logits_numpy[:, :int(len_logits / 2), :]
                        logits_start = padding_1d_2d_3d_4d_dim(logits_start, pad_len=pad_len, pad_side=pad_side, mode="constant", value=0)
                        logits_end = padding_1d_2d_3d_4d_dim(logits_end, pad_len=pad_len, pad_side=pad_side, mode="constant", value=0)
                        logits_numpy = np.concatenate([logits_start, logits_end], axis=1)
            else:
                inputs_numpy = inputs.get("labels").detach().cpu().numpy()
                logits_numpy = logits.detach().cpu().numpy()
                ### 动态encode的时候, 左右padding, 用于适配np.append, np.append需要相同的维度
                if self.config.flag_dynamic_encode and self.config.task_type.upper() in [_SL_MODEL_SOFTMAX, _SL_MODEL_CRF, _SL_MODEL_GRID]:
                    pad_len = self.config.max_len - inputs_numpy.shape[1]
                    if self.config.task_type.upper() in [_SL_MODEL_GRID]:  # sl-grid/people-ner, <32, 3, 128, 128>
                        pad_len = self.config.max_len - inputs_numpy.shape[-1]
                    if pad_len > 0:
                        pad_side = self.config.padding_side
                        inputs_numpy = padding_1d_2d_3d_4d_dim(inputs_numpy, pad_len=pad_len, pad_side=pad_side, mode="constant", value=0)
                        logits_numpy = padding_1d_2d_3d_4d_dim(logits_numpy, pad_len=pad_len, pad_side=pad_side, mode="constant", value=0)
                        myz = 0
            # 追加, extend
            if ys_pred_id is not None:
                ys_pred_id = np.append(ys_pred_id, logits_numpy, axis=0)
                ys_true_id = np.append(ys_true_id, inputs_numpy, axis=0)
            else:
                ys_pred_id = logits_numpy
                ys_true_id = inputs_numpy
        eval_loss = eval_loss / eval_steps
        # 最后输出, top-1
        if self.config.task_type.upper() in [_SL_MODEL_SOFTMAX, _SL_MODEL_CRF]:
            # ys_pred_id = np.argmax(ys_pred_id, axis=-1) if self.config.task_type.upper() in [_SL_MODEL_SOFTMAX] else ys_pred_id
            ys_true_id = ys_true_id.tolist()
            ys_pred_id = ys_pred_id.tolist()
            res_myz_true = []
            res_myz_pred = []
            count = 0
            for x, y, z in zip(ys_pred_id, ys_true_id, texts[:len(ys_pred_id)]):
                len_text = len(z)
                if self.config.padding_side.upper() == "LEFT":
                    x1 = [self.config.i2l[str(int(xi))] for xi in x[1:-1]][-len_text:]
                    y1 = [self.config.i2l[str(int(yi))] for yi in y[1:-1]][-len_text:]
                else:
                    x1 = [self.config.i2l[str(int(xi))] for xi in x[1:-1]][:len_text]
                    y1 = [self.config.i2l[str(int(yi))] for yi in y[1:-1]][:len_text]
                pos_pred = get_pos_from_common(z, x1)
                pos_true = get_pos_from_common(z, y1)
                res_myz_pred += [{"text": z, "label": pos_pred}]
                res_myz_true += [{"text": z, "label": pos_true}]
                if count < 5:
                    self.logger.info("y_pred:{}".format(x1))
                    self.logger.info("y_true:{}".format(y1))
                    self.logger.info("pos_pred:{}".format(pos_pred))
                    self.logger.info("pos_true:{}".format(pos_true))
                count += 1
            i2l = {}
            label_set = set()
            for i, lab in self.config.i2l.items():
                lab_1 = lab.split("-")[-1]
                if lab_1 not in label_set:
                    i2l[str(len(i2l))] = lab_1
                    label_set.add(lab_1)
        elif self.config.task_type.upper() in [_SL_MODEL_SPAN]:
            res_myz_true = []
            res_myz_pred = []
            count = 0
            for logits, labels, text_i in zip(ys_pred_id, ys_true_id, texts[:len(ys_pred_id)]):
                len_text = len(text_i)
                len_logits = logits.shape[0]
                if self.config.padding_side.upper() == "LEFT":
                    logits_start = logits[:int(len_logits / 2), :][1:-1][-len_text:]  # <max_len, num_labels>, like <256, 4>
                    logits_end = logits[int(len_logits / 2):, :][1:-1][-len_text:]
                    labels_start = labels[:int(len_logits / 2)][1:-1][-len_text:]  # <max_len>, like <256>
                    labels_end = labels[int(len_logits / 2):][1:-1][-len_text:]
                else:
                    logits_start = logits[:int(len_logits / 2), :][1:-1][:len_text]
                    logits_end = logits[int(len_logits / 2):, :][1:-1][:len_text]
                    labels_start = labels[:int(len_logits / 2)][1:-1][:len_text]
                    labels_end = labels[int(len_logits / 2):][1:-1][:len_text]
                pos_logits = get_pos_from_span(logits_start.tolist(), logits_end.tolist(), self.config.i2l)
                pos_true, pos_pred = [], []
                for ps_i in pos_logits:
                    pos_pred.append({"type": ps_i[0], "pos": [ps_i[1], ps_i[2]], "ent": text_i[ps_i[1]:ps_i[2]+1]})
                pos_label = get_pos_from_span(labels_start.tolist(), labels_end.tolist(), self.config.i2l, flag_index=True)
                for ps_i in pos_label:
                    pos_true.append({"type": ps_i[0], "pos": [ps_i[1], ps_i[2]], "ent": text_i[ps_i[1]:ps_i[2]+1]})
                res_myz_pred += [{"text": text_i, "label": pos_pred}]
                res_myz_true += [{"text": text_i, "label": pos_true}]
                if count < 5:
                    self.logger.info("pos_pred:{}".format(pos_pred))
                    self.logger.info("pos_true:{}".format(pos_true))
                count += 1
            i2l = self.config.i2l
        elif self.config.task_type.upper() in [_SL_MODEL_GRID]:
            ys_true_id = ys_true_id.tolist()
            ys_pred_id = ys_pred_id.tolist()
            res_myz_true = []
            res_myz_pred = []
            count = 0
            for x, y, z in zip(ys_pred_id, ys_true_id, texts[:len(ys_pred_id)]):
                x, y = np.array(x), np.array(y)
                pos_pred, pos_true = [], []
                len_text = len(z)
                if self.config.padding_side.upper() == "LEFT":
                    x = x[:, -len_text-1:-1, :]
                    x = x[:, :, -len_text-1:-1]
                    y = y[:, -len_text-1:-1, :]
                    y = y[:, :, -len_text-1:-1]
                else:
                    x = x[:, 1:len_text+1, :]
                    x = x[:, :, 1:len_text+1]
                    y = y[:, 1:len_text + 1, :]
                    y = y[:, :, 1:len_text + 1]

                # x[:, [0, -1]] -= np.inf
                # x[:, :, [0, -1]] -= np.inf
                # y[:, [0, -1]] -= np.inf
                # y[:, :, [0, -1]] -= np.inf
                for pos_type, pos_start, pos_end in zip(*np.where(x > self.config.grid_pointer_threshold)):
                    # pos_start, pos_end = pos_start-1, pos_end-1
                    pos_type = self.config.i2l[str(int(pos_type))]
                    if pos_type != "O":
                        line = {"type": pos_type, "pos": [pos_start, pos_end], "ent": z[pos_start:pos_end+1]}
                        pos_pred.append(line)
                for pos_type, pos_start, pos_end in zip(*np.where(y > self.config.grid_pointer_threshold)):
                    # pos_start, pos_end = pos_start-1, pos_end-1
                    pos_type = self.config.i2l[str(int(pos_type))]
                    if pos_type != "O":
                        line = {"type": pos_type, "pos": [pos_start, pos_end], "ent": z[pos_start:pos_end+1]}
                        pos_true.append(line)
                res_myz_pred += [{"text": z, "label": pos_pred}]
                res_myz_true += [{"text": z, "label": pos_true}]
                if count < 5:
                    # self.logger.info("y_pred:{}".format(x))
                    # self.logger.info("y_true:{}".format(y))
                    self.logger.info("pos_pred:{}".format(pos_pred[:5]))
                    self.logger.info("pos_true:{}".format(pos_true[:5]))
                count += 1
            i2l = self.config.i2l
        # 评估
        self.logger.info("i2l:{}".format(i2l))
        mertics_dict, mertics_report, mcm_report, y_error_dict = mertics_report_sequence_labeling(res_myz_true, res_myz_pred, i2l)
        self.logger.info(y_error_dict[:5])
        # self.logger.info(res_myz_pred[:5])
        self.logger.info("confusion_matrix:\n " + mcm_report)
        self.logger.info("mertics: \n" + mertics_report)
        self.logger.info("model_save_path is: {}".format(self.config.model_save_path))
        result = {"loss": eval_loss}
        result.update(mertics_dict)
        self.model.graph_config.flag_train = model_graph_config_flag_train  # 恢复初始的状态
        return result, mertics_report

    def train(self, d1, d2):
        """  训练迭代epoch  
             return global_steps, best_mertics
        """
        from tensorboardX import SummaryWriter

        #  配置好优化器与训练工作计划(主要是学习率预热warm-up与衰减weight-decay, 以及不作用的层超参)
        no_decay = ["LayerNorm.weight", "bias"]
        pretrain_params = list(self.model.bert.named_parameters())
        if self.config.task_type.upper() in [_SL_MODEL_SPAN]:
            fc_params = list(self.model.fc_span_start.named_parameters()) + list(self.model.fc_span_end.named_parameters())
        elif self.config.task_type.upper() in [_SL_MODEL_SOFTMAX]:
            fc_params = list(self.model.fc.named_parameters())
        else:
            fc_params = list(self.model.layer_crf.named_parameters())
        parameters_no_decay = [
            {"params": [p for n, p in pretrain_params if not any(nd in n for nd in no_decay)],
             "weight_decay": self.config.weight_decay, "lr": self.config.lr},
            {"params": [p for n, p in pretrain_params if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, "lr": self.config.lr},

            {"params": [p for n, p in fc_params if not any(nd in n for nd in no_decay)],
             "weight_decay": self.config.weight_decay, "lr": self.config.dense_lr},
            {"params": [p for n, p in fc_params if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0, "lr": self.config.dense_lr}
        ]

        optimizer = AdamW(parameters_no_decay, lr=self.config.lr, eps=self.config.adam_eps)
        # 训练轮次
        times_batch_size = len(d1) // self.config.grad_accum_steps
        num_training_steps = int(times_batch_size * self.config.epochs)
        # 如果选择-1不设置则为 1/10个epoch
        num_warmup_steps = int((len(d1) // self.config.grad_accum_steps // 10)) if self.config.warmup_steps == -1 else self.config.warmup_steps
        scheduler = get_scheduler(optimizer=optimizer, name=self.config.scheduler_name, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        tensorboardx_witer = SummaryWriter(logdir=self.config.model_save_path)

        # adv
        if self.config.flag_adv:
            fgm = FGM(self.model, emb_name=self.config.adv_emb_name, epsilon=self.config.adv_eps)
        # 开始训练
        epochs_store = []
        global_steps = 0
        best_mertics = {}
        best_report = ""
        for epochs_i in trange(self.config.epochs, desc="epoch"):  # epoch
            # ### 先召回, 然后精确
            # if epochs_i < (self.config.epochs/2):
            #     self.model.loss_type = "PRIOR_MARGIN_LOSS"
            # else:
            #     self.model.loss_type = "FOCAL_LOSS"
            # ### 召回/精确 交替进行
            # if epochs_i % 2 == 0:
            #     self.model.loss_type = "PRIOR_MARGIN_LOSS"
            # else:
            #     self.model.loss_type = "FOCAL_LOSS"

            for idx, batch_data in enumerate(tqdm(d1, desc="step")):  # step
                # 数据与模型, 获取输入的json
                batch_text = batch_data[-1]
                batch_data = [bd.to(self.device) for bd in batch_data[:-1]]  # device
                # warning: 必须按照该顺序zip
                # SPAN:  tensor_input, tensor_attention_mask, tensor_token_type, tensor_start, tensor_end
                # CRF-SOFTMAX-GRID:  tensor_input, tensor_attention_mask, tensor_token_type, tensor_label
                headers = ["input_ids", "attention_mask", "token_type_ids", "labels", "labels_start", "labels_end"]
                if self.config.task_type.upper() in [_SL_MODEL_SOFTMAX, _SL_MODEL_CRF, _SL_MODEL_GRID]:  # CRF or softmax
                    inputs = dict(zip(headers[:-2], batch_data))
                elif self.config.task_type.upper() in [_SL_MODEL_SPAN]:  # SPAN
                    inputs = dict(zip(headers[:-3] + headers[-2:], batch_data))
                else:
                    raise ValueError("invalid data_loader, length of batch_data must be 4 or 6!")
                # model
                outputs = self.model(**inputs)
                loss = outputs[0] / self.config.grad_accum_steps
                loss.backward()
                global_steps += 1
                # self.logger.info(loss)
                #  对抗训练
                if self.config.flag_adv:
                    fgm.attack()  # 在embedding上添加对抗扰动
                    outputs = self.model(**inputs)
                    loss = outputs[0] / self.config.grad_accum_steps  # 梯度累计
                    loss.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    fgm.restore()  # 恢复embedding参数
                #  梯度累计
                if (idx + 1) % self.config.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                # 至少跑完一个epoch, 评估算法/打印日志/存储模型, 1个epoch/到达保存的步数/或者是最后一轮最后一步
                # if epochs_i > 0 and (self.config.evaluate_steps > 0
                if (self.config.evaluate_steps > 0 and global_steps % self.config.evaluate_steps == 0)\
                        or (idx == times_batch_size-1) \
                        or (epochs_i+1 == self.config.epochs and idx+1 == len(d1)):
                    # 评估, is_train要置为False
                    # self.model.graph_config.flag_train = False
                    self.model.eval()
                    res, report = self.evaluate(d2)
                    self.logger.info("epoch_global: {}, step_global: {}, step: {}".format(epochs_i, global_steps, idx))
                    self.logger.info("best_report:\n" + best_report)
                    self.logger.info("current_mertics:\n {}".format(res))
                    # idx_score = res.get("micro", {}).get("f1", 0)  # "macro", "micro", "weighted"
                    for k,v in res.items():  # tensorboard日志, 其中抽取中文、数字和英文, 避免一些不支持的符号, 比如说 /\|等特殊字符
                        if type(v) == dict:  # 空格和一些特殊字符tensorboardx.add_scalar不支持
                            k = chinese_extract_extend(k)
                            k = str(k.replace(" ", ""))
                            k = k if k else "empty"
                            for k_2, v_2 in v.items():
                                tensorboardx_witer.add_scalar(k + "/" + k_2, v_2, global_steps)
                        elif type(v) == float:
                            tensorboardx_witer.add_scalar(k, v, global_steps)
                            tensorboardx_witer.add_scalar("lr", scheduler.get_lr()[-1], global_steps)  #  pytorch==1.4.0版本往后: scheduler.get_last_lr()
                    save_best_mertics_key = self.config.save_best_mertics_key  # 模型存储的判别指标
                    abmk_1 = save_best_mertics_key[0]  # like "micro_avg"
                    abmk_2 = save_best_mertics_key[1]  # like "f1-score"
                    if res.get(abmk_1, {}).get(abmk_2, 0) > best_mertics.get(abmk_1, {}).get(abmk_2, 0) \
                            or best_mertics.get(abmk_1, {}).get(abmk_2, 0)==0.:  # 只保留最优的指标 or 第一次都保存
                        epochs_store.append((epochs_i, idx))
                        res["total"] = {"epochs": epochs_i, "global_steps": global_steps, "step_current": idx}
                        best_mertics = res
                        best_report = report
                        if self.config.flag_save_model_state:
                            self.save_model_state()
                        else:
                            self.save_model()
                        # 保存有优化的模型
                        if not self.config.flag_save_best:
                            path_model_dir_epoch_step = os.path.join(self.config.model_save_path, "epoch_" + str(epochs_i)
                                            + "_globalstep_" + str(global_steps) + "_step_" + str(idx) + self.config.model_name)
                            torch.save(self.model.state_dict(), path_model_dir_epoch_step)

                    # 重置is_train为True
                    self.model.train()  # train-type
                    # 早停, 连续stop_epochs指标不增长则自动停止
                    if epochs_store and epochs_i - epochs_store[-1][0] >= self.config.stop_epochs:
                        break
        return global_steps, best_mertics, best_report


if __name__ == '__main__':
    myz = 0


    # 预训练模型地址, 本地win10默认只跑2步就评估保存模型
    if platform.system().lower() == 'windows':
        pretrained_model_dir = "E:/DATA/bert-model/00_pytorch"
        path_corpus_dir = "D:/workspace/Pytorch-NLU/pytorch_nlu/corpus/sequence_labeling"
        evaluate_steps = 128  # 评估步数
        save_steps = 128  # 存储步数
    else:
        # pretrained_model_dir = "/pretrain_models/pytorch"
        # evaluate_steps = 320  # 评估步数
        # save_steps = 320  # 存储步数
        pretrained_model_dir = "/opt/pretrain_models/pytorch"
        evaluate_steps = 3200  # 评估步数
        save_steps = 3200  # 存储步数
        ee = 0
    # 预训练模型适配的class
    model_type = ["BERT", "ERNIE", "BERT_WWM", "ALBERT", "ROBERTA", "XLNET", "ELECTRA"]
    pretrained_model_name_or_path = {
        "BERT_WWM": pretrained_model_dir + "/chinese_wwm_pytorch",
        "ROBERTA": pretrained_model_dir + "/chinese_roberta_wwm_ext_pytorch",
        "ALBERT": pretrained_model_dir + "/albert_base_v1",
        "XLNET": pretrained_model_dir + "/chinese_xlnet_mid_pytorch",
        # "ERNIE": pretrained_model_dir + "/ERNIE_stable-1.0.1-pytorch",
        "ERNIE": pretrained_model_dir + "/ernie-tiny",  # 小模型
        # "BERT": pretrained_model_dir + "/uer_roberta-base-word-chinese-cluecorpussmall",
        "BERT": pretrained_model_dir + "/bert-base-chinese",
        # "BERT": pretrained_model_dir + "/uer_roberta-base-wwm-chinese-cluecorpussmall"
        "ELECTRA": pretrained_model_dir + "/hfl_chinese-electra-180g-base-discriminator"
    }

    task_version = "ner_3e5_12epoch"
    path_train = os.path.join(path_corpus_dir, "org_ner_china_people_daily_1998/train.conll")
    path_dev = os.path.join(path_corpus_dir, "org_ner_china_people_daily_1998/dev.conll")
    path_tet = os.path.join(path_corpus_dir, "org_ner_china_people_daily_1998/test.conll")

    model_config["evaluate_steps"] = evaluate_steps  # 评估步数
    model_config["save_steps"] = save_steps  # 存储步数
    model_config["path_train"] = path_train  # 训练模语料, 必须
    model_config["path_dev"] = path_dev  # 验证语料, 可为None
    model_config["path_tet"] = path_tet  # 测试语料, 可为None
    # model_config["CUDA_VISIBLE_DEVICES"] = "0"
    # 一种格式 文件以.conll结尾, 或者corpus_type=="DATA-CONLL"
    # 另一种格式 文件以.span结尾, 或者corpus_type=="DATA-SPAN"
    # 任务类型, "SL-SOFTMAX", "SL-CRF", "SL-SPAN", "SL-GRID", "sequence_labeling"
    # model_config["task_type"] = "SL-SOFTMAX"
    # model_config["task_type"] = "SL-SPAN"
    model_config["task_type"] = "SL-GRID"
    # model_config["task_type"] = "SL-CRF"
    # model_config["corpus_type"] = "DATA-SPAN"  # 语料数据格式, "DATA-CONLL", "DATA-SPAN"
    model_config["loss_type"] = "BCE"  # 损失函数类型, 可选 None(BCE), BCE, MSE, FOCAL_LOSS,
    # multi-label:  MARGIN_LOSS, PRIOR_MARGIN_LOSS, CIRCLE_LOSS等
    # 备注: "SL-GRID"类型不要用BCE、PRIOR_MARGIN_LOSS
    model_config["grad_accum_steps"] = 1
    model_config["warmup_steps"] = 4  # 1024  # 0.01  # 预热步数, -1为取 0.5 的epoch步数
    model_config["batch_size"] = 32
    model_config["max_len"] = 128
    model_config["epochs"] = 12

    model_config["xy_keys"] = ["text", "label"]  # SPAN格式的数据, text, label在file中对应的keys
    model_config["dense_lr"] = 1e-3  # CRF层学习率/全连接层学习率, 1e-5, 1e-4, 1e-3
    model_config["sl_ctype"] = "BIO"  # 数据格式sl-type, BIO, BMES, BIOES, BO, 只在"corpus_type": "MYX", "task_type": "SL-CRL"或"SL-SOFTMAX"时候生效
    model_config["lr"] = 5e-5  # 学习率, 1e-5, 2e-5, 5e-5, 8e-5, 1e-4, 4e-4
    model_config["num_workers"] = 0

    idx = 1  # 0   # 选择的预训练模型类型---model_type, 0为BERT,
    model_config["pretrained_model_name_or_path"] = pretrained_model_name_or_path[model_type[idx]]
    # model_config["model_save_path"] = "../output/sequence_labeling/model_{}".format(model_type[idx] + "_" + str(get_current_time()))
    model_config["model_save_path"] = "../output/sequence_labeling/model_{}_{}".format(task_version, model_type[idx])
    model_config["model_type"] = model_type[idx]

    # data
    from argparse import Namespace
    config = Namespace(**model_config)
    logger = get_logger(os.path.join(config["model_save_path"], "log"))

    cdl = CreateDataLoader(config, logger=logger)
    train_data_loader, dev_data_loader, tet_data_loader, \
                tokenizer, data_config = cdl.create_for_train(config)

    # train
    office = Office(data_config, tokenizer, logger=logger)
    office.train(train_data_loader, dev_data_loader)
    office.evaluate(tet_data_loader)
