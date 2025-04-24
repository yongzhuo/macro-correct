# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/11/17 21:36
# @author  : Mo
# @function: office of transformers, 训练-主工作流


import logging as logger
import numpy as np
import traceback
import operator
import random
import codecs
import copy
import json
import os

from transformers import AdamW, get_linear_schedule_with_warmup, get_scheduler
# from torch.utils.data import DataLoader, RandomSampler
import torch

from macro_correct.pytorch_textcorrection.tcTools import chinese_extract_extend, mertics_report, sigmoid, softmax
from macro_correct.pytorch_textcorrection.tcTools import sent_mertic_det, sent_mertic_cor
from macro_correct.pytorch_textcorrection.tcTools import char_mertic_det_cor, save_json
from macro_correct.pytorch_textcorrection.tcTqdm import tqdm, trange
from macro_correct.pytorch_textcorrection.tcAdversarial import FGM
from macro_correct.pytorch_textcorrection.tcGraph import Graph


class Office:
    def __init__(self, config, tokenizer, train_corpus=None, dev_corpus=None, tet_corpus=None, logger=logger):
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
        self.logger = logger
        logger.info(config)
        self.device = "cuda:{}".format(config.CUDA_VISIBLE_DEVICES) if (torch.cuda.is_available() \
            and self.config.flag_cuda and self.config.CUDA_VISIBLE_DEVICES != "-1") else "cpu"
        self.loss_type = self.config.loss_type if self.config.loss_type else "BCE"
        self.logger.info(self.device)
        self.model = Graph(config, tokenizer).to(self.device)  # 初始化模型网络架构
        self.set_random_seed(self.config.seed)  # 初始化随机种子
        self.train_corpus = train_corpus
        self.dev_corpus = dev_corpus
        self.tet_corpus = tet_corpus

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
        all_inputs, all_predictions, all_labels = [], [], []
        det_acc_labels = []
        cor_acc_labels = []
        eval_loss = 0.0
        eval_steps = 0
        results = []
        for batch_data in tqdm(data_loader, desc="evaluate"):
            batch_text = batch_data[-1]
            batch_data = [bd.to(self.device) for bd in batch_data[:-1]]  # device
            with torch.no_grad():
                # 注意顺序
                attention_mask = batch_data[1]
                token_type_ids = batch_data[2]
                text_labels = batch_data[3]
                det_labels = batch_data[4]
                input_ids = batch_data[0]
                inputs = {"attention_mask": attention_mask,
                          "token_type_ids": token_type_ids,
                          "text_labels": text_labels,
                          "det_labels": det_labels,
                          "input_ids": input_ids,
                          }
                outputs = self.model(**inputs)
                # logits = outputs[-1]
                loss = self.config.loss_det_rate * outputs[0] \
                       + (1 - self.config.loss_det_rate) * outputs[1]
                det_y_hat = (outputs[-2] > 0.5).long()
                cor_y_hat = torch.argmax((outputs[-1]), dim=-1)
                cor_y_hat *= attention_mask
                eval_loss += loss.mean().item()
                eval_steps += 1

            for src, tgt, mask, predict, det_predict, det_label in zip(input_ids, text_labels, attention_mask, cor_y_hat, det_y_hat, det_labels):
                len_sent = sum(mask.cpu().numpy().tolist())
                _src = src[1:len_sent + 1].cpu().numpy().tolist()
                _tgt = tgt[1:len_sent + 1].cpu().numpy().tolist()
                _predict = predict[1:len_sent + 1].cpu().numpy().tolist()
                cor_acc_labels.append(1 if operator.eq(_tgt, _predict) else 0)
                det_acc_labels.append(det_predict[1:len_sent + 1].equal(det_label[1:len_sent + 1]))
                # results.append((_src, _tgt, _predict,))
                ### 对token有影响
                min_len = min(len(_predict), len(_tgt), len(_src))
                all_predictions.append(_predict[:min_len])
                all_labels.append(_tgt[:min_len])
                all_inputs.append(_src[:min_len])

        loss = eval_loss / eval_steps

        self.logger.info("#" * 128)
        # print("#" * 128)
        common_det_acc, common_det_precision, common_det_recall, common_det_f1 = sent_mertic_det(
            all_inputs, all_predictions, all_labels, self.logger)
        common_cor_acc, common_cor_precision, common_cor_recall, common_cor_f1 = sent_mertic_cor(
            all_inputs, all_predictions, all_labels, self.logger)

        common_det_mertics = f'common Sentence Level detection: acc:{common_det_acc:.4f}, precision:{common_det_precision:.4f}, recall:{common_det_recall:.4f}, f1:{common_det_f1:.4f}'
        common_cor_mertics = f'common Sentence Level correction: acc:{common_cor_acc:.4f}, precision:{common_cor_precision:.4f}, recall:{common_cor_recall:.4f}, f1:{common_cor_f1:.4f}'
        # print("#" * 128)
        self.logger.info(f'flag_eval: common')
        # print(f'flag_eval: common')
        self.logger.info(common_det_mertics)
        # print(common_det_mertics)
        self.logger.info(common_cor_mertics)
        # print(common_cor_mertics)
        self.logger.info("#" * 128)
        # print("#" * 128)

        strict_det_acc, strict_det_precision, strict_det_recall, strict_det_f1 = sent_mertic_det(
            all_inputs, all_predictions, all_labels, self.logger, flag_eval="strict")
        strict_cor_acc, strict_cor_precision, strict_cor_recall, strict_cor_f1 = sent_mertic_cor(
            all_inputs, all_predictions, all_labels, self.logger, flag_eval="strict")

        strict_det_mertics = f'strict Sentence Level detection: acc:{strict_det_acc:.4f}, precision:{strict_det_precision:.4f}, recall:{strict_det_recall:.4f}, f1:{strict_det_f1:.4f}'
        strict_cor_mertics = f'strict Sentence Level correction: acc:{strict_cor_acc:.4f}, precision:{strict_cor_precision:.4f}, recall:{strict_cor_recall:.4f}, f1:{strict_cor_f1:.4f}'
        self.logger.info("#" * 128)
        # print("#" * 128)
        self.logger.info(f'flag_eval: strict')
        # print(f'flag_eval: strict')
        self.logger.info(strict_det_mertics)
        # print(strict_det_mertics)
        self.logger.info(strict_cor_mertics)
        # print(strict_cor_mertics)
        self.logger.info("#" * 128)
        # print("#" * 128)

        sent_mertics = {
            "eval_loss": eval_loss,

            "common_det_acc": common_det_acc,
            "common_det_precision": common_det_precision,
            "common_det_recall": common_det_recall,
            "common_det_f1": common_det_f1,

            "common_cor_acc": common_cor_acc,
            "common_cor_precision": common_cor_precision,
            "common_cor_recall": common_cor_recall,
            "common_cor_f1": common_cor_f1,

            "strict_det_acc": strict_det_acc,
            "strict_det_precision": strict_det_precision,
            "strict_det_recall": strict_det_recall,
            "strict_det_f1": strict_det_f1,

            "strict_cor_acc": strict_cor_acc,
            "strict_cor_precision": strict_cor_precision,
            "strict_cor_recall": strict_cor_recall,
            "strict_cor_f1": strict_cor_f1,
        }

        detection_precision, detection_recall, detection_f1, \
        correction_precision, correction_recall, correction_f1 = char_mertic_det_cor(all_inputs, all_labels, all_predictions, self.logger)
        token_result = {"det_precision": detection_precision, "det_recall": detection_recall, "det_f1": detection_f1,
                        "cor_precision": correction_precision, "cor_recall": correction_recall, "cor_f1": correction_f1,
                        }

        token_det_mertics = f'common Token Level correction: precision:{detection_precision:.4f}, recall:{detection_recall:.4f}, f1:{detection_f1:.4f}'
        token_cor_mertics = f'common TOken Level correction: precision:{correction_precision:.4f}, recall:{correction_recall:.4f}, f1:{correction_f1:.4f}'
        self.logger.info("#" * 128)
        # print("#" * 128)
        self.logger.info(f'flag_eval: token')
        # print(f'flag_eval: token')
        self.logger.info(token_det_mertics)
        # print(token_det_mertics)
        self.logger.info(token_cor_mertics)
        # print(token_cor_mertics)
        self.logger.info("#" * 128)
        # print("#" * 128)

        dev_result = {"det_loss": float(outputs[0].detach().cpu().numpy()),
                      "mlm_loss": float(outputs[1].detach().cpu().numpy()),
                      "all_loss": loss, }
        result = {"sentence": sent_mertics, "token": token_result, "val": dev_result}
        report = ""
        for k, v in result.items():
            for vk, vv in v.items():
                report += "\n" + str(k) + "-" + str(vk) + "：" + str(vv)
        return result, report

    def predict(self, data_loader, flag_logits=False, **kwargs):
        """
        预测, pred
        config:
            data_loader: tensor, eg. tensor(1,2,3)
        Returns:
            ys_prob: list<json>
        """
        # pin_memory预先加载到cuda-gpu
        logits_pred_id = None
        probs_pred_id = None
        ys_pred_id = None
        # 预测 batch-size
        self.model.eval()
        for batch_data in data_loader:
            # batch_text = batch_data[-1]
            batch_data = [bd.to(self.device) for bd in batch_data[:-1]]  # device
            with torch.no_grad():
                # 注意顺序
                attention_mask = batch_data[1]
                token_type_ids = batch_data[2]
                # text_labels = batch_data[4]
                # det_labels = batch_data[3]
                input_ids = batch_data[0]
                inputs = {"attention_mask": attention_mask,
                          "token_type_ids": token_type_ids,
                          # "text_labels": text_labels,
                          # "det_labels": det_labels,
                          "input_ids": input_ids,
                          }
                outputs = self.model(**inputs)
                probs_pred = torch.max(torch.softmax(outputs[-1], dim=-1), dim=-1)
                cor_y_hat = torch.argmax((outputs[-1]), dim=-1)
                cor_y_hat *= attention_mask
                logits_pred = outputs[-1]
            logits_pred_numpy = logits_pred.detach().cpu().numpy()
            probs_pred_numpy = probs_pred.values.detach().cpu().numpy()
            cor_y_hat_numpy = cor_y_hat.detach().cpu().numpy()
            if ys_pred_id is not None:
                logits_pred_id = np.append(logits_pred_id, logits_pred_numpy, axis=0)
                probs_pred_id = np.append(probs_pred_id, probs_pred_numpy, axis=0)
                ys_pred_id = np.append(ys_pred_id, cor_y_hat_numpy, axis=0)
            else:
                logits_pred_id = logits_pred_numpy
                probs_pred_id = probs_pred_numpy
                ys_pred_id = cor_y_hat_numpy
        # 只返回logits形式
        if flag_logits:
            return logits_pred_id, probs_pred_id
        return ys_pred_id.tolist(), probs_pred_id.tolist()

    def train(self, d1, d2):
        """  训练迭代epoch
        return global_steps, best_mertics
        """
        from tensorboardX import SummaryWriter

        #  配置好优化器与训练工作计划(主要是学习率预热warm-up与衰减weight-decay, 以及不作用的层超参)
        params_no_decay = ["LayerNorm.weight", "bias"]
        parameters_no_decay = [
            {"params": [p for n, p in self.model.named_parameters() if not any(pnd in n for pnd in params_no_decay)],
             "weight_decay": self.config.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(pnd in n for pnd in params_no_decay)],
             "weight_decay": 0.0}
            ]
        # optimizer = AdamW(parameters_no_decay, lr=self.config.lr, eps=self.config.adam_eps)
        optimizer = torch.optim.AdamW(parameters_no_decay, lr=self.config.lr, eps=self.config.adam_eps)
        # 训练轮次
        times_batch_size = len(d1) // self.config.grad_accum_steps
        num_training_steps = int(times_batch_size * self.config.epochs)
        # 如果选择-1不设置则为 半个epoch
        if self.config.warmup_steps <= 0:
            num_warmup_steps = int((len(d1) // self.config.grad_accum_steps // 10)) if self.config.warmup_steps < 1 else self.config.warmup_steps
        elif 0 < self.config.warmup_steps and self.config.warmup_steps < 1:
            num_warmup_steps = int((len(d1) // self.config.grad_accum_steps) * self.config.warmup_steps)
        else:
            num_warmup_steps = self.config.warmup_steps

        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        scheduler = get_scheduler(optimizer=optimizer, name=self.config.scheduler_name, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        tensorboardx_witer = SummaryWriter(logdir=self.config.model_save_path)
        # current_lr = scheduler.get_last_lr()

        # adv
        if self.config.flag_adv:
            fgm = FGM(self.model, emb_name=self.config.adv_emb_name, epsilon=self.config.adv_eps)
        # 开始训练
        self.model.graph_config.flag_train = True
        epochs_store = []
        global_steps = 0
        best_mertics = {}
        best_report = ""
        for epochs_i in trange(self.config.epochs, desc="epoch"):  # epoch
            self.model.train()  # train-type
            for idx, batch_data in enumerate(tqdm(d1, desc="epoch_{} step".format(epochs_i))):  # step
                # 数据与模型
                batch_text = batch_data[-1]
                batch_data = [bd.to(self.device) for bd in batch_data[:-1]]  # device
                labels = batch_data[3]
                # 注意顺序
                inputs = {"attention_mask": batch_data[1],
                          "token_type_ids": batch_data[2],
                          "text_labels": batch_data[3],
                          "det_labels": batch_data[4],
                          "input_ids": batch_data[0],
                          }
                ### 最后一个epoch不MASK
                # if self.config.flag_mft and epochs_i != self.config.epochs-1:
                ### 50%的几率MASK(避免过度纠错)
                float_mft = random.random()
                if self.config.flag_mft and float_mft < 0.5:
                    sources_copy = copy.deepcopy(inputs["input_ids"])

                    input_ids_dynamic_mask = self.dynamic_mask_token(sources_copy, inputs["text_labels"], self.tokenizer,
                                                                     self.device, mask_mode="noerror", noise_probability=0.2)
                    inputs["input_ids"] = input_ids_dynamic_mask
                    inputs["det_labels"] = (input_ids_dynamic_mask != inputs["text_labels"]).float()
                outputs = self.model(**inputs)
                loss = self.config.loss_det_rate * outputs[0] \
                       + (1-self.config.loss_det_rate) * outputs[1]

                # logits = outputs[-1]
                # _, prd_ids = torch.max(logits, -1)  # (batch,seq)
                # def decode(x):
                #     return self.tokenizer.convert_ids_to_tokens(x, skip_special_tokens=True)
                # prd_tokens = [decode(p) for p in prd_ids]

                # loss = self.w * outputs[1] + (1 - self.w) * outputs[0]
                # loss = self.calculate_loss(logits, labels)
                loss = loss / self.config.grad_accum_steps
                loss.backward()
                global_steps += 1
                #  对抗训练
                if self.config.flag_adv:
                    fgm.attack()  # 在embedding上添加对抗扰动
                    outputs = self.model(**inputs)
                    loss = self.config.loss_det_rate * outputs[0] \
                           + (1 - self.config.loss_det_rate) * outputs[1]
                    # loss = outputs[0] + outputs[1]
                    # logits = self.model(**inputs)
                    # loss = self.calculate_loss(logits, labels)
                    loss = loss / self.config.grad_accum_steps  # 梯度累计
                    loss.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    fgm.restore()  # 恢复embedding参数
                #  梯度累计
                if (idx + 1) % self.config.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    scheduler.step()
                    optimizer.step()
                    self.model.zero_grad()
                # 评估算法/打印日志/存储模型, 1个epoch/到达保存的步数/或者是最后一轮最后一步
                if (self.config.evaluate_steps > -1 and global_steps % self.config.evaluate_steps == 0) or (idx == times_batch_size-1) \
                        or (epochs_i+1 == self.config.epochs and idx+1 == len(d1)):
                    self.model.eval()
                    self.logger.info("epoch: " + str(epochs_i))
                    self.logger.info("global_steps: " + str(global_steps))
                    res, report = self.evaluate(d2)
                    res["total"] = {"epochs": epochs_i, "global_steps": global_steps, "step_current": idx}
                    self.logger.info("best_report\n" + best_report)
                    self.logger.info("current_mertics: {}".format(res))
                    self.logger.info("epoch_global: {}, step_global: {}, step: {}".format(epochs_i, global_steps, idx))
                    # idx_score = res.get("micro", {}).get("f1", 0)  # "macro", "micro", "weighted"
                    for k, v in res.items():  # tensorboard日志, 其中抽取中文、数字和英文, 避免一些不支持的符号, 比如说 /\|等特殊字符
                        if type(v) == dict:  # 空格和一些特殊字符tensorboardx.add_scalar不支持
                            k = chinese_extract_extend(k)
                            k = str(k.replace(" ", ""))
                            k = k if k else "empty"
                            for k_2, v_2 in v.items():
                                tensorboardx_witer.add_scalar(k + "/" + k_2, v_2, global_steps)
                            # tensorboardx_witer.add_scalar(k + "/train_loss", loss, global_steps)
                        elif type(v) == float:
                            tensorboardx_witer.add_scalar(k, v, global_steps)
                            tensorboardx_witer.add_scalar("lr", scheduler.get_lr()[-1], global_steps)
                    tensorboardx_witer.add_scalar("train/lr", scheduler.get_lr()[-1], global_steps)
                    tensorboardx_witer.add_scalar("train/det_loss", float(outputs[0].detach().cpu().numpy()), global_steps)
                    tensorboardx_witer.add_scalar("train/mlm_loss", float(outputs[1].detach().cpu().numpy()), global_steps)
                    tensorboardx_witer.add_scalar("train/all_loss", loss, global_steps)

                    self.model.train()  # 预测时候的, 回转回来
                    save_best_mertics_key = self.config.save_best_mertics_key  # 模型存储的判别指标
                    abmk_1 = save_best_mertics_key[0]  # like "micro_avg"
                    abmk_2 = save_best_mertics_key[1]  # like "f1-score"
                    if res.get(abmk_1, {}).get(abmk_2, 0) > best_mertics.get(abmk_1, {}).get(abmk_2, 0):  # 只保留最优的指标
                        epochs_store.append((epochs_i, idx))
                        best_mertics = copy.deepcopy(res)
                        best_report = copy.deepcopy(report)
                        self.save_model_state(mertics=best_mertics)  # bert权重
                        self.save_model(mertics=best_mertics)        # 完整模型权重
                    # 重置is_train为True
                    # self.model.train()  # train-type
                    # 早停, 连续stop_epochs轮指标不增长则自动停止
                    if epochs_store and epochs_i - epochs_store[-1][0] >= self.config.stop_epochs:
                        break
        return global_steps, best_mertics, best_report

    def dynamic_mask_token(self, inputs, targets, tokenizer, device, mask_mode="noerror", noise_probability=0.2):
        '''
        the masked-FT proposed in 'Rethinking Masked Language Model for Chinese Spelling Correction'
        '''
        inputs = inputs.clone()
        probability_matrix = torch.full(inputs.shape, noise_probability).to(device)
        # do not mask sepcail tokens
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool).to(device)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # mask_mode in ["all","error","noerror"]
        if mask_mode == "noerror":
            probability_matrix.masked_fill_(inputs != targets, value=0.0)
        elif mask_mode == "error":
            probability_matrix.masked_fill_(inputs == targets, value=0.0)
        else:
            assert mask_mode == "all"
        masked_indices = torch.bernoulli(probability_matrix).bool()
        inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        return inputs

    def load_model_state(self, path_dir=""):
        """  仅加载模型参数(推荐使用)  """
        try:
            if path_dir:
                path_model = path_dir
            else:
                path_model = os.path.join(self.config.model_save_path, self.config.model_name)
            # model_dict_std = self.model.state_dict()
            model_dict_new = torch.load(path_model, map_location=torch.device(self.device))
            # model_dict_new = {k:v for k,v in model_dict_new.items()}
            # self.model.load_state_dict(model_dict_new)
            # model_dict_new = {"bert." + k if not k.startswith("bert.") else k: v for k, v in model_dict_new.items()}
            model_dict_new = {"bert." + k if not k.startswith("bert.bert.") else k: v for k, v in model_dict_new.items()}
            self.model.load_state_dict(model_dict_new, strict=False)
            self.model.to(self.device)
            self.logger.info("******model loaded success******")
            self.logger.info("self.device: {}".format(self.device))
        except Exception as e:
            self.logger.info(str(traceback.print_exc()))
            raise Exception("******load model error******")

    def save_model_state(self, path_dir="", mertics={}):
        """  仅保存模型参数(推荐使用), 包括预训练模型的部分config  """
        if path_dir:
            model_save_path = path_dir
        else:
            model_save_path = self.config.model_save_path
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        # save mertics
        path_mertics = os.path.join(self.config.model_save_path, "train_mertics_best.json")
        save_json(mertics, path_mertics)
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
        # ### save model(全部模型架构)
        # path_model = os.path.join(self.config.model_save_path, self.config.model_name)
        # torch.save(self.model.state_dict(), path_model)
        ### 只存储bert模型等
        self.model.bert.save_pretrained(self.config.model_save_path, safe_serialization=False)
        self.logger.info("****** model_save_path is {}******".format(self.config.model_save_path))

    def save_onnx(self, path_onnx_dir=""):
        """  存储为ONNX格式  """
        ### ONNX---路径
        if not path_onnx_dir:
            path_onnx_dir = os.path.join(self.config.model_save_path, "onnx")
        if not os.path.exists(path_onnx_dir):
            os.makedirs(path_onnx_dir)
        batch_data = [[[1, 2, 3, 4]*32]*32, [[1, 0]*64]*32, [[0, 1]*64]*32]
        # for name, param in self.model.named_parameters():  # 查看可优化的参数有哪些
        #     # if param.requires_grad:
        #         print(name)
        # batch_data = [bd.to(self.device) for bd in batch_data]  # device
        with torch.no_grad():
            inputs = {"attention_mask": torch.tensor(batch_data[1]).to(self.device),
                      "token_type_ids": torch.tensor(batch_data[2]).to(self.device),
                      "input_ids": torch.tensor(batch_data[0]).to(self.device),
                      }
            _ = self.model(**inputs)
            input_names = ["input_ids", "attention_mask", "token_type_ids"]
            output_names = ["outputs"]
            torch.onnx.export(self.model,
            (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
            os.path.join(path_onnx_dir, "csc_model.onnx"),
            input_names=input_names,
            output_names=output_names,  ## Be carefule to write this names
            opset_version=10,  # 7,8,9
            do_constant_folding=True,
            use_external_data_format=True,
            dynamic_axes = {
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "token_type_ids": {0: "batch", 1: "sequence"},
                output_names[0]: {0: "batch"}
            })

    def load_model(self, path_dir=""):
        """  加载模型  """
        try:
            if path_dir:
                path_model = path_dir
            else:
                # path_model = os.path.join(self.config.model_save_path, self.config.model_name)
                path_model = os.path.join(self.config.model_save_path, "csc.model")
            # ### load pytorch_model.bin
            self.model = torch.load(path_model, map_location=torch.device(self.device))
            # model_dict_new = torch.load(path_model, map_location=torch.device(self.device))
            # model_dict_new = {"bert." + k if not k.startswith("bert.bert.")
            #                   else k: v for k, v in model_dict_new.items()}
            # self.model.load_state_dict(model_dict_new, strict=False)
            self.model.to(self.device)
            self.logger.info("******model loaded success******")
        except Exception as e:
            self.logger.info(str(traceback.print_exc()))
            raise Exception("******load model error******")

    def save_model(self, path_dir="", mertics={}):
        """  存储模型, 包括预训练模型的部分config  """
        if path_dir:
            self.config.model_save_path = path_dir
        if not os.path.exists(self.config.model_save_path):
            os.makedirs(self.config.model_save_path)
        # save mertics
        path_mertics = os.path.join(self.config.model_save_path, "train_mertics_best.json")
        save_json(mertics, path_mertics)
        # save bert.config
        self.model.pretrained_config.save_pretrained(save_directory=self.config.model_save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(save_directory=self.config.model_save_path)
        # save csc.config
        self.config.flag_train = False  # 存储后的用于预测
        path_config = os.path.join(self.config.model_save_path, self.config.config_name)
        with codecs.open(filename=path_config, mode="w", encoding="utf-8") as fc:
            self.config.flag_train = False
            self.config.num_workers = 0
            json.dump(vars(self.config), fc, indent=4, ensure_ascii=False)
            fc.close()
        ### save csc.model
        # path_model = os.path.join(self.config.model_save_path, self.config.model_name)
        path_model = os.path.join(self.config.model_save_path, "csc.model")
        torch.save(self.model, path_model)
        ### 只存储bert模型等
        # self.model.bert.save_pretrained(self.config.model_save_path, safe_serialization=False)
        self.logger.info("****** model_save_path is {}******".format(self.config.model_save_path))
