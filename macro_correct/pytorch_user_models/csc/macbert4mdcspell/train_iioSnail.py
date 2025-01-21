# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/1/18 21:34
# @author  : Mo
# @function: MDCSpell
# @paper   : [MDCSpell: A Multi-task Detector-Corrector Framework for Chinese Spelling Correction](https://aclanthology.org/2022.findings-acl.98/).
# @code    : most code copy from https://github.com/iioSnail/MDCSpell_pytorch, small modfix


import logging as logger
import pickle
import copy
import json
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(path_root)
print("path_root: " + path_root)
from macro_correct.pytorch_user_models.csc.mdcspell.config import csc_config as args
os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES or "0"
os.environ["USE_TORCH"] = args.USE_TORCH or "1"
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm
import torch

from macro_correct.pytorch_user_models.csc.mdcspell.dataset import sent_mertic_det, sent_mertic_cor


# 句子的长度，作者并没有说明。我这里就按经验取一个
max_length = 128
# 作者使用的batch_size
batch_size = 16  # 32
# epoch数，作者并没有具体说明，按经验取一个
epochs = 50

# 每${log_after_step}步，打印一次日志
log_after_step = 20

# 模型存放的位置。
model_path_dir = args.model_save_path
os.makedirs(model_path_dir, exist_ok=True)
model_path = model_path_dir + 'MDCSpell_Model.pt'
pretrained_model_name_or_path = args.pretrained_model_name_or_path
path_train = args.path_train
path_dev = args.path_dev
path_tet = args.path_tet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)


class CorrectionNetwork(nn.Module):

    def __init__(self):
        super(CorrectionNetwork, self).__init__()
        # BERT分词器，作者并没提到自己使用的是哪个中文版的bert，我这里就使用一个比较常用的
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        # BERT
        self.bert = AutoModel.from_pretrained(pretrained_model_name_or_path)
        # BERT的word embedding，本质就是个nn.Embedding
        self.word_embedding_table = self.bert.get_input_embeddings()
        # 预测层。hidden_size是词向量的大小，len(self.tokenizer)是词典大小
        self.dense_layer = nn.Linear(self.bert.config.hidden_size, len(self.tokenizer))

    def forward(self, inputs, word_embeddings, detect_hidden_states):
        """
        Correction Network的前向传递
        :param inputs: inputs为tokenizer对中文文本的分词结果，
                       里面包含了token对一个的index，attention_mask等
        :param word_embeddings: 使用BERT的word_embedding对token进行embedding后的结果
        :param detect_hidden_states: Detection Network输出hidden state
        :return: Correction Network对个token的预测结果。
        """
        # 1. 使用bert进行前向传递
        bert_outputs = self.bert(token_type_ids=inputs['token_type_ids'],
                                 attention_mask=inputs['attention_mask'],
                                 inputs_embeds=word_embeddings)
        # 2. 将bert的hidden_state和Detection Network的hidden state进行融合。
        hidden_states = bert_outputs['last_hidden_state'] + detect_hidden_states
        # 3. 最终使用全连接层进行token预测
        return self.dense_layer(hidden_states)

    def get_inputs_and_word_embeddings(self, sequences, max_length=128):
        """
        对中文序列进行分词和word embeddings处理
        :param sequences: 中文文本序列。例如: ["鸡你太美", "哎呦，你干嘛！"]
        :param max_length: 文本的最大长度，不足则进行填充，超出进行裁剪。
        :return: tokenizer的输出和word embeddings.
        """
        inputs = self.tokenizer(sequences, padding='max_length', max_length=max_length, return_tensors='pt',
                                truncation=True).to(device)
        # 使用BERT的work embeddings对token进行embedding，这里得到的embedding并不包含position embedding和segment embedding
        word_embeddings = self.word_embedding_table(inputs['input_ids'])
        return inputs, word_embeddings


class DetectionNetwork(nn.Module):

    def __init__(self, position_embeddings, transformer_blocks, hidden_size):
        """
        :param position_embeddings: bert的position_embeddings，本质是一个nn.Embedding
        :param transformer: BERT的前两层transformer_block，其是一个ModuleList对象
        """
        super(DetectionNetwork, self).__init__()
        self.position_embeddings = position_embeddings
        self.transformer_blocks = transformer_blocks

        # 定义最后的预测层，预测哪个token是错误的
        self.dense_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, word_embeddings):
        # 获取token序列的长度，这里为128
        sequence_length = word_embeddings.size(1)
        # 生成position embedding
        position_embeddings = self.position_embeddings(torch.LongTensor(range(sequence_length)).to(device))
        # 融合work_embedding和position_embedding
        x = word_embeddings + position_embeddings
        # 将x一层一层的使用transformer encoder进行向后传递
        for transformer_layer in self.transformer_blocks:
            x = transformer_layer(x)[0]

        # 最终返回Detection Network输出的hidden states和预测结果
        hidden_states = x
        return hidden_states, self.dense_layer(hidden_states)


class MDCSpellModel(nn.Module):

    def __init__(self):
        super(MDCSpellModel, self).__init__()
        # 构造Correction Network
        self.correction_network = CorrectionNetwork()
        self._init_correction_dense_layer()

        # 构造Detection Network
        # position embedding使用BERT的
        position_embeddings = self.correction_network.bert.embeddings.position_embeddings
        # 作者在论文中提到的，Detection Network的Transformer使用BERT的权重
        # 所以我这里直接克隆BERT的前两层Transformer来完成这个动作
        transformer = copy.deepcopy(self.correction_network.bert.encoder.layer[:2])
        # 提取BERT的词向量大小
        hidden_size = self.correction_network.bert.config.hidden_size

        # 构造Detection Network
        self.detection_network = DetectionNetwork(position_embeddings, transformer, hidden_size)

    def forward(self, sequences, max_length=128):
        # 先获取word embedding，Correction Network和Detection Network都要用
        inputs, word_embeddings = self.correction_network.get_inputs_and_word_embeddings(sequences, max_length)
        # Detection Network进行前向传递，获取输出的Hidden State和预测结果
        hidden_states, detection_outputs = self.detection_network(word_embeddings)
        # Correction Network进行前向传递，获取其预测结果
        correction_outputs = self.correction_network(inputs, word_embeddings, hidden_states)
        # 返回Correction Network 和 Detection Network 的预测结果。
        # 在计算损失时`[PAD]`token不需要参与计算，所以这里将`[PAD]`部分全都变为0
        return correction_outputs, detection_outputs.squeeze(2) * inputs['attention_mask']

    def _init_correction_dense_layer(self):
        """
        原论文中提到，使用Word Embedding的weight来对Correction Network进行初始化
        """
        self.correction_network.dense_layer.weight.data = self.correction_network.word_embedding_table.weight.data


class MDCSpellLoss(nn.Module):

    def __init__(self, coefficient=0.85):
        super(MDCSpellLoss, self).__init__()
        # 定义Correction Network的Loss函数
        self.correction_criterion = nn.CrossEntropyLoss(ignore_index=0)
        # 定义Detection Network的Loss函数，因为是二分类，所以用Binary Cross Entropy
        self.detection_criterion = nn.BCELoss()
        # 权重系数
        self.coefficient = coefficient

    def forward(self, correction_outputs, correction_targets, detection_outputs, detection_targets):
        """
        :param correction_outputs: Correction Network的输出，Shape为(batch_size, sequence_length, hidden_size)
        :param correction_targets: Correction Network的标签，Shape为(batch_size, sequence_length)
        :param detection_outputs: Detection Network的输出，Shape为(batch_size, sequence_length)
        :param detection_targets: Detection Network的标签，Shape为(batch_size, sequence_length)
        :return:
        """
        # 计算Correction Network的loss，因为Shape维度为3，所以要把batch_size和sequence_length进行合并才能计算
        correction_loss = self.correction_criterion(correction_outputs.view(-1, correction_outputs.size(2)),
                                                    correction_targets.view(-1))
        # 计算Detection Network的loss
        detection_loss = self.detection_criterion(detection_outputs, detection_targets)
        # 对两个loss进行加权平均
        return self.coefficient * correction_loss + (1 - self.coefficient) * detection_loss


def load_json(path, encoding="utf-8"):
    """   加载json文件   """
    with open(path, "r", encoding=encoding) as fj:
        model_json = json.load(fj)
        fj.close()
    return model_json


class CSCDataset(Dataset):

    def __init__(self, path_train):
        super(CSCDataset, self).__init__()
        self.task_name = "espell_law"
        train_data = load_json(path_train)
        self.data = train_data

    def __getitem__(self, index):
        if "original_text" in self.data[index]:
            src = self.data[index].get("original_text", "")
            trg = self.data[index].get("correct_text", "")
        else:
            src = self.data[index].get("source", "")
            trg = self.data[index].get("target", "")
        return src, trg

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    src, tgt = zip(*batch)
    src, tgt = list(src), list(tgt)

    src_tokens = tokenizer(src, padding='max_length', max_length=128, return_tensors='pt', truncation=True)['input_ids']
    tgt_tokens = tokenizer(tgt, padding='max_length', max_length=128, return_tensors='pt', truncation=True)['input_ids']

    correction_targets = tgt_tokens
    detection_targets = (src_tokens != tgt_tokens).float()
    return src, correction_targets, detection_targets, src_tokens  # src_tokens在计算Correction的精准率时要用到


def evaluation(model, test_data):
    d_recall_numerator = 0  # Detection的Recall的分子
    d_recall_denominator = 0  # Detection的Recall的分母
    d_precision_numerator = 0  # Detection的precision的分子
    d_precision_denominator = 0  # Detection的precision的分母
    c_recall_numerator = 0  # Correction的Recall的分子
    c_recall_denominator = 0  # Correction的Recall的分母
    c_precision_numerator = 0  # Correction的precision的分子
    c_precision_denominator = 0  # Correction的precision的分母
    model = model.eval()

    predict_tokens_list = []
    src_tokens_list = []
    tgt_tokens_list = []

    prograss = tqdm(range(len(test_data)))
    for i in prograss:
        # src, tgt = test_data[i]['src'], test_data[i]['tgt']
        src, tgt = test_data.data[i]['source'], test_data.data[i]['target']

        src_tokens = tokenizer(src, return_tensors='pt', max_length=128, truncation=True)['input_ids'][0][1:-1]
        tgt_tokens = tokenizer(tgt, return_tensors='pt', max_length=128, truncation=True)['input_ids'][0][1:-1]

        # 正常情况下，src和tgt的长度应该是一致的
        if len(src_tokens) != len(tgt_tokens):
            print("第%d条数据异常" % i)
            continue

        correction_outputs, _ = model(src)
        predict_tokens = correction_outputs[0][1:len(src_tokens) + 1].argmax(1).detach().cpu()

        predict_tokens_list.append(predict_tokens.numpy().tolist())
        src_tokens_list.append(src_tokens.numpy().tolist())
        tgt_tokens_list.append(tgt_tokens.numpy().tolist())

        # 计算错误token的数量
        d_recall_denominator += (src_tokens != tgt_tokens).sum().item()
        # 计算在这些错误token，有多少网络也认为它是错误的
        d_recall_numerator += (predict_tokens != src_tokens)[src_tokens != tgt_tokens].sum().item()
        # 计算网络找出的错误token的数量
        d_precision_denominator += (predict_tokens != src_tokens).sum().item()
        # 计算在网络找出的这些错误token中，有多少是真正错误的
        d_precision_numerator += (src_tokens != tgt_tokens)[predict_tokens != src_tokens].sum().item()
        # 计算Detection的recall、precision和f1-score
        d_recall = d_recall_numerator / (d_recall_denominator + 1e-9)
        d_precision = d_precision_numerator / (d_precision_denominator + 1e-9)
        d_f1_score = 2 * (d_recall * d_precision) / (d_recall + d_precision + 1e-9)

        # 计算错误token的数量
        c_recall_denominator += (src_tokens != tgt_tokens).sum().item()
        # 计算在这些错误token中，有多少网络预测对了
        c_recall_numerator += (predict_tokens == tgt_tokens)[src_tokens != tgt_tokens].sum().item()
        # 计算网络找出的错误token的数量
        c_precision_denominator += (predict_tokens != src_tokens).sum().item()
        # 计算网络找出的错误token中，有多少是正确修正的
        c_precision_numerator += (predict_tokens == tgt_tokens)[predict_tokens != src_tokens].sum().item()

        # 计算Correction的recall、precision和f1-score
        c_recall = c_recall_numerator / (c_recall_denominator + 1e-9)
        c_precision = c_precision_numerator / (c_precision_denominator + 1e-9)
        c_f1_score = 2 * (c_recall * c_precision) / (c_recall + c_precision + 1e-9)
        res = {
            'd_recall': d_recall,
            'd_precision': d_precision,
            'd_f1_score': d_f1_score,
            'c_recall': c_recall,
            'c_precision': c_precision,
            'c_f1_score': c_f1_score,
        }
        prograss.set_postfix(res)

    print("#" * 128)
    det_acc, det_precision, det_recall, det_f1 = sent_mertic_det(src_tokens_list, predict_tokens_list,
                                                                 tgt_tokens_list, logger)
    cor_acc, cor_precision, cor_recall, cor_f1 = sent_mertic_cor(src_tokens_list, predict_tokens_list,
                                                                 tgt_tokens_list, logger)

    print(
        f'Sentence Level detection: acc:{det_acc:.4f}, precision:{det_precision:.4f}, recall:{det_recall:.4f}, f1:{det_f1:.4f}')
    print(
        f'Sentence Level correction: acc:{cor_acc:.4f}, precision:{cor_precision:.4f}, recall:{cor_recall:.4f}, f1:{cor_f1:.4f}')
    print("#" * 128)
    result = {
        "det_acc": det_acc,
        "det_precision": det_precision,
        "det_recall": det_recall,
        "det_f1": det_f1,

        "cor_acc": cor_acc,
        "cor_precision": cor_precision,
        "cor_recall": cor_recall,
        "cor_f1": cor_f1,

        # "lr": scheduler.get_lr()[-1]
    }
    for k, v in result.items():
        print(k, v)

train_data = CSCDataset(path_train)
dev_data = CSCDataset(path_dev)
tet_data = CSCDataset(path_tet)

train_data.__getitem__(0)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
dev_loader = DataLoader(dev_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
tet_loader = DataLoader(tet_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

model = MDCSpellModel().to(device)
correction_outputs, detection_outputs = model(["鸡你太美", "哎呦，你干嘛！"])
print("correction_outputs shape:", correction_outputs.size())
print("detection_outputs shape:", detection_outputs.size())
print("correction_outputs:", correction_outputs)
print("detection_outputs:", detection_outputs)


criterion = MDCSpellLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
start_epoch = 0  # 从哪个epoch开始
total_step = 0  # 一共更新了多少次参数

# # 恢复之前的训练
# if os.path.exists(model_path):
#     if not torch.cuda.is_available():
#         checkpoint = torch.load(model_path, map_location='cpu')
#     else:
#         checkpoint = torch.load(model_path)
#     model.load_state_dict(checkpoint['model'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     start_epoch = checkpoint['epoch']
#     total_step = checkpoint['total_step']
#     print("恢复训练，epoch:", start_epoch)

model = model.to(device)
model = model.train()
total_loss = 0.  # 记录loss

d_recall_numerator = 0  # Detection的Recall的分子
d_recall_denominator = 0  # Detection的Recall的分母
d_precision_numerator = 0  # Detection的precision的分子
d_precision_denominator = 0  # Detection的precision的分母
c_recall_numerator = 0  # Correction的Recall的分子
c_recall_denominator = 0  # Correction的Recall的分母
c_precision_numerator = 0  # Correction的precision的分子
c_precision_denominator = 0  # Correction的precision的分母

for epoch in range(start_epoch, epochs):

    step = 0

    for sequences, correction_targets, detection_targets, correction_inputs in train_loader:
        correction_targets, detection_targets = correction_targets.to(device), detection_targets.to(device)
        correction_inputs = correction_inputs.to(device)
        correction_outputs, detection_outputs = model(sequences)
        loss = criterion(correction_outputs, correction_targets, detection_outputs, detection_targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        step += 1
        total_step += 1

        total_loss += loss.detach().item()

        # 计算Detection的recall和precision指标
        # 大于0.5，认为是错误token，反之为正确token
        d_predicts = detection_outputs >= 0.5
        # 计算错误token中被网络正确预测到的数量
        d_recall_numerator += d_predicts[detection_targets == 1].sum().item()
        # 计算错误token的数量
        d_recall_denominator += (detection_targets == 1).sum().item()
        # 计算网络预测的错误token的数量
        d_precision_denominator += d_predicts.sum().item()
        # 计算网络预测的错误token中，有多少是真错误的token
        d_precision_numerator += (detection_targets[d_predicts == 1]).sum().item()

        # 计算Correction的recall和precision
        # 将输出映射成index，即将correction_outputs的Shape由(32, 128, 21128)变为(32,128)
        correction_outputs = correction_outputs.argmax(2)
        # 对于填充、[CLS]和[SEP]这三个token不校验
        correction_outputs[(correction_targets == 0) | (correction_targets == 101) | (correction_targets == 102)] = 0
        # correction_targets的[CLS]和[SEP]也要变为0
        correction_targets[(correction_targets == 101) | (correction_targets == 102)] = 0
        # Correction的预测结果，其中True表示预测正确，False表示预测错误或无需预测
        c_predicts = correction_outputs == correction_targets
        # 计算错误token中被网络正确纠正的token数量
        c_recall_numerator += c_predicts[detection_targets == 1].sum().item()
        # 计算错误token的数量
        c_recall_denominator += (detection_targets == 1).sum().item()
        # 计算网络纠正token的数量
        correction_inputs[(correction_inputs == 101) | (correction_inputs == 102)] = 0
        c_precision_denominator += (correction_outputs != correction_inputs).sum().item()
        # 计算在网络纠正的这些token中，有多少是真正被纠正对的
        c_precision_numerator += c_predicts[correction_outputs != correction_inputs].sum().item()

        if total_step % log_after_step == 0:
            loss = total_loss / log_after_step
            d_recall = d_recall_numerator / (d_recall_denominator + 1e-9)
            d_precision = d_precision_numerator / (d_precision_denominator + 1e-9)
            c_recall = c_recall_numerator / (c_recall_denominator + 1e-9)
            c_precision = c_precision_numerator / (c_precision_denominator + 1e-9)

            print("Epoch {}, "
                  "Step {}/{}, "
                  "Total Step {}, "
                  "loss {:.5f}, "
                  "detection recall {:.4f}, "
                  "detection precision {:.4f}, "
                  "correction recall {:.4f}, "
                  "correction precision {:.4f}".format(epoch, step, len(train_loader), total_step,
                                                       loss,
                                                       d_recall,
                                                       d_precision,
                                                       c_recall,
                                                       c_precision))

            total_loss = 0.
            total_correct = 0
            total_num = 0
            d_recall_numerator = 0
            d_recall_denominator = 0
            d_precision_numerator = 0
            d_precision_denominator = 0
            c_recall_numerator = 0
            c_recall_denominator = 0
            c_precision_numerator = 0
            c_precision_denominator = 0

    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + 1,
        'total_step': total_step,
    }, model_path)
    evaluation(model, dev_data)

"""
espell_law
Epoch 30, Step 110/123, Total Step 3800, loss 0.00196, detection recall 0.5412, detection precision 0.9650, correction recall 1.0000, correction precision 1.0000
100%|鈻堚枅鈻堚枅鈻堚枅鈻堚枅鈻堚枅| 500/500 [00:09<00:00, 51.60it/s, d_recall=0.456, d_precision=0.994, d_f1_score=0.626, c_recall=0.392, c_precision=0.855, c_f1_score=0.538]
################################################################################################################################
Sentence Level detection: acc:0.6960, precision:0.7075, recall:0.4078, f1:0.5174
Sentence Level correction: acc:0.6660, precision:0.6054, recall:0.3490, f1:0.4428
"""
