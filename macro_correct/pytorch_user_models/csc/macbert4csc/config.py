# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/1/18 21:34
# @author  : Mo
# @function: Macbert4CSC
# @github   : [Macbert4CSC](https://github.com/shibing624/pycorrector/tree/master/examples/macbert)
# @paper   :  [SoftMaskedBert4CSC](https://arxiv.org/abs/2004.13922)


from argparse import Namespace
import platform


if platform.system().lower() == "windows":
    csc_config = {
        # "pretrained_model_name_or_path": "bert-base-chinese", # "bert-base-chinese",
        "pretrained_model_name_or_path": "hfl/chinese-macbert-base",
        # "path_relm": "",  # relm权重, 论文提供的预训练wiki修改句对的权重

        "path_train": "../../../corpus/text_correction/espell/csc_espell_law.train",
        "path_dev": "../../../corpus/text_correction/espell/csc_espell_law.test",
        "path_tet": "../../../corpus/text_correction/espell/csc_espell_law.test",
        "model_save_path": "../../../output/text_correction",
        "task_name": "espell_law_of_macbert4csc",

        # "path_train": "../../../corpus/text_correction/sighan/sighan2015.train.json",
        # "path_dev": "../../../corpus/text_correction/sighan/sighan2015.dev.json",
        # "path_tet": "../../../corpus/text_correction/sighan/sighan2015.dev.json",
        # "model_save_path": "../../../output/text_correction",
        # "task_name": "sighan_std_of_macbert4csc",

        "do_lower_case": True,
        "do_train": True,
        "do_eval": True,
        "do_test": True,
        "gradient_accumulation_steps": 4,
        "warmup_proportion": 0.1,
        "num_warmup_steps": None,  # 优先num_warmup_steps, 否则warmup_proportion
        "max_train_steps": 5000,  # 优先max_train_steps, 否则num_train_epochs, 5000
        "num_train_epochs": None,  # std=10
        "train_batch_size": 16,  # 16*4=(noerror-mask--实际是error); # 32*4=0.8948(only-error-mask)
        "eval_batch_size": 16,
        "learning_rate": 3e-5,
        "max_seq_length": 128,  # 这里应该是双倍实际长度, constant=256; 256;384;512
        "max_grad_norm": 1.0,
        "weight_decay": 5e-4,  # 5e-4
        "save_steps": 100,
        "anchor": None,
        "seed": 42,
        "lr_scheduler_type": "linear",  # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "inverse_sqrt", "reduce_lr_on_plateau"
        "loss_type": "focal_loss",  # "SOFT_MARGIN_LOSS/FOCAL_LOSS/CIRCLE_LOSS/DICE_LOSS/LABEL_SMOOTH/BCE_LOGITS/BCE_MULTI/MSE/BCE"
        "mask_mode": "noerror",  # "noerror", "error" or "all"
        "loss_det_rate": 0.3,  # det-loss的权重, 0.15/0.3
        "prompt_length": 0,  # 1, 3, 5
        "mask_rate": 0.3,  # 0.15, 0.2, 0,3
        "threshold": 0.5,
        "flag_dynamic_encode": False,  ### 是否动态编码encode, False则padding到最大长度
        "flag_fast_tokenizer": True,
        "flag_loss_period": False,  # 训练阶段, 前4/5时间进行bce, 后1/5时间进行focal_loss
        "flag_cpo_loss": False,  # 是否使用cpo_loss(ecopo)
        "flag_pin_memory": True,
        "flag_train": True,
        "flag_fp16": False,
        "flag_cuda": True,
        "flag_skip": True,  # 跳过特殊字符
        "flag_mft": True,
        "num_workers": 0,
        "CUDA_VISIBLE_DEVICES": "0",
        "USE_TORCH": "1"
    }

else:
    csc_config = {
        # "pretrained_model_name_or_path": "bert-base-chinese", # "bert-base-chinese",
        "pretrained_model_name_or_path": "hfl/chinese-macbert-base",
        # "pretrained_model_name_or_path": "",  # 预训练模型权重目录, 如bert-base权重
        "path_relm": "",  # relm权重, 论文提供的预训练wiki修改句对的权重

        # "path_train": "../../../corpus/text_correction/espell/csc_espell_law.train",
        # "path_dev": "../../../corpus/text_correction/espell/csc_espell_law.test",
        # "path_tet": "../../../corpus/text_correction/espell/csc_espell_law.test",
        # "model_save_path": "../../../output/text_correction",
        # "task_name": "espell_law_std_of_macbert4csc",

        "path_train": "../../../corpus/text_correction/sighan/sighan2015.train.json",
        "path_dev": "../../../corpus/text_correction/sighan/sighan2015.dev.json",
        "path_tet": "../../../corpus/text_correction/sighan/sighan2015.dev.json",
        "model_save_path": "../../../output/text_correction",
        "task_name": "sighan_std_of_macbert4csc",

        "do_lower_case": True,
        "do_train": True,
        "do_eval": True,
        "do_test": True,
        "gradient_accumulation_steps": 4,
        "warmup_proportion": 0.1,
        "num_warmup_steps": None,  # 优先num_warmup_steps, 否则warmup_proportion
        "max_train_steps": None,  # 优先max_train_steps, 否则num_train_epochs, 5000
        "num_train_epochs": 10,  # 10
        "train_batch_size": 32,  # 16*4=(noerror-mask--实际是error); # 32*4=0.8948(only-error-mask)
        "eval_batch_size": 32,
        "learning_rate": 3e-5,
        "max_seq_length": 128,  # 这里应该是双倍实际长度, constant=256; 256;384;512
        "max_grad_norm": 1.0,
        "weight_decay": 5e-4,  # 5e-4
        "save_steps": 1000,
        "anchor": None,
        "seed": 42,
        "lr_scheduler_type": "cosine",  # "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "inverse_sqrt", "reduce_lr_on_plateau"
        "loss_type": "focal_loss",  # "SOFT_MARGIN_LOSS/FOCAL_LOSS/CIRCLE_LOSS/DICE_LOSS/LABEL_SMOOTH/BCE_LOGITS/BCE_MULTI/MSE/BCE"
        "mask_mode": "noerror",  # "noerror", "error" or "all"
        "loss_det_rate": 0.3,  # det-loss的权重, 0.15/0.3
        "prompt_length": 0,  # 1, 3, 5
        "mask_rate": 0.3,  # 0.15, 0.2, 0,3
        "threshold": 0.5,
        "flag_dynamic_encode": False,  ### 是否动态编码encode, False则padding到最大长度
        "flag_fast_tokenizer": True,
        "flag_loss_period": False,  # 训练阶段, 前4/5时间进行bce, 后1/5时间进行focal_loss
        "flag_cpo_loss": False,  # 是否使用cpo_loss(ecopo)
        "flag_pin_memory": True,
        "flag_train": True,
        "flag_fp16": False,
        "flag_cuda": True,
        "flag_skip": True,  # 跳过特殊字符
        "flag_mft": True,
        "num_workers": 4,
        "CUDA_VISIBLE_DEVICES": "0",
        "USE_TORCH": "1"
    }

csc_config = Namespace(**csc_config)


