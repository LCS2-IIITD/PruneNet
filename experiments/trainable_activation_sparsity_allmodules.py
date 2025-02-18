import logging
import os

logging.basicConfig()
logger = logging.getLogger(__name__)

import argparse
from datasets import load_dataset, DatasetDict
from evaluate import load
from functools import partial
from itertools import product
import json
import pandas as pd
from pathlib import Path
import random
import statistics
from safetensors.torch import load_file
import time
import torch
from tqdm import tqdm
from copy import deepcopy as cp
import numpy as np
from scipy import stats

from torch.profiler import profile, record_function, ProfilerActivity
from fvcore.nn import FlopCountAnalysis
import gc

import wandb
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from peft import LoraConfig, TaskType, get_peft_model

from bo_options import lora_target_map
from slicegpt import data_utils, gpu_utils, hf_utils, utils
from slicegpt.config import config

from bernoulligpt_utils import LeakyGeLU, LeakySiLU

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.distributions.uniform import Uniform
from torch.distributions import Categorical, Bernoulli
from torch.nn.utils.parametrizations import orthogonal

import ddks

calculation = ddks.methods.ddKS()


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / 1024**3


def random_slicing(model_name, layer, sparsity):
    if "opt" in model_name:
        feat_len = layer.fc1.out_features
    elif "phi" in model_name:
        feat_len = layer.mlp.fc1.out_features
    elif "llama" in model_name:
        feat_len = layer.mlp.gate_proj.out_features
    elif "falcon" in model_name:
        feat_len = layer.mlp.dense_h_to_4h.out_features
    else:
        raise ValueError(
            "Model type is not supported. Only OPT, Phi, Llama and Falcon models are supported."
        )

    row_indices = random.sample(range(feat_len), int((1 - sparsity) * feat_len))

    slicing(model_name, layer, row_indices)


def slicing(model_name, layer, row_indices):
    if "opt" in model_name:
        # slice the intermediate and output weight matrices appropriately
        layer.fc1.out_features = len(row_indices)
        layer.fc1.weight.data = layer.fc1.weight[row_indices, :]

        # try:
        #    layer.fc1.lora_B.default.weight.data = (
        #        layer.fc1.lora_B.default.weight[row_indices, :]
        #    )
        # except:
        #    pass

        layer.fc1.bias.data = layer.fc1.bias[row_indices]

        # revert changes on output layer
        layer.fc2.in_features = len(row_indices)
        layer.fc2.weight.data = layer.fc2.weight[:, row_indices]

        # try:
        #    layer.fc2.lora_A.default.weight.data = (
        #        layer.fc2.lora_A.default.weight[:, row_indices]
        #    )
        # except:
        #    pass

    elif "phi" in model_name:
        # slice the intermediate and output weight matrices appropriately
        layer.mlp.fc1.out_features = len(row_indices)
        layer.mlp.fc1.weight.data = layer.mlp.fc1.weight[row_indices, :]

        # try:
        #    layer.fc1.lora_B.default.weight.data = (
        #        layer.fc1.lora_B.default.weight[row_indices, :]
        #    )
        # except:
        #    pass

        layer.mlp.fc1.bias.data = layer.mlp.fc1.bias[row_indices]

        # revert changes on output layer
        layer.mlp.fc2.in_features = len(row_indices)
        layer.mlp.fc2.weight.data = layer.mlp.fc2.weight[:, row_indices]

        # try:
        #    layer.fc2.lora_A.default.weight.data = (
        #        layer.fc2.lora_A.default.weight[:, row_indices]
        #    )
        # except:
        #    pass

    elif "llama" in model_name:
        # slice the intermediate and output weight matrices appropriately
        layer.mlp.gate_proj.out_features = len(row_indices)
        layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight[
            row_indices, :
        ]

        # try:
        #    layer.mlp.gate_proj.lora_B.default.weight.data = (
        #        layer.mlp.gate_proj.lora_B.default.weight[row_indices, :]
        #    )
        # except:
        #    pass

        layer.mlp.up_proj.out_features = len(row_indices)
        layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight[row_indices, :]

        # try:
        #    layer.mlp.up_proj.lora_B.default.weight.data = (
        #        layer.mlp.up_proj.lora_B.default.weight[row_indices, :]
        #    )
        # except:
        #    pass

        # revert changes on output layer
        layer.mlp.down_proj.in_features = len(row_indices)
        layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight[
            :, row_indices
        ]

        # try:
        #    layer.mlp.down_proj.lora_A.default.weight.data = (
        #        layer.mlp.down_proj.lora_A.default.weight[:, row_indices]
        #    )
        # except:
        #    pass

    elif "falcon" in model_name:
        # slice the intermediate and output weight matrices appropriately
        layer.mlp.dense_h_to_4h.out_features = len(row_indices)
        layer.mlp.dense_h_to_4h.weight.data = layer.mlp.dense_h_to_4h.weight[
            row_indices, :
        ]

        # try:
        #    layer.mlp.dense_h_to_4h.lora_B.default.weight.data = (
        #        layer.mlp.dense_h_to_4h.lora_B.default.weight[row_indices, :]
        #    )
        # except:
        #    pass

        # revert changes on output layer
        layer.mlp.dense_4h_to_h.in_features = len(row_indices)
        layer.mlp.dense_4h_to_h.weight.data = layer.mlp.dense_4h_to_h.weight[
            :, row_indices
        ]

        # try:
        #    layer.mlp.dense_4h_to_h.lora_A.default.weight.data = (
        #        layer.mlp.dense_4h_to_h.lora_A.default.weight[:, row_indices]
        #    )
        # except:
        #    pass

    return layer


def slicing_qkv(model_name, layer, row_indices):
    if "llama" in model_name:
        # slice the intermediate and output weight matrices appropriately
        layer.self_attn.k_proj.out_features = len(row_indices)
        layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight[
            row_indices, :
        ]

        layer.self_attn.q_proj.out_features = len(row_indices)
        layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight[
            row_indices, :
        ]

        layer.self_attn.v_proj.out_features = len(row_indices)
        layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight[
            row_indices, :
        ]

        layer.self_attn.o_proj.in_features = len(row_indices)
        layer.self_attn.o_proj.weight.data = layer.self_attn.o_proj.weight[
            :, row_indices
        ]

    else:
        raise ValueError("qkv slicing only works with llama2")

    return layer


def get_all_layers(model_name, model):
    if "opt" in model_name:
        all_layers = model.base_model.model.model.decoder.layers
    elif "phi" in model_name:
        all_layers = model.base_model.model.model.layers
    elif "llama" in model_name:
        all_layers = model.base_model.model.model.layers
    elif "falcon" in model_name:
        all_layers = model.base_model.model.transformer.h
    else:
        raise ValueError(
            "Model type is not supported. Only OPT, Phi, Llama and Falcon models are supported."
        )

    return all_layers


def get_all_layers_before_lora(model_name, model):
    if "opt" in model_name:
        all_layers = model.model.decoder.layers
    elif "phi" in model_name:
        all_layers = model.model.layers
    elif "llama" in model_name:
        all_layers = model.model.layers
    elif "falcon" in model_name:
        all_layers = model.model.transformer.h
    else:
        raise ValueError(
            "Model type is not supported. Only OPT, Phi, Llama and Falcon models are supported."
        )

    return all_layers


def get_memory_consumption(model):
    mem_params = sum(
        [
            param.nelement() * param.element_size()
            for param in model.parameters()
        ]
    )
    mem_bufs = sum(
        [buf.nelement() * buf.element_size() for buf in model.buffers()]
    )
    mem = mem_params + mem_bufs  # in bytes
    return mem


def get_closest_even_number(x, binsize):
    if int(x // binsize) % 2 == 0:
        return int(binsize * int(x // binsize))
    else:
        return int(binsize * int(x // binsize) - binsize)


class SparsityPredictor(torch.nn.Module):
    def __init__(
        self, hidden_size=768, intermediate_size=3072, sparsity_level=0.2
    ):
        super(SparsityPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.proj_intermediate = nn.Linear(
            hidden_size, intermediate_size, bias=True
        )
        self.proj_query = nn.Linear(hidden_size, hidden_size, bias=True)

        self.row_sparsities = nn.Parameter(
            torch.rand(intermediate_size, 1), requires_grad=True
        )  # (3072, 1)
        self.row_sparsities_bias = nn.Parameter(
            torch.rand(1, 1), requires_grad=True
        )  # (3072, 1)

        self.row_sparsities_query = nn.Parameter(
            torch.rand(hidden_size, 1), requires_grad=True
        )  # (3072, 1)
        self.row_sparsities_bias_query = nn.Parameter(
            torch.rand(1, 1), requires_grad=True
        )  # (3072, 1)

        # self.col_sparsities = nn.Parameter(torch.rand(hidden_size, 1), requires_grad=True)  # (768, 1)
        self.sparsity_level = sparsity_level
        self.density_level = 1 - sparsity_level

        self.singular_value = None

    def calculate_KLD(self):
        return (
            -1 * torch.log(self.alpha) * (1 - self.alpha)
            - self.alpha * torch.log(1 - self.alpha)
            + torch.log(torch.tensor(0.5)).to(self.alpha.device)
        ).sum()

    def calculate_l1_loss(self):
        return torch.sum(torch.abs(self.keep_probs - self.density_level))

    def calculate_total_loss(self):
        return self.calculate_KLD()

    def forward(self, weight_matrix):
        if weight_matrix.shape[0] == self.intermediate_size:  # (3072, 768)
            proj_ = self.proj_intermediate(weight_matrix)  # (3072, 3072)
            alpha = nn.Sigmoid()(proj_ @ self.row_sparsities)[:, 0]  # (3072, )
        else:
            proj_ = self.proj_query(weight_matrix)  # (3072, 3072)
            alpha = nn.Sigmoid()(proj_ @ self.row_sparsities_query)[
                :, 0
            ]  # (3072, )

        self.alpha = alpha

        m = Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        eps = m.sample((alpha.shape[0],)).to(weight_matrix.device)[
            :, 0
        ]  # (3072, )

        # Calculate the probabilities using reparametrization trick
        keep_probs = nn.Sigmoid()(
            torch.log(eps)
            - torch.log(1 - eps)
            + torch.log(alpha)
            - torch.log(1 - alpha)
        )
        self.keep_probs = keep_probs

        # Use the keep_probs as a mask to determine which rows to keep
        # rows_to_keep = keep_probs <= 0.5

        return keep_probs


def calculate_activation_reward(s1, weight_matrix2):
    if weight_matrix2.dtype == torch.float16:
        weight_matrix2 = weight_matrix2.to(torch.float32)

    # _,s1,_ = torch.svd(weight_matrix1)
    _, s2, _ = torch.svd(weight_matrix2)

    # dist = calculation(s1.unsqueeze(1),s2.unsqueeze(1))
    dist = stats.ks_2samp(
        s1.detach().cpu().numpy(), s2.detach().cpu().numpy()
    ).statistic

    # dist = s2.max() #torch.abs(s2.max() - 1)
    if dist == 0:
        return 99999
    else:
        return 1 / dist


def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    # r = r[::-1].cumsum()[::-1]
    # return r - r.mean()
    return r


def get_optimizer_and_scheduler(model, train_dataset, config):
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )

    kwargs_lr_scheduler = {
        "optimizer": optimizer,
        "num_warmup_steps": config.num_warmup_steps,
        "num_training_steps": (
            (len(train_dataset) - 1)
            // (
                config.finetune_train_batch_size
                * config.gradient_accumulation_steps
            )
            + 1
        )
        * config.epochs,
    }
    if config.lr_scheduler_type in ("cosine", "cosine_with_warmup"):
        lr_scheduler = transformers.get_cosine_schedule_with_warmup(
            **kwargs_lr_scheduler
        )
    elif config.lr_scheduler_type in ("linear", "linear_with_warmup"):
        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            **kwargs_lr_scheduler
        )
    else:
        raise NotImplementedError

    return optimizer, lr_scheduler


class CustomTrainer(Trainer):
    def __init__(self, *args, train_loader=None, test_loader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=self.model.config.pad_token_id
        )
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_train_dataloader(self) -> DataLoader:
        return self.train_loader

    def get_eval_dataloader(self, _) -> DataLoader:
        return self.test_loader


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)


def finetune(args, model, tokenizer, skip_lora=False, base_model=False):
    # get the dataset for finetuning
    finetune_ds = data_utils.get_dataset(args.finetune_dataset)
    finetune_train_loader = data_utils.prepare_dataloader(
        dataset=finetune_ds["train"],
        tokenizer=tokenizer,
        max_seqlen=args.finetune_train_seqlen,
        batch_size=args.finetune_train_batch_size,
        nsamples=args.finetune_train_nsamples,
        varied_seqlen=args.varied_seqlen,
        seed=args.seed,
    )
    finetune_test_loader = data_utils.prepare_dataloader(
        dataset=finetune_ds["test"],
        tokenizer=tokenizer,
        max_seqlen=args.finetune_test_seqlen,
        batch_size=args.finetune_test_batch_size,
        nsamples=args.finetune_test_nsamples,
        varied_seqlen=args.varied_seqlen,
        seed=args.seed,
    )

    if skip_lora == False:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            task_type=TaskType.CAUSAL_LM,
            target_modules=lora_target_map(args.model_name)[
                args.lora_target_option
            ],
        )

        lora_model = get_peft_model(model, lora_config)
    else:
        lora_model = model

    log_mode = "wandb" if args.no_wandb == False else "tensorboard"

    if args.finetune:
        lora_model.print_trainable_parameters()
        # create optimizer and scheduler
        optimizer, lr_scheduler = get_optimizer_and_scheduler(
            lora_model, finetune_ds["train"], args
        )

        try:
            wandb_trainer_name = args.wandb_trainer_name
        except:
            wandb_trainer_name = "huggingface"

        training_args = TrainingArguments(
            # output_dir=args.st_checkpoint_dir,  # output directory
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.finetune_train_batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.finetune_test_batch_size,  # batch size for evaluation
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            disable_tqdm=False,
            load_best_model_at_end=True,
            eval_steps=args.eval_steps,
            evaluation_strategy=args.evaluation_strategy,
            report_to=log_mode,
            metric_for_best_model="eval_loss",
            greater_is_better=False,  # lower eval_loss is better,
            gradient_checkpointing=True,
            output_dir=wandb_trainer_name,
        )

        trainer = CustomTrainer(
            model=lora_model,
            tokenizer=tokenizer,
            train_loader=finetune_train_loader,
            test_loader=finetune_test_loader,
            args=training_args,
            optimizers=(optimizer, lr_scheduler),
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stopping_patience
                )
            ],
        )

        # required to enable gradient_checkpointing
        lora_model.enable_input_require_grads()

        lora_model.train()
        if base_model == False:
            try:
                trainer.train()
            except:
                pass
    return lora_model


def _get_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        dest="loglevel",
        default="DEBUG",
        help="Set the log level for the logging module.",
    )
    parser.add_argument(
        "--use_gpu",
        dest="use_gpu",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to use the gpu or not.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for sampling the calibration data.",
    )

    parser.add_argument(
        "--model_name",
        dest="model_name",
        default="",
        help="Name of the model checkpoint.",
    )
    parser.add_argument(
        "--model_path",
        dest="model_path",
        default="",
        help="The path containing the model checkpoint.",
    )
    parser.add_argument(
        "--model_save_path",
        dest="model_save_path",
        default="",
        help="The path to save the sparsity action model.",
    )

    parser.add_argument(
        "--num_episodes",
        dest="num_episodes",
        default=20,
        help="Number of episodes to train model.",
    )

    parser.add_argument(
        "--sparsity_level",
        dest="sparsity_level",
        default=0.2,
        help="Sparsity level of the model.",
    )

    parser.add_argument(
        "--sparsity_technique",
        dest="sparsity_technique",
        default="bernoulligpt",
        help="Type of sparsity injection - bernoulli or random",
    )

    parser.add_argument(
        "--activation",
        type=str,
        help="Default activation or new activation like LeakySiLU, LeakyGeLU",
        default="",
    )
    parser.add_argument(
        "--learning-rate-action",
        dest="learning_rate_action",
        default=0.001,
        help="Learning rate to train sparsity model.",
    )

    # Perplexity evaluation command-line arguments
    parser.add_argument(
        "--ppl-eval-dataset",
        type=str,
        help="Dataset to evaluate perplexity.",
        choices=["wikitext2", "ptb", "c4", "alpaca"],
        default="wikitext2",
    )
    parser.add_argument(
        "--ppl-eval-nsamples",
        type=int,
        help="Number of samples of the perplexity eval dataset to load.",
        default=128,
    )
    parser.add_argument(
        "--ppl-eval-batch-size",
        type=int,
        default=8,
        help="Batch size for evaluating the perplexity.",
    )
    parser.add_argument(
        "--ppl-eval-seqlen",
        type=int,
        default=2048,
        help="Sequence length for evaluating the perplexity.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        help="Data type to use.",
        choices=["fp32", "fp16"],
        default="fp16",
    )
    parser.add_argument(
        "--varied-seqlen",
        action="store_true",
        help="Varied sequence lengths in the calibration data.",
    )

    parser.add_argument(
        "--finetune", action="store_true", help="Whether to finetune model."
    )

    # finetuning command-line arguments
    parser.add_argument(
        "--finetune-dataset",
        type=str,
        help="Dataset to finetune on.",
        choices=["wikitext2", "ptb", "c4", "alpaca"],
        default="wikitext2",
    )
    parser.add_argument(
        "--finetune-train-nsamples",
        type=int,
        help="Number of samples to load from the train set for finetuning.",
        default=4096,
    )
    parser.add_argument(
        "--finetune-test-nsamples",
        type=int,
        help="Number of samples to load from the test set for finetuning.",
        default=128,
    )
    parser.add_argument(
        "--finetune-train-batch-size",
        type=int,
        default=1,
        help="Batch size for finetuning training.",
    )
    parser.add_argument(
        "--finetune-test-batch-size",
        type=int,
        default=8,
        help="Batch size for finetuning testing.",
    )
    parser.add_argument(
        "--finetune-train-seqlen",
        type=int,
        default=2048,
        help="Sequence length for finetuning training.",
    )
    parser.add_argument(
        "--finetune-test-seqlen",
        type=int,
        default=2048,
        help="Sequence length for finetuning testing.",
    )

    parser.add_argument(
        "--use-kld", action="store_true", help="To use KLD loss"
    )

    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-scheduler-type", type=str, default="linear")
    parser.add_argument("--num-warmup-steps", type=int, default=400)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=5)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--evaluation-strategy", type=str, default="steps")
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=100)

    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-bias", type=str, default="none")

    parser.add_argument(
        "--st_checkpoint_dir",
        type=str,
        default="../models/",
        help="Path for syne-tune to save finetuning checkpoints.",
    )
    parser.add_argument(
        "--lora-target-option",
        default="attn_head_and_mlp",
        help="target module option to apply lora to (names of attn i/p, attn o/p and mlp in LayerAdapter)",
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="bernoulligpt-training",
        help="wandb project name.",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable wandb."
    )

    args = parser.parse_args()

    if args.no_wandb == False:
        config = vars(args)
        wandb.login()
        wandb.init(project=args.wandb_project, config=config)

    try:
        os.makedirs(args.model_save_path)
    except:
        pass

    return args


def get_model_with_activation():
    model_adapter, _ = hf_utils.get_model_and_tokenizer(args.model_name)
    model = cp(model_adapter.model).to(device)

    if args.activation != "":
        for layer in get_all_layers_before_lora(args.model_name, model):
            if "opt" in args.model_name or "phi" in args.model_name:
                layer.activation_fn = act_fn  # (3072, 768)
            elif "llama" in args.model_name:
                layer.mlp.act_fn = act_fn
            elif "falcon" in args.model_name:
                layer.mlp.act_fn = act_fn
            else:
                ValueError(
                    "Model type is not supported. Only OPT, Phi, Llama and Falcon models are supported."
                )
                # optimizer.zero_grad()

    del model_adapter

    return model


if __name__ == "__main__":
    args = _get_parse_args()

    args.num_episodes = int(args.num_episodes)
    args.sparsity_level = float(args.sparsity_level)
    args.learning_rate_action = float(args.learning_rate_action)

    logger.setLevel(getattr(logging, args.loglevel.upper()))
    logger.debug(f"args: {args}")

    model_checkpoint_save_path = os.path.join(
        args.model_save_path,
        "model={}_finetune={}_sparsity={}.ckpt".format(
            args.model_name.split("/")[-1], "False", args.sparsity_level
        ),
    )

    wandb_trainer_name = "model={}_sparsity={}_activation={}".format(
        args.model_name.split("/")[-1],
        args.sparsity_level,
        args.activation.lower(),
    )

    args.wandb_trainer_name = wandb_trainer_name

    if args.activation.lower() == "leakysilu":
        act_fn = LeakySiLU()
    elif args.activation.lower() == "leakygelu":
        act_fn = LeakyGeLU()
    elif args.activation.lower() == "relu":
        act_fn = nn.ReLU()

    # otherwise, continue with the experiments
    # set random seed
    set_seed(args.seed)

    device = "cuda" if (torch.cuda.is_available() and args.use_gpu) else "cpu"

    if args.dtype == "fp16":
        config.dtype = torch.float16
    elif args.dtype == "fp32":
        config.dtype = torch.float32
    else:
        raise argparse.ArgumentTypeError(
            f"Data type should be one of 'fp16', 'fp32'"
        )

    # config = AutoConfig.from_pretrained(args.model_name)

    model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(args.model_name)

    model_adapter.model.to(device)

    config = model_adapter.model.config

    # get the dataset for perplexity evaluation
    ppl_ds = data_utils.get_dataset(args.ppl_eval_dataset)
    ppl_eval_loader = data_utils.prepare_dataloader(
        dataset=ppl_ds["validation"],
        tokenizer=tokenizer,
        max_seqlen=args.ppl_eval_seqlen,
        batch_size=args.ppl_eval_batch_size,
        nsamples=args.ppl_eval_nsamples,
        varied_seqlen=args.varied_seqlen,
        seed=args.seed,
    )

    model_adapter.model.eval()

    # print (model_adapter.model)

    dataset_ppl = gpu_utils.evaluate_ppl(
        model_adapter.model, config.pad_token_id, ppl_eval_loader
    )

    print(f"PPL before finetuning: {dataset_ppl:.4f}")

    if args.no_wandb == False:
        #    wandb.log({"Memory Before": get_memory_consumption(model_orig)})
        wandb.log({"PPL Before": dataset_ppl})
    #    #wandb.log({"Inference time before": start.elapsed_time(end)})
    #    #wandb.log({"pre_finetune_ppl": dataset_ppl})

    orig_parameters = sum(p.numel() for p in model_adapter.model.parameters())

    print(orig_parameters)

    all_svds_main_model = {}
    for i, layer in enumerate(
        get_all_layers_before_lora(args.model_name, model_adapter.model)
    ):
        if i <= 9999:
            if "opt" in args.model_name:
                main_w = layer.fc1.weight.data
            elif "phi" in args.model_name:
                main_w = layer.mlp.fc1.weight.data
            elif "llama" in args.model_name:
                main_w = layer.mlp.gate_proj.weight.data
            elif "falcon" in args.model_name:
                main_w = layer.mlp.dense_h_to_4h.weight.data
            else:
                ValueError(
                    "Model type is not supported. Only OPT, Phi, Llama and Falcon models are supported."
                )

            if main_w.dtype == torch.float16:
                main_w = main_w.to(torch.float32)

            _, s1, _ = torch.svd(main_w)

            all_svds_main_model[str(i) + "inter"] = s1

            if "llama" in args.model_name:
                main_w = layer.self_attn.k_proj.weight.data
                if main_w.dtype == torch.float16:
                    main_w = main_w.to(torch.float32)
                _, s1, _ = torch.svd(main_w)
                all_svds_main_model[str(i) + "key"] = s1

    del model_adapter

    gc.collect()

    if args.sparsity_level > 0:
        if args.sparsity_technique != "random":
            all_rewards = []
            # best_score = 999999999
            best_score = 0

            if "opt" in args.model_name:
                action_model = SparsityPredictor(
                    config.hidden_size, config.ffn_dim, args.sparsity_level
                )
            elif "llama" in args.model_name or "phi" in args.model_name:
                action_model = SparsityPredictor(
                    config.hidden_size,
                    config.intermediate_size,
                    args.sparsity_level,
                )
            elif "falcon" in args.model_name:
                action_model = SparsityPredictor(
                    config.hidden_size,
                    config.ffn_hidden_size,
                    args.sparsity_level,
                )
            else:
                raise ValueError(
                    "Model type is not supported. Only OPT, Phi, Llama and Falcon models are supported."
                )

            if os.path.exists(model_checkpoint_save_path):
                action_model.load_state_dict(
                    torch.load(model_checkpoint_save_path, weights_only=True)
                )
                action_model.to(device)
            else:
                action_model.to(device)
                action_model.train()
                print(action_model)

                print(
                    "Total number of parameters",
                    sum(
                        p.numel()
                        for p in action_model.parameters()
                        if p.requires_grad
                    ),
                )

                optimizer = torch.optim.AdamW(
                    action_model.parameters(), lr=args.learning_rate_action
                )

                best_accuracy = 0

                scaler = torch.cuda.amp.GradScaler()

                for episode in tqdm(range(args.num_episodes)):
                    model = get_model_with_activation()

                    state_pool = []
                    action_pool = []
                    reward_pool = []

                    total_loss = 0
                    count = 0
                    total_reward = 0

                    for i, layer in enumerate(
                        get_all_layers_before_lora(args.model_name, model)
                    ):
                        if i <= 9999:
                            if "opt" in args.model_name:
                                weight = layer.fc1.weight.data  # (3072, 768)
                            elif "phi" in args.model_name:
                                weight = (
                                    layer.mlp.fc1.weight.data
                                )  # (3072, 768)
                            elif "llama" in args.model_name:
                                weight = layer.mlp.gate_proj.weight.data
                            elif "falcon" in args.model_name:
                                weight = layer.mlp.dense_h_to_4h.weight.data
                            else:
                                ValueError(
                                    "Model type is not supported. Only OPT, Phi, Llama and Falcon models are supported."
                                )

                            state = Variable(cp(weight))

                            # print (weight)

                            with torch.autocast(
                                device_type=device, dtype=torch.float16
                            ):
                                o = action_model(state)  # (3072, )
                                feat_len = state.shape[0]
                                o = torch.nan_to_num(o).to(o.device)
                                row_indices = (
                                    torch.multinomial(
                                        o,
                                        int(
                                            (1 - args.sparsity_level) * feat_len
                                        ),
                                        replacement=False,
                                    )
                                    .sort()
                                    .values
                                )

                            slicing(args.model_name, layer, row_indices)

                            if "llama" in args.model_name:
                                weight = layer.self_attn.k_proj.weight.data
                                state2 = Variable(cp(weight))
                                with torch.autocast(
                                    device_type=device, dtype=torch.float16
                                ):
                                    o = action_model(state2)  # (3072, )
                                    feat_len = state2.shape[0]
                                    o = torch.nan_to_num(o).to(o.device)
                                    slice_num = get_closest_even_number(
                                        int(
                                            (1 - args.sparsity_level) * feat_len
                                        ),
                                        layer.self_attn.config.num_attention_heads,
                                    )
                                    row_indices2 = (
                                        torch.multinomial(
                                            o, slice_num, replacement=False
                                        )
                                        .sort()
                                        .values
                                    )

                                slicing_qkv(
                                    args.model_name, layer, row_indices2
                                )

                            else:
                                state2 = None

                            # get the updated rewards
                            if "opt" in args.model_name:
                                new_w = layer.fc1.weight.data
                            elif "phi" in args.model_name:
                                new_w = layer.mlp.fc1.weight.data
                            elif "llama" in args.model_name:
                                new_w = layer.mlp.gate_proj.weight.data
                            elif "falcon" in args.model_name:
                                new_w = layer.mlp.dense_h_to_4h.weight.data
                            else:
                                ValueError(
                                    "Model type is not supported. Only OPT, Phi, Llama and Falcon models are supported."
                                )

                            reward = calculate_activation_reward(
                                all_svds_main_model[str(i) + "inter"], new_w
                            )  # - action_model.calculate_l1_loss()

                            state_pool.append(state)
                            action_pool.append(row_indices)
                            reward_pool.append(reward.item())

                            if "llama" in args.model_name:
                                new_w = layer.self_attn.k_proj.weight.data
                                reward2 = calculate_activation_reward(
                                    all_svds_main_model[str(i) + "key"], new_w
                                )

                            if state2 is not None:
                                state_pool.append(state2)
                                action_pool.append(row_indices2)
                                reward_pool.append(reward2.item())

                            total_reward += reward.item() + reward2.item()
                            count += 1

                            # print (reward)

                    reward_pool = discount_rewards(reward_pool)

                    for i in range(len(state_pool)):
                        with torch.autocast(
                            device_type=device, dtype=torch.float16
                        ):
                            state = state_pool[i]
                            action = Variable(action_pool[i])
                            reward = reward_pool[i]

                            o = action_model(state)  # (3072, )
                            # y = Bernoulli(o)

                            loss = (
                                -1
                                * torch.gather(torch.log(o), 0, action).sum()
                                * reward
                            )

                            total_loss += loss.item()

                            if args.use_kld:
                                kld_loss = action_model.calculate_total_loss()
                                loss += kld_loss

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        torch.nn.utils.clip_grad_norm_(
                            action_model.parameters(), 1.0
                        )

                    print("Episode ", episode, " loss ", total_loss / count)
                    if args.no_wandb == False:
                        wandb.log({"Episodic Loss": total_loss / count})
                        wandb.log({"Episodic Reward": total_reward})

                    state_pool = []
                    action_pool = []
                    reward_pool = []

                    print(
                        "Episode", episode, "Avg reward", total_reward / count
                    )

                    # if total_loss < best_score:
                    #    best_score =  total_loss
                    #    torch.save(action_model.state_dict(), model_checkpoint_save_path)
                    if total_reward > best_score:
                        best_score = total_reward
                        torch.save(
                            action_model.state_dict(),
                            model_checkpoint_save_path,
                        )

            ######## inference ############
            if os.path.exists(model_checkpoint_save_path):
                action_model.load_state_dict(
                    torch.load(model_checkpoint_save_path, weights_only=True)
                )
            else:
                pass

            action_model.eval()

            model = get_model_with_activation()

            for i, layer in enumerate(
                get_all_layers_before_lora(args.model_name, model)
            ):
                if i <= 9999:
                    if "opt" in args.model_name:
                        weight = layer.fc1.weight.data  # (3072, 768)
                    elif "phi" in args.model_name:
                        weight = layer.mlp.fc1.weight.data  # (3072, 768)
                    elif "llama" in args.model_name:
                        weight = layer.mlp.gate_proj.weight.data
                    elif "falcon" in args.model_name:
                        weight = layer.mlp.dense_h_to_4h.weight.data
                    else:
                        ValueError(
                            "Model type is not supported. Only OPT, Phi, Llama and Falcon models are supported."
                        )

                    state = Variable(cp(weight))

                    # print (weight)

                    with torch.autocast(
                        device_type=device, dtype=torch.float16
                    ):
                        with torch.no_grad():
                            o = action_model(state)  # (3072, )
                            feat_len = state.shape[0]
                            o = torch.nan_to_num(o).to(o.device)
                            row_indices = (
                                torch.multinomial(
                                    o,
                                    int((1 - args.sparsity_level) * feat_len),
                                    replacement=False,
                                )
                                .sort()
                                .values
                            )

                    slicing(args.model_name, layer, row_indices)

                    if "llama" in args.model_name:
                        weight = layer.self_attn.k_proj.weight.data
                        state = Variable(cp(weight))
                        with torch.autocast(
                            device_type=device, dtype=torch.float16
                        ):
                            with torch.no_grad():
                                o = action_model(state)  # (3072, )
                                feat_len = state.shape[0]
                                o = torch.nan_to_num(o).to(o.device)
                                slice_num = get_closest_even_number(
                                    int((1 - args.sparsity_level) * feat_len),
                                    layer.self_attn.config.num_attention_heads,
                                )
                                row_indices2 = (
                                    torch.multinomial(
                                        o, slice_num, replacement=False
                                    )
                                    .sort()
                                    .values
                                )

                            slicing_qkv(args.model_name, layer, row_indices2)
                            layer.self_attn.head_dim = int(
                                slice_num
                                // layer.self_attn.config.num_attention_heads
                            )
                            # layer.self_attn.max_position_embeddings = int(slice_num//layer.self_attn.config.num_attention_heads) - 1
                            layer.self_attn._init_rope()
                            layer.self_attn.rotary_emb.to(device)
        else:
            model = get_model_with_activation()
            for layer in get_all_layers_before_lora(args.model_name, model):
                random_slicing(args.model_name, layer, args.sparsity_level)

    # model = torch.load("sliced_model.pt")

    print(model)

    new_parameters = sum(p.numel() for p in model.parameters())

    print(new_parameters)

    if args.finetune:
        model.train()
        model = finetune(args, model, tokenizer, skip_lora=False)
    model.eval()

    dataset_ppl2 = gpu_utils.evaluate_ppl(
        model, model.config.pad_token_id, ppl_eval_loader
    )

    print(f"PPL after finetuning: {dataset_ppl2:.4f}")
    # print(f'Total inference time after: {start.elapsed_time(end):.4f}')
    print(f"Sparsity achieved: {int((1-new_parameters/orig_parameters)*100)}%")

    if args.no_wandb == False:
        #    wandb.log({"Memory After": get_memory_consumption(model)})
        wandb.log({"PPL After": dataset_ppl2})
        wandb.log(
            {
                "Sparsity Achieved": int(
                    (1 - new_parameters / orig_parameters) * 100
                )
            }
        )
