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
    AutoModelForCausalLM
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
    return mem_usage_bytes / 1024 ** 3

def get_all_layers(model_name, model):
    if 'opt' in model_name:
        all_layers = model.base_model.model.model.decoder.layers
    elif  'phi' in model_name:
        all_layers = model.base_model.model.model.layers
    elif 'llama' in model_name:
        all_layers = model.base_model.model.model.layers
    elif 'falcon' in model_name:
        all_layers = model.base_model.model.transformer.h
    else:
        raise ValueError("Model type is not supported. Only OPT, Phi, Llama and Falcon models are supported.")    

    return all_layers

def get_all_layers_before_lora(model_name, model):
    if 'opt' in model_name:
        all_layers = model.model.decoder.layers
    elif 'phi' in model_name:
        all_layers = model.model.layers
    elif 'llama' in model_name:
        all_layers = model.model.layers
    elif 'falcon' in model_name:
        all_layers = model.model.transformer.h
    else:
        raise ValueError("Model type is not supported. Only OPT, Phi, Llama and Falcon models are supported.")    

    return all_layers


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
            (len(train_dataset) - 1) // (config.finetune_train_batch_size * config.gradient_accumulation_steps) + 1
        )
        * config.epochs,
    }
    if config.lr_scheduler_type in ("cosine", "cosine_with_warmup"):
        lr_scheduler = transformers.get_cosine_schedule_with_warmup(**kwargs_lr_scheduler)
    elif config.lr_scheduler_type in ("linear", "linear_with_warmup"):
        lr_scheduler = transformers.get_linear_schedule_with_warmup(**kwargs_lr_scheduler)
    else:
        raise NotImplementedError

    return optimizer, lr_scheduler

class CustomTrainer(Trainer):
    def __init__(self, *args, train_loader=None, test_loader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.model.config.pad_token_id)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_train_dataloader(self) -> DataLoader:
        return self.train_loader

    def get_eval_dataloader(self, _) -> DataLoader:
        return self.test_loader

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)

def finetune(args, model, tokenizer, skip_lora=False):
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
            target_modules=lora_target_map(args.model_name)[args.lora_target_option],
        )

        lora_model = get_peft_model(model, lora_config)
    else:
        lora_model = model

    if args.finetune:
        lora_model.print_trainable_parameters()
        # create optimizer and scheduler
        optimizer, lr_scheduler = get_optimizer_and_scheduler(lora_model, finetune_ds["train"], args)

        training_args = TrainingArguments(
            output_dir=args.st_checkpoint_dir,  # output directory
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
            report_to='tensorboard',
            metric_for_best_model="eval_loss",
            greater_is_better=False,  # lower eval_loss is better,
            gradient_checkpointing=True,
        )

        trainer = CustomTrainer(
            model=lora_model,
            tokenizer=tokenizer,
            train_loader=finetune_train_loader,
            test_loader=finetune_test_loader,
            args=training_args,
            optimizers=(optimizer, lr_scheduler),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
        )

        # required to enable gradient_checkpointing
        lora_model.enable_input_require_grads()

        lora_model.train()
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

    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    
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
        "--activation",
        type=str,
        help="Default activation or new activation like LeakySiLU, LeakyGeLU",
        default=''
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
    parser.add_argument("--ppl-eval-batch-size", type=int, default=8, help="Batch size for evaluating the perplexity.")
    parser.add_argument(
        "--ppl-eval-seqlen", type=int, default=2048, help="Sequence length for evaluating the perplexity."
    )

    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--varied-seqlen", action="store_true", help="Varied sequence lengths in the calibration data.")

    parser.add_argument("--finetune", action="store_true", help="Whether to finetune model.")

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
    parser.add_argument("--finetune-train-batch-size", type=int, default=1, help="Batch size for finetuning training.")
    parser.add_argument("--finetune-test-batch-size", type=int, default=8, help="Batch size for finetuning testing.")
    parser.add_argument(
        "--finetune-train-seqlen", type=int, default=2048, help="Sequence length for finetuning training."
    )
    parser.add_argument(
        "--finetune-test-seqlen", type=int, default=2048, help="Sequence length for finetuning testing."
    )

    parser.add_argument("--use-kld", action="store_true", help="To use KLD loss")

    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--adam-beta1', type=float, default=0.9)
    parser.add_argument('--adam-beta2', type=float, default=0.95)
    parser.add_argument('--adam-epsilon', type=float, default=1e-8)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--lr-scheduler-type', type=str, default="linear")
    parser.add_argument('--num-warmup-steps', type=int, default=400)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4)
    parser.add_argument('--early-stopping-patience', type=int, default=5)

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--evaluation-strategy', type=str, default="steps")
    parser.add_argument('--eval-steps', type=int, default=100)
    parser.add_argument('--save-steps', type=int, default=100)
    parser.add_argument('--save-total-limit', type=int, default=1)
    parser.add_argument('--logging-steps', type=int, default=100)

    parser.add_argument('--lora-alpha', type=float, default=32.0)
    parser.add_argument('--lora-dropout', type=float, default=0.1)
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-bias', type=str, default="none")

    parser.add_argument(
        '--st_checkpoint_dir', type=str, default="../models/", help="Path for syne-tune to save finetuning checkpoints."
    )
    parser.add_argument(
        '--lora-target-option',
        default="attn_head_and_mlp",
        help="target module option to apply lora to (names of attn i/p, attn o/p and mlp in LayerAdapter)",
    )

    parser.add_argument('--wandb-project', type=str, default="benoulligpt", help="wandb project name.")
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")

    args = parser.parse_args()

    args.finetune = True

    if args.no_wandb == False:
        config = vars(args)
        wandb.login()
        wandb.init(project=args.wandb_project,config=config)

    try:
        os.makedirs(args.model_save_path)
    except:
        pass

    return args


if __name__ == "__main__":
    args = _get_parse_args()

    logger.setLevel(getattr(logging, args.loglevel.upper()))
    logger.debug(f"args: {args}")

    if args.activation.lower() == 'leakysilu':
        act_fn = LeakySiLU()
    elif args.activation.lower() == 'leakygelu':
        act_fn = LeakyGeLU()
    elif args.activation.lower() == 'relu':
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
        raise argparse.ArgumentTypeError(f"Data type should be one of 'fp16', 'fp32'")

    #config = AutoConfig.from_pretrained(args.model_name)

    model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(args.model_name)

    model_orig = cp(model_adapter.model).to(device)

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
    
    if args.activation != '':
        for layer in get_all_layers_before_lora(args.model_name, model_orig):
            if 'opt' in args.model_name:
                layer.activation_fn = act_fn  # (3072, 768)
            if 'phi' in args.model_name:
                layer.activation_fn = act_fn  # (3072, 768)
            elif 'llama' in args.model_name:
                layer.mlp.act_fn = act_fn
            elif 'falcon' in args.model_name:
                layer.mlp.act_fn = act_fn
            else:
                ValueError("Model type is not supported. Only OPT, Phi, Llama and Falcon models are supported.")

    dataset_ppl = gpu_utils.evaluate_ppl(model_orig, model_orig.config.pad_token_id, ppl_eval_loader)
    print(f'PPL before finetuning: {dataset_ppl:.4f}')
    
    if args.no_wandb == False:
        wandb.log({"PPL Before": dataset_ppl})

    model_orig = finetune(args, model_orig, tokenizer)

    model_orig.eval()

    dataset_ppl = gpu_utils.evaluate_ppl(model_orig, model_orig.config.pad_token_id, ppl_eval_loader)

    print(f'PPL after finetuning: {dataset_ppl:.4f}')

    if args.no_wandb == False:
        wandb.log({"PPL After": dataset_ppl})
