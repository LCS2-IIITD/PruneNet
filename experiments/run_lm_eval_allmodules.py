# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import logging
import os

import lm_eval
import torch
import wandb
from lm_eval import tasks
from lm_eval import utils as lm_eval_utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import initialize_tasks

from bo_options import lora_target_map
from slicegpt import data_utils, gpu_utils, hf_utils, utils
from slicegpt.config import config

import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM
)
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from peft import LoraConfig, TaskType, get_peft_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.distributions.uniform import Uniform
from torch.distributions import Categorical, Bernoulli
from torch.nn.utils.parametrizations import orthogonal

from copy import deepcopy as cp

from trainable_activation_sparsity import slicing, get_optimizer_and_scheduler, finetune, set_seed, CustomTrainer, random_slicing
        
from trainable_activation_sparsity_allmodules import slicing_qkv, get_closest_even_number, SparsityPredictor
    
from bernoulligpt_utils import LeakyGeLU, LeakySiLU

def get_all_layers(model_name, model):
    if 'opt' in model_name:
        all_layers = model.base_model.decoder.layers
    elif  'phi' in model_name:
        all_layers = model.base_model.model.layers
    elif 'llama' in model_name:
        all_layers = model.base_model.layers
    elif 'falcon' in model_name:
        all_layers = model.base_model.transformer.h
    else:
        raise ValueError("Model type is not supported. Only OPT, Phi, Llama and Falcon models are supported.")    

    return all_layers

def get_all_layers_before_lora(model_name, model):
    if 'opt' in model_name:
        all_layers = model.model.decoder.layers
    elif  'phi' in model_name:
        all_layers = model.model.layers
    elif 'llama' in model_name:
        all_layers = model.model.layers
    elif 'falcon' in model_name:
        all_layers = model.model.transformer.h
    else:
        raise ValueError("Model type is not supported. Only OPT, Phi, Llama and Falcon models are supported.")    

    return all_layers

TASK_METRIC_MAP = {
    "mmlu_abstract_algebra": "acc,none",
    "mmlu_business_ethics": "acc,none",
    "mmlu_college_computer_science": "acc,none",
    "mmlu_college_mathematics": "acc,none",
    "mmlu_conceptual_physics": "acc,none",
    "mmlu_formal_logic": "acc,none",
    "mmlu_machine_learning": "acc,none",
    "mmlu_miscellaneous": "acc,none",
    "mmlu_philosophy": "acc,none",
    "mmlu_global_facts": "acc,none",
    "arc_challenge": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "piqa": "acc_norm,none",
    "winogrande": "acc,none",
}

def eval_arg_parser(interactive: bool = True) -> argparse.Namespace:
    initialize_tasks()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model to load",
    )
    path_group = parser.add_mutually_exclusive_group()
    path_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to load the model and tokenizer from (required for local models, not required for HF models)",
    )
    path_group.add_argument(
        "--model-save-path",
        type=str,
        help="The path to load the sparsity action model.",
        default='../models2/'
    )
    path_group.add_argument(
        "--sliced-model-path",
        type=str,
        help="Path to load the model to fine-tune (sliced) and tokenizer from",
        default=None,
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1))"
    )
    parser.add_argument(
        "--checkpoint_sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1))"
    )
    path_group.add_argument(
        "--activation",
        type=str,
        help="Default activation or new activation like LeakySiLU, LeakyGeLU",
        default=''
    )

    parser.add_argument(
        "--round-interval",
        type=int,
        default=8,
        help="Interval for rounding the weights (the best value may depend on your hardware)",
    )
    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluating with lm eval harness.")
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )

    parser.add_argument('--use-slicing', action="store_true", help="Use slicing.")
    parser.add_argument(
        "--sparsity_technique",
        dest="sparsity_technique",
        default='bernoulligpt',
        help="Type of sparsity injection - bernoulli or random",
    )
    parser.add_argument('--finetune', action="store_true", help="Fine tune model.")

    parser.add_argument('--wandb-project', type=str, default="bernoulligpt-lm-eval", help="wandb project name.")
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande"],
        choices=lm_eval_utils.MultiChoice(tasks.ALL_TASKS),
    )
    parser.add_argument('--num-fewshot', type=int, default=0, help="Number of fewshots for all tasks.")
    parser.add_argument("--save-dir", type=str, default=".", help="Path to save the lm eval results")

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

    parser.add_argument("--dtype", type=str, help="Data type to use.", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--varied-seqlen", action="store_true", help="Varied sequence lengths in the calibration data.")

    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    
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
    parser.add_argument('--eval-steps', type=int, default=16)
    parser.add_argument('--save-steps', type=int, default=16)
    parser.add_argument('--save-total-limit', type=int, default=1)
    parser.add_argument('--logging-steps', type=int, default=1)

    parser.add_argument('--lora-alpha', type=float, default=32.0)
    parser.add_argument('--lora-dropout', type=float, default=0.1)
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-bias', type=str, default="none")

    parser.add_argument(
        '--st_checkpoint_dir', type=str, default=".", help="Path for syne-tune to save finetuning checkpoints."
    )
    parser.add_argument(
        '--lora-target-option',
        default="attn_head_and_mlp",
        help="target module option to apply lora to (names of attn i/p, attn o/p and mlp in LayerAdapter)",
    )

    return parser.parse_args() if interactive else parser.parse_args('')


def process_eval_args(args: argparse.Namespace):
    logging.info(f'Parsed arguments:')
    for arg, argv in vars(args).items():
        logging.info(f'{arg} = {argv}')


def calculate_avg_accuracy(task_names: str, results: dict) -> float:
    n_tasks = len(task_names)
    acc_cumul = sum(result.get(TASK_METRIC_MAP[task]) for task, result in results.items() if 'mmlu' not in task)

    questions_per_mmlu_task = {
        task_name: lm_eval.tasks.get_task_dict([task_name])[task_name].dataset["test"].num_rows
        for task_name in task_names
        if 'mmlu' in task_name
    }

    if not questions_per_mmlu_task:
        return acc_cumul / n_tasks

    # Calculate average accuracy for mmlu tasks, weighted by number of questions in each task
    acc_mmlu = sum(
        result.get(TASK_METRIC_MAP[task]) * questions_per_mmlu_task[task]
        for task, result in results.items()
        if 'mmlu' in task
    )
    acc_mmlu_avg = acc_mmlu / sum(questions_per_mmlu_task.values())
    if args.no_wandb == False:
        wandb.log({'acc_mmlu_avg': acc_mmlu_avg})

    return (acc_cumul + acc_mmlu_avg) / (n_tasks - len(questions_per_mmlu_task) + 1)

def eval_main(args: argparse.Namespace) -> None:
    logging.info("Running SliceGPT LM eval experiment.")

    if args.activation.lower() == 'leakysilu' or args.activation.lower() == 'silu' :
        act_fn = LeakySiLU()
    elif args.activation.lower() == 'leakygelu':
        act_fn = LeakyGeLU()
    elif args.activation.lower() == 'relu':
        act_fn = nn.ReLU()
        
    logging.info(f"PyTorch device: {config.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    try:
        wandb.init(project=args.wandb_project, config=args, mode='disabled' if args.no_wandb else None)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb')
        wandb.init(project=args.wandb_project, mode='disabled')

    if args.sliced_model_path:
        # load the sliced model
        logging.info(f"Loading sliced {args.model} model from {args.sliced_model_path} with sparsity {args.sparsity}")
        model_adapter, tokenizer = hf_utils.load_sliced_model(
            args.model,
            args.sliced_model_path,
            sparsity=args.sparsity,
            token=args.hf_token,
            round_interval=args.round_interval,
        )
    else:
        # load the original model
        logging.info(f"Loading {args.model} model")
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(args.model, args.model_path, token=args.hf_token)

    # the lm eval harness ties the weights, but this should not be done for sliced models unless the lm_head was sliced
    model_adapter.model.tie_weights = lambda: None

    if args.distribute_model:
        # distribute model across available GPUs
        gpu_utils.distribute_model(model_adapter)
    else:
        model_adapter.model.to(config.device)

    ### LM Eval Harness ###

    model = cp(model_adapter.model)

    if args.use_slicing:
        if args.sparsity_technique != 'random':
            if 'opt' in args.model:
                action_model = SparsityPredictor(
                    model.config.hidden_size, model.config.ffn_dim, args.sparsity
                )
            elif 'llama' in args.model or 'phi' in args.model:
                action_model = SparsityPredictor(
                    model.config.hidden_size, model.config.intermediate_size, args.sparsity
                )
            elif 'falcon' in args.model:
                action_model = SparsityPredictor(
                    model.config.hidden_size, model.config.ffn_hidden_size, args.sparsity
                )
            else:
                raise ValueError("Model type is not supported. Only OPT, Phi, Llama and Falcon models are supported.")
            
            if args.checkpoint_sparsity == 0:
                args.checkpoint_sparsity = args.sparsity

            model_checkpoint_save_path = os.path.join(args.model_save_path, \
            "model={}_finetune={}_sparsity={}.ckpt".format(args.model.split("/")[-1], "False", args.checkpoint_sparsity))

            action_model.to(config.device)
            
            action_model.load_state_dict(torch.load(model_checkpoint_save_path, weights_only=True))
            action_model.eval()

            orig_parameters = sum(p.numel() for p in model.parameters())

            for layer in get_all_layers_before_lora(args.model, model):
                if 'opt' in args.model:
                    weight = layer.fc1.weight.data  # (3072, 768)
                elif 'phi' in args.model:
                    weight = layer.mlp.fc1.weight.data  # (3072, 768)
                elif 'llama' in args.model:
                    weight = layer.mlp.gate_proj.weight.data
                elif 'falcon' in args.model:
                    weight = layer.mlp.dense_h_to_4h.weight.data
                else:
                    ValueError("Model type is not supported. Only OPT, Llama and Falcon models are supported.")

                state = Variable(cp(weight))
                
                # print (weight)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    with torch.no_grad():
                        o = action_model(state)  # (3072, )
                        y = Bernoulli(o).sample()
                        feat_len = state.shape[0]
                        #y = (o > args.sparsity).long().to(o.device)  # (3072, )
                        #row_indices = (y != 0).nonzero()[:, 0]  # len less than 3072
                        o = torch.nan_to_num(o).to(o.device)
                        row_indices = torch.multinomial(o, int((1 - args.sparsity)*feat_len), replacement=False).sort().values

                slicing(args.model, layer, row_indices)

                if 'llama' in args.model:
                    weight = layer.self_attn.k_proj.weight.data
                    state = Variable(cp(weight))
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        with torch.no_grad():
                            o = action_model(state)  # (3072, )
                            feat_len = state.shape[0]
                            slice_num = get_closest_even_number(int((1 - args.sparsity)*feat_len), layer.self_attn.config.num_attention_heads)
                            o = torch.nan_to_num(o).to(o.device)
                            row_indices2 = torch.multinomial(o, slice_num, replacement=False).sort().values
                        
                        layer.hidden_size = slice_num
                        model.config.hidden_size = slice_num
                        slicing_qkv(args.model_name, layer, row_indices2)
                        layer.self_attn.head_dim = int(slice_num//layer.self_attn.config.num_attention_heads)
                        #layer.self_attn.max_position_embeddings = int(slice_num//layer.self_attn.config.num_attention_heads) - 1
                        layer.self_attn._init_rope()
                        layer.self_attn.rotary_emb.to(config.device)
        else:
            orig_parameters = sum(p.numel() for p in model.parameters())

            for layer in get_all_layers_before_lora(args.model, model):
                random_slicing(args.model, layer, args.sparsity)

        new_parameters = sum(p.numel() for p in model.parameters())
        print ("Sparsity achieved {}%".format(int(100*(1-new_parameters/orig_parameters))))

    if args.activation != '':
        for layer in get_all_layers_before_lora(args.model, model):
            if 'opt' in args.model or 'phi' in args.model:
                layer.activation_fn = act_fn  # (3072, 768)
            elif 'llama' in args.model:
                layer.mlp.act_fn = act_fn
            elif 'falcon' in args.model:
                layer.mlp.act_fn = act_fn
            else:
                ValueError("Model type is not supported. Only OPT, Llama and Falcon models are supported.")

    print(model)
    if args.finetune:
        model = finetune(args, model, tokenizer, skip_lora=False)

    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)

    logging.info(f"Selected Tasks: {task_names}")

    for task in task_names:
        if task not in TASK_METRIC_MAP:
            raise NotImplementedError(
                f"Please specify the metric to use for {task} in TASK_METRIC_MAP. Available info {TASK_METRIC_MAP}"
            )

    # results is a dict with keys: results, configs, versions, n-shot, samples, config, git_hash
    results_main = lm_eval.simple_evaluate(hflm, tasks=task_names, num_fewshot=args.num_fewshot, batch_size=args.batch_size, log_samples=True)
    
    results = results_main[
        'results'
    ]

    #print (results_main["samples"])

    import json
    with open("model={}_sparsity={}.json".format(args.model.split("/")[-1], args.sparsity), 'w') as fp:
        json.dump(results_main["samples"], fp)

    #from lm_eval.loggers import WandbLogger
    #wandb_logger = WandbLogger()  # or empty if wandb.init(...) already called before
    #wandb_logger.post_init(results_main)
    #wandb_logger.log_eval_result()
    #wandb_logger.log_eval_samples(results_main["samples"])  # if log_samples

    logging.info(results)

    # name the output file
    outfile = f"{args.save_dir}/full_results_{args.num_fewshot}_shot"
    #if args.use_slicing:
    #    if args.slice_with_action_model:
    #        outfile += "_actionModelSlicing"
    #    else:
    #        outfile += "_uniformSlicing"
    outfile += ".json"

    with open(outfile, "w") as f:
        json.dump(results, f, indent=4)

    metric_vals = {task: round(result.get(TASK_METRIC_MAP[task]), 4) for task, result in results.items()}
    acc_avg = calculate_avg_accuracy(task_names, results)
    metric_vals['average'] = round(acc_avg, 4)

    # save this in the task results output file
    task_outfile = f"{args.save_dir}/{args.num_fewshot}_shot_task_results"
    #if args.use_slicing:
    #    if args.slice_with_action_model:
    #        task_outfile += "_actionModelSlicing"
    #    else:
    #        task_outfile += "_uniformSlicing"
    task_outfile += ".json"

    with open(task_outfile, "w") as f:
        json.dump(metric_vals, f, indent=4)

    if args.no_wandb == False:
        wandb.log(metric_vals)

    if args.no_wandb == False:
        wandb.log({'acc_avg': acc_avg})

    logging.info(json.dumps(metric_vals, indent=4))
    logging.info(f"Average accuracy across tasks: {acc_avg}")


if __name__ == "__main__":
    # Use the logger from lm_eval, adding a file handler to write the log to file
    logging = lm_eval_utils.eval_logger
    logging.addHandler(utils.create_file_handler(log_dir="log"))

    os.environ["WANDB__SERVICE_WAIT"] = "300"

    args = eval_arg_parser()
    process_eval_args(args)
    args.model_name = args.model
    eval_main(args)
