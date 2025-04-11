import argparse
import copy
import logging
import os
from pathlib import Path
import numpy as np

import random
from torch import nn
from torch.autograd import Variable
from torch.distributions.uniform import Uniform
import torch
import wandb

from slicegpt import (
    data_utils,
    gpu_utils,
    hf_utils,
    layernorm_fusion,
    rotate,
    utils,
)
from slicegpt.config import config
from slicegpt.slicing_scheduler import ConstSlicingScheduler
from scipy import stats
from transformers import AutoModel
from tqdm import tqdm


class SparsityPredictor(torch.nn.Module):
    def __init__(
        self, hidden_size=768, intermediate_size=3072, sparsity_level=0.2
    ):
        super(SparsityPredictor, self).__init__()

        self.intermediate_size = intermediate_size
        self.proj_intermediate = nn.Linear(
            hidden_size, intermediate_size, bias=True
        )
        self.row_sparsities = nn.Parameter(
            torch.rand(intermediate_size, 1), requires_grad=True
        )  # (3072, 1)

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
            raise ValueError("The layer does not support sparsity operation")

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

        return keep_probs


def _set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)


def prunenet_arg_parser(interactive: bool = True) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/opt-125m",
        help="HF checkpoint to load the model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The random seed used for multinomial sampling.",
    )
    parser.add_argument(
        "--compression_ratio",
        type=float,
        default=0.0,
        help="A measure of how much compression is applied (in the range [0, 1))",
    )
    parser.add_argument(
        "--num_episodes",
        dest="num_episodes",
        default=20,
        help="Number of episodes to train the policy learner.",
    )
    parser.add_argument(
        "--learning_rate_action",
        default=0.001,
        help="Learning rate to train the policy learner.",
    )
    parser.add_argument(
        "--use_kld", action="store_true", help="To use KLD loss"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Path to save the compressed model and the action model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device to use. Example values are 'cpu', 'cuda', 'cuda:0'. If not specified it will be defaulted to 'cuda' if available and 'cpu' otherwise.",
    )

    return parser.parse_args() if interactive else parser.parse_args("")


def process_prunenet_args(args):
    for arg, argv in vars(args).items():
        logging.debug(f"{arg} = {argv}")

    if not 0 <= args.compression_ratio < 1:
        raise argparse.ArgumentTypeError(
            f"Compression ratio should be in the range [0, 1)"
        )

    if args.device:
        config.device = torch.device(args.device)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)


def _get_all_layers(model_name, model):
    if "opt" in model_name:
        all_layers = model.decoder.layers
    elif "phi" in model_name:
        all_layers = model.layers
    elif "llama" in model_name:
        all_layers = model.layers
    elif "falcon" in model_name:
        all_layers = model.transformer.h
    else:
        raise ValueError(
            "Model type is not supported. Only OPT, Phi, Llama and Falcon models are supported."
        )

    return all_layers


def _get_layer_weight(model_name, layer):
    if "opt" in model_name:
        main_w = layer.fc1.weight.data
    elif "phi" in model_name:
        main_w = layer.mlp.fc1.weight.data
    elif "llama" in model_name:
        main_w = layer.mlp.gate_proj.weight.data
    elif "falcon" in model_name:
        main_w = layer.mlp.dense_h_to_4h.weight.data
    else:
        ValueError(
            "Model type is not supported. Only OPT, Phi, Llama and Falcon models are supported."
        )
    return main_w


def _get_action_model(model_name, model_config, compression_ratio):
    if "opt" in model_name:
        action_model = SparsityPredictor(
            model_config.hidden_size, model_config.ffn_dim, compression_ratio
        )
    elif "llama" in args.model_name or "phi" in args.model_name:
        action_model = SparsityPredictor(
            model_config.hidden_size,
            model_config.intermediate_size,
            compression_ratio,
        )
    elif "falcon" in args.model_name:
        action_model = SparsityPredictor(
            model_config.hidden_size,
            model_config.ffn_hidden_size,
            compression_ratio,
        )
    else:
        raise ValueError(
            "Model type is not supported. Only OPT, Phi, Llama and Falcon models are supported."
        )
    return action_model


def _slicing(model_name, layer, row_indices):
    if "opt" in model_name:
        # slice the intermediate and output weight matrices appropriately
        layer.fc1.out_features = len(row_indices)
        layer.fc1.weight.data = layer.fc1.weight[row_indices, :]
        layer.fc1.bias.data = layer.fc1.bias[row_indices]

        # revert changes on output layer
        layer.fc2.in_features = len(row_indices)
        layer.fc2.weight.data = layer.fc2.weight[:, row_indices]

    elif "phi" in model_name:
        # slice the intermediate and output weight matrices appropriately
        layer.mlp.fc1.out_features = len(row_indices)
        layer.mlp.fc1.weight.data = layer.mlp.fc1.weight[row_indices, :]
        layer.mlp.fc1.bias.data = layer.mlp.fc1.bias[row_indices]

        # revert changes on output layer
        layer.mlp.fc2.in_features = len(row_indices)
        layer.mlp.fc2.weight.data = layer.mlp.fc2.weight[:, row_indices]

    elif "llama" in model_name:
        # slice the intermediate and output weight matrices appropriately
        layer.mlp.gate_proj.out_features = len(row_indices)
        layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight[
            row_indices, :
        ]

        layer.mlp.up_proj.out_features = len(row_indices)
        layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight[row_indices, :]

        # revert changes on output layer
        layer.mlp.down_proj.in_features = len(row_indices)
        layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight[
            :, row_indices
        ]

    elif "falcon" in model_name:
        # slice the intermediate and output weight matrices appropriately
        layer.mlp.dense_h_to_4h.out_features = len(row_indices)
        layer.mlp.dense_h_to_4h.weight.data = layer.mlp.dense_h_to_4h.weight[
            row_indices, :
        ]

        # revert changes on output layer
        layer.mlp.dense_4h_to_h.in_features = len(row_indices)
        layer.mlp.dense_4h_to_h.weight.data = layer.mlp.dense_4h_to_h.weight[
            :, row_indices
        ]


def _calculate_activation_reward(s1, weight_matrix2):
    if weight_matrix2.dtype == torch.float16:
        weight_matrix2 = weight_matrix2.to(torch.float32)

    _, s2, _ = torch.svd(weight_matrix2)

    dist = stats.ks_2samp(
        s1.detach().cpu().numpy(), s2.detach().cpu().numpy()
    ).statistic

    # dist = s2.max() #torch.abs(s2.max() - 1)
    if dist == 0:
        return 99999
    else:
        return 1 / dist


def _discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    return r


def _training(
    prunenet_args,
    original_model,
    action_model,
    all_svds_main_model,
    action_model_checkpoint,
):
    action_model.train()
    optimizer = torch.optim.AdamW(
        action_model.parameters(), lr=prunenet_args.learning_rate_action
    )
    scaler = torch.amp.GradScaler("cuda")
    best_score = 0

    for episode in tqdm(range(prunenet_args.num_episodes)):
        # get a fresh copy of the model
        model = copy.deepcopy(original_model)
        model.to(config.device)

        state_pool = []
        action_pool = []
        reward_pool = []

        total_loss = 0
        count = 0
        total_reward = 0

        # compress all layers, and get the states/actions/rewards
        for i, layer in enumerate(
            _get_all_layers(prunenet_args.model_name, model)
        ):
            weight = _get_layer_weight(prunenet_args.model_name, layer)
            state = Variable(copy.deepcopy(weight))

            # get importance scores per row
            # and sample row indices
            with torch.autocast(
                device_type=config.device.type, dtype=torch.float16
            ):
                o = action_model(state)  # (3072, )
                feat_len = state.shape[0]
                row_indices = (
                    torch.multinomial(
                        o,
                        int((1 - prunenet_args.compression_ratio) * feat_len),
                        replacement=False,
                    )
                    .sort()
                    .values
                )

            # slice the weights using the row indices
            _slicing(prunenet_args.model_name, layer, row_indices)

            # get the updated rewards
            new_w = _get_layer_weight(prunenet_args.model_name, layer)
            reward = _calculate_activation_reward(all_svds_main_model[i], new_w)

            state_pool.append(state)
            action_pool.append(row_indices)
            reward_pool.append(reward.item())

            total_reward += reward.item()
            count += 1

        # compute discounted rewards for the episode
        reward_pool = _discount_rewards(reward_pool)

        # compute policy loss
        for i in range(len(state_pool)):
            with torch.autocast(
                device_type=config.device.type, dtype=torch.float16
            ):
                state = state_pool[i]
                action = Variable(action_pool[i])
                reward = reward_pool[i]
                o = action_model(state)  # (3072, )
                loss = -1 * torch.gather(torch.log(o), 0, action).sum() * reward
                total_loss += loss.item()

                if prunenet_args.use_kld:
                    kld_loss = action_model.calculate_total_loss()
                    loss += kld_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(action_model.parameters(), 1.0)

        logging.info(f"Episode: {episode}, loss: {total_loss/count}")
        # if not (prunenet_args.no_wandb):
        #     wandb.log({"Episodic Loss": total_loss / count})
        #     wandb.log({"Episodic Reward": total_reward})
        logging.info(f"Episode: {episode}, Avg. reward: {total_reward / count}")

        state_pool = []
        action_pool = []
        reward_pool = []

        if total_reward > best_score:
            best_score = total_reward
            logging.info(
                f"Got a better action model. Saving to {action_model_checkpoint}"
            )
            torch.save(
                action_model.state_dict(),
                action_model_checkpoint,
            )


if __name__ == "__main__":
    utils.configure_logging(
        log_to_console=True, log_to_file=False, level=logging.DEBUG
    )
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    prunenet_args = prunenet_arg_parser()
    process_prunenet_args(prunenet_args)

    # get the save path for the action model
    action_model_checkpoint = (
        Path(prunenet_args.save_dir)
        / f"action_model={prunenet_args.model_name.split('/')[-1]}_sparsity={prunenet_args.compression_ratio}.ckpt"
    )

    # set the random seed
    _set_seed(prunenet_args.seed)

    # get the pre-trained model
    model = AutoModel.from_pretrained(prunenet_args.model_name)
    model.to(config.device)
    model.eval()
    model.seqlen = model.config.max_position_embeddings

    # the original parameter count
    original_param_count = sum(int(p.nelement()) for p in model.parameters())
    logging.info(f"Original model parameters: {original_param_count:,d}")

    # getting all svds for the uncompressed model
    all_svds_main_model = {}
    for i, layer in enumerate(_get_all_layers(prunenet_args.model_name, model)):
        main_w = _get_layer_weight(prunenet_args.model_name, layer)
        if main_w.dtype == torch.float16:
            main_w = main_w.to(torch.float32)
        _, s1, _ = torch.svd(main_w)
        all_svds_main_model[i] = s1

    # now, depending on the pruning type
    # we either prune randomly, or using an action model

    # get the action model
    action_model = _get_action_model(
        prunenet_args.model_name, model.config, prunenet_args.compression_ratio
    )
    # if os.path.exists(model_checkpoint_save_path):
    #     action_model.load_state_dict(
    #         torch.load(model_checkpoint_save_path, weights_only=True)
    #     )
    #     action_model.to(device)
    action_model.to(config.device)

    # the training loop
    _training(
        prunenet_args,
        model,
        action_model,
        all_svds_main_model,
        action_model_checkpoint,
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

    for layer in get_all_layers_before_lora(args.model_name, model):
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

        with torch.autocast(device_type=device, dtype=torch.float16):
            with torch.no_grad():
                o = action_model(state)  # (3072, )
                feat_len = state.shape[0]
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


"""
python3 - \
    --model_name facebook/opt-125m \
    --seed 42 \
    --compression_ratio 0.3 \
    --save_dir /home/codetalker7/PruneNet/opt \
    --device cuda:0
"""
