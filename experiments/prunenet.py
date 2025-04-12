import argparse
import copy
import logging
import os
from pathlib import Path

from torch.autograd import Variable
import torch

from slicegpt import utils
from slicegpt.config import config
from transformers import AutoModel
from tqdm import tqdm

from prunenet_utils import (
    _set_seed,
    _get_all_layers,
    _get_layer_weight,
    _get_action_model,
    _slicing,
    _calculate_activation_reward,
    _discount_rewards,
)


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
        logging.info(f"Episode: {episode}, Avg. reward: {total_reward / count}")

        state_pool = []
        action_pool = []
        reward_pool = []

        if total_reward > best_score:
            best_score = total_reward
            logging.info(
                f"Got a better action model. Saving to {action_model_checkpoint}."
            )
            torch.save(
                action_model.state_dict(),
                action_model_checkpoint,
            )


def _prune_model(prunenet_args, model, action_model):
    action_model.eval()

    # prune each layer
    for layer in _get_all_layers(prunenet_args.model_name, model):
        weight = _get_layer_weight(prunenet_args.model_name, layer)
        state = Variable(copy.deepcopy(weight))
        with torch.autocast(
            device_type=config.device.type, dtype=torch.float16
        ):
            with torch.no_grad():
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
        _slicing(prunenet_args.model_name, layer, row_indices)


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
    compressed_model_checkpoint = (
        Path(prunenet_args.save_dir)
        / f"model={prunenet_args.model_name.split('/')[-1]}_sparsity={prunenet_args.compression_ratio}"
    )

    # set the random seed
    _set_seed(prunenet_args.seed)

    # get the pre-trained model
    model = AutoModel.from_pretrained(prunenet_args.model_name)
    model.to(config.device)
    model.eval()
    model.seqlen = model.config.max_position_embeddings

    # get the action model
    action_model = _get_action_model(prunenet_args.model_name, model.config)
    action_model.to(config.device)
    if action_model_checkpoint.is_file():
        # no need to re-train the action model
        logging.info(
            f"Action model already exists at {action_model_checkpoint}. Now pruning."
        )
        action_model.load_state_dict(
            torch.load(action_model_checkpoint, weights_only=True)
        )
        _prune_model(prunenet_args, model, action_model)
        logging.info(
            f"Saving the pruned model at {compressed_model_checkpoint}."
        )
        model.save_pretrained(compressed_model_checkpoint)
        exit(0)

    # otherwise, we train the action model
    # getting all svds for the uncompressed model
    all_svds_main_model = {}
    for i, layer in enumerate(_get_all_layers(prunenet_args.model_name, model)):
        main_w = _get_layer_weight(prunenet_args.model_name, layer)
        if main_w.dtype == torch.float16:
            main_w = main_w.to(torch.float32)
        _, s1, _ = torch.svd(main_w)
        all_svds_main_model[i] = s1

    # the training loop
    _training(
        prunenet_args,
        model,
        action_model,
        all_svds_main_model,
        action_model_checkpoint,
    )

    _prune_model(prunenet_args, model, action_model)
    logging.info(f"Saving the pruned model at {compressed_model_checkpoint}.")
    model.save_pretrained(compressed_model_checkpoint)
