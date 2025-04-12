import numpy as np
import random
from scipy import stats
import torch
from SparsityPredictor import SparsityPredictor


def _set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)


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


def _get_action_model(model_name, model_config):
    if "opt" in model_name:
        action_model = SparsityPredictor(
            model_config.hidden_size, model_config.ffn_dim
        )
    elif "llama" in model_name or "phi" in model_name:
        action_model = SparsityPredictor(
            model_config.hidden_size,
            model_config.intermediate_size,
        )
    elif "falcon" in model_name:
        action_model = SparsityPredictor(
            model_config.hidden_size,
            model_config.ffn_hidden_size,
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


if __name__ == "__main__":
    print("Importing works")
