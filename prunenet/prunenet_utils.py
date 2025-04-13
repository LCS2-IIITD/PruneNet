import numpy as np
import random
from scipy import stats
import torch
from SparsityPredictor import SparsityPredictor


def _set_seed(seed):
    """
    Set the random seed for reproducibility across torch and Python's random module.
    This is used during the multinomial sampling employed in the training and compression
    loops.

    Parameters
    ----------
    seed : int
        The seed value to use.
    """
    torch.manual_seed(seed)
    random.seed(seed)


def _get_all_layers(model_name, model):
    """
    Retrieve the list of transformer layers for supported model architectures.

    Parameters
    ----------
    model_name : str
        The name of the model.
    model : torch.nn.Module
        The model instance.

    Returns
    -------
    list
        A list of layers from the model.

    Raises
    ------
    ValueError
        If the model type is not supported.
    """
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
    """
    Extract the first MLP weight matrix from a given layer depending on the model type.

    Parameters
    ----------
    model_name : str
        Name of the model type.
    layer : torch.nn.Module
        The layer object from the transformer model.

    Returns
    -------
    torch.Tensor
        The weight matrix used in the first projection of the MLP layer.

    Raises
    ------
    ValueError
        If the model type is not supported.
    """
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
    """
    Create a SparsityPredictor model specific to the given model configuration.

    Parameters
    ----------
    model_name : str
        Name of the model architecture.
    model_config : object
        Configuration object containing model-specific dimensions.

    Returns
    -------
    SparsityPredictor
        An instance of the sparsity prediction module.

    Raises
    ------
    ValueError
        If the model type is not supported.
    """
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
    """
    Apply row slicing to the intermediate and output projection weights of a transformer MLP layer.

    Parameters
    ----------
    model_name : str
        Model architecture name.
    layer : torch.nn.Module
        The transformer layer to modify.
    row_indices : list[int]
        Indices of rows to retain in the intermediate projection layers.

    Notes
    -----
    This function updates weight and bias tensors in-place.
    """
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
    """
    Compute a reward based on the similarity of two singular value spectra using the
    Kolmogorov-Smirnov statistic.

    Parameters
    ----------
    s1 : torch.Tensor
        Reference singular values (1D tensor).
    weight_matrix2 : torch.Tensor
        New weight matrix to compare against.

    Returns
    -------
    float
        Inverse of the KS statistic (or large constant if distributions match perfectly).
    """
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
    """
    Apply discounting to a list of rewards using exponential decay.

    Parameters
    ----------
    rewards : list[float]
        List of reward values.
    gamma : float, optional
        Discount factor in the range (0, 1), by default 0.99.

    Returns
    -------
    np.ndarray
        Array of discounted rewards.
    """
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    return r


if __name__ == "__main__":
    print("Importing works")
