from torch import nn
from torch.distributions import Uniform
import torch


class SparsityPredictor(torch.nn.Module):
    """
    A neural module that predicts sparsity patterns over MLP weight matrices
    using a learned distribution over rows. Implements differentiable
    sparsity via a reparameterization trick for RL-based optimization.

    Parameters
    ----------
    hidden_size : int, optional
        Dimensionality of the input hidden representation, by default 768.
    intermediate_size : int, optional
        Number of rows in the target MLP weight matrix (typically the intermediate
        size of the MLP), by default 3072.

    Attributes
    ----------
    proj_intermediate : nn.Linear
        Learnable linear projection used to produce row-wise sparsity logits.
    row_sparsities : nn.Parameter
        Learnable parameter representing initial sparsity scores for each row.

    Returns 
    -------
    keep_probs : torch.Tensor
        Final sparsity sampling probabilities obtained using reparameterization.
    """

    def __init__(self, hidden_size=768, intermediate_size=3072):
        super(SparsityPredictor, self).__init__()

        self.intermediate_size = intermediate_size
        self.proj_intermediate = nn.Linear(
            hidden_size, intermediate_size, bias=True
        )
        self.row_sparsities = nn.Parameter(
            torch.rand(intermediate_size, 1), requires_grad=True
        )  # (3072, 1)

    def calculate_KLD(self):
        """
        Compute KL divergence between the learned sparsity distribution (alpha)
        and a Bernoulli(0.5) prior to encourage sparsity regularization.

        Returns
        -------
        torch.Tensor
            Scalar representing the total KL divergence loss.
        """
        return (
            -1 * torch.log(self.alpha) * (1 - self.alpha)
            - self.alpha * torch.log(1 - self.alpha)
            + torch.log(torch.tensor(0.5)).to(self.alpha.device)
        ).sum()

    def calculate_total_loss(self):
        """
        Compute the total loss for training the sparsity predictor.

        Returns
        -------
        torch.Tensor
            The total loss (currently only the KL divergence).
        """
        return self.calculate_KLD()

    def forward(self, weight_matrix):
        """
        Forward pass to compute differentiable sparsity probabilities over rows
        of the input weight matrix.

        Parameters
        ----------
        weight_matrix : torch.Tensor
            The MLP weight matrix of shape (intermediate_size, hidden_size).

        Returns
        -------
        torch.Tensor
            Keep probabilities for each row of the weight matrix (shape: [intermediate_size]).

        Raises
        ------
        ValueError
            If the input matrix does not have the expected shape.
        """
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
