from torch import nn
from torch.distributions import Uniform
import torch


class SparsityPredictor(torch.nn.Module):
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
