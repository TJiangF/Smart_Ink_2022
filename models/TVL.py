import torch

class TVLoss(torch.nn.Module):
    def __init__(self, weight: float = 1) -> None:
        """Total Variation Loss

        Args:
            weight (float): weight of TV loss
        """
        super().__init__()
        self.weight = weight

    def forward(self, x):
        batch_size, c, h, w = x.size()
        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        return self.weight * (tv_h + tv_w) / (batch_size * c * h * w)