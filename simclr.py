import torch
import torch.nn as nn
import torchvision
from torch.nn.functional import cosine_similarity


def info_nce_loss(feats: torch.Tensor, temperature: float):
    # Размер `[b, hidden_dim]
    cos_sim = cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(
        cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
    # InfoNCE loss
    cos_sim = cos_sim / temperature
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()
    return nll


class Encoder(nn.Module):

    def __init__(self, hidden_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.convnet = torchvision.models.resnet18(num_classes=4 * hidden_dim)
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Проход через свёрточную сеть.

        Parameters
        ----------
        x : torch.Tensor
            Вход размером `[b, 3, 224, 224]`.

        Returns
        -------
        torch.Tensor
            Выход размером `[b, hidden_dim]`.
        """
        return self.convnet(x)
