"""Base Function and Class are defined here."""
# This is a lightweight ML agent trained by self-play.
# After sharing this notebook,
# we will add Hungry Geese environment in our HandyRL library.
# https://github.com/DeNA/HandyRL
# We hope you enjoy reinforcement learning!
# Load PyTorch Model


from typing import (
    Self,
)

import numpy as np
import torch
import torch.nn.functional as func
from torch import (
    Tensor,
    nn,
)

# Neural Network for Hungry Geese

class TorusConv2d(nn.Module):
    """Conv2d for Torus which looks like a donut and has a shape with a hole in the middle."""
    def __init__(self: Self, input_dim: int, output_dim: int, kernel_size: tuple[int, int], bn: bool): # noqa: FBT001
        """Args:
        input_dim (int): Number of channels in the input image
        output_dim (int): Number of channels produced by the convolution
        kernel_size (tuple[int, int]):  Size of the convolving kernel (height, width)
        bn (bool): If True, Batch Normalization will be applied
        """
        super().__init__()

        """edge_size (tuple[int]): This means the size of the edge i.e. (height, width) to deal with the torus."""
        self.edge_size: tuple = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=kernel_size,
            )
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self: Self, x: Tensor) -> Tensor:
        """Args:
            x (Tensor): This dimension is (batch_size, channel, height, width)

        Returns:
            Tensor: This dimension is (batch_size, channel, height, width)
        """
        h: Tensor = torch.cat(
            tensors = [
                x[:,:,:,-self.edge_size[1]:],
                x,
                x[:,:,:,:self.edge_size[1]],
            ],
            dim=3,
        )
        h = torch.cat(
            tensors = [
                h[:,:,-self.edge_size[0]:],
                h,
                h[:,:,:self.edge_size[0]],
            ],
            dim=2,
        )
        h = self.conv(h)
        return self.bn(h) if self.bn is not None else h


class GeeseNet(nn.Module):
    """Neural Network for Hungry Geese."""
    def __init__(self: Self, layers: int, filters: int):
        """Args:
        layers (int): Number of layers for the neural network
        filters (int): Number of filters for the neural network
        """
        super().__init__()

        input_dim = 17
        kernel_size: tuple[int,int] = (3, 3)
        is_bn = True

        self.conv0 = TorusConv2d(
            input_dim=input_dim,
            output_dim=filters,
            kernel_size = kernel_size,
            bn = is_bn,
        )
        self.blocks = nn.ModuleList(
            [
                TorusConv2d(
                    input_dim = filters,
                    output_dim = filters,
                    kernel_size=kernel_size,
                    bn = is_bn,
                )
                for _ in range(layers)],
            )

        out_features_p = 4
        out_features_v = 1

        self.head_p = nn.Linear(
            in_features=filters,
            out_features = out_features_p,
            bias=False,
        )
        self.head_v = nn.Linear(
            in_features = filters * 2,
            out_features= out_features_v,
            bias=False,
        )

    def forward(self: Self, x: Tensor) -> dict[str, Tensor]:
        """Args:
            x (Tensor): This dimension is (batch_size, channel, height, width)

        Returns:
            Tensor: This dimension is (batch_size, channel, height, width)
        """
        h = func.relu_(self.conv0(x))
        for block in self.blocks:
            h: Tensor = func.relu_(h + block(h))
        h_head: Tensor = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        h_avg: Tensor = h.view(h.size(0), h.size(1), -1).mean(-1)
        p: Tensor = self.head_p(h_head)
        v: Tensor = torch.tanh(self.head_v(torch.cat([h_head, h_avg], 1)))

        return {"policy": p, "value": v}


# Input for Neural Network

def make_input(obses):
    b = np.zeros((17, 7 * 11), dtype=np.float32)
    obs = obses[-1]

    for p, pos_list in enumerate(obs["geese"]):
        # head position
        for pos in pos_list[:1]:
            b[0 + (p - obs["index"]) % 4, pos] = 1
        # tip position
        for pos in pos_list[-1:]:
            b[4 + (p - obs["index"]) % 4, pos] = 1
        # whole position
        for pos in pos_list:
            b[8 + (p - obs["index"]) % 4, pos] = 1

    # previous head position
    if len(obses) > 1:
        obs_prev = obses[-2]
        for p, pos_list in enumerate(obs_prev["geese"]):
            for pos in pos_list[:1]:
                b[12 + (p - obs["index"]) % 4, pos] = 1

    # food
    for pos in obs["food"]:
        b[16, pos] = 1

    return b.reshape(-1, 7, 11)

# # Undefined PARAM, layers and filters
#
# state_dict = pickle.loads(base64.b64decode(PARAM))
# model = GeeseNet(layers, filters)
# model.load_state_dict(state_dict)
# model.eval()
#


# Main Function of Agent

obses = []

def agent(obs, _, model: GeeseNet):
    """Note: Unusef function...?"""
    obses.append(obs)
    x = make_input(obses)
    with torch.no_grad():
        xt = torch.from_numpy(x).unsqueeze(0)
        o = model(xt)
    p = o["policy"].squeeze(0).detach().numpy()

    actions = ["NORTH", "SOUTH", "WEST", "EAST"]
    return actions[np.argmax(p)]
