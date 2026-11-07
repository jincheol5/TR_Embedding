import torch
import torch.nn as nn
from module import MLP,TRNet_Module,ResidualGRU

class TRNet(nn.Module):
    def __init__(self,
            mem_dim:int,
            latent_dim:int
        ):
        super().__init__()
        self.mem_dim=mem_dim
        self.latent_dim=latent_dim


