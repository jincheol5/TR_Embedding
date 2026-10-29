import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,
            input_dim:int,
            latent_dim:int,
            output_dim:int
        ):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.Linear(in_features=input_dim,out_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim,out_features=output_dim)
        )
    def forward(self,input:torch.Tensor):
        return self.mlp(input) 

### Original MLP-Mixer / GraphMixer Github module
class MLPMixerBlock(nn.Module):
    def __init__(self,
            n_token:int,
            embed_dim:int,
            token_dim_expansion_factor:float=0.5,
            channel_dim_expansion_factor:float=4.0,
            dropout:float=0.0
        ):
        super().__init__()
        self.token_norm=nn.LayerNorm(embed_dim)
        self.token_mlp=nn.Sequential(
            nn.Linear(n_token,int(token_dim_expansion_factor*n_token)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(token_dim_expansion_factor*n_token),n_token),
            nn.Dropout(dropout)
        )
        self.channel_norm=nn.LayerNorm(embed_dim)
        self.channel_mlp=nn.Sequential(
            nn.Linear(embed_dim,int(channel_dim_expansion_factor*embed_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(channel_dim_expansion_factor*embed_dim),embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self,z:torch.Tensor)->torch.Tensor:
        # Normalize features in [B, m, d], then mix tokens in [B, d, m].
        token_input=self.token_norm(z).transpose(1,2)
        z=z+self.token_mlp(token_input).transpose(1,2)
        return z+self.channel_mlp(self.channel_norm(z))

### TPNet Github module
class TPNet_MLPMixerBlock(nn.Module):
    def __init__(self,
            n_token:int,
            embed_dim:int,
            token_dim_expansion_factor:float=0.5,
            channel_dim_expansion_factor:float=4.0,
            dropout:float=0.0
        ):
        super().__init__()
        self.token_norm=nn.LayerNorm(n_token)
        self.token_mlp=nn.Sequential(
            nn.Linear(n_token,int(token_dim_expansion_factor*n_token)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(token_dim_expansion_factor*n_token),n_token),
            nn.Dropout(dropout)
        )
        self.channel_norm=nn.LayerNorm(embed_dim)
        self.channel_mlp=nn.Sequential(
            nn.Linear(embed_dim,int(channel_dim_expansion_factor*embed_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(channel_dim_expansion_factor*embed_dim),embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self,z:torch.Tensor)->torch.Tensor:
        # Token mixing: [B, m, d] -> [B, d, m] -> [B, m, d]
        token_input=self.token_norm(z.transpose(1,2))
        z=z+self.token_mlp(token_input).transpose(1,2)
        # Channel mixing: mix the d features at each sequence position.
        return z+self.channel_mlp(self.channel_norm(z))