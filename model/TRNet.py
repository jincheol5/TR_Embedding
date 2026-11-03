import torch
import torch.nn as nn
from module import TRNet_Module,MLP 

class TRNet(nn.Module):
    def __init__(self,
            mem_dim:int,
            latent_dim:int
        ):
        super().__init__()
        self.mem_dim=mem_dim
        self.latent_dim=latent_dim
        
        # encoder
        self.encoder=MLP(
            input_dim=2*mem_dim,
            latent_dim=latent_dim,
            output_dim=mem_dim
        )

        # decoder
        self.decoder=MLP(
            input_dim=3*mem_dim,
            latent_dim=latent_dim,
            output_dim=1
        )

    def set_module(self,n_node:int):
        self.n_node=n_node
        # set module
        self.module=TRNet_Module(
            n_node=self.n_node,
            mem_dim=self.mem_dim
        )

    def update_mem_vec(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor
        ):
        """
        """

    def forward(self,
            src:torch.Tensor,
            dst:torch.Tensor
        ):
        """
        TR sample에 대한 Temporal Reachability 예측.
        """
        # src_vec=self.module.get_init_mem_vec(src) # [B,mem_dim]
        init_src_vec=self.module.get_init_mem_vec(src)
        src_vec=self.module.get_cumul_mem_vec(src)
        dst_vec=self.module.get_cumul_mem_vec(dst) # [B,mem_dim]

        ### decode 
        pair_vec=torch.concat([init_src_vec,src_vec,dst_vec],dim=-1)
        pred_logit=self.decoder(pair_vec) # [B,1]
        return pred_logit.squeeze(-1) # [B,]
