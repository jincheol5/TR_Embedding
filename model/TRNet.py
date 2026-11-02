import torch
import torch.nn as nn
from graph import GNN_Graph
from module import TRNet_Module,MLP 

class TRNet(nn.Module):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            latent_dim:int,
            mem_dim:int,
            embed_dim:int
        ):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.latent_dim=latent_dim
        self.mem_dim=mem_dim
        self.embed_dim=embed_dim

        # decoder
        self.decoder=MLP(
            input_dim=2*mem_dim+1,
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

    def forward(self,
            src:torch.Tensor,
            dst:torch.Tensor
        ):
        """
        TR sample에 대한 Temporal Reachability 예측.
        """
        ### get mem_vec
        src_vec=self.module.get_init_mem_vec(node=src)
        dst_vec=self.module.get_cumul_mem_vec(node=dst)
        dst_scale=self.module.get_cumul_log_scale(node=dst)
        # 영벡터의 -inf는 decoder에 유한한 값으로 전달한다.
        dst_scale=torch.where(torch.isfinite(dst_scale),dst_scale,0.0)

        ### decode 
        pair_vec=torch.concat([src_vec,dst_vec,dst_scale.unsqueeze(-1)],dim=-1)
        pred_logit=self.decoder(pair_vec) # [B,1]
        return pred_logit.squeeze(-1) # [B,]