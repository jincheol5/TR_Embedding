import torch
import torch.nn as nn
from module import TRNet_Module,ResidualGRU,MLP

class TRNet(nn.Module):
    def __init__(self,
            mem_dim:int,
            latent_dim:int
        ):
        super().__init__()
        self.mem_dim=mem_dim
        self.latent_dim=latent_dim
        
        # encoder
        self.encoder=ResidualGRU(
            input_dim=mem_dim,
            hidden_dim=mem_dim
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

    def update_mem(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor
        ):
        """
        시간순 이벤트를 처리하고 Source 초기 식별 정보를 residual로 전달한다.
        """
        for i in range(src.numel()):
            src_node=src[i].reshape(1)
            dst_node=dst[i].reshape(1)
            t=event_t[i]
            src_vec=(
                self.module.prev_mem[src_node]
                if self.module.last_t[src_node].item()==t.item()
                else self.module.cumul_mem[src_node]
            )
            dst_vec=self.module.cumul_mem[dst_node]
            src_init=self.module.init_mem[src_node]
            if self.module.last_t[dst_node].item()<t.item():
                self.module.update_prev_mem(dst_node,dst_vec)
            updated_dst=self.encoder(
                x=src_vec,
                h_prev=dst_vec,
                init_src=src_init
            )
            self.module.update_cumul_mem(dst_node,updated_dst)
            self.module.update_last_t(dst_node,t.reshape(1))

    def forward(self,
            src:torch.Tensor,
            dst:torch.Tensor
        ):
        """
        고정 Source 식별 벡터와 최신 Destination 메모리로 Temporal Reachability 예측.
        """
        init_src_vec=self.module.init_mem[src] # [B,mem_dim]
        src_vec=self.module.cumul_mem[src] # [B,mem_dim]
        dst_vec=self.module.cumul_mem[dst] # [B,mem_dim]

        ### decode 
        pair_vec=torch.concat([init_src_vec,src_vec,dst_vec],dim=-1)
        pred_logit=self.decoder(pair_vec) # [B,1]
        return pred_logit.squeeze(-1) # [B,]
