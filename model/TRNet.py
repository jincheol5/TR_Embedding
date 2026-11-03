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
            input_dim=2*mem_dim,
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
        src node의 last_t를 확인하여서 현재 edge event의 event_t가 last_t보다 큰 경우 
        -> dst의 cumul_mem_vec와 last_t를 업데이트 한다.
        
        cumul_mem_vec update 방법:
        - encoder에 src의 cumul_mem_vec과 dst의 cumul_mem_vec을 concat하여 전달 후 output을 dst의 cumul_mem_vec으로 업데이트
        - dst의 last_t를 event_t로 업데이트 

        """
        for i in range(src.size(0)):
            src_node=src[i:i+1]
            dst_node=dst[i:i+1]
            current_t=event_t[i:i+1]
            src_vec=self.module.get_cumul_mem_vec(src_node).clone()
            dst_vec=self.module.get_cumul_mem_vec(dst_node).clone()

            if (self.module.get_last_t(src_node)<current_t).item():
                updated_dst_vec=self.encoder(torch.cat((src_vec,dst_vec),dim=-1))
                self.module.update_cumul_mem_vec(dst_node,updated_dst_vec)
                self.module.update_last_t(dst_node,current_t)

    def forward(self,
            src:torch.Tensor,
            dst:torch.Tensor
        ):
        """
        TR sample에 대한 Temporal Reachability 예측.
        """
        src_vec=self.module.get_init_mem_vec(src) # [B,mem_dim]
        dst_vec=self.module.get_cumul_mem_vec(dst) # [B,mem_dim]

        ### decode 
        pair_vec=torch.concat([src_vec,dst_vec],dim=-1)
        pred_logit=self.decoder(pair_vec) # [B,1]
        return pred_logit.squeeze(-1) # [B,]
