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
        self.encoder=nn.GRUCell(
            input_size=mem_dim,
            hidden_size=mem_dim
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
        1. 시간순으로 전달된 이벤트를 하나씩 처리한다.
        2. 현재 이벤트의 src, dst, t를 가져온다.
        3. last_t[src]==t이면 prev_mem_vec[src], 아니면 mem_vec[src]를 Source Feature로 선택한다.
        4. 현재 mem_vec[dst]를 Destination Feature로 가져온다.
        5. last_t[dst]<t이면 업데이트 전 Destination Feature를 prev_mem_vec[dst]에 저장한다.
        6. GRUCell(Source Feature, Destination Feature)로 새 Destination Feature를 계산한다.
        7. index_copy 결과를 self.module.mem_vec에 대입해 Destination을 즉시 갱신한다.
        8. last_t[dst]=t로 업데이트 시각을 기록한다.
        9. 배치의 모든 이벤트 처리가 끝나면 최신 mem_vec로 Prediction을 수행한다.
        10. 학습에서는 역전파와 Optimizer 업데이트 후 상태를 detach()하고, 값과 Snapshot은 다음 배치에 유지한다.
        """
        for i in range(src.numel()):
            src_node=src[i].reshape(1)
            dst_node=dst[i].reshape(1)
            t=event_t[i]
            src_vec=(
                self.module.prev_mem_vec[src_node]
                if self.module.last_t[src_node].item()==t.item()
                else self.module.mem_vec[src_node]
            )
            dst_vec=self.module.mem_vec[dst_node]
            if self.module.last_t[dst_node].item()<t.item():
                self.module.prev_mem_vec=self.module.prev_mem_vec.index_copy(0,dst_node,dst_vec)
            self.module.mem_vec=self.module.mem_vec.index_copy(0,dst_node,self.encoder(src_vec,dst_vec))
            self.module.last_t[dst_node]=t

    def forward(self,
            src:torch.Tensor,
            dst:torch.Tensor
        ):
        """
        고정 Source 식별 벡터와 최신 Destination 메모리로 Temporal Reachability 예측.
        """
        src_vec=self.module.init_mem_vec[src] # [B,mem_dim]
        dst_vec=self.module.get_mem_vec(dst) # [B,mem_dim]

        ### decode 
        pair_vec=torch.concat([src_vec,dst_vec],dim=-1)
        pred_logit=self.decoder(pair_vec) # [B,1]
        return pred_logit.squeeze(-1) # [B,]
