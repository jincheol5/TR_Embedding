import torch 
import torch.nn as nn

class TPNet_Matrix(nn.Module):
    def __init__(self,
            n_node:int,
            random_ft_dim:int=32,
            walk_len:int=5,
            time_decay_coef:float=0.0001
        ):
        super().__init__()
        self.n_node=n_node
        self.random_ft_dim=random_ft_dim
        self.walk_len=walk_len
        self.time_decay_coef=time_decay_coef
        self.register_buffer(
            "random_ft",
            torch.zeros(
                (self.walk_len+1,self.n_node+1,self.random_ft_dim)
            )
        )
        self.register_buffer(
            "last_t",
            torch.zeros(
                (self.n_node+1,)
            )
        )
        self.init_random_ft()
        self.init_last_t()

    def init_random_ft(self):
        """
        Node Representation Maintaining (NRM) 초기화.

        random_ft: [walk_len+1, n_node+1, random_ft_dim]
            random_ft[0]: 실제 노드의 0-hop representation, N(0, 1/random_ft_dim).
            random_ft[1:]: 1~walk_len-hop representation, zero vector.
            random_ft[:,0,:]: 모든 hop의 padding node (node_id=0), zero vector.
        """
        self.random_ft.zero_()
        self.random_ft[0,1:].normal_(
            mean=0.0,
            std=self.random_ft_dim**-0.5
        )
        self.random_ft[:,0,:].zero_()

    def init_last_t(self):
        self.last_t.zero_()

    @torch.no_grad()
    def update_random_ft(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor
        ):
        """
        Node Representation Maintaining (NRM)

        하이퍼 파라미터:
        - λ: time_decay_coef
        - k: walk_len (hop 수, layer 수)

        논문 방식:
        - 모든 노드들에 대한 H^(0...k) 초기화
            H^0 -> 
            H^(1...k) -> 
        - interaction (u,v,t)에 대해서 업데이트 시 (l=현재 업데이트 할 hop(layer) 수),
            H^l_u(t+) = H^l_u(t) + exp(λ * t) * H^(l-1)_v(t)
            H^l_v(t+) = H^l_v(t) + exp(λ * t) * H^(l-1)_u(t)

        TPNet 공식 Github 방식:

            
        """
        next_t=event_t[-1]
        
        ### 시간 스케일 보정 -> 기존에 저장된 노드 표현을 시간이 흐른 만큼 줄이는 단계 
        # 노드별로 이전 갱신 이후 얼마나 시간이 흘렀는지 계산하여 계산한 비율을 기존 벡터의 모든 원소에 곱
        # hop이 클수록 더 많이 줄어듬
        elapsed=next_t-self.last_t
        for hop in range(1,self.walk_len+1):
            time_correction_factor=torch.exp(-self.time_decay_coef*hop*elapsed)
            self.random_ft[hop].mul_(time_correction_factor.unsqueeze(-1))

        ### update random_ft
        time_decay_weight=torch.exp(
            -self.time_decay_coef*(next_t-event_t)
        ).unsqueeze(-1)
        for hop in range(self.walk_len,0,-1):
            src_messages=self.random_ft[hop-1][dst]*time_decay_weight
            dst_messages=self.random_ft[hop-1][src]*time_decay_weight
            self.random_ft[hop].scatter_add_(
                dim=0,
                index=src[:,None].expand(-1,self.random_ft_dim),
                src=src_messages
            )
            self.random_ft[hop].scatter_add_(
                dim=0,
                index=dst[:,None].expand(-1,self.random_ft_dim),
                src=dst_messages
            )
        self.random_ft[:,0,:].zero_()
        self.last_t.copy_(next_t.expand_as(self.last_t))

    def get_pairwise_ft(self,
            src:torch.Tensor,
            dst:torch.Tensor
        ) -> torch.Tensor:
        """
        Pairwise Feature Decoding의 MLP 입력을 반환.

        interaction (u,v,t)에 대해서,
            F_u = [H^0_u, H^1_u, ... , H_l_u] = [walk_len+1,random_ft_dim]
            F_uv = [F_u,F_v] = [2*(walk_len+1),random_ft_dim]
            f_uv = F_uv x F_uv^T = [2*(walk_len+1),2*(walk_len+1)]
                -> pairwise information (모든 Hop Representation 사이의 내적을 계산) 
            flattened_f_uv = [4*(walk_len+1)**2]
        log(ReLU(flattened_f_uv)+1) 반환.
        """
        # [hop,batch,dim] -> [batch,hop,dim]
        src_ft=self.random_ft[:,src,:].permute(1,0,2)
        dst_ft=self.random_ft[:,dst,:].permute(1,0,2)

        # F_{u,v}: [B,2*(walk_len+1),dim]
        pair_ft=torch.cat((src_ft,dst_ft),dim=1)
        pairwise_info=torch.bmm(pair_ft,pair_ft.transpose(1,2)) # [B,2*(walk_len+1),2*(walk_len+1)], 각 노드 쌍마다 표현 벡터끼리 내적한 표
        raw_pairwise_ft=pairwise_info.flatten(start_dim=1)
        return torch.log1p(torch.relu(raw_pairwise_ft)) # [B,4*(walk_len+1)**2], log(1+x)까지 계산 후 반환

