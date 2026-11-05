import torch 
import torch.nn as nn

class TRNet_Module(nn.Module):
    def __init__(self,
            n_node:int,
            mem_dim:int=32
        ):
        super().__init__()
        self.n_node=n_node
        self.mem_dim=mem_dim

        # 초기 메모리 벡터
        self.register_buffer(
            "init_mem",
            torch.zeros((self.n_node+1,self.mem_dim))
        )

        # 누적 메모리 벡터
        self.register_buffer(
            "cumul_mem",
            torch.zeros((self.n_node+1,self.mem_dim))
        )

        # 이전 누적 메모리 벡터 -> 해당 노드가 마지막으로 업데이트된 Timestamp 직전의 Feature Snapshot
        self.register_buffer(
            "prev_mem",
            torch.zeros((self.n_node+1,self.mem_dim))
        )

        # 업데이트 timestamp
        self.register_buffer(
            "last_t",
            torch.zeros((self.n_node+1,),dtype=torch.int64)
        )
        self.init_module()

    @torch.no_grad()
    def init_module(self):
        self.init_mem.normal_(
            mean=0.0,
            std=self.mem_dim**-0.5
        )
        self.cumul_mem.copy_(self.init_mem)
        self.prev_mem.copy_(self.init_mem)
        self.last_t.zero_()

    def detach_module(self):
        self.init_mem.detach_()
        self.cumul_mem.detach_()
        self.prev_mem.detach_()
        self.last_t.detach_()

    def update_cumul_mem(self,
            node:torch.Tensor,
            updated_cumul_mem:torch.Tensor
        ):
        """
        Input:
            node: [N,]
            updated_cumul_mem: [N,mem_dim]
        """
        self.cumul_mem=self.cumul_mem.index_copy(0,node,updated_cumul_mem)

    def update_prev_mem(self,
            node:torch.Tensor,
            updated_prev_mem:torch.Tensor
        ):
        """
        Input:
            node: [N,]
            updated_prev_mem: [N,mem_dim]
        """
        self.prev_mem=self.prev_mem.index_copy(0,node,updated_prev_mem)

    def update_last_t(self,
            node:torch.Tensor,
            updated_last_t:torch.Tensor
        ):
        """
        Input:
            node: [N,]
            updated_last_t: [N,]
        """
        self.last_t[node]=updated_last_t

class ResidualGRU(nn.Module):
    def __init__(self,
            input_dim,
            hidden_dim
        ):
        super().__init__()
        self.gru=nn.GRUCell(input_dim,hidden_dim)
        self.residual=nn.Linear(input_dim,hidden_dim,bias=False)

        # sigmoid(-2) ≈ 0.119
        self.residual_logit=nn.Parameter(torch.tensor(-2.0))

    def forward(self,
            x,
            h_prev,
            init_src
        ):
        h_new=self.gru(x,h_prev)
        alpha=torch.sigmoid(self.residual_logit)
        return h_new+alpha*self.residual(init_src)
