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

        # 누적 메모리 벡터
        self.register_buffer(
            "mem_vec",
            torch.zeros((self.n_node+1,self.mem_dim))
        )

        # 이전 누적 메모리 벡터 -> 해당 노드가 마지막으로 업데이트된 Timestamp 직전의 Feature Snapshot
        self.register_buffer(
            "prev_mem_vec",
            torch.zeros((self.n_node+1,self.mem_dim))
        )

        # 업데이트 timestamp
        self.register_buffer(
            "last_t",
            torch.zeros((self.n_node+1,),dtype=torch.int64)
        )
        self.init_module_vec()
        self.register_buffer("init_mem_vec",self.mem_vec.clone())

    @torch.no_grad()
    def init_module_vec(self):
        self.mem_vec.normal_(
            mean=0.0,
            std=self.mem_dim**-0.5
        )
        self.prev_mem_vec.copy_(self.mem_vec)
        self.last_t.zero_()

    def detach_module_vec(self):
        self.mem_vec.detach_()
        self.prev_mem_vec.detach_()
        self.last_t.detach_()

    def get_mem_vec(self,node:torch.Tensor|None=None):
        if node is None:
            return self.mem_vec
        else:
            return self.mem_vec[node]

    def get_prev_mem_vec(self,node:torch.Tensor|None=None):
        if node is None:
            return self.prev_mem_vec
        else:
            return self.prev_mem_vec[node]

    def get_last_t(self,node:torch.Tensor|None=None):
        if node is None:
            return self.last_t
        else:
            return self.last_t[node]

    def update_mem_vec(self,
            node:torch.Tensor,
            updated_mem_vec:torch.Tensor
        ):
        """
        Input:
            node: [N,]
            updated_cumul_mem_vec: [N,mem_dim]
        """
        self.mem_vec[node]=updated_mem_vec

    def update_prev_mem_vec(self,
            node:torch.Tensor,
            updated_prev_mem_vec:torch.Tensor
        ):
        """
        Input:
            node: [N,]
            updated_cumul_mem_vec: [N,mem_dim]
        """
        self.prev_mem_vec[node]=updated_prev_mem_vec

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
