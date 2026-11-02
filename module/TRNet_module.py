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
        # 초기 벡터 (for src vec)
        self.register_buffer(
            "init_mem_vec",
            torch.zeros(
                (self.n_node+1,self.mem_dim)
            )
        )
        # 누적 벡터의 단위 방향. 실제 벡터 = exp(cumul_log_scale) * cumul_mem_vec
        self.register_buffer(
            "cumul_mem_vec",
            torch.zeros(
                (self.n_node+1,self.mem_dim)
            )
        )
        self.register_buffer(
            "cumul_log_scale",
            torch.full((self.n_node+1,), -torch.inf)
        )
        self.register_buffer(
            "last_t",
            torch.zeros(
                (self.n_node+1,),
                dtype=torch.int64
            )
        )
        # 노드가 현재 시각에 처음 갱신되기 직전의 상태.
        # 같은 시각의 후속 이벤트와 다음 batch에서도 source는 이 상태를 읽는다.
        self.register_buffer("pre_update_vec",torch.zeros_like(self.cumul_mem_vec))
        self.register_buffer("pre_update_scale", torch.full_like(self.cumul_log_scale, -torch.inf))
        self.register_buffer("pre_update_t", torch.zeros_like(self.last_t))
        self.register_buffer("has_pre_update", torch.zeros(self.n_node+1, dtype=torch.bool))
        self.init_module_vec()

    @torch.no_grad()
    def init_module_vec(self):
        self.init_mem_vec.normal_(
            mean=0.0,
            std=self.mem_dim**-0.5
        )
        # 각 노드는 자신의 초기 정보를 가진 상태에서 전파를 시작한다.
        lengths=torch.linalg.vector_norm(self.init_mem_vec,dim=-1)
        nonzero=lengths>0
        self.cumul_mem_vec.zero_()
        self.cumul_log_scale.fill_(-torch.inf)
        self.cumul_mem_vec[nonzero]=self.init_mem_vec[nonzero]/lengths[nonzero,None]
        self.cumul_log_scale[nonzero]=lengths[nonzero].log()
        self.last_t.zero_()
        self.pre_update_vec.zero_()
        self.pre_update_scale.fill_(-torch.inf)
        self.pre_update_t.zero_()
        self.has_pre_update.zero_()

    def detach_module_vec(self):
        self.init_mem_vec.detach_()
        self.cumul_mem_vec.detach_()
        self.cumul_log_scale.detach_()
        self.last_t.detach_()
        self.pre_update_vec.detach_()
        self.pre_update_scale.detach_()

    def get_init_mem_vec(self,
            node:torch.Tensor|None=None
        ):
        """
        Input:
            node: [N,]
        Output:
            init_mem_vec: [N,mem_dim]
        """
        if node is None:
            return self.init_mem_vec
        else:
            return self.init_mem_vec[node]

    def get_cumul_mem_vec(self,
            node:torch.Tensor|None=None
        ):
        """
        Input:
            node: [N,]
        Output:
            cumul_mem_vec: [N,mem_dim], 누적 벡터의 단위 방향
        """
        if node is None:
            return self.cumul_mem_vec
        else:
            return self.cumul_mem_vec[node]

    def get_cumul_log_scale(self, node:torch.Tensor|None=None):
        """
        누적 벡터의 log L2 norm. 
        영벡터는 -inf로 표현한다.
        """
        return self.cumul_log_scale if node is None else self.cumul_log_scale[node]

    @torch.no_grad()
    def update_module_vec(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor
        ):
        """
        시간순으로 x[v] += x[u]를 방향/로그 크기 표현에서 계산한다.

        입력 이벤트는 batch 사이까지 포함해 시간순이어야 한다.
        동일 시각의 source는 해당 시각 이전 상태만 전달하고,
        destination은 동일 시각의 모든 메시지를 누적한다.

        실제 크기를 복원하지 않아 큰 누적값의 overflow를 피한다.
        부동소수점 반올림과 작은 기여의 소실까지 방지하지는 않는다.
        """
        for u,v,t in zip(src,dst,event_t):
            if self.last_t[v]<=t:
                # 첫 destination 갱신 전에 보존한다 (self-loop도 동일 규칙).
                if not self.has_pre_update[v] or self.pre_update_t[v] != t:
                    self.pre_update_vec[v].copy_(self.cumul_mem_vec[v])
                    self.pre_update_scale[v] = self.cumul_log_scale[v]
                    self.pre_update_t[v] = t
                    self.has_pre_update[v] = True
                if self.has_pre_update[u] and self.pre_update_t[u] == t:
                    src_vec = self.pre_update_vec[u]
                    src_scale = self.pre_update_scale[u]
                else:
                    src_vec = self.cumul_mem_vec[u]
                    src_scale = self.cumul_log_scale[u]
                dst_scale=self.cumul_log_scale[v]
                common_scale=torch.maximum(src_scale,dst_scale)
                # 둘 다 영벡터면 -inf - (-inf) 계산을 건너뛴다.
                if torch.isfinite(common_scale):
                    combined=(
                        torch.exp(dst_scale-common_scale)*self.cumul_mem_vec[v]
                        + torch.exp(src_scale-common_scale)*src_vec
                    )
                    length=torch.linalg.vector_norm(combined)
                    if length>0:
                        new_scale=common_scale+length.log()
                        self.cumul_mem_vec[v].copy_(combined/length)
                        self.cumul_log_scale[v]=new_scale
                    else:
                        # 정확한 상쇄로 생긴 영벡터 (두 벡터가 서로 정확히 반대인 경우).
                        self.cumul_mem_vec[v].zero_()
                        self.cumul_log_scale[v]=-torch.inf
                self.last_t[v]=t
