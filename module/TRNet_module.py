import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class TRNet_Module(nn.Module):
    def __init__(self,
            n_node:int,
            mem_dim:int=32
        ):
        super().__init__()
        self.n_node=n_node
        self.mem_dim=mem_dim
        self._aux_names=()

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
        for name in self._aux_names:
            getattr(self,f"aux_{name}").zero_()

    def configure_state(self,state):
        """Register zero-initialized recurrent auxiliary buffers per node.

        Auxiliary state belongs to destinations only. Source messages read the
        timestamp-snapshotted vector memory, so no auxiliary timestamp snapshot
        is necessary.
        """
        if self._aux_names:
            raise RuntimeError("Auxiliary state is already configured")
        for name,value in state.items():
            if value.shape[0]!=self.n_node+1:
                raise ValueError("Auxiliary state must have one row per node")
            self.register_buffer(f"aux_{name}",value)
        self._aux_names=tuple(state)

    def auxiliary_state(self):
        return {name:getattr(self,f"aux_{name}") for name in self._aux_names}

    def update_aux_state(self,name,node,value):
        buffer=getattr(self,f"aux_{name}")
        setattr(self,f"aux_{name}",buffer.index_copy(0,node,value))

    def _state_buffer_names(self):
        return ("init_mem","cumul_mem","prev_mem","last_t")+tuple(
            f"aux_{name}" for name in self._aux_names)

    def detach_module(self):
        for name in self._state_buffer_names():
            getattr(self,name).detach_()

    @torch.no_grad()
    def reset_state(self):
        """Reset the stream without resampling the fixed node identifiers."""
        self.detach_module()
        self.cumul_mem.copy_(self.init_mem)
        self.prev_mem.copy_(self.init_mem)
        self.last_t.zero_()
        for name in self._aux_names:
            getattr(self,f"aux_{name}").zero_()

    def snapshot_state(self):
        return {
            name:getattr(self,name).detach().clone()
            for name in self._state_buffer_names()
        }

    @torch.no_grad()
    def restore_state(self,state):
        self.detach_module()
        for name in self._state_buffer_names():
            getattr(self,name).copy_(state[name])

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


class NormalizedResidualGRU(ResidualGRU):
    """Nonlinear control isolating source-message LayerNorm from encoder choice."""
    def __init__(self,input_dim,hidden_dim):
        super().__init__(input_dim,hidden_dim)
        self.input_norm=nn.LayerNorm(input_dim)

    def forward(self,x,h_prev,init_src):
        return super().forward(self.input_norm(x),h_prev,init_src)


class _LinearInput(nn.Module):
    """Normalized graph message and a shared, weak source-ID residual.

    Linear refers to the destination-state recurrence conditional on its input.
    Source messages themselves come from graph states and are normalized, so the
    entire temporal graph dynamical system is not a linear operator.
    """
    def __init__(self,input_dim,hidden_dim):
        super().__init__()
        self.input_norm=nn.LayerNorm(input_dim)
        self.input_proj=nn.Linear(input_dim,hidden_dim,bias=False)
        self.residual=nn.Linear(input_dim,hidden_dim,bias=False)
        self.residual_logit=nn.Parameter(torch.tensor(-2.0))

    def message(self,x,init_src):
        return self.input_proj(self.input_norm(x))+torch.sigmoid(self.residual_logit)*self.residual(init_src)


class DiagonalLinearRNN(_LinearInput):
    """Real stable diagonal recurrence, initialized with long memory timescales."""
    def __init__(self,input_dim,hidden_dim):
        super().__init__(input_dim,hidden_dim)
        radius=torch.linspace(0.8,0.999,hidden_dim)
        self.decay_logit=nn.Parameter(torch.logit(radius))

    def forward(self,x,h_prev,init_src):
        a=torch.sigmoid(self.decay_logit)
        return a*h_prev+torch.sqrt((1-a.square()).clamp_min(1e-6))*self.message(x,init_src)


class LinearRecurrentUnit(_LinearInput):
    """LRU core with a complex diagonal represented by two real state halves.

    Uses Orvieto et al. (2023) annulus initialization and learned gamma scaling.
    This is a graph adaptation of the recurrent core, not the full LRU sequence
    block: the unchanged TRNet MLP reads the real/imaginary state directly.
    """
    def __init__(self,input_dim,hidden_dim):
        if hidden_dim%2:
            raise ValueError("lru requires an even mem_dim (real/imaginary halves)")
        super().__init__(input_dim,hidden_dim)
        n=hidden_dim//2
        radius=torch.sqrt(torch.rand(n)*(0.999**2-0.8**2)+0.8**2)
        phase=torch.rand(n)*math.pi/10+1e-3
        self.nu_log=nn.Parameter(torch.log(-torch.log(radius)))
        self.theta_log=nn.Parameter(torch.log(phase))
        self.gamma_log=nn.Parameter(torch.log(torch.sqrt(1-radius.square())))

    def forward(self,x,h_prev,init_src):
        radius=torch.exp(-torch.exp(self.nu_log))
        phase=torch.exp(self.theta_log)
        re=radius*torch.cos(phase)
        im=radius*torch.sin(phase)
        h_re,h_im=h_prev.chunk(2,dim=-1)
        rotated=torch.cat((re*h_re-im*h_im,im*h_re+re*h_im),dim=-1)
        gamma=torch.exp(self.gamma_log).repeat(2)
        return rotated+gamma*self.message(x,init_src)


class GatedLinearRNN(_LinearInput):
    """Input-dependent real diagonal recurrence inspired by Griffin RG-LRU.

    h'=a*h+sqrt(1-a²)*input_gate*message, with a=exp(-8*softplus(A)*decay_gate).
    Dense gates replace Griffin's block-diagonal gates in this graph adaptation.
    """
    def __init__(self,input_dim,hidden_dim):
        super().__init__(input_dim,hidden_dim)
        self.gates=nn.Linear(input_dim,2*hidden_dim)
        target=torch.linspace(0.9,0.999,hidden_dim)
        self.a_param=nn.Parameter(torch.log(torch.expm1(-torch.log(target)/4)))

    def forward(self,x,h_prev,init_src):
        normalized=self.input_norm(x)
        decay_gate,input_gate=torch.sigmoid(self.gates(normalized)).chunk(2,dim=-1)
        a=torch.exp(-8*F.softplus(self.a_param)*decay_gate)
        message=self.input_proj(normalized)+torch.sigmoid(self.residual_logit)*self.residual(init_src)
        return a*h_prev+torch.sqrt((1-a.square()).clamp_min(1e-6))*input_gate*message


class MinimalGRU(_LinearInput):
    """minGRU update (Feng et al., 2024), with graph-message normalization.

    The candidate is linear as in the basic sequential minGRU equations; the
    positive/log-space parallel-scan variant is not used for graph event replay.
    """
    def __init__(self,input_dim,hidden_dim):
        super().__init__(input_dim,hidden_dim)
        self.gate=nn.Linear(input_dim,hidden_dim)
        nn.init.constant_(self.gate.bias,-2.0)

    def forward(self,x,h_prev,init_src):
        normalized=self.input_norm(x)
        z=torch.sigmoid(self.gate(normalized))
        candidate=self.input_proj(normalized)+torch.sigmoid(self.residual_logit)*self.residual(init_src)
        return (1-z)*h_prev+z*candidate
