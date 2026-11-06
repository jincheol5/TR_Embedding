"""Persistent matrix-memory graph adaptations of mLSTM and DeltaNet cores.

The inputs are historical source-node vectors, while each destination owns its
matrix state. These are recurrent cores with normalized graph input, an output
projection, and TRNet's source-ID residual. They omit sequence convolutions and
the complete language-model stacks. Rectangular key/value heads make the output
width independent of the matrix-state budget.

Primary recurrence references:
  https://github.com/NX-AI/xlstm/blob/main/xlstm/blocks/mlstm/backends.py
  https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/delta_rule/naive.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MatrixGraphCell(nn.Module):
    def __init__(self,input_dim,hidden_dim,heads=4,key_dim=8):
        super().__init__()
        if heads<1 or key_dim<1 or hidden_dim%heads:
            raise ValueError("Matrix heads must divide mem_dim; key_dim must be positive")
        self.hidden_dim=hidden_dim
        self.heads=heads
        self.key_dim=key_dim
        self.value_dim=hidden_dim//heads
        self.input_norm=nn.LayerNorm(input_dim)
        self.q_proj=nn.Linear(input_dim,heads*key_dim,bias=False)
        self.k_proj=nn.Linear(input_dim,heads*key_dim,bias=False)
        self.v_proj=nn.Linear(input_dim,hidden_dim,bias=False)
        self.output_gate=nn.Linear(input_dim,hidden_dim)
        self.output_norm=nn.GroupNorm(heads,hidden_dim)
        self.output_proj=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.residual=nn.Linear(input_dim,hidden_dim,bias=False)
        self.residual_logit=nn.Parameter(torch.tensor(-2.0))

    def projections(self,x):
        x=self.input_norm(x)
        q=self.q_proj(x).reshape(-1,self.heads,self.key_dim)
        k=self.k_proj(x).reshape(-1,self.heads,self.key_dim)
        v=self.v_proj(x).reshape(-1,self.heads,self.value_dim)
        return x,q,k,v

    def readout(self,retrieved,normalized_input,init_src):
        # Head-wise normalization and an output gate follow the recurrent core;
        # none of these transformations change the fixed TRNet decoder.
        retrieved=retrieved.reshape(-1,self.hidden_dim)
        output=self.output_norm(retrieved)*torch.sigmoid(self.output_gate(normalized_input))
        return self.output_proj(output)+torch.sigmoid(self.residual_logit)*self.residual(init_src)


class MatrixLSTM(_MatrixGraphCell):
    """xLSTM mLSTM covariance update with stabilized exponential input gating.

    C is [B,H,K,V], n is [B,H,K], m is [B,H,1]. The forget gate is sigmoid,
    matching NX-AI's recurrent backend, and the input gate is exponential. The
    log-domain denominator avoids overflow for extreme stabilizer values.
    """
    def __init__(self,input_dim,hidden_dim,heads=4,key_dim=8):
        super().__init__(input_dim,hidden_dim,heads,key_dim)
        self.input_gate=nn.Linear(input_dim,heads)
        self.forget_gate=nn.Linear(input_dim,heads)
        nn.init.zeros_(self.input_gate.bias)
        with torch.no_grad():
            self.forget_gate.bias.copy_(torch.linspace(3.0,6.0,heads))

    def initial_state(self,n_nodes,device=None,dtype=None):
        opts={"device":device,"dtype":dtype}
        return {"C":torch.zeros(n_nodes,self.heads,self.key_dim,self.value_dim,**opts),
                "n":torch.zeros(n_nodes,self.heads,self.key_dim,**opts),
                "m":torch.zeros(n_nodes,self.heads,1,**opts)}

    def recurrent_step(self,q,k,v,input_preact,forget_preact,state):
        log_f=F.logsigmoid(forget_preact).unsqueeze(-1)
        log_i=input_preact.unsqueeze(-1)
        m=torch.maximum(log_f+state["m"],log_i)
        f=torch.exp(log_f+state["m"]-m)
        i=torch.exp(log_i-m)
        k=k/math.sqrt(self.key_dim)
        C=f.unsqueeze(-1)*state["C"]+i.unsqueeze(-1)*k.unsqueeze(-1)*v.unsqueeze(-2)
        n=f*state["n"]+i*k
        new_state={"C":C,"n":n,"m":m}
        return self.retrieve(q,new_state),new_state

    def retrieve(self,q,state):
        numerator=torch.einsum("bhk,bhkv->bhv",q,state["C"])
        dot=torch.einsum("bhk,bhk->bh",q,state["n"]).unsqueeze(-1)
        # max(abs(q*n),exp(-m)) + eps, computed without exp(-m) overflow.
        log_abs=dot.abs().clamp_min(torch.finfo(dot.dtype).tiny).log()
        log_max=torch.maximum(log_abs,-state["m"])
        log_denom=torch.logaddexp(log_max,log_max.new_tensor(math.log(1e-6)))
        return numerator*torch.exp(-log_denom)

    def read_state(self,query_input,init_src,state,raw=False):
        """Address a destination matrix; optionally expose retrieval magnitude."""
        normalized=self.input_norm(query_input)
        q=self.q_proj(normalized).reshape(-1,self.heads,self.key_dim)
        retrieved=self.retrieve(q,state)
        if raw:
            return retrieved.reshape(-1,self.hidden_dim)
        return self.readout(retrieved,normalized,init_src)

    def forward_state(self,x,h_prev,init_src,state):
        normalized,q,k,v=self.projections(x)
        retrieved,new_state=self.recurrent_step(
            q,k,v,self.input_gate(normalized),self.forget_gate(normalized),state)
        return self.readout(retrieved,normalized,init_src),new_state


class DeltaNet(_MatrixGraphCell):
    """Normalized-key DeltaNet delta rule with persistent destination matrices.

    C' = C + beta * k outer (v - k^T C), readout=q^T C'/sqrt(K).
    beta is sigmoid and q/k use L2 normalization, as in DeltaNet layers. There
    is no additional forget/decay gate: this is DeltaNet, not Gated DeltaNet.
    """
    def __init__(self,input_dim,hidden_dim,heads=4,key_dim=8):
        super().__init__(input_dim,hidden_dim,heads,key_dim)
        self.beta=nn.Linear(input_dim,heads)

    def initial_state(self,n_nodes,device=None,dtype=None):
        return {"C":torch.zeros(n_nodes,self.heads,self.key_dim,self.value_dim,
                                device=device,dtype=dtype)}

    def recurrent_step(self,q,k,v,beta,state):
        predicted=torch.einsum("bhk,bhkv->bhv",k,state["C"])
        error=(v-predicted)*beta.unsqueeze(-1)
        C=state["C"]+k.unsqueeze(-1)*error.unsqueeze(-2)
        retrieved=torch.einsum("bhk,bhkv->bhv",q,C)/math.sqrt(self.key_dim)
        return retrieved,{"C":C}

    def read_state(self,query_input,init_src,state,raw=False):
        normalized=self.input_norm(query_input)
        q=self.q_proj(normalized).reshape(-1,self.heads,self.key_dim)
        q=F.normalize(F.silu(q),p=2,dim=-1,eps=1e-6)
        retrieved=torch.einsum("bhk,bhkv->bhv",q,state["C"])/math.sqrt(self.key_dim)
        if raw:
            return retrieved.reshape(-1,self.hidden_dim)
        return self.readout(retrieved,normalized,init_src)

    def forward_state(self,x,h_prev,init_src,state):
        normalized,q,k,v=self.projections(x)
        q=F.normalize(F.silu(q),p=2,dim=-1,eps=1e-6)
        k=F.normalize(F.silu(k),p=2,dim=-1,eps=1e-6)
        retrieved,new_state=self.recurrent_step(q,k,v,torch.sigmoid(self.beta(normalized)),state)
        return self.readout(retrieved,normalized,init_src),new_state
