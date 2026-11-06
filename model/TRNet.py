import torch
import torch.nn as nn
from module import (TRNet_Module,ResidualGRU,NormalizedResidualGRU,MLP,DiagonalLinearRNN,
                    LinearRecurrentUnit,GatedLinearRNN,MinimalGRU)
from module.TRNet_matrix import MatrixLSTM,DeltaNet
from module.TRNet_bayesian import BayesianMinimalGRU

class TRNet(nn.Module):
    def __init__(self,
            mem_dim:int,
            latent_dim:int,
            encoder_type:str="deltanet",
            decoder_input:str="concat",
            event_backend:str="sequential",
            matrix_heads:int=4,
            matrix_key_dim:int=8,
            bayes_prior_std:float=1.0,
            bayes_posterior_rho_init:float=-5.0
        ):
        super().__init__()
        self.mem_dim=mem_dim
        self.latent_dim=latent_dim
        self.encoder_type=encoder_type
        self.decoder_input=decoder_input
        self.event_backend=event_backend
        self.matrix_heads=matrix_heads
        self.matrix_key_dim=matrix_key_dim
        if event_backend not in ("sequential","wave"):
            raise ValueError(f"Unknown event_backend: {event_backend}")
        if decoder_input not in ("concat","interaction","hybrid","matrix_query","matrix_query_raw"):
            raise ValueError(f"Unknown decoder_input: {decoder_input}")
        if decoder_input in ("matrix_query","matrix_query_raw") and encoder_type not in ("mlstm","deltanet"):
            raise ValueError("matrix_query requires a matrix-memory encoder")
        
        # encoder
        encoders={
            "residual_gru":ResidualGRU,
            "normalized_gru":NormalizedResidualGRU,
            "diagonal":DiagonalLinearRNN,
            "lru":LinearRecurrentUnit,
            "gated_linear":GatedLinearRNN,
            "mingru":MinimalGRU,
            "mlstm":MatrixLSTM,
            "deltanet":DeltaNet,
            "bayesian_mingru":BayesianMinimalGRU,
        }
        if encoder_type not in encoders:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        encoder_kwargs={"input_dim":mem_dim,"hidden_dim":mem_dim}
        if encoder_type in ("mlstm","deltanet"):
            encoder_kwargs.update(heads=matrix_heads,key_dim=matrix_key_dim)
        if encoder_type=="bayesian_mingru":
            encoder_kwargs.update(prior_std=bayes_prior_std,
                                posterior_rho_init=bayes_posterior_rho_init)
        self.encoder=encoders[encoder_type](**encoder_kwargs)

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
        if hasattr(self.encoder,"initial_state"):
            self.module.configure_state(self.encoder.initial_state(
                self.n_node+1,device=self.module.init_mem.device,dtype=self.module.init_mem.dtype))

    def reset_memory(self):
        self.module.reset_state()

    def update_mem(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor
        ):
        """
        시간순 이벤트를 처리하고 Source 초기 식별 정보를 residual로 전달한다.
        """
        if src.numel()==0:
            return
        src_ids=src.detach().cpu().tolist()
        dst_ids=dst.detach().cpu().tolist()
        times=event_t.detach().cpu().tolist()
        # Cache only the touched rows, avoiding an N*d index_copy for every event.
        nodes=list(dict.fromkeys(src_ids+dst_ids))
        index=torch.tensor(nodes,dtype=torch.long,device=self.module.cumul_mem.device)
        current_rows=self.module.cumul_mem[index]
        previous_rows=self.module.prev_mem[index]
        initial_rows=self.module.init_mem[index]
        current={node:current_rows[j:j+1] for j,node in enumerate(nodes)}
        previous={node:previous_rows[j:j+1] for j,node in enumerate(nodes)}
        initial={node:initial_rows[j:j+1] for j,node in enumerate(nodes)}
        aux_rows={name:value[index] for name,value in self.module.auxiliary_state().items()}
        auxiliary={name:{node:value[j:j+1] for j,node in enumerate(nodes)}
                for name,value in aux_rows.items()}
        last=dict(zip(nodes,self.module.last_t[index].detach().cpu().tolist()))
        updated_nodes=list(dict.fromkeys(dst_ids))
        if self.event_backend=="wave":
            # A conservative touch dependency preserves the original order for
            # every shared source or destination, including source-only reads.
            # Events at one depth have disjoint node sets and can run together.
            # This changes scheduling only, never the scientific update rule.
            last_touch={}
            waves=[]
            for src_node,dst_node,t in zip(src_ids,dst_ids,times):
                depth=1+max(last_touch.get(src_node,-1),last_touch.get(dst_node,-1))
                last_touch[src_node]=last_touch[dst_node]=depth
                if depth==len(waves):
                    waves.append([])
                waves[depth].append((src_node,dst_node,t))
            for wave in waves:
                x=torch.cat([previous[s] if last[s]==t else current[s] for s,d,t in wave])
                h=torch.cat([current[d] for s,d,t in wave])
                init=torch.cat([initial[s] for s,d,t in wave])
                for s,d,t in wave:
                    if last[d]<t:
                        previous[d]=current[d]
                if auxiliary:
                    state={name:torch.cat([values[d] for s,d,t in wave])
                        for name,values in auxiliary.items()}
                    output,new_state=self.encoder.forward_state(x=x,h_prev=h,init_src=init,state=state)
                    for name,values in new_state.items():
                        for j,(s,d,t) in enumerate(wave):
                            auxiliary[name][d]=values[j:j+1]
                else:
                    output=self.encoder(x=x,h_prev=h,init_src=init)
                for j,(s,d,t) in enumerate(wave):
                    current[d]=output[j:j+1]
                    last[d]=t
        else:
            for src_node,dst_node,t in zip(src_ids,dst_ids,times):
                src_vec=previous[src_node] if last[src_node]==t else current[src_node]
                dst_vec=current[dst_node]
                src_init=initial[src_node]
                if last[dst_node]<t:
                    previous[dst_node]=dst_vec
                if auxiliary:
                    state={name:values[dst_node] for name,values in auxiliary.items()}
                    updated_dst,new_state=self.encoder.forward_state(
                        x=src_vec,h_prev=dst_vec,init_src=src_init,state=state)
                    for name,value in new_state.items():
                        auxiliary[name][dst_node]=value
                else:
                    updated_dst=self.encoder(
                        x=src_vec,
                        h_prev=dst_vec,
                        init_src=src_init
                    )
                current[dst_node]=updated_dst
                last[dst_node]=t
        updated_index=torch.tensor(updated_nodes,dtype=torch.long,device=index.device)
        self.module.update_cumul_mem(updated_index,torch.cat([current[node] for node in updated_nodes]))
        self.module.update_prev_mem(updated_index,torch.cat([previous[node] for node in updated_nodes]))
        self.module.update_last_t(updated_index,torch.tensor([last[node] for node in updated_nodes],device=index.device,dtype=torch.int64))
        for name,values in auxiliary.items():
            self.module.update_aux_state(name,updated_index,torch.cat([values[node] for node in updated_nodes]))

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
        if self.decoder_input in ("matrix_query","matrix_query_raw"):
            # Keep source/destination activity and expose a source-ID-addressed
            # read from the destination matrix. No recurrent state is mutated.
            state={name:value[dst] for name,value in self.module.auxiliary_state().items()}
            associative=self.encoder.read_state(init_src_vec,init_src_vec,state,
                                            raw=self.decoder_input=="matrix_query_raw")
            pair_vec=torch.concat([src_vec,dst_vec,associative],dim=-1)
        elif self.decoder_input=="hybrid":
            # Preserve current source/destination activity alongside an explicit
            # source-ID/destination product; decoder dimensions stay unchanged.
            pair_vec=torch.concat([src_vec,dst_vec,init_src_vec*dst_vec],dim=-1)
        elif self.decoder_input=="interaction":
            # Matched 3*d decoder shape. Fixed-ID/destination interactions give
            # the MLP explicit evidence for source identity within dst memory.
            pair_vec=torch.concat([init_src_vec*dst_vec,torch.abs(init_src_vec-dst_vec),dst_vec],dim=-1)
        else:
            pair_vec=torch.concat([init_src_vec,src_vec,dst_vec],dim=-1)
        pred_logit=self.decoder(pair_vec) # [B,1]
        return pred_logit.squeeze(-1) # [B,]
