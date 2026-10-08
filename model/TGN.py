import torch
import torch.nn as nn
from typing_extensions import Literal
from graph import TGN_Graph
from module import TimeEncoder,Memory,GraphAttnEmbedding,GRUMemoryUpdater,TR_Decoder

class TGN(nn.Module):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            time_dim:int,
            latent_dim:int,
            msg_dim:int,
            mem_dim:int,
            embed_dim:int,
            graph:TGN_Graph,
            n_layer:int,
            n_neighbor:int,
            n_head:int,
            msg_fn:Literal["concat","mlp"]="concat",
            aggr_fn:Literal["last","mean"]="last"
        ):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.time_dim=time_dim
        self.latent_dim=latent_dim
        self.msg_dim=msg_dim
        self.mem_dim=mem_dim
        self.embed_dim=embed_dim

        self.n_layer=n_layer
        self.n_neighbor=n_neighbor
        self.n_head=n_head
        self.msg_fn=msg_fn
        self.aggr_fn=aggr_fn
        
        # graph
        self.graph=graph

        # memory
        self.n_node=self.graph.n_node
        self.memory=Memory(n_node=self.n_node,mem_dim=self.mem_dim)

        # time encoder
        self.time_encoder=TimeEncoder(time_dim=time_dim)

        # memory updater
        self.memory_updater=GRUMemoryUpdater(
            mem_dim=mem_dim,
            edge_dim=edge_dim,
            msg_dim=msg_dim,
            time_dim=time_dim,
            time_encoder=self.time_encoder,
            graph=self.graph,
            msg_fn=msg_fn,
            aggr_fn=aggr_fn
        )

        # encoder
        self.encoder=GraphAttnEmbedding(
            node_dim=node_dim,
            edge_dim=edge_dim,
            time_dim=time_dim,
            latent_dim=latent_dim,
            mem_dim=mem_dim,
            embed_dim=embed_dim,
            graph=self.graph,
            n_layer=n_layer,
            n_neighbor=n_neighbor,
            n_head=n_head,
            use_last_state=False,
            use_memory=True,
            time_encoder=self.time_encoder
        )

        # decoder
        self.decoder=TR_Decoder(
            embed_dim=embed_dim,
            latent_dim=latent_dim
        )

    def update_model_memory(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            edge:torch.Tensor,
            event_t:torch.Tensor
        ):
        """
        eventstream에 대해서 model의 memory state 업데이트
        """
        mem_vec=self.memory.get_mem_vec()
        mem_t=self.memory.get_mem_t()
        updated_result=self.memory_updater.update_memory(
            src=src,
            dst=dst,
            edge=edge,
            event_t=event_t,
            mem_vec=mem_vec,
            mem_t=mem_t
        )
        updated_node=updated_result["node"]
        updated_mem_vec=updated_result["mem_vec"]
        updated_mem_t=updated_result["mem_t"]
        self.memory.update_memory_state(
            node=updated_node,
            mem_vec=updated_mem_vec,
            mem_t=updated_mem_t
        )

    def embedding(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor
        ):
        """
        TR sample의 src, dst node들의 embedding vector 반환.
        """
        mem_vec=self.memory.get_mem_vec()
        batch_size=src.size(0) 
        tar=torch.concat([src,dst],dim=0) 
        tar_t=torch.cat([event_t,event_t],dim=0)
        embedded_tar_vec=self.encoder.compute_embedding(
            tar=tar,
            tar_t=tar_t,
            mem_vec=mem_vec,
            n_layer=self.n_layer
        )
        src_vec=embedded_tar_vec[:batch_size]
        dst_vec=embedded_tar_vec[batch_size:]
        return {
            "src_vec":src_vec,
            "dst_vec":dst_vec
        }

    def forward(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor
        ):
        """
        TR sample에 대한 Temporal Reachability 예측.
        """
        ### embedding
        embedded_result=self.embedding(
            src=src,
            dst=dst,
            event_t=event_t
        )
        src_vec=embedded_result["src_vec"] # [B,embed_dim]
        dst_vec=embedded_result["dst_vec"] # [B,embed_dim]

        ### decode 
        pred_logit=self.decoder(
            src_vec=src_vec,
            dst_vec=dst_vec
        ) # [B,1]
        return pred_logit