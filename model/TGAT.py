import torch
import torch.nn as nn
from graph import TGN_Graph
from module import TimeEncoder,GraphAttnEmbedding,TR_Decoder

class TGAT(nn.Module):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            time_dim:int,
            latent_dim:int,
            embed_dim:int,
            graph:TGN_Graph,
            n_layer:int,
            n_neighbor:int,
            n_head:int
        ):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.time_dim=time_dim
        self.latent_dim=latent_dim
        self.embed_dim=embed_dim
        self.graph=graph
        self.n_layer=n_layer
        self.n_neighbor=n_neighbor
        self.n_head=n_head

        # time encoder
        self.time_encoder=TimeEncoder(time_dim=time_dim)

        # encoder
        self.encoder=GraphAttnEmbedding(
            node_dim=node_dim,
            edge_dim=edge_dim,
            time_dim=time_dim,
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            graph=self.graph,
            n_layer=n_layer,
            n_neighbor=n_neighbor,
            n_head=n_head,
            use_last_state=False,
            use_memory=False,
            time_encoder=self.time_encoder
        )

        # decoder
        self.decoder=TR_Decoder(
            embed_dim=embed_dim,
            latent_dim=latent_dim
        )

    def embedding(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor
        ):
        """
        TR sample의 src, dst node들의 embedding vector 반환.
        """
        batch_size=src.size(0) 
        tar=torch.concat([src,dst],dim=0) 
        tar_t=torch.cat([event_t,event_t],dim=0)
        embedded_tar_vec=self.encoder.compute_embedding(
            tar=tar,
            tar_t=tar_t,
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