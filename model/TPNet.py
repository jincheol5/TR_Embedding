import torch
import torch.nn as nn
from graph import GNN_Graph
from module import TimeEncoder,MLP,TPNet_MLPMixerBlock,TPNet_Matrix

class TPNet(nn.Module):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            time_dim:int,
            latent_dim:int,
            random_ft_dim:int,
            embed_dim:int,
            walk_len:int,
            time_decay_coef:float,
            n_last_event:int,
            mlp_mixer_n_layer:int
        ):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.time_dim=time_dim
        self.latent_dim=latent_dim
        self.random_ft_dim=random_ft_dim
        self.embed_dim=embed_dim
        self.walk_len=walk_len
        self.time_decay_coef=time_decay_coef
        self.n_last_event=n_last_event
        self.mlp_mixer_n_layer=mlp_mixer_n_layer

        # time encoder
        self.time_encoder=TimeEncoder(time_dim=time_dim)

        # MLP for Pairwise Feature Decoding, 공식 구현 코드의 input/output dim 참고 
        self.pairwise_ft_decoder=MLP(
            input_dim=4*(self.walk_len+1)**2,
            latent_dim=4*4*(self.walk_len+1)**2, 
            output_dim=4*(self.walk_len+1)**2
        )

        # MLP for Auxiliary Feature Learning
        self.sequence_encoder=MLP(
            input_dim=node_dim+edge_dim+time_dim+2*(4*(self.walk_len+1)**2),
            latent_dim=latent_dim,
            output_dim=embed_dim
        )

        # MLP-Mixer for Auxiliary Feature Learning
        self.mixers=nn.ModuleList([
            TPNet_MLPMixerBlock(n_token=n_last_event,embed_dim=embed_dim)
            for _ in range(mlp_mixer_n_layer)
        ])

        # link prediction decoder
        self.decoder=MLP(
            input_dim=(2*embed_dim)+(4*(self.walk_len+1)**2),
            latent_dim=latent_dim,
            output_dim=1
        )

    def set_graph(self,graph:GNN_Graph):
        self.graph=graph
        self.n_node=graph.n_node
        ### set TPNet module
        self.matrix=TPNet_Matrix(
            n_node=self.n_node,
            random_ft_dim=self.random_ft_dim,
            walk_len=self.walk_len,
            time_decay_coef=self.time_decay_coef
        )

    def embedding(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor
        ):
        """
        Auxiliary Feature Learning
        """
        batch_size=src.size(0)

        ### u,v의 과거 이력을 m개씩 가져와서 node, edge, timespan, pairwise feature 계산.
        nodes=torch.cat((src,dst),dim=0)
        history=self.graph.get_temporal_neighbor(
            tar=nodes,
            tar_t=torch.cat((event_t,event_t),dim=0),
            n_neighbor=self.n_last_event
        )
        neighbor=history["neighbor"]
        edge=history["neighbor_edge"]
        mask=history["neighbor_mask"]

        node_ft=self.graph.get_node_ft(neighbor)
        edge_ft=self.graph.get_edge_ft(edge)
        time_ft=self.time_encoder(history["neighbor_ts"].unsqueeze(-1))

        # For every history neighbor w, concatenate f_{u,w} and f_{v,w}.
        src_ids=torch.cat((src,src),dim=0)[:,None].expand_as(neighbor)
        dst_ids=torch.cat((dst,dst),dim=0)[:,None].expand_as(neighbor)
        flat_neighbor=neighbor.reshape(-1)
        src_pair_ft=self.pairwise_ft_decoder(self.matrix.get_pairwise_ft(
            src_ids.reshape(-1),flat_neighbor
        ))
        dst_pair_ft=self.pairwise_ft_decoder(self.matrix.get_pairwise_ft(
            dst_ids.reshape(-1),flat_neighbor
        ))
        relative_ft=torch.cat((src_pair_ft,dst_pair_ft),dim=-1).reshape(
            2*batch_size,self.n_last_event,2*(4*(self.walk_len+1)**2)
        )

        ### node, edge, timespan, pairwise feature을 연결한 뒤 sequence_encoder
        sequence_ft=torch.cat((node_ft,edge_ft,time_ft,relative_ft),dim=-1)
        sequence_ft=sequence_ft.masked_fill(~mask.unsqueeze(-1),0.0)
        z=self.sequence_encoder(sequence_ft)
        z=z.masked_fill(~mask.unsqueeze(-1),0.0)

        ### MLP-Mixer
        for mixer in self.mixers:
            z=mixer(z)
        h=z.mean(dim=1)
        src_h,dst_h=h.split(batch_size,dim=0)

        ### get pairwise_vec for link prediction
        pairwise_vec=self.pairwise_ft_decoder(self.matrix.get_pairwise_ft(src,dst))
        return {
            "src_vec":src_h, # [B,embed_dim]
            "dst_vec":dst_h, # [B,embed_dim]
            "pairwise_vec":pairwise_vec # [B,4*(walk_len+1)**2]
        }

    def forward(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor
        ):
        embedded_result=self.embedding(
            src=src,
            dst=dst,
            event_t=event_t
        )
        src_vec=embedded_result["src_vec"]
        dst_vec=embedded_result["dst_vec"]
        pairwise_vec=embedded_result["pairwise_vec"]
        pair_vec=torch.concat([src_vec,dst_vec,pairwise_vec],dim=-1) # [B,2*embed_dim + 4*(walk_len+1)**2]
        pred_logit=self.decoder(pair_vec) # [B,1]
        return pred_logit.squeeze(-1) # [B,]