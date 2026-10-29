import torch
import torch.nn as nn
from graph import GNN_Graph
from module import TimeEncoder,MLP,MLPMixerBlock,GraphMixer_Module

class GraphMixer(nn.Module):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            time_dim:int,
            latent_dim:int,
            embed_dim:int,
            n_last_event:int
        ):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.time_dim=time_dim
        self.latent_dim=latent_dim
        self.embed_dim=embed_dim
        self.n_last_event=n_last_event

        # time encoder
        self.time_encoder=TimeEncoder(
            time_dim=time_dim,
            parameter_requires_grad=False
        )

        # MLP-Mixer
        self.link_encoder=nn.Linear(edge_dim+time_dim,embed_dim)
        self.mlp_mixer=MLPMixerBlock(
            n_token=n_last_event,
            embed_dim=embed_dim
        )
        self.link_norm=nn.LayerNorm(embed_dim)
        self.link_output=nn.Linear(embed_dim,embed_dim)

        # decoder
        self.decoder=MLP(
            input_dim=2*(embed_dim+node_dim),
            latent_dim=latent_dim,
            output_dim=1
        )

    def set_graph(self,graph:GNN_Graph):
        self.graph=graph
        self.n_node=graph.n_node
        self.module=GraphMixer_Module(n_last_event=self.n_last_event)
        self.module.set_graph(graph=graph)

    def embedding(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor
        ):
        """
        Return:
            src_vec, dst_vec: [B,embed_dim+node_dim]
        """
        batch_size=src.size(0)
        tar=torch.cat((src,dst),dim=0)
        tar_t=torch.cat((event_t,event_t),dim=0)
        history=self.graph.get_temporal_neighbor(
            tar=tar,
            tar_t=tar_t,
            n_neighbor=self.n_last_event
        )

        # 시간 차이는 정수 상태에서 계산된 neighbor_ts를 사용한다.
        edge_vec=self.module.get_temporal_link_matrix(tar=tar,tar_t=tar_t)
        time_vec=self.time_encoder(history["neighbor_ts"].unsqueeze(-1))
        link_vec=self.link_encoder(torch.cat((edge_vec,time_vec),dim=-1))

        # projection bias와 time encoding을 포함해 패딩 토큰 전체를 0으로 만든다.
        link_vec=link_vec.masked_fill(~history["neighbor_mask"].unsqueeze(-1),0)

        # 최신 이벤트부터 배치하고, 패딩은 뒤쪽에 둔다.
        link_vec=link_vec.flip(dims=[1])

        link_vec=self.mlp_mixer(link_vec)
        link_vec=self.link_output(self.link_norm(link_vec).mean(dim=1))
        node_vec=self.module.get_encoded_node_vec(tar=tar,tar_t=tar_t)
        encoded_tar_vec=torch.cat((link_vec,node_vec),dim=-1)
        return {
            "src_vec":encoded_tar_vec[:batch_size],
            "dst_vec":encoded_tar_vec[batch_size:]
        }

    def forward(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor
        ):
        embedded_vec=self.embedding(
            src=src,
            dst=dst,
            event_t=event_t
        )
        src_vec=embedded_vec["src_vec"]
        dst_vec=embedded_vec["dst_vec"]

        ### decode 
        pair_vec=torch.concat([src_vec,dst_vec],dim=-1)
        pred_logit=self.decoder(pair_vec) # [B,1]
        return pred_logit.squeeze(-1) # [B,]