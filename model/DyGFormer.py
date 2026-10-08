import torch
import torch.nn as nn
from graph import DyGFormer_Graph
from module import TimeEncoder,TransformerEncoderBlock,DyGFormer_Module,TR_Decoder

class DyGFormer(nn.Module):
    def __init__(self,
            node_dim:int,
            edge_dim:int,
            latent_dim:int,
            time_dim:int,
            co_dim:int,
            common_dim:int,
            embed_dim:int,
            graph:DyGFormer_Graph,
            n_layer:int,
            n_head:int,
            max_history_len:int,
            patch_size:int
        ):
        super().__init__()
        self.node_dim=node_dim
        self.edge_dim=edge_dim
        self.latent_dim=latent_dim
        self.time_dim=time_dim
        self.co_dim=co_dim
        self.common_dim=common_dim
        self.embed_dim=embed_dim
        self.graph=graph
        self.n_layer=n_layer
        self.n_head=n_head
        self.max_history_len=max_history_len
        self.patch_size=patch_size

        # DyGFormer Module
        self.module=DyGFormer_Module(graph=graph)

        # time encoder
        self.time_encoder=TimeEncoder(time_dim=time_dim)

        # Neighbor Co-occurrence Encoding
        self.NCoE=nn.Sequential(
            nn.Linear(in_features=1,out_features=latent_dim),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim,out_features=co_dim)
        )

        # patch_encoder
        self.patch_encoder=nn.ModuleDict(
            {
                "node":nn.Linear(
                    in_features=patch_size*node_dim,
                    out_features=common_dim
                ),
                "edge":nn.Linear(
                    in_features=patch_size*edge_dim,
                    out_features=common_dim
                ),
                "time":nn.Linear(
                    in_features=patch_size*time_dim,
                    out_features=common_dim
                ),
                "co":nn.Linear(
                    in_features=patch_size*co_dim,
                    out_features=common_dim
                )
            }
        )

        # transformer encoder
        self.transformer_encoders=nn.ModuleList(
            [
                TransformerEncoderBlock(
                    attn_dim=4*common_dim,
                    latent_dim=latent_dim,
                    n_head=n_head
                )
                for _ in range(n_layer)
            ]
        )

        # Time-aware Node Representation
        self.output_layer=nn.Linear(
            in_features=4*common_dim,
            out_features=embed_dim
        )

        # decoder
        self.decoder=TR_Decoder(
            embed_dim=embed_dim,
            latent_dim=latent_dim
        )

    def forward(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor,
        ):
        """
        TR sample에 대한 Temporal Reachability 예측.
        
        < STEP > 
        In CPU:
            1. src, dst에 대한 history sequence 가져오기.
            2. padding sequence using patch_size.
            3. src_seq, dst_seq로 co-occurrence vector 구하기.
        In GPU:
            5. time_encoder로 ts_vec 인코딩.
            6. NCoE로 co_vec 인코딩.
            7. patching
            8. patch_encoder로 마지막 차원 통일.
            9. ... 

        Input: sampling 된 pos/neg pair의 src,dst,event_t (=query time)
            src: [B,] 
            dst: [B,]
            event_t: [B,]
        Return:
            pred_logit: [B,1]
        """
        device=src.device

        ### 1. get sequence data
        src_result=self.graph.get_history_seq(
            node=src,
            event_t=event_t,
            max_history_len=self.max_history_len
        )
        src_node_seq_list=src_result["node"]
        src_edge_seq_list=src_result["edge"]
        src_ts_seq_list=src_result["ts"]

        dst_result=self.graph.get_history_seq(
            node=dst,
            event_t=event_t,
            max_history_len=self.max_history_len
        )
        dst_node_seq_list=dst_result["node"]
        dst_edge_seq_list=dst_result["edge"]
        dst_ts_seq_list=dst_result["ts"]

        ### 2. get padded sequence
        src_result=self.module.get_padded_seq_vec(
            node_seq_list=src_node_seq_list,
            edge_seq_list=src_edge_seq_list,
            ts_seq_list=src_ts_seq_list,
            patch_size=self.patch_size
        )
        src_seq=src_result["node_seq"]
        src_ts_seq=src_result["ts_seq"]
        src_node_seq_vec=src_result["node_seq_vec"]
        src_edge_seq_vec=src_result["edge_seq_vec"]

        dst_result=self.module.get_padded_seq_vec(
            node_seq_list=dst_node_seq_list,
            edge_seq_list=dst_edge_seq_list,
            ts_seq_list=dst_ts_seq_list,
            patch_size=self.patch_size
        )
        dst_seq=dst_result["node_seq"]
        dst_ts_seq=dst_result["ts_seq"]
        dst_node_seq_vec=dst_result["node_seq_vec"]
        dst_edge_seq_vec=dst_result["edge_seq_vec"]

        ### 3. compute co-occurrence vector
        co_result=self.module.get_co_occurrence_vec(
            src_seq=src_seq,
            dst_seq=dst_seq
        )
        src_co_vec=co_result["src_co_vec"]
        dst_co_vec=co_result["dst_co_vec"]

        ### 4. move to GPU
        src_seq=src_seq.to(device)
        src_node_seq_vec=src_node_seq_vec.to(device)
        src_edge_seq_vec=src_edge_seq_vec.to(device)
        src_ts_seq=src_ts_seq.to(device)
        src_co_vec=src_co_vec.to(device)

        dst_seq=dst_seq.to(device)
        dst_node_seq_vec=dst_node_seq_vec.to(device)
        dst_edge_seq_vec=dst_edge_seq_vec.to(device)
        dst_ts_seq=dst_ts_seq.to(device)
        dst_co_vec=dst_co_vec.to(device)

        ### 5. encoding timespan using time_encoder
        src_ts_seq_vec=self.time_encoder(src_ts_seq) 
        dst_ts_seq_vec=self.time_encoder(dst_ts_seq)

        # 공식 구현과 동일하게 패딩 위치의 시간 벡터를 0으로 설정
        src_ts_seq_vec[src_seq==0]=0.0
        dst_ts_seq_vec[dst_seq==0]=0.0

        ### 6. encoding co_vec using NCoE
        src_co_vec_0=src_co_vec[:,:,0:1] 
        src_co_vec_1=src_co_vec[:,:,1:2] 
        dst_co_vec_0=dst_co_vec[:,:,0:1] 
        dst_co_vec_1=dst_co_vec[:,:,1:2] 

        src_co_vec_0=self.NCoE(src_co_vec_0)
        src_co_vec_1=self.NCoE(src_co_vec_1)
        dst_co_vec_0=self.NCoE(dst_co_vec_0)
        dst_co_vec_1=self.NCoE(dst_co_vec_1)

        src_co_vec=src_co_vec_0+src_co_vec_1 # [B,seq_len,co_dim]
        dst_co_vec=dst_co_vec_0+dst_co_vec_1 # [B,seq_len,co_dim]

        ### 7. get patching vector
        src_result=self.module.get_patching_vec(
            node_seq_vec=src_node_seq_vec,
            edge_seq_vec=src_edge_seq_vec,
            ts_seq_vec=src_ts_seq_vec,
            co_seq_vec=src_co_vec,
            patch_size=self.patch_size
        )
        src_M_n=src_result["node"]
        src_M_e=src_result["edge"]
        src_M_t=src_result["ts"]
        src_M_c=src_result["co"]

        dst_result=self.module.get_patching_vec(
            node_seq_vec=dst_node_seq_vec,
            edge_seq_vec=dst_edge_seq_vec,
            ts_seq_vec=dst_ts_seq_vec,
            co_seq_vec=dst_co_vec,
            patch_size=self.patch_size
        )
        dst_M_n=dst_result["node"]
        dst_M_e=dst_result["edge"]
        dst_M_t=dst_result["ts"]
        dst_M_c=dst_result["co"]

        ### 8. apply patch encoder
        src_M_n=self.patch_encoder["node"](src_M_n) 
        src_M_e=self.patch_encoder["edge"](src_M_e) 
        src_M_t=self.patch_encoder["time"](src_M_t) 
        src_M_c=self.patch_encoder["co"](src_M_c) 

        dst_M_n=self.patch_encoder["node"](dst_M_n) 
        dst_M_e=self.patch_encoder["edge"](dst_M_e) 
        dst_M_t=self.patch_encoder["time"](dst_M_t) 
        dst_M_c=self.patch_encoder["co"](dst_M_c) 

        ### 9. get updated node vec z
        src_z=torch.concat([src_M_n,src_M_e,src_M_t,src_M_c],dim=-1) # [B,src_l,4 x patch_dim]
        dst_z=torch.concat([dst_M_n,dst_M_e,dst_M_t,dst_M_c],dim=-1) # [B,dst_l,4 x patch_dim]
        src_l=src_z.size(1)
        dst_l=dst_z.size(1)
        z=torch.concat([src_z,dst_z],dim=1) 

        ### 10. apply transformer encoder
        for transformer_encoder in self.transformer_encoders:
            z=transformer_encoder(z)
        src_z,dst_z=torch.split(z,[src_l,dst_l],dim=1)

        ### 11. get Time-aware Node Representation h
        src_h=src_z.mean(dim=1) # [B,4 x patch_dim]
        dst_h=dst_z.mean(dim=1) # [B,4 x patch_dim]
        h=torch.concat([src_h,dst_h],dim=0) # [2B,4 x patch_dim]
        h=self.output_layer(h) # [2B,output_dim]
        src_h,dst_h=torch.chunk(h,chunks=2,dim=0) # [B,output_dim]

        ### 12. predict TR
        pred_logit=self.decoder(
            src_vec=src_h,
            dst_vec=dst_h
        ) # [B,1]
        return pred_logit