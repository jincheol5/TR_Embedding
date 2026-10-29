import numpy as np
import torch
from graph import DyGFormer_Graph

class DyGFormer_Module:
    def __init__(self):
        pass

    def set_graph(self,graph:DyGFormer_Graph):
        self.graph=graph
        self.n_node=graph.n_node
        self.node_ft=graph.get_node_ft().to(device="cpu")
        self.edge_ft=graph.get_edge_ft().to(device="cpu")

    def get_padded_seq_vec(self,
            node_seq_list:list,
            edge_seq_list:list,
            ts_seq_list:list,
            patch_size:int
        ):
        """
        batch 안의 여러 노드에 대해 길이가 서로 다른 historical interaction sequence를 같은 길이로 맞추고, 그 길이를 patch_size의 배수로 만든다.
        batch 내 sequence의 최대 sequence length를 patch_size의 배수로 맞춰서(부족할경우 올려서) 최종 shape를 [B,final_seq_len,element_dim]으로 맞춘다 (final_seq_len = 최종 기준 seq len). 
        CPU에서 수행한다.

        Input:
            node_seq_list: list of each node's history node sequence 
            edge_seq_list: list of each node's history edge sequence
            ts_seq_list: list of each node's history timespan sequence
            patch_size: int
        Return:
            node_seq: [B,seq_len] 
            node_seq_vec: [B,seq_len,node_dim] 
            edge_seq_vec: [B,seq_len,edge_dim]
            ts_seq: [B,seq_len,1]
        """
        batch_size=len(node_seq_list)

        # batch 내 최대 sequence 길이
        max_batch_seq_len=max(
            len(seq) for seq in node_seq_list
        )

        # patch_size의 배수로 올림
        seq_len=((max_batch_seq_len+patch_size-1)//patch_size)*patch_size

        # padding array 생성
        # padding value = 0
        node_seq=np.zeros(
            (batch_size,seq_len),
            dtype=np.int64
        )
        edge_seq=np.zeros(
            (batch_size,seq_len),
            dtype=np.int64
        )
        ts_seq=np.zeros(
            (batch_size,seq_len),
            dtype=np.int64
        )

        # sequence를 앞에서부터 저장
        # 남는 오른쪽 영역은 padding
        for i in range(batch_size):
            n=len(node_seq_list[i])
            node_seq[i,:n]=node_seq_list[i]
            edge_seq[i,:n]=edge_seq_list[i]
            ts_seq[i,:n]=ts_seq_list[i]

        # numpy -> torch
        node_seq=torch.from_numpy(node_seq)
        edge_seq=torch.from_numpy(edge_seq)
        ts_seq=torch.from_numpy(ts_seq).unsqueeze(-1)

        # feature lookup
        node_seq_vec=self.node_ft[node_seq]
        edge_seq_vec=self.edge_ft[edge_seq]
        return {
            "node_seq":node_seq,
            "node_seq_vec":node_seq_vec,
            "edge_seq_vec":edge_seq_vec,
            "ts_seq":ts_seq
        }

    def get_co_occurrence_vec(self,
            src_seq:torch.Tensor,
            dst_seq:torch.Tensor
        ):
        """
        Compute Neighbor Co-occurrence Vector.
        CPU에서 수행

        torch.bincount(): 0 이상의 정수 텐서에서 각 정수가 몇 번 등장했는지 세는 함수
        
        padding node (id=0) 에 대해서는 [0,0]
        Input:
            src_seq: [B,src_len]
            dst_seq: [B,dst_len]

        Return:
            src_co_vec: [B,src_len,2]
            dst_co_vec: [B,dst_len,2]
        """
        src_co_vec=[]
        dst_co_vec=[]
        for src,dst in zip(src_seq,dst_seq):
            max_id=torch.max(src.max(),dst.max()).item()+1
            src_count=torch.bincount(src,minlength=max_id)
            dst_count=torch.bincount(dst,minlength=max_id)

            src_co_vec.append(
                torch.stack([
                    src_count[src],
                    dst_count[src]
                ], dim=-1)
            )
            dst_co_vec.append(
                torch.stack([
                    src_count[dst],
                    dst_count[dst]
                ], dim=-1)
            )

        src_co_vec=torch.stack(src_co_vec).float()
        dst_co_vec=torch.stack(dst_co_vec).float()

        # padding node 처리
        src_co_vec[src_seq==0]=0
        dst_co_vec[dst_seq==0]=0
        return {
            "src_co_vec": src_co_vec,
            "dst_co_vec": dst_co_vec
        }

    def get_patching_vec(self,
            node_seq_vec:torch.Tensor,
            edge_seq_vec:torch.Tensor,
            ts_seq_vec:torch.Tensor,
            co_seq_vec:torch.Tensor,
            patch_size:int
        ):
        """
        Patching Technique in DyGFormer.
        GPU에서 수행.

        p=patch_size
        l=seq_len/patch_size

        Input:
            node_seq_vec: [batch_size,seq_len,node_dim]
            edge_seq_vec: [batch_size,seq_len,edge_dim]
            ts_seq_vec: [batch_size,seq_len,time_dim]
            co_seq_vec:[batch_size,seq_len,co_dim]
            patch_size: int
        Return:
            M_n: [B,l,node_dim x p]
            M_e: [B,l,edge_dim x p]
            M_t: [B,l,time_dim x p]
            M_c: [B,l,co_dim x p]
        """
        batch_size,seq_len,_=node_seq_vec.size()
        l=seq_len//patch_size
        M_n=node_seq_vec.reshape(batch_size,l,-1)
        M_e=edge_seq_vec.reshape(batch_size,l,-1)
        M_t=ts_seq_vec.reshape(batch_size,l,-1)
        M_c=co_seq_vec.reshape(batch_size,l,-1)
        return {
            "node":M_n,
            "edge":M_e,
            "ts":M_t,
            "co":M_c
        }