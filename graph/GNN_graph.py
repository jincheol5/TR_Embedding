import pandas as pd
import numpy as np
import torch
from .temporal_graph import TemporalGraph

class GNN_Graph(TemporalGraph):
    def __init__(self,
            graph_df:pd.DataFrame,
            node_ft:np.ndarray=None,
            edge_ft:np.ndarray=None,
            node_dim:int=32,
            edge_dim:int=32,
            one_hot:bool=False
        ):
        super().__init__(graph_df=graph_df,)
        self.set_node_ft(node_ft=node_ft,node_dim=node_dim,one_hot=one_hot)
        self.set_edge_ft(edge_ft=edge_ft,edge_dim=edge_dim)

    def to_device(self,device:torch.device):
        self.node_ft=self.node_ft.to(device)
        self.edge_ft=self.edge_ft.to(device)

    def set_node_ft(self,
            node_ft:np.ndarray=None,
            node_dim:int=32,
            one_hot:bool=False
        ):
        """
        Input:
            node_ft_np: 
                np.ndarray of shape (n_node+1, node_dim) 또는 None.
                0번째 행은 padding node feature (zero vector).
            node_dim: int
            one_hot: True이면 node_ft, node_dim 대신 n_node 차원의 원 핫 벡터 생성.
                node_id=i (1~n_node)는 i-1번째 원소가 1, node_id=0은 제로 벡터.
        """
        if one_hot:
            self.node_dim=self.n_node
            self.node_ft=torch.zeros(
                (self.n_node+1,self.node_dim),
                dtype=torch.float32
            )
            self.node_ft[1:].diagonal().fill_(1)
        elif node_ft is None:
            self.node_dim=node_dim
            self.node_ft=torch.zeros(
                (self.n_node+1,node_dim),
                dtype=torch.float32
            )
        else:
            self.node_dim=node_ft.shape[1]
            self.node_ft=torch.as_tensor(node_ft,dtype=torch.float32)

    def set_edge_ft(self,
            edge_ft:np.ndarray=None,
            edge_dim:int=32
        ):
        """
        Input:
            edge_ft_np: 
                np.ndarray of shape (n_edge+1, edge_dim)
                0번째 행은 padding edge의 feature(보통 zero vector).
                None이면 one feature 생성.
            edge_dim: int
        """
        if edge_ft is None:
            self.edge_dim=edge_dim
            self.edge_ft=torch.zeros(
                (self.n_event+1,edge_dim),
                dtype=torch.float32
            )
        else:
            self.edge_dim=edge_ft.shape[1]
            self.edge_ft=torch.as_tensor(edge_ft,dtype=torch.float32)

    def get_node_ft(self,node:torch.Tensor|None=None):
        """
        Input:
            node: [B,]
        Return:
            node_ft
        """
        if node is None:
            return self.node_ft
        else:
            return self.node_ft[node]

    def get_edge_ft(self,edge:torch.Tensor|None=None):
        """
        Input:
            edge: [B,]
        Return:
            edge_ft
        """
        if edge is None:
            return self.edge_ft
        else:
            return self.edge_ft[edge]

    def get_temporal_neighbor(self,
            tar:torch.Tensor,
            tar_t:torch.Tensor,
            n_neighbor:int
        ):
        """
        Input:
            tar: [n_tar,]
            tar_t: [n_tar,]
            n_neighbor: int

        Return:
            neighbor: [n_tar,n_neighbor]
            neighbor_mask: [n_tar,n_neighbor]
            neighbor_t: [n_tar,n_neighbor]
            neighbor_ts: [n_tar,n_neighbor]
            neighbor_edge: [n_tar,n_neighbor]
        """
        device=tar.device
        n_tar=tar.size(0)

        neighbor=torch.zeros((n_tar,n_neighbor),dtype=torch.long)
        neighbor_edge=torch.zeros((n_tar,n_neighbor),dtype=torch.long)
        neighbor_t=torch.zeros((n_tar,n_neighbor),dtype=torch.int64)
        neighbor_ts=torch.zeros((n_tar,n_neighbor),dtype=torch.int64)

        tar=tar.detach().cpu().tolist()
        tar_t=tar_t.detach().cpu().tolist()
        for i,(tar_id,cut_time) in enumerate(zip(tar,tar_t)):
            neighbors=self.in_adj[tar_id]
            edges=self.in_adj_edge[tar_id]
            times=self.in_adj_t[tar_id]
            if neighbors.size==0:
                continue

            # t < cut_time 인 첫 위치
            cut_idx=np.searchsorted(
                times,
                cut_time,
                side="left"
            )

            # 최근 n_neighbor개 선택
            start_idx=max(0,cut_idx-n_neighbor)
            selected_neighbors=neighbors[start_idx:cut_idx]
            selected_edges=edges[start_idx:cut_idx]
            selected_times=times[start_idx:cut_idx]

            n_selected=len(selected_neighbors)
            if n_selected==0:
                continue

            # 앞쪽은 padding, 뒤쪽에 실제 neighbor 저장
            offset=n_neighbor-n_selected
            neighbor[i,offset:]=torch.as_tensor(
                selected_neighbors,
                dtype=torch.long
            )
            neighbor_edge[i,offset:]=torch.as_tensor(
                selected_edges,
                dtype=torch.long
            )
            neighbor_t[i,offset:]=torch.as_tensor(
                selected_times,
                dtype=torch.int64
            )
            neighbor_ts[i,offset:]=torch.as_tensor(
                cut_time-selected_times,
                dtype=torch.int64
            )
        neighbor_mask=neighbor!=0
        return {
            "neighbor":neighbor.to(device=device),
            "neighbor_mask":neighbor_mask.to(device=device),
            "neighbor_t":neighbor_t.to(device=device),
            "neighbor_ts":neighbor_ts.to(device=device),
            "neighbor_edge":neighbor_edge.to(device=device)
        }
