import pandas as pd
import numpy as np
import torch
from .temporal_graph import TemporalGraph

class TGN_Graph(TemporalGraph):
    def __init__(self,
            graph_df:pd.DataFrame,
            node_ft:np.ndarray=None,
            edge_ft:np.ndarray=None,
            node_dim:int=32,
            edge_dim:int=32
        ):
        super().__init__(graph_df=graph_df)
        # set node_ft, edge_ft, node_dim, edge_dim
        if node_ft is None: 
            self.set_node_ft(node_dim=node_dim)
        else: 
            self.set_node_ft(node_ft=node_ft)
        if edge_ft is None: 
            self.set_edge_ft(edge_dim=edge_dim)
        else: 
            self.set_edge_ft(edge_ft=edge_ft)

    def to_device(self,device:torch.device):
        self.node_ft=self.node_ft.to(device)
        self.edge_ft=self.edge_ft.to(device)

    def set_node_ft(self,
            node_ft:np.ndarray=None,
            node_dim:int=32
        ):
        """
        Node Feature 설정.
        node_ft=None 인 경우 모두 zero vector로 설정.

        Input:
            node_ft:
                np.ndarray of shape (n_node+1,node_dim) 또는 None.
                0번째 행은 padding node feature (zero vector).
            node_dim:
                node_ft가 None인 경우 사용할 feature dimension.
        """
        if node_ft is None:
            self.node_dim=node_dim
            self.node_ft=torch.zeros(
                (self.n_node+1,self.node_dim),
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
        Edge Feature 설정.
        edge_ft=None 인 경우 모두 zero vector로 설정.
        
        Input:
            edge_ft:
                np.ndarray of shape (n_event+1,edge_dim) 또는 None.
                0번째 행은 padding edge feature (zero vector).
            edge_dim:
                edge_ft가 None인 경우 사용할 feature dimension.
        """
        if edge_ft is None:
            self.edge_dim=edge_dim
            self.edge_ft=torch.zeros(
                (self.n_event+1,self.edge_dim),
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
        neighbor_t=torch.zeros((n_tar,n_neighbor),dtype=torch.float32)
        neighbor_ts=torch.zeros((n_tar,n_neighbor),dtype=torch.float32)

        tar=tar.detach().cpu().tolist()
        tar_t=tar_t.detach().cpu().tolist()
        for i,(tar_id,cut_time) in enumerate(zip(tar,tar_t)):
            neighbors=self.in_adj[tar_id]
            edges=self.in_adj_edge[tar_id]
            times=self.in_adj_t[tar_id]
            if neighbors.size==0:
                continue

            # t < cut_time 인 첫 위치
            cut_idx=np.searchsorted(times,cut_time,side="left")

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
            neighbor[i,offset:]=torch.as_tensor(selected_neighbors,dtype=torch.long)
            neighbor_edge[i,offset:]=torch.as_tensor(selected_edges,dtype=torch.long)
            neighbor_t[i,offset:]=torch.as_tensor(selected_times,dtype=torch.float32)
            neighbor_ts[i,offset:]=torch.as_tensor(cut_time-selected_times,dtype=torch.float32)
        neighbor_mask=neighbor!=0
        return {
            "neighbor":neighbor.to(device=device),
            "neighbor_mask":neighbor_mask.to(device=device),
            "neighbor_t":neighbor_t.to(device=device),
            "neighbor_ts":neighbor_ts.to(device=device),
            "neighbor_edge":neighbor_edge.to(device=device)
        }