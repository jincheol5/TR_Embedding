import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

class GNN_Graph:
    """
    Directed Temporal Graph
    """
    def __init__(self,
            graph_df:pd.DataFrame,
            node_ft:np.ndarray|None=None,
            edge_ft:np.ndarray|None=None,
            node_dim:int=32,
            edge_dim:int=32,
            one_hot:bool=False
        ):
        # set graph info
        self.n_node=int(max(graph_df["u"].max(),graph_df["i"].max()))
        self.n_event=len(graph_df)

        # set node, edge feature
        self.set_node_ft(node_ft=node_ft,node_dim=node_dim,one_hot=one_hot)
        self.set_edge_ft(edge_ft=edge_ft,edge_dim=edge_dim)

        # incoming adj for GNN
        self.in_adj=[[] for _ in range(self.n_node+1)]
        self.in_adj_edge=[[] for _ in range(self.n_node+1)]
        self.in_adj_t=[[] for _ in range(self.n_node+1)]

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

    def init_in_adj(self):
        # initialize incoming adj for GNN
        self.in_adj=[[] for _ in range(self.n_node+1)]
        self.in_adj_edge=[[] for _ in range(self.n_node+1)]
        self.in_adj_t=[[] for _ in range(self.n_node+1)]

    def update_in_adj(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor,
            edge:torch.Tensor
        ):
        src_list=src.cpu().tolist()
        dst_list=dst.cpu().tolist()
        event_t_list=event_t.cpu().tolist()
        edge_list=edge.cpu().tolist()
        for src,dst,t,edge_id in zip(src_list,dst_list,event_t_list,edge_list):
            self.in_adj[dst].append(src)
            self.in_adj_edge[dst].append(edge_id)
            self.in_adj_t[dst].append(t)

    def set_in_adj_for_evaluate(self,
            train_loader:DataLoader,
            val_loader:DataLoader
        ):
        ### set train_loader eventstream
        for src,dst,event_t,edge in train_loader:
            self.update_in_adj(
                src=src,
                dst=dst,
                event_t=event_t,
                edge=edge
            )
        ### set val_loader eventstream
        for src,dst,event_t,edge in val_loader:
            self.update_in_adj(
                src=src,
                dst=dst,
                event_t=event_t,
                edge=edge
            )

    def get_temporal_neighbor(self,
            tar:torch.Tensor,
            tar_t:torch.Tensor,
            n_neighbor:int
        ):
        """
        현재까지 update_in_adj로 업데이트된 eventstream으로 temporal_neighbor 계산.

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
            if tar_id==0 or n_neighbor==0:
                continue
            neighbors=self.in_adj[tar_id]
            edges=self.in_adj_edge[tar_id]
            times=self.in_adj_t[tar_id]
            if not neighbors:
                continue

            # 최근 n_neighbor개 선택
            selected_neighbors=neighbors[-n_neighbor:]
            selected_edges=edges[-n_neighbor:]
            selected_times=times[-n_neighbor:]

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
            selected_times=torch.as_tensor(
                selected_times,
                dtype=torch.int64
            )
            neighbor_t[i,offset:]=selected_times
            neighbor_ts[i,offset:]=cut_time-selected_times
        neighbor_mask=neighbor!=0
        return {
            "neighbor":neighbor.to(device=device),
            "neighbor_mask":neighbor_mask.to(device=device),
            "neighbor_t":neighbor_t.to(device=device),
            "neighbor_ts":neighbor_ts.to(device=device),
            "neighbor_edge":neighbor_edge.to(device=device)
        }
