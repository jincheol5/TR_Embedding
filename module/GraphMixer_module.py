import torch
from graph import GNN_Graph

class GraphMixer_Module:
    def __init__(self,n_last_event:int):
        self.n_last_event=n_last_event
    
    def set_graph(self,graph:GNN_Graph):
        self.graph=graph
        self.n_node=graph.n_node

    def get_temporal_link_matrix(self,
            tar:torch.Tensor,
            tar_t:torch.Tensor
        ):
        """
        각 노드 tar가 참여한 tar_t 미만의 최근 n_last_event개 temporal link의
        edge feature를 시간 오름차순으로 쌓는다. 부족한 행은 앞쪽에 zero-padding.

        Input: 
            tar: [N,]
            tar_t: [N,]
        Return:
            temporal_link_matrix: [N,n_last_event,edge_dim]
        """
        device=tar.device
        temporal_neighbor=self.graph.get_temporal_neighbor(
            tar=tar,
            tar_t=tar_t,
            n_neighbor=self.n_last_event
        )
        neighbor_edge=temporal_neighbor["neighbor_edge"].to(device)
        temporal_link_matrix=self.graph.get_edge_ft(edge=neighbor_edge)
        return temporal_link_matrix.to(device)

    def get_encoded_node_vec(self,
            tar:torch.Tensor,
            tar_t:torch.Tensor
        ):
        """
        tar_t 미만의 최근 n_last_event개 이벤트에서 중복 이웃과 self-loop를
        제외하고 node feature를 mean-pooling한 뒤 자기 node feature를 더한다.

        Input:
            tar: [N,]
            tar_t: [N,]
        Return:
            encoded_tar_vec: [N,node_dim], 이웃이 없으면 자기 node feature
        """
        device=tar.device
        temporal_neighbor=self.graph.get_temporal_neighbor(
            tar=tar,
            tar_t=tar_t,
            n_neighbor=self.n_last_event
        )
        neighbor=temporal_neighbor["neighbor"]
        neighbor_mask=temporal_neighbor["neighbor_mask"]

        # 정렬 후 동일한 node ID의 첫 위치만 남긴다.
        neighbor,sort_idx=neighbor.sort(dim=1)
        neighbor_mask=neighbor_mask.gather(dim=1,index=sort_idx)
        unique_mask=torch.ones_like(neighbor_mask)
        unique_mask[:,1:]=neighbor[:,1:]!=neighbor[:,:-1]
        neighbor_mask=neighbor_mask & unique_mask & (neighbor!=tar.unsqueeze(-1))
        neighbor_vec=self.graph.get_node_ft(node=neighbor)

        # 패딩과 중복을 제외한 실제 이웃의 수로 나누어 mean-pooling.
        neighbor_vec=neighbor_vec.masked_fill(~neighbor_mask.unsqueeze(-1),0)
        n_neighbor=neighbor_mask.sum(dim=1,keepdim=True).clamp(min=1)
        encoded_tar_vec=self.graph.get_node_ft(node=tar)+neighbor_vec.sum(dim=1)/n_neighbor
        return encoded_tar_vec.to(device)