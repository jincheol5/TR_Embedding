import pandas as pd
import numpy as np
import torch
from .GNN_graph import GNN_Graph

class CAW_N_Graph(GNN_Graph):
    def __init__(self,
            graph_df:pd.DataFrame,
            node_ft:np.ndarray=None,
            edge_ft:np.ndarray=None,
            node_dim:int=32,
            edge_dim:int=32,
            n_walk:int=10,
            walk_len:int=5,
            alpha:float=0.0
        ):
        super().__init__(
            graph_df=graph_df,
            node_ft=node_ft,
            edge_ft=edge_ft,
            node_dim=node_dim,
            edge_dim=edge_dim
        )
        self.n_walk=n_walk
        self.walk_len=walk_len
        self.alpha=alpha

    def _get_walks(self,
            root:int,
            root_t:int
        ):
        """
        Temporal Walk for Causality Extraction.
        exp(alpha * (t - t_p))
            t: 후보 link time
            t_p: 이전에 샘플링한 link의 time
            alpha: 음수가 아닌 하이퍼 파라미터
                - 값이 클수록 최근 link를 더 강조
                - 값이 0이면 균등 샘플링 수행
        
        Return:
            walks: list of walk (size = n_walk)
                walk: list of (node_id, time, edge_id) (len = walk_len + 1, 시작 위치 포함)
                과거 이력이 없으면 남은 위치는 (0,0,0)으로 padding.
                시작 위치와 padding의 incoming_edge_id는 0.
        """
        walks=[]
        for _ in range(self.n_walk):
            walk=[(0,0,0)]*(self.walk_len+1)
            node,cut_time=int(root),int(root_t)
            walk[0]=(node,cut_time,0)
            for step in range(1,self.walk_len+1):
                history_times=self.in_adj_t[node]
                cut_idx=int(np.searchsorted(history_times,cut_time,side="left"))
                if cut_idx==0:
                    break

                ### sampling
                if self.alpha==0: # random 
                    selected=int(np.random.randint(cut_idx))
                else: # weighted
                    with np.errstate(over="ignore",under="ignore"): # np의 오버플로와 언더플로 경고 숨김
                        log_weights=self.alpha*(history_times[:cut_idx]-cut_time)
                        log_weights-=log_weights.max()
                        weights=np.exp(log_weights)
                    probabilities=weights/weights.sum()
                    selected=int(np.random.choice(cut_idx,p=probabilities))
                edge_id=int(self.in_adj_edge[node][selected])
                node,cut_time=int(self.in_adj[node][selected]),int(history_times[selected]) # 선택한 link의 시각을 다음 탐색의 기준 시각으로 사용
                walk[step]=(node,cut_time,edge_id)
            walks.append(walk)
        return walks

    def causality_extraction(self,
            node:torch.Tensor,
            event_t:torch.Tensor
        ):
        """
        Input:
            node: [B,]
            event_t: [B,]
            alpha: finite non-negative recency bias (0 means uniform sampling).
        Return:
            batch_node_walks: (batch_size,n_walk,walk_len+1) list
                각 원소는 (node_id, time, edge_id) 튜플이며 입력 node 순서를 유지.
                빈 배치이면 빈 list 반환.
        """
        
        batch_node_walks=[]
        roots=node.detach().cpu().tolist()
        root_times=event_t.detach().cpu().tolist()
        for root,root_t in zip(roots,root_times):
            walks=self._get_walks(root=root,root_t=root_t)
            batch_node_walks.append(walks)
        return batch_node_walks

    def _get_anonymizated_vec(self,
            target_node:int,
            walks:list
        ):
        """
        target_node의 walk_list에서의 등장 횟수 벡터 계산.
        
        Return:
            anonymizated_vec: [walk_len+1,] float32 tensor
        """
        anonymizated_vec=torch.zeros(self.walk_len+1,dtype=torch.float32)
        if target_node==0:
            return anonymizated_vec
        for walk in walks:
            for step,(node_id,*_) in enumerate(walk):
                if node_id==target_node:
                    anonymizated_vec[step]+=1
        return anonymizated_vec

    def set_based_anonymization(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor
        ):
        """
        Input:
            src: [B,]
            dst: [B,]
            event_t: [B,]
        Return:
            batch_src_caws: (B, n_walk, walk_len+1) list
            batch_dst_caws: (B, n_walk, walk_len+1) list
            caws element = (anonymizated_vec_set,time) = ([2,anonymizated_vec] tensor, time)
                src_anonymizated_vec and dst_anonymizated_vec is stacked to anonymizated_vec_set using torch.stack 
            batch_src_walks
            batch_dst_walks
        """
        batch_src_caws=[]
        batch_dst_caws=[]
        batch_src_walks=self.causality_extraction(node=src,event_t=event_t)
        batch_dst_walks=self.causality_extraction(node=dst,event_t=event_t)
        for src_walks,dst_walks in zip(batch_src_walks,batch_dst_walks):
            node_ids={
                node_id 
                for walks in (src_walks,dst_walks)
                for walk in walks 
                for node_id,*_ in walk
                if node_id!=0
            }
            anonymizated_vec_set_dict={
                node_id:torch.stack((
                    self._get_anonymizated_vec(node_id,src_walks),
                    self._get_anonymizated_vec(node_id,dst_walks)
                ))
                for node_id in node_ids
            }
            anonymizated_vec_set_dict[0]=torch.zeros((2,self.walk_len+1),dtype=torch.float32) # padding node 처리
            batch_src_caws.append([
                [(anonymizated_vec_set_dict[node_id],time) for node_id,time,_ in walk]
                for walk in src_walks
            ])
            batch_dst_caws.append([
                [(anonymizated_vec_set_dict[node_id],time) for node_id,time,_ in walk]
                for walk in dst_walks
            ])
        return {
            "batch_src_caws":batch_src_caws,
            "batch_src_walks":batch_src_walks,
            "batch_dst_caws":batch_dst_caws,
            "batch_dst_walks":batch_dst_walks
        }