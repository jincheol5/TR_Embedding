import pandas as pd
import numpy as np
import torch
from typing import Any,Literal
from tqdm import tqdm
from torch.utils.data import DataLoader
from graph import GNN_Graph,CAW_N_Graph,DyGFormer_Graph

class GraphAlgorithm:
    @staticmethod
    def compute_TR_using_Time_Centric(
            eventstream:list[tuple],
            n_node:int,
            gamma_table:dict[int,dict[int,dict[str,Any]]]|None=None,
        )->dict[int,dict[int,dict[str,Any]]]:
        """
        Compute All Pairs Temporal Reachability using Time-Centric Algorithm
        - shortest-hop path 중 가장 빨리 도착하는 path
        - Strictly increasing temporal path
        """
        INF=float("inf")
        NEG_INF=float("-inf")

        ### Gamma Table
        if gamma_table is None:
            gamma_table={
                source:{
                    node:{
                        "r": 1 if node==source else 0,
                        "hop": 0 if node==source else INF,
                        "first_t": 0.0 if node==source else INF,
                        "last_t": 0.0 if node==source else INF,
                        "entries":( 
                            [(0,NEG_INF,0.0)] # (hop,arrival_t,first_t)
                            if node==source
                            else []
                        )
                    }
                    for node in range(1,n_node+1) 
                }
                for source in range(1,n_node+1)
            }

        ### Compute Temporal Reachability
        for src,dst,t in eventstream:
            for source in range(1,n_node+1):
                src_entries=gamma_table[source][src]["entries"]
                if not src_entries:
                    continue
                new_entries=[]
                for hop,arrival_t,first_t in src_entries:
                    # strictly increasing
                    if t<=arrival_t:
                        continue
                    new_hop=hop+1
                    new_first_t=(t if hop==0 else first_t)
                    new_entries.append((new_hop,t,new_first_t))

                for new_hop,new_arrival_t,new_first_t in new_entries:
                    dst_entries=gamma_table[source][dst]["entries"]

                    # 기존 entry가 새 entry를 dominate하면 추가 X
                    if any(
                        old_hop<=new_hop
                        and old_arrival_t<=new_arrival_t
                        for old_hop,old_arrival_t,_ in dst_entries
                    ):
                        continue

                    # 새 entry가 dominate하는 기존 entry 제거
                    gamma_table[source][dst]["entries"]=[
                        entry
                        for entry in dst_entries
                        if not (
                            new_hop<=entry[0]
                            and new_arrival_t<=entry[1]
                        )
                    ]
                    gamma_table[source][dst]["entries"].append((new_hop,new_arrival_t,new_first_t))

                    # minimum hop 결과 갱신
                    if new_hop<gamma_table[source][dst]["hop"]:
                        gamma_table[source][dst]["r"]=1
                        gamma_table[source][dst]["hop"]=new_hop
                        gamma_table[source][dst]["first_t"]=new_first_t
                        gamma_table[source][dst]["last_t"]=new_arrival_t

                    # 같은 minimum hop이면 더 빠른 path 선택
                    elif (
                        new_hop==gamma_table[source][dst]["hop"]
                        and new_arrival_t<gamma_table[source][dst]["last_t"]
                    ):
                        gamma_table[source][dst]["first_t"]=new_first_t
                        gamma_table[source][dst]["last_t"]=new_arrival_t
        return gamma_table

class GraphUtils:
    @staticmethod
    def generate_model_graph(
            model_name:Literal[
                "TGAT",
                "TGN",
                "CAW-N",
                "GraphMixer",
                "DyGFormer",
                "TPNet"
            ],
            graph_df:pd.DataFrame,
            node_ft:np.ndarray=None,
            edge_ft:np.ndarray=None,
            node_dim:int=32,
            edge_dim:int=32,
            one_hot:bool=False,
            **graph_config
        ):
        match model_name:
            case "TGAT"|"TGN"|"GraphMixer"|"TPNet":
                graph=GNN_Graph(
                    graph_df=graph_df,
                    node_ft=node_ft,
                    edge_ft=edge_ft,
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    one_hot=one_hot
                )
            case "CAW-N":
                graph=CAW_N_Graph(
                    graph_df=graph_df,
                    node_ft=node_ft,
                    edge_ft=edge_ft,
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    n_walk=graph_config["n_walk"],
                    walk_len=graph_config["walk_len"],
                    alpha=graph_config["alpha"]
                )
            case "DyGFormer":
                graph=DyGFormer_Graph(
                    graph_df=graph_df,
                    node_ft=node_ft,
                    edge_ft=edge_ft,
                    node_dim=node_dim,
                    edge_dim=edge_dim
                )
        return graph

    @staticmethod
    def convert_batch_to_eventstream(
            src:torch.Tensor,
            dst:torch.Tensor,
            event_t:torch.Tensor
        )->list[tuple[int,int,int]]:
        """
        (src, dst, event_t) 배치를 시간 순서를 유지하며 양방향 이벤트로 변환하여 저장
        """
        eventstream=[]
        for source,destination,timestamp in zip(
                src.tolist(),dst.tolist(),event_t.tolist()
            ):
                source=int(source)
                destination=int(destination)
                timestamp=int(timestamp)
                eventstream.append((source,destination,timestamp))
                if source!=destination:
                    eventstream.append((destination,source,timestamp))
        return eventstream

    @staticmethod
    def get_TR_result_using_Time_Centric(
            n_node:int,
            graph_loader:DataLoader,
            gamma_table:dict[int,dict[int,dict[str,Any]]]|None=None,
        )->dict[str,Any]:
        """
        Input:
            n_node
            graph_loader
            gamma_table: 이전 계산 상태. 전달한 테이블은 제자리에서 갱신됨.
                동일한 노드 ID와 시간 기준을 사용하고, 이전 입력에 이어 시간순으로 전달해야 함.
        Return:
            TR_result: dict
                label: [batch_seq_len,N+1,N+1], boolean tensor
                hop: [batch_seq_len,N+1,N+1], int16 tensor
                first_t: [batch_seq_len,N+1,N+1], int32 tensor
                last_t: [batch_seq_len,N+1,N+1], int32 tensor
            gamma_table: 다음 호출에 전달할 최종 계산 상태 (빈 입력이면 입력 상태 유지)
        """
        TR_label=torch.zeros(
            (len(graph_loader),n_node+1,n_node+1),
            dtype=torch.bool
        )
        TR_hop=torch.zeros(
            (len(graph_loader),n_node+1,n_node+1),
            dtype=torch.int16
        )
        TR_first_t=torch.zeros(
            (len(graph_loader),n_node+1,n_node+1),
            dtype=torch.int32
        )
        TR_last_t=torch.zeros(
            (len(graph_loader),n_node+1,n_node+1),
            dtype=torch.int32
        )
        for seq_idx,(src,dst,event_t,_) in enumerate(tqdm(graph_loader,desc="Compute TR result tensor...")):
            eventstream=GraphUtils.convert_batch_to_eventstream(src=src,dst=dst,event_t=event_t)
            gamma_table=GraphAlgorithm.compute_TR_using_Time_Centric(
                eventstream=eventstream,
                n_node=n_node,
                gamma_table=gamma_table
            )
            for src in range(1,n_node+1):
                for dst in range(1,n_node+1):
                    info=gamma_table[src][dst]
                    if not info["r"]:
                        continue
                    TR_label[seq_idx,src,dst]=True
                    TR_hop[seq_idx,src,dst]=int(info["hop"])
                    TR_first_t[seq_idx,src,dst]=int(info["first_t"])
                    TR_last_t[seq_idx,src,dst]=int(info["last_t"])
        TR_result={
            "label":TR_label,
            "hop":TR_hop,
            "first_t":TR_first_t,
            "last_t":TR_last_t
        }
        return {
            "TR_result":TR_result,
            "gamma_table":gamma_table
        }
