import random
import pandas as pd
import numpy as np
from typing import Any
from bisect import bisect_right

class TemporalGraph:
    def __init__(self,graph_df:pd.DataFrame):
        self.graph_df=graph_df

        ### graph information
        self.n_node=int(max(graph_df["u"].max(),graph_df["i"].max()))
        self.n_event=int(graph_df["idx"].max())
        self.max_u=int(graph_df["u"].max())
        self.max_t=float(graph_df["t"].max())

        ### save in_adj, out_adj info
        self.edge_events=[]

        # in_adj for GNN
        in_adj=[[] for _ in range(self.n_node+1)]
        in_adj_edge=[[] for _ in range(self.n_node+1)]
        in_adj_t=[[] for _ in range(self.n_node+1)]

        # out_adj for algorithm
        out_adj=[[] for _ in range(self.n_node+1)]
        out_adj_edge=[[] for _ in range(self.n_node+1)]
        out_adj_t=[[] for _ in range(self.n_node+1)]
        for event in graph_df.itertuples(index=False): # col: [u,i,t,idx=edge_id]
            src=int(event.u)
            dst=int(event.i)
            t=float(event.t)
            edge_id=int(event.idx)

            # edge event 저장
            self.edge_events.append((src,dst,t,edge_id))

            # incoming adj 저장
            in_adj[dst].append(src)
            in_adj_edge[dst].append(edge_id)
            in_adj_t[dst].append(t)

            # outgoing adj 저장
            out_adj[src].append(dst)
            out_adj_edge[src].append(edge_id)
            out_adj_t[src].append(t)

        # edge_events를 t 기준 오름차순 정렬
        self.edge_events.sort(key=lambda x:x[2])

        ### convert list -> numpy array
        # incoming adj
        self.in_adj=[
            np.asarray(values,dtype=np.int64)
            for values in in_adj
        ]
        self.in_adj_edge=[
            np.asarray(values,dtype=np.int64)
            for values in in_adj_edge
        ]
        self.in_adj_t=[
            np.asarray(values,dtype=np.float64)
            for values in in_adj_t
        ]

        # outgoing adj
        self.out_adj=[
            np.asarray(values,dtype=np.int64)
            for values in out_adj
        ]
        self.out_adj_edge=[
            np.asarray(values,dtype=np.int64)
            for values in out_adj_edge
        ]
        self.out_adj_t=[
            np.asarray(values,dtype=np.float64)
            for values in out_adj_t
        ]

    def get_edge_events(self,query_time:float|None=None):
        if query_time is None:
            return self.edge_events
        else:
            # bisect 검색을 위한 timestamp 리스트
            self.edge_event_times=[
                event[2] for event in self.edge_events
            ]
            idx=bisect_right(
                self.edge_event_times,
                query_time
            )
            return self.edge_events[:idx]

    def compute_TR_using_Time_Centric(self,
            query_time:float|None=None
        )->dict[int,dict[int,dict[str,Any]]]:
        """
        Compute Temporal Reachability using Time-Centric
        - shortest-hop path 중 가장 빨리 도착하는 path
        - Strictly increasing temporal path
        """
        INF=float("inf")
        NEG_INF=float("-inf")

        ### 최종 결과
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
                for node in range(1,self.n_node+1) 
            }
            for source in range(1,self.n_node+1)
        }

        edge_events=self.get_edge_events(query_time=query_time)
        for src,dst,t,_ in edge_events:
            for source in range(1,self.n_node+1):
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

    def compute_TR_using_Temporal_BFS(self,
            source:int,
            query_time:float|None=None
        ):
        """
        Compute Temporal Reachability using Temporal BFS
        - shortest-hop path 중 가장 빨리 도착하는 path
        - Strictly increasing temporal path
        """
        max_hop=self.n_node-1
        INF=float("inf")
        NEG_INF=float("-inf")
        TR_info={
            node:{
                "r":0,
                "hop":INF,
                "first_t":INF,
                "last_t":INF
            }
            for node in range(1,self.n_node+1)
        }
        TR_info[source]={
            "r":1,
            "hop":0,
            "first_t":0.0,
            "last_t":NEG_INF
        }

        # 탐색용 상태: node -> (arrival_t, first_t)
        current={
            source:(NEG_INF,0.0)
        }

        # 지금까지 각 node에 가장 일찍 도착한 시간
        best_arrival={
            source:NEG_INF
        }

        for hop in range(1,max_hop+1):
            next_state={}

            for node,(arrival_t,first_t) in current.items():
                times=self.out_adj_t[node]
                neighbors=self.out_adj[node]

                start=np.searchsorted(
                    times,
                    arrival_t,
                    side="right"
                )
                end=(
                    len(times)
                    if query_time is None
                    else np.searchsorted(
                        times,
                        query_time,
                        side="right"
                    )
                )

                for idx in range(start,end):
                    dst=int(neighbors[idx])
                    t=float(times[idx])

                    # 이전 hop에서 이미 더 일찍 도착했으면
                    # 현재 경로는 탐색 가치 없음
                    if best_arrival.get(dst,INF)<=t:
                        continue

                    # 같은 hop에서 이미 더 일찍 도착했으면 제거
                    if dst in next_state and next_state[dst][0]<=t:
                        continue

                    next_state[dst]=(
                        t,
                        t if hop==1 else first_t
                    )

            if not next_state:
                break

            for node,(arrival_t,first_t) in next_state.items():
                # 탐색용 earliest arrival은 계속 갱신
                best_arrival[node]=arrival_t

                # 결과는 최초 발견 때만 저장
                # hop-layer BFS이므로 최초 발견 hop = minimum hop
                if TR_info[node]["r"]==0:
                    TR_info[node]={
                        "r":1,
                        "hop":hop,
                        "first_t":first_t,
                        "last_t":arrival_t
                    }
            current=next_state
        return TR_info