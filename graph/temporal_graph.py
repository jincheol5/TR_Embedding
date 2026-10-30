import pandas as pd
import numpy as np

class TemporalGraph:
    """
    Directed Temporal Graph
    """
    def __init__(self,graph_df:pd.DataFrame):
        self.set_graph(graph_df=graph_df)

    def set_graph(self,graph_df:pd.DataFrame):
        ### graph information
        self.graph_df=graph_df
        self.n_event=len(graph_df)
        self.n_node=int(max(graph_df["u"].max(),graph_df["i"].max()))

        # incoming adj for GNN
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
            t=int(event.t)
            edge_id=int(event.idx)

            # incoming adj 저장
            in_adj[dst].append(src)
            in_adj_edge[dst].append(edge_id)
            in_adj_t[dst].append(t)

            # outgoing adj 저장
            out_adj[src].append(dst)
            out_adj_edge[src].append(edge_id)
            out_adj_t[src].append(t)

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
            np.asarray(values,dtype=np.int64)
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






