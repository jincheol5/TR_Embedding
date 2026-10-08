import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from graph import TemporalGraph

class GraphUtils:
    @staticmethod
    def get_TR_result_using_Time_Centric(
            graph:TemporalGraph,
            data_loader:DataLoader
        )->dict[str,torch.Tensor]:
        """
        Input:
            graph
            data_loader
            max_hop
        Return:
            TR_result: dict
                label: [batch_seq_len,N+1,N+1], boolean tensor
                hop: [batch_seq_len,N+1,N+1], int16 tensor
                last_t: [batch_seq_len,N+1,N+1], int16 tensor
                first_t: [batch_seq_len,N+1,N+1], int16 tensor
        """
        n_node=graph.n_node
        TR_result=torch.zeros(
            (len(data_loader),n_node+1,n_node+1),
            dtype=torch.bool
        )
        TR_hop=torch.zeros(
            (len(data_loader),n_node+1,n_node+1),
            dtype=torch.int16
        )
        TR_last_t=torch.zeros(
            (len(data_loader),n_node+1,n_node+1),
            dtype=torch.int16
        )
        TR_first_t=torch.zeros(
            (len(data_loader),n_node+1,n_node+1),
            dtype=torch.int16
        )
        for seq_idx,(_,_,event_t,_) in enumerate(tqdm(data_loader,desc="Compute TR result tensor...")):
            query_time=event_t.max().item()
            gamma_table=graph.compute_TR_using_Time_Centric(query_time=query_time) # 모든 source-dst에 대한 TR + minimum hop 계산
            for src in range(1,n_node+1):
                for dst in range(1,n_node+1):
                    info=gamma_table[src][dst]
                    if not info["r"]:
                        continue
                    TR_result[seq_idx,src,dst]=True
                    TR_hop[seq_idx,src,dst]=int(info["hop"])
                    TR_first_t[seq_idx,src,dst]=int(info["first_t"])
                    TR_last_t[seq_idx,src,dst]=int(info["last_t"])
        return {
            "label":TR_result,
            "hop":TR_hop,
            "last_t":TR_last_t,
            "first_t":TR_first_t
        }

    @staticmethod
    def get_TR_result_using_Temporal_BFS(
            graph:TemporalGraph,
            data_loader:DataLoader
        )->dict[str,torch.Tensor]:
        """
        Input:
            graph
            data_loader
            max_hop
        Return:
            TR_result: dict
                label: [batch_seq_len,N+1,N+1], boolean tensor
                hop: [batch_seq_len,N+1,N+1], int16 tensor
                last_t: [batch_seq_len,N+1,N+1], int16 tensor
                first_t: [batch_seq_len,N+1,N+1], int16 tensor
        """
        n_node=graph.get_num_node()
        TR_result=torch.zeros(
            (len(data_loader),n_node+1,n_node+1),
            dtype=torch.bool
        )
        TR_hop=torch.zeros(
            (len(data_loader),n_node+1,n_node+1),
            dtype=torch.int16
        )
        TR_last_t=torch.zeros(
            (len(data_loader),n_node+1,n_node+1),
            dtype=torch.int16
        )
        TR_first_t=torch.zeros(
            (len(data_loader),n_node+1,n_node+1),
            dtype=torch.int16
        )
        for seq_idx,(_,_,event_t,_) in enumerate(tqdm(data_loader,desc="Compute TR result tensor...")):
            query_time=event_t.max().item()
            for source in range(1,n_node+1):
                TR_info=graph.compute_TR_using_Temporal_BFS(
                    source=source,
                    query_time=query_time
                )
                for dst in range(1,n_node+1):
                    info=TR_info[dst]
                    if not info["r"]:
                        continue

                    TR_result[seq_idx,source,dst]=True
                    TR_hop[seq_idx,source,dst]=int(info["hop"])
                    TR_first_t[seq_idx,source,dst]=int(info["first_t"])

                    # source node의 last_t는 -inf이므로 padding 값 0을 유지
                    if dst!=source:
                        TR_last_t[seq_idx,source,dst]=int(info["last_t"])
        return {
            "label":TR_result,
            "hop":TR_hop,
            "last_t":TR_last_t,
            "first_t":TR_first_t
        }