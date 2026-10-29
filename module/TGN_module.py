import torch
import torch.nn as nn
from typing import Literal
from graph import GNN_Graph
from .time_module import TimeEncoder

class Memory(nn.Module):
    def __init__(self,
            n_node:int,
            mem_dim:int=32
        ):
        """
        mem_dim: memory vector dim
        mem_vec: memory vector
        mem_t: last update time of memory 
        """
        super().__init__()
        self.n_node=n_node
        self.mem_dim=mem_dim
        self.register_buffer(
            "mem_vec",
            torch.zeros(self.n_node+1,self.mem_dim),
        )
        self.register_buffer(
            "mem_t",
            torch.zeros(self.n_node+1,dtype=torch.int64),
        )
        self.init_memory_state()
    
    def init_memory_state(self):
        """
        initialize all the mem_ft and mem_t to zero vectors, which should be called at the start of each epoch
        """
        self.mem_vec.data.zero_()
        self.mem_t.data.zero_()

    def memory_detach(self):
        self.mem_vec.detach_()
        self.mem_t.detach_()

    def get_mem_vec(self,
            node:torch.Tensor|None=None
        ):
        """
        Input:
            node: [N,]
        Output:
            mem_vec: [N,mem_dim]
        """
        if node is None:
            return self.mem_vec
        else:
            return self.mem_vec[node]

    def get_mem_t(self,
            node:torch.Tensor|None=None
        ):
        """
        Input:
            node: [N,]
        Output:
            mem_vec: [N,]
        """
        if node is None:
            return self.mem_t
        else:
            return self.mem_t[node]

    def get_node_timespan(self,
            node:torch.Tensor,
            event_t:torch.Tensor
        ):
        """
        Input:
            node: [N,]
            event_t: [N,]
        Return:
            node_ts: [N,1]
        """
        node_ts=torch.abs(
            event_t-self.mem_t[node]
        )
        return node_ts.unsqueeze(-1) # [N,1]

    def update_memory_state(self,
            node:torch.Tensor,
            mem_vec:torch.Tensor,
            mem_t:torch.Tensor
        ):
        """
        Batch 내의 N개의 node들의 새로운 mem_vec와 mem_t 업데이트 
        Input:
            node: [N,]
            mem_vec: [N,mem_dim]
            event_t: [N,]
        """
        self.mem_vec[node]=mem_vec
        self.mem_t[node]=mem_t

class MemoryUpdater(nn.Module):
    def __init__(self,
            mem_dim:int,
            edge_dim:int,
            time_dim:int,
            msg_dim:int,
            time_encoder:TimeEncoder,
            msg_fn:Literal["concat","mlp"]="concat",
            aggr_fn:Literal["last","mean"]="last"
        ):
        super().__init__()
        self.mem_dim=mem_dim
        self.edge_dim=edge_dim
        self.time_dim=time_dim
        self.msg_dim=msg_dim
        self.msg_fn=msg_fn
        self.aggr_fn=aggr_fn

        # module
        self.time_encoder=time_encoder
        if msg_fn=="mlp":
            self.src_mlp=nn.Sequential(
                nn.Linear(
                    in_features=mem_dim+mem_dim+time_dim+edge_dim,
                    out_features=msg_dim
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features=msg_dim,
                    out_features=msg_dim
                )
            )
            self.dst_mlp=nn.Sequential(
                nn.Linear(
                    in_features=mem_dim+mem_dim+time_dim+edge_dim,
                    out_features=msg_dim
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features=msg_dim,
                    out_features=msg_dim
                )
            )

    def set_graph(self,graph:GNN_Graph):
        self.graph=graph
        self.n_node=graph.n_node
    
    def create_message(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            edge:torch.Tensor,
            event_t:torch.Tensor,
            mem_vec:torch.Tensor,
            mem_t:torch.Tensor
        ):
        """
        Input:
            src: [B,]
            dst: [B,]
            edge: [B,]
            event_t: [B,]
            mem_vec: [N,mem_dim]
            mem_t: [N,]
        Output:
            src_msg: [B,msg_dim]
            dst_msg: [B,msg_dim]
        """
        src_mem=mem_vec[src]
        src_ts=torch.abs(
            event_t-mem_t[src]
        ).unsqueeze(-1)
        src_ts_vec=self.time_encoder(src_ts)

        dst_mem=mem_vec[dst]
        dst_ts=torch.abs(
            event_t-mem_t[dst]
        ).unsqueeze(-1)
        dst_ts_vec=self.time_encoder(dst_ts)

        edge_ft=self.graph.get_edge_ft(edge=edge)

        src_msg=torch.concat(
            [
                src_mem,
                dst_mem,
                src_ts_vec,
                edge_ft
            ],
            dim=-1
        ) # [B,mem_dim+mem_dim+time_dim+edge_dim]

        dst_msg=torch.concat(
            [
                dst_mem,
                src_mem,
                dst_ts_vec,
                edge_ft
            ],
            dim=-1
        ) # [B,mem_dim+mem_dim+time_dim,edge_dim]

        if self.msg_fn=="mlp":
            src_msg=self.src_mlp(src_msg) # [B,msg_dim]
            dst_msg=self.dst_mlp(dst_msg) # [B,msg_dim]
        return src_msg,dst_msg

    def aggregate_message(self,
            src,
            dst,
            src_msg,
            dst_msg,
            event_t
        ):
        """
        Message Aggregation

        Input:
            src: [B,]
            dst: [B,]
            src_msg: [B,msg_dim]
            dst_msg: [B,msg_dim]
            event_t: [B,]
        Output:
            aggr_node: [unique_N,]
            aggr_msg: [unique_N,msg_dim]
            aggr_event_t: [unique_N,]
        """
        device=src.device

        nodes=torch.concat([src,dst],dim=0) # [2B,]
        msgs=torch.concat([src_msg,dst_msg],dim=0) # [2B,msg_dim]
        times=torch.concat([event_t,event_t],dim=0) # [2B,]

        # event time 순으로 오름차순 정렬
        sorted_idx=torch.argsort(times)
        nodes=nodes[sorted_idx]
        msgs=msgs[sorted_idx]
        times=times[sorted_idx]

        msg_dict={}
        for node,msg,t in zip(nodes,msgs,times):
            node=node.item()
            if node not in msg_dict:
                msg_dict[node]=[]
            msg_dict[node].append((msg,t))

        aggr_node=[]
        aggr_msg=[]
        aggr_event_t=[]
        match self.aggr_fn:
            case "last":
                for node in msg_dict.keys():
                    last_msg,last_t=msg_dict[node][-1]
                    aggr_node.append(node)
                    aggr_msg.append(last_msg)
                    aggr_event_t.append(last_t)
            case "mean":
                """
                공식 TGN 코드에서는 mean aggregation 시 message vector만 평균하고, memory의 last_update에 사용할 시간은 그 노드의 가장 최근 message의 timestamp를 사용
                """
                for node in msg_dict.keys():
                    msg_list=[msg for msg,_ in msg_dict[node]]
                    last_t=msg_dict[node][-1][1]
                    aggr_node.append(node)
                    aggr_msg.append(
                        torch.mean(
                            torch.stack(msg_list,dim=0),
                            dim=0,
                        )
                    )
                    # interaction time은 가장 최근 시각 사용
                    aggr_event_t.append(last_t)
        aggr_node=torch.tensor(aggr_node,device=device) # [unique_N,]
        aggr_msg=torch.stack(aggr_msg,dim=0) # [unique_N,msg_dim]
        aggr_event_t=torch.stack(aggr_event_t,dim=0) # [unique_N,]
        return aggr_node,aggr_msg,aggr_event_t
    
    def update_memory(self,
            src:torch.Tensor,
            dst:torch.Tensor,
            edge:torch.Tensor,
            event_t:torch.Tensor,
            mem_vec:torch.Tensor,
            mem_t:torch.Tensor
        ):
        """
        자식 class의 update_memory 실행으로 자식 class에서 구현된 update_memory_implement 호출
        
        Input:
            src: [B,]
            tar: [B,]
            edge: [B,]
            event_t: [B,]
            mem_vec: [N,mem_dim]
            mem_t: [N,]
        Return:
            node: [unique_N,]
            mem_vec: [unique_N,mem_dim]
            mem_t: [unique_N,]
        """
        src_msg,dst_msg=self.create_message(
            src=src,
            dst=dst,
            edge=edge,
            event_t=event_t,
            mem_vec=mem_vec,
            mem_t=mem_t
        )
        aggr_node,aggr_msg,aggr_event_t=self.aggregate_message(
            src=src,
            dst=dst,
            src_msg=src_msg,
            dst_msg=dst_msg,
            event_t=event_t
        )
        updated_result=self.update_memory_implement(
            aggr_node=aggr_node,
            aggr_msg=aggr_msg,
            aggr_event_t=aggr_event_t,
            mem_vec=mem_vec
        )
        return updated_result

    def update_memory_implement(self,
            aggr_node,
            aggr_msg,
            aggr_event_t,
            mem_vec
        ):
        return NotImplemented

class GRUMemoryUpdater(MemoryUpdater):
    def __init__(self,
            mem_dim:int,
            edge_dim:int,
            time_dim:int,
            msg_dim:int,
            time_encoder:TimeEncoder,
            msg_fn:Literal["concat","mlp"]="concat",
            aggr_fn:Literal["last","mean"]="last"
        ):
        super(GRUMemoryUpdater,self).__init__(
            mem_dim=mem_dim,
            edge_dim=edge_dim,
            time_dim=time_dim,
            msg_dim=msg_dim,
            time_encoder=time_encoder,
            msg_fn=msg_fn,
            aggr_fn=aggr_fn
        )
        if self.msg_fn=="concat":
            input_size=mem_dim+mem_dim+time_dim+edge_dim
        else:
            input_size=msg_dim
        self.memory_updater=nn.GRUCell(
            input_size=input_size,
            hidden_size=mem_dim
        )

    def update_memory_implement(self,
            aggr_node:torch.Tensor,
            aggr_msg:torch.Tensor,
            aggr_event_t:torch.Tensor,
            mem_vec:torch.Tensor
        ):
        """
        Input:
            aggr_node: [unique_N,]
            aggr_msg: [unique_N,msg_dim]
            aggr_event_t: [unique_N,]
            mem_vec: [N,mem_dim]
        Return:
            node: [unique_N,]
            mem_vec: [unique_N,mem_dim]
            mem_t: [unique_N,]
        """
        pre_mem_vec=mem_vec[aggr_node] # [unique_N,mem_dim]
        updated_mem_vec=self.memory_updater(aggr_msg,pre_mem_vec) # [unique_N,mem_dim]
        return {
            "node":aggr_node, # [unique_N]
            "mem_vec":updated_mem_vec, # [unique_N,mem_dim]
            "mem_t":aggr_event_t # [unique_N,]
        }