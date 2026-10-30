import os
import pandas as pd
import numpy as np
import torch
from typing import Literal

BASE_PATH=os.path.join("..","data")
class DataUtils:
    @staticmethod
    def preprocess_SNAP_dataset(
            dataset_name:Literal[
                "CollegeMsg"
                "soc-sign-bitcoinalpha",
                "soc-sign-bitcoinotc"
            ]
        ):
        """
        Stanford Network Analysis Project (SNAP) link: https://snap.stanford.edu/data

        중복 edge event가 없는 Undirected Temporal Graph로 전처리.

        1. Load graph_df (node_ft, edge_ft if exists.)
        2. Preprocess timestamp to unix timestamp second 
        3. Remove self-loop and 중복 edge event, timestamp 기준으로 오름차순 정렬
        4. timestamp range를 0 ~ (max_t-min_t)로 상대적 조정
        5. Remap edge_id
        6. Remap node_id
        """
        ### 1. Load graph_df (node_ft, edge_ft if exists.)
        match dataset_name:
            case "CollegeMsg":
                dataset_path=os.path.join(BASE_PATH,f"temporal_graph","SNAP",dataset_name,f"{dataset_name}.txt")
                graph_df=pd.read_csv(
                    dataset_path,
                    header=None,
                    sep=r"\s+",
                    usecols=[0,1,2],
                    names=["u","i","t"],
                )
                node_ft=None
                edge_ft=None
            case "soc-sign-bitcoinotc"|"soc-sign-bitcoinalpha":
                dataset_path=os.path.join(BASE_PATH,f"temporal_graph","SNAP",dataset_name,f"{dataset_name}.csv")
                graph_df=pd.read_csv(
                    dataset_path,
                    header=None,
                    usecols=[0,1,2,3],
                    names=["u","i","edge_w","t"],
                )
                node_ft=None
                edge_ft=graph_df.pop("edge_w").to_numpy(dtype=np.float32).reshape(-1,1) # [E,1]
        graph_df["idx"]=np.arange(len(graph_df)) # 원본 edge feature의 행 위치를 보존 (패딩이 없으므로 0부터 시작)

        ### 2. Preprocess timestamp to unix timestamp second
        graph_df["t"]=(
            pd.to_numeric(graph_df["t"],errors="raise")
            .astype(np.int64)
        )

        ### 3. Remove self-loop and 방향을 무시한 중복 edge event, timestamp 기준으로 오름차순 정렬
        graph_df=graph_df[graph_df["u"]!=graph_df["i"]]
        # (u,i,t)와 (i,u,t)를 같은 이벤트로 취급하고 첫 번째 원본 행만 보존
        edge_keys=pd.DataFrame({
            "min_node":graph_df[["u","i"]].min(axis=1),
            "max_node":graph_df[["u","i"]].max(axis=1),
            "t":graph_df["t"],
        })
        graph_df=(
            graph_df.loc[~edge_keys.duplicated(keep="first")]
            .sort_values(by="t",kind="stable")
            .reset_index(drop=True)
        )

        ### 4. timestamp range를 0 ~ (max_t-min_t)로 상대적 조정
        min_event_time=graph_df["t"].min()
        graph_df["t"]=(graph_df["t"]-min_event_time).astype(np.int64)

        ### 5. Remap edge_id
        if edge_ft is not None:
            edge_ft=np.vstack([
                np.zeros((1,edge_ft.shape[1]),dtype=edge_ft.dtype),
                edge_ft[graph_df["idx"].to_numpy()]
            ])
        graph_df["idx"]=np.arange(1,len(graph_df)+1)

        ### 6. Remap node_id
        used_nodes=np.sort(np.unique(graph_df[["u","i"]].to_numpy()))
        node_map={
            old_id:new_id
            for new_id,old_id in enumerate(used_nodes,start=1)
        }
        graph_df[["u","i"]]=(
            graph_df[["u","i"]].replace(node_map).astype(int)
        )
        if node_ft is not None:
            node_ft=np.vstack([
                np.zeros((1,node_ft.shape[1]),dtype=node_ft.dtype),
                node_ft[used_nodes]
            ])

        return {
            "graph_df":graph_df,
            "node_ft":node_ft,
            "edge_ft":edge_ft
        }

    @staticmethod
    def save_TR_result_to_pt(
            TR_result:dict[str,torch.Tensor],
            dataset_name:Literal[
                "CollegeMsg",
                "soc-sign-bitcoinalpha",
                "soc-sign-bitcoinotc"
            ],
            purpose:Literal["train","val","test"],
            batch_size:int
        ):
        """
        SNAP dataset들의 TR_result를 저장

        file_name example: CollegeMsg_train_B200.pt
        """
        file_name=f"{dataset_name}_{purpose}_B{batch_size}.pt"
        file_path=os.path.join(BASE_PATH,"temporal_graph","SNAP",dataset_name,"TR_result",file_name)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        torch.save(TR_result,file_path)
        print(f"Save {file_name}!")

    @staticmethod
    def load_TR_result(
            dataset_name:Literal[
                "CollegeMsg",
                "soc-sign-bitcoinalpha",
                "soc-sign-bitcoinotc"
            ],
            purpose:Literal["train","val","test"],
            batch_size:int
        )->dict[str,torch.Tensor]:
        file_name=f"{dataset_name}_{purpose}_B{batch_size}.pt"
        file_path=os.path.join(BASE_PATH,"temporal_graph","SNAP",dataset_name,"TR_result",file_name)
        TR_result=torch.load(file_path)
        return TR_result

    @staticmethod
    def save_sample_loader(
            sample_loader:list[dict[str,torch.Tensor]],
            dataset_name:Literal[
                "CollegeMsg",
                "soc-sign-bitcoinalpha",
                "soc-sign-bitcoinotc"
            ],
            purpose:Literal["val","test"],
            batch_size:int,
            seed:int
        ):
        file_name=f"{dataset_name}_{purpose}_B{batch_size}_S{seed}.pt"
        file_path=os.path.join(BASE_PATH,"TR_Embedding","sample_loader",dataset_name,file_name)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        torch.save(sample_loader,file_path)
        print(f"Save {file_name}!")

    @staticmethod
    def load_sample_loader(
            dataset_name:Literal[
                "CollegeMsg",
                "soc-sign-bitcoinalpha",
                "soc-sign-bitcoinotc"
            ],
            purpose:Literal["val","test"],
            batch_size:int,
            seed:int
        ):
        file_name=f"{dataset_name}_{purpose}_B{batch_size}_S{seed}.pt"
        file_path=os.path.join(BASE_PATH,"TR_Embedding","sample_loader",dataset_name,file_name)
        sample_loader=torch.load(file_path)
        return sample_loader