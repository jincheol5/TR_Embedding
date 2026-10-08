import os
import pandas as pd
import numpy as np
import torch
from typing import Literal

BASE_PATH=os.path.join("..","data")
class DataUtils:
    @staticmethod
    def preprocess_snap_dataset(
            dataset_name:Literal[
                "CollegeMsg"
                "bitcoin-otc",
                "bitcoin-alpha"
            ]
        ):
        """
        timestamp: UTC unix timestamp day 단위로 변환
        """
        match dataset_name:
            case "CollegeMsg":
                dataset_path=os.path.join(BASE_PATH,f"temporal_graph",dataset_name,f"{dataset_name}.txt")
                graph_df=pd.read_csv(
                    dataset_path,
                    header=None,
                    sep=r"\s+",
                    usecols=[0,1,2],
                    names=["u","i","t"],
                )
            case "bitcoin-otc"|"bitcoin-alpha":
                dataset_path=os.path.join(BASE_PATH,f"temporal_graph",dataset_name,f"{dataset_name}.csv")
                graph_df=pd.read_csv(
                    dataset_path,
                    header=None,
                    usecols=[0,1,3],
                    names=["u","i","t"],
                )

        # 숫자로 변환할 수 없는 값은 결측값으로 처리한 뒤 제거
        graph_df=graph_df[["u","i","t"]].apply(
            pd.to_numeric,
            errors="coerce"
        )
        graph_df=(
            graph_df.dropna(subset=["u","i","t"])
            .astype(np.int64)
            .reset_index(drop=True)
        )

        # self-loop 제거
        self_loop_mask=graph_df["u"]==graph_df["i"]
        graph_df=graph_df.loc[~self_loop_mask].copy()

        # Unix timestamp(초)를 day 단위로 변환하고 최초 시각을 0으로 조정
        graph_df["t"]=(graph_df["t"]//(24*60*60)).astype(np.int64)
        min_event_time=graph_df["t"].min()
        graph_df["t"]=(graph_df["t"]-min_event_time).astype(np.int64)

        # 같은 방향의 동일 edge event는 하나만 유지
        graph_df=graph_df.drop_duplicates(
            subset=["u","i","t"],
            keep="first"
        )

        # event time 순으로 정렬
        graph_df=(
            graph_df.sort_values("t",kind="stable")
            .reset_index(drop=True)
        )

        # 모든 node ID 수집
        node_ids=sorted(
            set(graph_df["u"]).union(graph_df["i"])
        )

        # 기존 node ID -> 1 ~ N 매핑
        node_mapping={
            original_id:mapped_id
            for mapped_id,original_id in enumerate(
                node_ids,
                start=1,
            )
        }

        # node ID 재매핑
        graph_df["u"]=graph_df["u"].map(node_mapping).astype(np.int64)
        graph_df["i"]=graph_df["i"].map(node_mapping).astype(np.int64)

        # 중복 제거된 시간순 event에 맞춰 edge ID를 1 ~ E로 재매핑
        graph_df["idx"]=np.arange(1,len(graph_df)+1,dtype=np.int64)

        return {
            "graph_df":graph_df,
            "n_node":len(node_ids),
            "max_u":graph_df["u"].max(),
            "node_dim":None,
            "edge_dim":None,
            "node_ft":None,
            "edge_ft":None
        }

    @staticmethod
    def save_TR_result_to_pt(
            TR_result:dict[str,torch.Tensor],
            dataset_name:Literal[
                "CollegeMsg",
                "bitcoin-alpha",
                "bitcoin-otc"
            ],
            purpose:Literal["train","val","test"],
            batch_size:int
        ):
        """
        SNAP dataset들의 TR_result를 저장

        file_name example: CollegeMsg_train_B100.pt
        """
        file_name=f"{dataset_name}_{purpose}_B{batch_size}.pt"
        file_path=os.path.join(BASE_PATH,"temporal_graph","TR_result",dataset_name,purpose,file_name)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        torch.save(TR_result,file_path)
        print(f"Save {file_name}!")

    @staticmethod
    def load_TR_result(
            dataset_name:Literal[
                "CollegeMsg",
                "bitcoin-alpha",
                "bitcoin-otc"
            ],
            purpose:Literal["train","val","test"],
            batch_size:int
        )->dict[str,torch.Tensor]:
        file_name=f"{dataset_name}_{purpose}_B{batch_size}.pt"
        file_path=os.path.join(BASE_PATH,"temporal_graph","TR_result",dataset_name,purpose,file_name)
        TR_result=torch.load(file_path)
        return TR_result