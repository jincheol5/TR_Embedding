import os
import random
import pandas as pd
import numpy as np
import torch
from typing import Literal
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from .sampling_utils import SamplingUtils

class TemporalGraphDataset(Dataset):
    def __init__(self,df:pd.DataFrame):
        self.src=torch.tensor(df["u"].values,dtype=torch.long)
        self.dst=torch.tensor(df["i"].values,dtype=torch.long)
        self.t=torch.tensor(df["t"].values,dtype=torch.float32)
        self.edge=torch.tensor(df["idx"].values,dtype=torch.long)

    def __len__(self):
        return len(self.src)

    def __getitem__(self,idx):
        return self.src[idx],self.dst[idx],self.t[idx],self.edge[idx]

class EarlyStopper:
    def __init__(self,
            patience:int=1
        ):
        self.patience=patience
        self.patience_count=0
        self.best_loss=np.inf
        self.best_state=None
        self.early_stop=False
    def __call__(self,
            val_loss:float,
            model:torch.nn.Module
        ):
        # val_loss가 NaN, Inf이면 즉시 early stop
        if not np.isfinite(val_loss): 
            print("Loss is NaN or Inf!")
            self.early_stop=True
            if self.best_state is not None:
                model.load_state_dict(self.best_state)
            return model

        # 첫 번째 validation에서는 비교할 이전 best가 없으므로 현재 loss와 모델을 그대로 best로 저장
        if self.best_state is None: 
            self.best_loss=val_loss
            self.best_state={
                key: value.detach().clone()
                for key,value in model.state_dict().items()
            }
            return model

        # val_loss가 개선 되지 않은 경우
        if self.best_loss<=val_loss: 
            self.patience_count+=1
            if self.patience<=self.patience_count:
                self.early_stop=True
                model.load_state_dict(self.best_state)
            return model

        # val_loss가 개선 된 경우
        self.patience_count=0
        self.best_loss=val_loss
        self.best_state={
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        return model

class TrainUtils:
    @staticmethod
    def set_seed(seed:int):
        os.environ["PYTHONHASHSEED"]=str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False

    @staticmethod
    def split_graph_df(
            df:pd.DataFrame,
            train_ratio:float=0.7,
            val_ratio:float=0.15
        ):
        """
        Input:
            df
            train_ratio
            val_ratio
        Return:
            train_df
            val_df
            test_df
        """
        n=len(df)
        train_end=int(n*train_ratio)
        val_end=int(n*(train_ratio+val_ratio))
        train_df=df.iloc[:train_end].reset_index(drop=True)
        val_df=df.iloc[train_end:val_end].reset_index(drop=True)
        test_df=df.iloc[val_end:].reset_index(drop=True)
        return train_df,val_df,test_df

    @staticmethod
    def get_TR_sample_list(
            data_loader:DataLoader,
            n_sample:int=1000,
            n_pair:int=10,
            TR_result:dict[str,torch.Tensor]|None=None,
            sampling:Literal["random","focused","hop_range"]=f"random"
        )->list[dict[str,torch.Tensor]]:
        """
        Input:
            data_loader
            n_pair
            TR_result
            sampling
        Return:
            TR_sample_list
        """
        TR_label=TR_result["label"]
        TR_hop=TR_result["hop"]
        TR_sample_list=[]
        for batch_idx,(_,dst,event_t,_) in tqdm(
                enumerate(data_loader),
                desc="Generating TR samples..."
            ):
            query_time=event_t.max().item()
            match sampling:
                case "random":
                    TR_sample=SamplingUtils.random_TR_sampling(
                        n_sample=n_sample,
                        n_pair=n_pair,
                        query_time=query_time,
                        TR_label=TR_label[batch_idx]
                    )
                case "focused":
                    TR_sample=SamplingUtils.dst_focused_random_TR_sampling(
                        n_sample=n_sample,
                        n_pair=n_pair,
                        dst=dst,
                        query_time=query_time,
                        TR_label=TR_label[batch_idx]
                    )
                case "hop_range":
                    TR_sample=SamplingUtils.hop_range_random_TR_sampling(
                        n_sample=n_sample,
                        query_time=query_time,
                        TR_label=TR_label[batch_idx],
                        TR_hop=TR_hop[batch_idx]
                    )
            TR_sample_list.append(TR_sample)
        return TR_sample_list