import os
import random
import pandas as pd
import numpy as np
import torch
from typing import Literal
from tqdm import tqdm
from torch.utils.data import Dataset
from graph import GNN_Graph

class TemporalGraphDataset(Dataset):
    def __init__(self,graph_df:pd.DataFrame):
        self.src=torch.tensor(graph_df["u"].values,dtype=torch.long)
        self.dst=torch.tensor(graph_df["i"].values,dtype=torch.long)
        self.t=torch.tensor(graph_df["t"].values,dtype=torch.int64)
        self.edge=torch.tensor(graph_df["idx"].values,dtype=torch.long)

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
        os.environ["PYTHONHASHSEED"]=str(seed) # Python의 문자열 등에 사용하는 해시 시드를 설정
        random.seed(seed) # Python 표준 random 모듈의 난수 시드를 고정
        np.random.seed(seed) # NumPy의 기존 전역 난수 생성기 시드를 고정
        torch.manual_seed(seed) # PyTorch의 CPU 및 CUDA 등 장치의 난수 시드를 설정
        torch.backends.cudnn.deterministic=True # cuDNN이 사용하는 알고리즘을 결정적인 알고리즘으로 제한
        torch.backends.cudnn.benchmark=False # cuDNN이 여러 알고리즘의 속도를 측정해 실행 알고리즘을 선택하는 기능 끔

    @staticmethod
    def split_graph_df(
            graph_df:pd.DataFrame,
            train_ratio:float=0.7,
            val_ratio:float=0.15
        ):
        """
        Input:
            graph_df
            train_ratio
            val_ratio
        Return:
            train_df
            val_df
            test_df
        """
        n=len(graph_df)
        train_end=int(n*train_ratio)
        val_end=int(n*(train_ratio+val_ratio))
        train_df=graph_df.iloc[:train_end].reset_index(drop=True)
        val_df=graph_df.iloc[train_end:val_end].reset_index(drop=True)
        test_df=graph_df.iloc[val_end:].reset_index(drop=True)
        return train_df,val_df,test_df
