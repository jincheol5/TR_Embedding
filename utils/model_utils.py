import os
import torch
import torch.nn as nn
from typing import Literal

BASE_PATH=os.path.join("..","data","TR_Embedding","inference")
class ModelUtils:
    @staticmethod
    def save_model_parameter(
            model:nn.Module,
            model_name:Literal[
                "TGAT",
                "TGN",
                "CAW-N",
                "GraphMixer",
                "DyGFormer",
                "TPNet",
                "ReaCH-TGN"
            ],
            seed:int,
            lr:float,
            batch_size:int,
            dataset_name:Literal[
                "CollegeMsg",
                "soc-sign-bitcoinalpha",
                "soc-sign-bitcoinotc"
            ]
        ):
        file_name=f"{model_name}_S{seed}_LR{lr}_B{batch_size}_{dataset_name}.pt"
        file_path=os.path.join(BASE_PATH,dataset_name,file_name)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        torch.save(model.state_dict(),file_path)
        print(f"Save {file_name} model parameter!")

    @staticmethod
    def load_model_parameter(
            model:nn.Module,
            model_name:Literal[
                "TGAT",
                "TGN",
                "CAW-N",
                "GraphMixer",
                "DyGFormer",
                "TPNet",
                "ReaCH-TGN"
            ],
            seed:int,
            lr:float,
            batch_size:int,
            dataset_name:Literal[
                "CollegeMsg",
                "soc-sign-bitcoinalpha",
                "soc-sign-bitcoinotc"
            ]
        ):
        file_name=f"{model_name}_S{seed}_LR{lr}_B{batch_size}_{dataset_name}.pt"
        file_path=os.path.join(BASE_PATH,dataset_name,file_name)
        model.load_state_dict(torch.load(file_path))
        print(f"Load {file_name} model parameter!")
        return model