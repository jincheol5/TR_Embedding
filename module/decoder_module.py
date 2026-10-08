import torch
import torch.nn as nn

class TR_Decoder(nn.Module):
    def __init__(self,
            embed_dim:int=32,
            latent_dim:int=32
        ):
        """
        MLP로 src,dst embedding 사이의 상호작용 생성하여 src->dst, dst->src 결과 다르도록 설정
        """
        super().__init__()
        self.decoder=nn.Sequential(
            nn.Linear(
                in_features=embed_dim+embed_dim,
                out_features=latent_dim
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=latent_dim,
                out_features=1
            )
        )
    def forward(self,
            src_vec:torch.Tensor,
            dst_vec:torch.Tensor
        ):
        """
        Input:
            src_vec: [B,embed_dim]
            dst_vec: [B,embed_dim]
        Return:
            pred_logit: [B,1]
        """
        pair_vec=torch.concat([src_vec,dst_vec],dim=-1) # [B,embed_dim+embed_dim]
        pred_logit=self.decoder(pair_vec) # [B,1]
        return pred_logit