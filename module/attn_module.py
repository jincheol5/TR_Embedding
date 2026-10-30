import math
import torch
import torch.nn as nn

class TemporalGraphAttn(nn.Module):
    def __init__(self,
            input_dim:int,
            edge_dim:int,
            time_dim:int,
            embed_dim:int,
            n_head:int=1,
            dropout:float=0.1
        ):
        super().__init__()
        self.input_dim=input_dim
        self.edge_dim=edge_dim
        self.time_dim=time_dim
        self.embed_dim=embed_dim
        self.n_head=n_head
        self.raw_dim=input_dim+edge_dim+time_dim
        self.attn_dim=((self.raw_dim+n_head-1)//n_head)*n_head
        self.head_dim=self.attn_dim//n_head
        self.q_dim=self.raw_dim
        self.kv_dim=self.raw_dim

        # Project Q/K/V to a dimension divisible by the number of heads.
        self.query_proj=nn.Linear(self.raw_dim,self.attn_dim,bias=False)
        self.key_proj=nn.Linear(self.raw_dim,self.attn_dim,bias=False)
        self.value_proj=nn.Linear(self.raw_dim,self.attn_dim,bias=False)
        for projection in (self.query_proj,self.key_proj,self.value_proj):
            nn.init.normal_(projection.weight,mean=0,
                            std=math.sqrt(2.0/(self.raw_dim+self.head_dim)))
        self.output_proj=nn.Linear(self.attn_dim,self.raw_dim)
        nn.init.xavier_normal_(self.output_proj.weight)
        self.attn_dropout=nn.Dropout(dropout)
        self.output_dropout=nn.Dropout(dropout)
        self.layer_norm=nn.LayerNorm(self.raw_dim)
        self.MLPs=nn.Sequential(
            nn.Linear(self.raw_dim+input_dim,input_dim),
            nn.ReLU(),
            nn.Linear(input_dim,embed_dim)
        )
        nn.init.xavier_normal_(self.MLPs[0].weight)
        nn.init.xavier_normal_(self.MLPs[2].weight)

    def forward(self,
            tar_vec:torch.Tensor,
            tar_ts_vec:torch.Tensor,
            neighbor_vec:torch.Tensor,
            neighbor_ts_vec:torch.Tensor,
            neighbor_edge_ft:torch.Tensor,
            neighbor_mask:torch.Tensor
        ):
        batch_size,n_neighbor=neighbor_mask.shape
        # Root has no incoming edge. Preserve this unprojected query as residual.
        query=torch.cat((tar_vec,tar_vec.new_zeros(batch_size,self.edge_dim),tar_ts_vec),dim=-1).unsqueeze(1)
        key=torch.cat((neighbor_vec,neighbor_edge_ft,neighbor_ts_vec),dim=-1)
        residual=query
        q=self.query_proj(query).reshape(batch_size,1,self.n_head,self.head_dim).transpose(1,2)
        k=self.key_proj(key).reshape(batch_size,n_neighbor,self.n_head,self.head_dim).transpose(1,2)
        v=self.value_proj(key).reshape(batch_size,n_neighbor,self.n_head,self.head_dim).transpose(1,2)

        scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.head_dim)
        scores=scores.masked_fill(~neighbor_mask[:,None,None,:],-1e10)
        weights=self.attn_dropout(torch.softmax(scores,dim=-1))
        output=torch.matmul(weights,v).transpose(1,2).reshape(batch_size,1,self.attn_dim)
        output=self.output_dropout(self.output_proj(output))
        # With no history, contribute no neighbor message, but retain the query.
        output=output.masked_fill(~neighbor_mask.any(dim=-1)[:,None,None],0)
        output=self.layer_norm(output+residual).squeeze(1)
        return self.MLPs(torch.cat((output,tar_vec),dim=-1)) # [B,embed_dim]