import torch
import torch.nn.functional as F

class ReaCH_TGN_Utils:
    @staticmethod
    def augment_event_t(
            event_t:torch.Tensor,
            gaussian_noise_std:float=1.0,
            uniform_jitter:float=0.1
        )->torch.Tensor:
        """
        Temporal Augmentation
        각 temporal edge의 timestamp에 Gaussian noise와 작은 uniform jitter를 추가한다.

        Input:
            gaussian_noise_std: Gaussian noise의 표준편차
            uniform_jitter: Uniform jitter 범위 (day), 예: 0.1 = 약 2.4시간
        Return:
            augmented_event_t: [B,]
        """
        ### Gaussian noise
        gaussian_noise=(torch.randn_like(event_t)*gaussian_noise_std)

        ### Small uniform jitter
        uniform_noise=(torch.rand_like(event_t)*2.0-1.0)*uniform_jitter

        ### Timestamp jitter
        augmented_event_t=event_t+gaussian_noise+uniform_noise
        return augmented_event_t

    @staticmethod
    def compute_NT_Xent_Loss(
            Z_src_A:torch.Tensor,
            Z_dst_A:torch.Tensor,
            Z_src_B:torch.Tensor,
            Z_dst_B:torch.Tensor,
            temperature:float=0.5
        )->torch.Tensor:
        """
        Compute NT-Xent Loss

        동일 sample에서 생성된 View A와 View B의 embedding을 positive pair로 사용한다.

        Positive pairs:
            Z_src_A[i] <-> Z_src_B[i]
            Z_dst_A[i] <-> Z_dst_B[i]

        Input:
            Z_src_A: [B, D]
            Z_dst_A: [B, D]
            Z_src_B: [B, D]
            Z_dst_B: [B, D]
            temperature: temperature parameter

        Return:
            loss: scalar
        """

        ### 각 view의 embedding 결합
        Z_A=torch.cat([Z_src_A,Z_dst_A],dim=0) # [2B,D]
        Z_B=torch.cat([Z_src_B,Z_dst_B],dim=0) # [2B,D]

        ### Cosine similarity를 위한 normalization
        Z_A=F.normalize(Z_A,dim=1)
        Z_B=F.normalize(Z_B,dim=1)

        ### 두 augmented view 결합
        Z=torch.cat([Z_A,Z_B],dim=0) # [4B,D]
        N=Z_A.size(0) # 2B

        ### Pairwise cosine similarity / temperature
        logits=Z@Z.T/temperature # [4B,4B]

        ### 자기 자신 제외
        self_mask=torch.eye(2*N,dtype=torch.bool,device=Z.device)
        logits=logits.masked_fill(self_mask,float("-inf"))

        ### log denominator
        log_prob=(logits-torch.logsumexp(logits,dim=1,keepdim=True))

        ### 각 anchor에 대응하는 positive index
        # A[i] -> B[i]
        # B[i] -> A[i]
        positive_idx=torch.cat([
            torch.arange(N,2*N,device=Z.device),
            torch.arange(0,N,device=Z.device)
        ])

        ### NT-Xent loss
        loss=-log_prob[
            torch.arange(2*N,device=Z.device),
            positive_idx
        ].mean()
        return loss

    @staticmethod
    def compute_hop_based_penalty(
            pair_hop:torch.Tensor,
            max_hop:int,
            penalty_exponent:float=1.0
        )->torch.Tensor:
        """
        Hop-Based Penalty
        Positive node pair loss에 대해서만 적용
        Hop이 짧을수록 더 큰 weight를 부여
        
        w_hop = (max_hop-pair_hop+1)^penalty_exponent

        Input:
            pair_hop: [N_pos,] long tensor
            max_hop: 최대 hop, n_node-1
            penalty_exponent: hop weight 감소 정도
        Return:
            weight: [N_pos,] float tensor
        """
        weight=(max_hop-pair_hop+1).float().pow(penalty_exponent)
        return weight 

    @staticmethod
    def compute_time_gap_penalty(
            pair_first_t:torch.Tensor,
            query_time:float,
            decay_coefficient:float=0.01
        )->torch.Tensor:
        """
        Time-Gap Penalty
        Positive node pair loss에 대해서만 적용
        Query time과 temporal path의 최초 event 시간 차이가 클수록 작은 weight를 부여

        w_time = exp(-decay_coefficient * Δt)
        Δt = query_time - first_t

        Input:
            pair_first_t: [N_pos,] float tensor
            query_time: query 시점
            decay_lambda: 시간 차이(time gap)가 커질 때 weight를 얼마나 빠르게 감소시킬지를 결정하는 지수 감쇠 계수

        Return:
            weight: [N_pos,] float tensor
        """
        time_gap=query_time-pair_first_t
        weight=torch.exp(-decay_coefficient*time_gap)
        return weight