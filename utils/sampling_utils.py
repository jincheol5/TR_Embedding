import torch

"""
undirected graph인 점을 고려하여 수정.


<< Random TR Sampling >>
- n_pos_sample=n_sample//2
- n_neg_sample=n_sample//2
- 고유한 event_dst들을 랜덤으로 정렬
- for문으로 event_dst 하나씩 돌며 
    Positive Sampling
    - 최대 n_pair개 newly reachable한 positive src를 candidates(해당 dst 제외)에서 샘플링
    - n_pair개를 못 채운 경우, 부족한 만큼 already reachable한 positive src를 candidates(해당 dst 제외)에서 샘플링
    - 그래도 못채우는 경우는 다음 event_dst로 넘어감 
    - n_pos_sample 다 채우면 for문 중간에 Positive Sampling 종료

    Negative Sampling
    - 최대 n_pair개 unreachable한 negative src를 candidates(해당 dst 제외)에서 샘플링
    - n_pair개를 못 채운 경우 다음 event_dst로 넘어감
    - n_neg_sample 다 채우면 for문 중간에 Negative Sampling 종료

- for문이 끝난 후에도 n_sample개를 못 채운 경우 candidates(event_dst들 제외)를 무작위 정렬
- for문으로 하나씩 돌며
    Positive Sampling
    - 최대 n_pair개 reachable한 positive src를 candidates(해당 dst 제외)에서 샘플링
    - n_pos_sample 다 채우면 for문 중간에 Positive Sampling 종료

    Negative Sampling
    - 최대 n_pair개 unreachable한 positive src를 candidates(해당 dst 제외)에서 샘플링
    - n_neg_sample 다 채우면 for문 중간에 Negative Sampling 종료
"""

class Sampler:
    @staticmethod
    def random_TR_sampling(
            n_sample:int,
            n_pair:int,
            query_time:int,
            event_dst:torch.Tensor,
            candidates:list[int],
            TR_label:torch.Tensor,
            pre_TR_label:torch.Tensor|None=None,
        ):
        """
        Input:
            n_sample
            n_pair
            query_time
            event_dst: [B,] tensor
            candidates: list of candidate node ids
            TR_label: [N+1,N+1] boolean tensor
            pre_TR_label: [N+1,N+1] boolean tensor
        Return: dict 
            src: [n_sample,]
            dst: [n_sample,]
            query_t: [n_sample,]
            label: [n_sample,]
            pos_mask: [n_sample,]
        """
        nodes=torch.as_tensor(candidates,dtype=torch.long).unique()
        nodes=nodes[(nodes>0)&(nodes<TR_label.size(0))]

        event_dst=torch.as_tensor(event_dst,dtype=torch.long).unique()
        event_dst=event_dst[torch.isin(event_dst,nodes)]
        event_dst=event_dst[torch.randperm(len(event_dst))]

        quota=n_sample//2
        pos,neg=[],[]

        # 1. Batch DST Sampling
        for d in event_dst.tolist():
            src=nodes[nodes!=d]

            # Positive Sampling
            if len(pos)<quota:
                positive_src=src[TR_label[src,d]]
                n_selected=0

                # Newly Reachable
                if pre_TR_label is not None:
                    newly_src=positive_src[~pre_TR_label[positive_src,d]]
                    n=min(n_pair,quota-len(pos),len(newly_src))
                    sampled=newly_src[torch.randperm(len(newly_src))[:n]]
                    pos.extend((s,d) for s in sampled.tolist())
                    n_selected=n
                    positive_src=positive_src[pre_TR_label[positive_src,d]]

                # Already Reachable
                n=min(n_pair-n_selected,quota-len(pos),len(positive_src))
                sampled=positive_src[torch.randperm(len(positive_src))[:n]]
                pos.extend((s,d) for s in sampled.tolist())

            # Negative Sampling
            if len(neg)<quota:
                negative_src=src[~TR_label[src,d]]
                n=min(n_pair,quota-len(neg),len(negative_src))
                sampled=negative_src[torch.randperm(len(negative_src))[:n]]
                neg.extend((s,d) for s in sampled.tolist())

            if len(pos)==quota and len(neg)==quota:
                break

        # 2. Remaining DST Sampling
        if len(pos)<quota or len(neg)<quota:
            remaining_dst=nodes[~torch.isin(nodes,event_dst)]
            remaining_dst=remaining_dst[torch.randperm(len(remaining_dst))]

            for d in remaining_dst.tolist():
                src=nodes[nodes!=d]

                # Positive Sampling
                if len(pos)<quota:
                    positive_src=src[TR_label[src,d]]
                    n=min(n_pair,quota-len(pos),len(positive_src))
                    sampled=positive_src[torch.randperm(len(positive_src))[:n]]
                    pos.extend((s,d) for s in sampled.tolist())

                # Negative Sampling
                if len(neg)<quota:
                    negative_src=src[~TR_label[src,d]]
                    n=min(n_pair,quota-len(neg),len(negative_src))
                    sampled=negative_src[torch.randperm(len(negative_src))[:n]]
                    neg.extend((s,d) for s in sampled.tolist())

                if len(pos)==quota and len(neg)==quota:
                    break

        # 3. Combine & Shuffle
        pairs=torch.tensor(pos+neg,dtype=torch.long).reshape(-1,2)
        labels=torch.cat([
            torch.ones(len(pos)),
            torch.zeros(len(neg)),
        ])
        perm=torch.randperm(len(pairs))
        return {
            "src": pairs[perm,0],
            "dst": pairs[perm,1],
            "query_t": torch.full((len(pairs),),query_time,dtype=torch.long),
            "label": labels[perm],
            "pos_mask": labels[perm].bool(),
        }

