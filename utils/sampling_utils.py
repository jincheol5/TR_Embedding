import torch

class SamplingUtils:
    @staticmethod
    def random_TR_sampling(
            n_sample:int,
            n_pair:int,
            query_time:float,
            TR_label:torch.Tensor
        )->dict[str,torch.Tensor]:
        """
        source를 랜덤 순회하며 총 n_sample개의 positive pair + negative pair (1:1비율)를 생성한다.
        source마다 양성/음성을 각각 최대 n_pair개, 작은 쪽에 맞춰 동일하게 선택한다.
        한쪽 후보가 없으면 건너뛰고 padding, 자기 자신, 중복 dst는 제외한다.
        후보가 소진되면 가능한 개수만 반환한다. 홀수 n_sample은 짝수로 내림한다.

        Input:
            n_sample
            n_pair
            query_time
            TR_label: [N+1,N+1] bool tensor
        Return:
            tensor dict:
                src
                dst
                label
                query_t
                pos_mask
        """
        pairs=[]
        remaining=n_sample
        for source in (torch.randperm(TR_label.size(0)-1)+1).tolist():
            if remaining<2:
                break
            pos=torch.where(TR_label[source])[0]
            neg=torch.where(~TR_label[source])[0]
            pos=pos[(pos!=0) & (pos!=source)]
            neg=neg[(neg!=0) & (neg!=source)]
            count=min(n_pair,len(pos),len(neg),remaining//2)
            if count<=0:
                continue

            pos=pos[torch.randperm(len(pos))[:count]]
            neg=neg[torch.randperm(len(neg))[:count]]
            dst=torch.cat((pos,neg))
            pairs.append(torch.stack((torch.full_like(dst,source),dst),dim=1))
            remaining-=2*count

        pairs=torch.cat(pairs) if pairs else torch.empty((0,2),dtype=torch.long)
        src,dst=pairs.unbind(dim=1)
        label=TR_label[src,dst].float()
        return {
            "src":src,
            "dst":dst,
            "label":label,
            "query_t":torch.full_like(label,float(query_time)),
            "pos_mask":label.bool()
        }

    @staticmethod
    def dst_focused_random_TR_sampling(
            n_sample:int,
            n_pair:int,
            dst:torch.Tensor,
            query_time:float,
            TR_label:torch.Tensor
        )->dict[str,torch.Tensor]:
        """
        Source를 랜덤 순회하며 Positive/Negative Pair를 1:1 비율로 샘플링.
        Source마다 각 클래스에서 최대 n_pair개씩 선택.
        입력 dst를 우선 샘플링하고 부족하면 나머지 후보에서 보충.
        각 Source마다 Padding Node(id=0), Source 자기 자신, 중복 dst 제외.
        후보가 소진되면 가능한 개수만 반환.
        홀수 n_sample은 짝수로 내림.

        Input:
            n_sample
            n_pair
            dst: [B,] 우선 샘플링할 node ids
            query_time
            TR_label: [N+1,N+1] bool tensor
        Return:
            tensor dict:
                src
                dst
                label
                query_t
                pos_mask
        """
        priority_dst=torch.unique(dst.long())
        pairs=[]
        remaining=n_sample
        for source in (torch.randperm(TR_label.size(0)-1)+1).tolist():
            if remaining<2:
                break

            ### Positive/Negative 후보
            pos=torch.where(TR_label[source])[0]
            neg=torch.where(~TR_label[source])[0]
            pos=pos[(pos!=0)&(pos!=source)]
            neg=neg[(neg!=0)&(neg!=source)]

            count=min(n_pair,len(pos),len(neg),remaining//2)
            if count<=0:
                continue

            ### 입력 dst 우선 샘플링
            sampled=[]
            for candidates in (pos,neg):
                mask=torch.isin(candidates,priority_dst)
                priority=candidates[mask]
                others=candidates[~mask]
                selected=priority[torch.randperm(len(priority))[:count]]
                extra=others[torch.randperm(len(others))[:count-len(selected)]]
                sampled.append(torch.cat([selected,extra]))

            ### Source-Destination Pair 생성
            sampled_dst=torch.cat(sampled)
            pairs.append(torch.stack((
                torch.full_like(sampled_dst,source),
                sampled_dst
            ),dim=1))
            remaining-=2*count

        pairs=torch.cat(pairs) if pairs else torch.empty((0,2),dtype=torch.long)
        src,dst=pairs.unbind(dim=1)
        label=TR_label[src,dst].float()
        return {
            "src":src,
            "dst":dst,
            "label":label,
            "query_t":torch.full_like(label,float(query_time)),
            "pos_mask":label.bool()
        }

    @staticmethod
    def hop_range_random_TR_sampling(
            n_sample:int,
            query_time:float,
            TR_label:torch.Tensor,
            TR_hop:torch.Tensor
        )->dict[str,torch.Tensor]:
        """
        평가용 TR Sampling.
        Source를 랜덤 순회하며 Hop Range별 Positive Pair를 샘플링.
        각 Hop Range에서 최대 n_sample//3개씩 선택.

        Hop Range:
            range 1: 1<=hop<3
            range 2: 3<=hop<5
            range 3: 5<=hop

        Padding Node(id=0), Source 자기 자신, 동일 Source 내 중복 dst 제외.
        후보가 부족하면 가능한 개수만 반환.
        """
        ### Hop Range별 목표 Sample 개수
        n_per_range=n_sample//3
        sampled=[[],[],[]]
        counts=[0,0,0]

        ### Source 랜덤 순회
        for source in (torch.randperm(TR_label.size(0)-1)+1).tolist():
            if all(count>=n_per_range for count in counts):
                break

            ### Positive 후보
            pos=torch.where(TR_label[source])[0]
            pos=pos[(pos!=0)&(pos!=source)]
            pos_hop=TR_hop[source,pos]

            ### Hop Range별 Sampling
            masks=[
                (1<=pos_hop)&(pos_hop<3),
                (3<=pos_hop)&(pos_hop<5),
                (5<=pos_hop)
            ]

            for i,mask in enumerate(masks):
                remaining=n_per_range-counts[i]
                if remaining<=0:
                    continue
                candidates=pos[mask]
                selected=candidates[torch.randperm(len(candidates))[:remaining]]
                if len(selected)>0:
                    sampled[i].append(torch.stack((torch.full_like(selected,source),selected),dim=1))
                    counts[i]+=len(selected)

        ### Hop Range 순서로 결합
        pairs=[
            torch.cat(group) if group else torch.empty((0,2),dtype=torch.long)
            for group in sampled
        ]
        pairs=torch.cat(pairs)
        src,dst=pairs.unbind(dim=1)

        ### Hop Range Mask
        n_sample=len(src)
        range_1_mask=torch.zeros(n_sample,dtype=torch.bool)
        range_2_mask=torch.zeros(n_sample,dtype=torch.bool)
        range_3_mask=torch.zeros(n_sample,dtype=torch.bool)

        range_1_mask[:counts[0]]=True
        range_2_mask[counts[0]:counts[0]+counts[1]]=True
        range_3_mask[counts[0]+counts[1]:]=True

        return {
            "src":src,
            "dst":dst,
            "label":torch.ones(n_sample,dtype=torch.float32),
            "query_t":torch.full((n_sample,),query_time,dtype=torch.float32),
            "range_1_mask":range_1_mask,
            "range_2_mask":range_2_mask,
            "range_3_mask":range_3_mask
        }