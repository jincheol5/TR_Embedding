import argparse
import numpy as np
from torch.utils.data import DataLoader
from utils import DataUtils,TrainUtils,TemporalGraphDataset,GraphAnalysis

def main(**kwargs):
    match kwargs["app_num"]:
        case 1:
            """
            """
            ### TR result 관련 파라미터
            batch_size=200
            max_hop=5

            ### load SR, TR result
            dataset_name=kwargs["dataset_name"]
            train_SR_result=DataUtils.load_SR_result(
                dataset_name=dataset_name,
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="train"
            )
            val_SR_result=DataUtils.load_SR_result(
                dataset_name=dataset_name,
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="val"
            )
            test_SR_result=DataUtils.load_SR_result(
                dataset_name=dataset_name,
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="test"
            )

            train_TR_result=DataUtils.load_TR_result(
                dataset_name=dataset_name,
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="train"
            )
            val_TR_result=DataUtils.load_TR_result(
                dataset_name=dataset_name,
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="val"
            )
            test_TR_result=DataUtils.load_TR_result(
                dataset_name=dataset_name,
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="test"
            )

            ### check train 
            SR_label=train_SR_result["SR_label"][-1]
            SR_hop=train_SR_result["SR_hop"][-1]
            TR_label=train_TR_result["TR_label"][-1]
            TR_hop=train_TR_result["TR_hop"][-1]
            TR_ratio=GraphAnalysis.check_reach_ratio(reach_label=TR_label)
            hard_pos_TR_ratio=GraphAnalysis.check_hard_positive_ratio(TR_label=TR_label,TR_hop=TR_hop)
            hard_neg_TR_ratio=GraphAnalysis.check_hard_negative_ratio(TR_label=TR_label,SR_label=SR_label,SR_hop=SR_hop)
            print(f"Check {dataset_name} Train")
            print(f"{dataset_name} TR_ratio in all pair: {TR_ratio}")
            print(f"{dataset_name} hard_pos_TR_ratio in all pos pair: {hard_pos_TR_ratio}")
            print(f"{dataset_name} hard_neg_TR_ratio in all neg pair: {hard_neg_TR_ratio}",end="\n\n")

            ### check val
            SR_label=val_SR_result["SR_label"][-1]
            SR_hop=val_SR_result["SR_hop"][-1]
            TR_label=val_TR_result["TR_label"][-1]
            TR_hop=val_TR_result["TR_hop"][-1]
            TR_ratio=GraphAnalysis.check_reach_ratio(reach_label=TR_label)
            hard_pos_TR_ratio=GraphAnalysis.check_hard_positive_ratio(TR_label=TR_label,TR_hop=TR_hop)
            hard_neg_TR_ratio=GraphAnalysis.check_hard_negative_ratio(TR_label=TR_label,SR_label=SR_label,SR_hop=SR_hop)
            print(f"Check {dataset_name} Val")
            print(f"{dataset_name} TR_ratio in all pair: {TR_ratio}")
            print(f"{dataset_name} hard_pos_TR_ratio in all pos pair: {hard_pos_TR_ratio}")
            print(f"{dataset_name} hard_neg_TR_ratio in all neg pair: {hard_neg_TR_ratio}",end="\n\n")

            ### check test
            SR_label=test_SR_result["SR_label"][-1]
            SR_hop=test_SR_result["SR_hop"][-1]
            TR_label=test_TR_result["TR_label"][-1]
            TR_hop=test_TR_result["TR_hop"][-1]
            TR_ratio=GraphAnalysis.check_reach_ratio(reach_label=TR_label)
            hard_pos_TR_ratio=GraphAnalysis.check_hard_positive_ratio(TR_label=TR_label,TR_hop=TR_hop)
            hard_neg_TR_ratio=GraphAnalysis.check_hard_negative_ratio(TR_label=TR_label,SR_label=SR_label,SR_hop=SR_hop)
            print(f"Check {dataset_name} Test")
            print(f"{dataset_name} TR_ratio in all pair: {TR_ratio}")
            print(f"{dataset_name} hard_pos_TR_ratio in all pos pair: {hard_pos_TR_ratio}")
            print(f"{dataset_name} hard_neg_TR_ratio in all neg pair: {hard_neg_TR_ratio}",end="\n\n")

        case 2:
            """
            """
            ### TR sample 관련 파라미터
            batch_size=200
            max_hop=5
            n_pair=10
            sampling=kwargs["sampling"]
            pos_hard_ratio=0.5
            neg_hard_ratio=0.5

            ### set data_loader
            data=DataUtils.preprocess_graph_dataset(
                dataset_name=kwargs["dataset_name"]
            )
            graph_df=data["graph_df"]
            train_df,val_df,test_df=TrainUtils.split_graph_df(df=graph_df)
            train_dataset=TemporalGraphDataset(df=train_df)
            val_dataset=TemporalGraphDataset(df=val_df)
            test_dataset=TemporalGraphDataset(df=test_df)
            train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
            val_loader=DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
            test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

            ### load SR, TR result
            train_SR_result=DataUtils.load_SR_result(
                dataset_name=kwargs["dataset_name"],
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="train"
            )
            val_SR_result=DataUtils.load_SR_result(
                dataset_name=kwargs["dataset_name"],
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="val"
            )
            test_SR_result=DataUtils.load_SR_result(
                dataset_name=kwargs["dataset_name"],
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="test"
            )
            train_TR_result=DataUtils.load_TR_result(
                dataset_name=kwargs["dataset_name"],
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="train"
            )
            val_TR_result=DataUtils.load_TR_result(
                dataset_name=kwargs["dataset_name"],
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="val"
            )
            test_TR_result=DataUtils.load_TR_result(
                dataset_name=kwargs["dataset_name"],
                max_hop=max_hop,
                batch_size=batch_size,
                purpose="test"
            )

            ### set train, val, test sample_list
            match sampling:
                case "random":
                    train_sample_list=TrainUtils.get_TR_sample_list(
                        n_pair=n_pair,
                        data_loader=train_loader,
                        TR_result=train_TR_result,
                        sampling=sampling
                    )
                    val_sample_list=TrainUtils.get_TR_sample_list(
                        n_pair=n_pair,
                        data_loader=val_loader,
                        TR_result=val_TR_result,
                        sampling=sampling
                    )
                    test_sample_list=TrainUtils.get_TR_sample_list(
                        n_pair=n_pair,
                        data_loader=test_loader,
                        TR_result=test_TR_result,
                        sampling=sampling
                    )
                case "hard":
                    train_sample_list=TrainUtils.get_TR_sample_list(
                        n_pair=n_pair,
                        data_loader=train_loader,
                        SR_result=train_SR_result,
                        TR_result=train_TR_result,
                        sampling=sampling,
                        pos_hard_ratio=pos_hard_ratio,
                        neg_hard_ratio=neg_hard_ratio
                    )
                    val_sample_list=TrainUtils.get_TR_sample_list(
                        n_pair=n_pair,
                        data_loader=val_loader,
                        SR_result=val_SR_result,
                        TR_result=val_TR_result,
                        sampling=sampling,
                        pos_hard_ratio=pos_hard_ratio,
                        neg_hard_ratio=neg_hard_ratio
                    )
                    test_sample_list=TrainUtils.get_TR_sample_list(
                        n_pair=n_pair,
                        data_loader=test_loader,
                        SR_result=test_SR_result,
                        TR_result=test_TR_result,
                        sampling=sampling,
                        pos_hard_ratio=pos_hard_ratio,
                        neg_hard_ratio=neg_hard_ratio
                    )

            ### check ratio of train
            pos_ratio_list=[]
            neg_ratio_list=[]
            for idx,sample in enumerate(train_sample_list):
                src=sample["src"]
                dst=sample["dst"]
                pos_mask=sample["pos_mask"]
                pos_src=src[pos_mask]
                pos_dst=dst[pos_mask]
                neg_src=src[~pos_mask]
                neg_dst=dst[~pos_mask]
                SR_label=train_SR_result["SR_label"][idx]
                SR_hop=train_SR_result["SR_hop"][idx]
                TR_hop=train_TR_result["TR_hop"][idx]
                pos_ratio=GraphAnalysis.check_hard_positive_ratio_in_sample(
                    pos_src=pos_src,
                    pos_dst=pos_dst,
                    TR_hop=TR_hop
                )
                pos_ratio_list.append(pos_ratio)
                neg_ratio=GraphAnalysis.check_hard_negative_ratio_in_sample(
                    neg_src=neg_src,
                    neg_dst=neg_dst,
                    SR_label=SR_label,
                    SR_hop=SR_hop
                )
                neg_ratio_list.append(neg_ratio)
            print(f"{sampling} sampling: positive train sample list 2-hop 이상 TR node pair 평균 비율: {np.mean(pos_ratio_list)}")
            print(f"{sampling} sampling: negative train sample list 2-hop 이상 SR and not TR node pair 평균 비율: {np.mean(neg_ratio_list)}",end="\n\n")

            ### check ratio of val
            pos_ratio_list=[]
            neg_ratio_list=[]
            for idx,sample in enumerate(val_sample_list):
                src=sample["src"]
                dst=sample["dst"]
                pos_mask=sample["pos_mask"]
                pos_src=src[pos_mask]
                pos_dst=dst[pos_mask]
                neg_src=src[~pos_mask]
                neg_dst=dst[~pos_mask]
                SR_label=val_SR_result["SR_label"][idx]
                SR_hop=val_SR_result["SR_hop"][idx]
                TR_hop=val_TR_result["TR_hop"][idx]
                pos_ratio=GraphAnalysis.check_hard_positive_ratio_in_sample(
                    pos_src=pos_src,
                    pos_dst=pos_dst,
                    TR_hop=TR_hop
                )
                pos_ratio_list.append(pos_ratio)
                neg_ratio=GraphAnalysis.check_hard_negative_ratio_in_sample(
                    neg_src=neg_src,
                    neg_dst=neg_dst,
                    SR_label=SR_label,
                    SR_hop=SR_hop
                )
                neg_ratio_list.append(neg_ratio)
            print(f"{sampling} sampling: positive val sample list 2-hop 이상 TR node pair 평균 비율: {np.mean(pos_ratio_list)}")
            print(f"{sampling} sampling: negative val sample list 2-hop 이상 SR and not TR node pair 평균 비율: {np.mean(neg_ratio_list)}",end="\n\n")

            ### check ratio of test
            pos_ratio_list=[]
            neg_ratio_list=[]
            for idx,sample in enumerate(test_sample_list):
                src=sample["src"]
                dst=sample["dst"]
                pos_mask=sample["pos_mask"]
                pos_src=src[pos_mask]
                pos_dst=dst[pos_mask]
                neg_src=src[~pos_mask]
                neg_dst=dst[~pos_mask]
                SR_label=test_SR_result["SR_label"][idx]
                SR_hop=test_SR_result["SR_hop"][idx]
                TR_hop=test_TR_result["TR_hop"][idx]
                pos_ratio=GraphAnalysis.check_hard_positive_ratio_in_sample(
                    pos_src=pos_src,
                    pos_dst=pos_dst,
                    TR_hop=TR_hop
                )
                pos_ratio_list.append(pos_ratio)
                neg_ratio=GraphAnalysis.check_hard_negative_ratio_in_sample(
                    neg_src=neg_src,
                    neg_dst=neg_dst,
                    SR_label=SR_label,
                    SR_hop=SR_hop
                )
                neg_ratio_list.append(neg_ratio)
            print(f"{sampling} sampling: positive test sample list 2-hop 이상 TR node pair 평균 비율: {np.mean(pos_ratio_list)}")
            print(f"{sampling} sampling: negative test sample list 2-hop 이상 SR and not TR node pair 평균 비율: {np.mean(neg_ratio_list)}",end="\n\n")

if __name__=="__main__":
    """
    Execute app
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--dataset_name",
        type=str,
        choices=[
            "CollegeMsg",
            "bitcoin-otc",
            "bitcoin-alpha",
            "enron"
        ],
        default=f"CollegeMsg"
    )
    parser.add_argument("--sampling",
        type=str,
        choices=["random","hard"],
        default=f"random"
    )
    args=parser.parse_args()
    app_config={
        "app_num":args.app_num,
        "dataset_name":args.dataset_name,
        "sampling":args.sampling
    }
    main(**app_config)
