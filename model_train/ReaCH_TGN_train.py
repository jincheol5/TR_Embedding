import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import TrainUtils,ReaCH_TGN_Utils,Metric,EarlyStopper
from model import ReaCH_TGN

class ReaCH_TGN_Trainer:
    """
    ReaCH-TGN 학습/평가
    """
    @staticmethod
    def train(
            model:ReaCH_TGN,
            train_loader:DataLoader,
            val_loader:DataLoader,
            val_sample_list:list,
            SR_result:dict[str,torch.Tensor],
            TR_result:dict[str,torch.Tensor],
            **kwargs
        ):
        """
        Set GPU, Optimizer
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model=model.to(device)
        model.graph.to_device(device=device)

        if kwargs["optimizer"]=="adam":
            optimizer=torch.optim.Adam(
                model.parameters(),
                lr=kwargs["lr"]
            )
        else:
            optimizer=torch.optim.SGD(
                model.parameters(),
                lr=kwargs["lr"]
            )

        """
        Set Early Stopper
        """
        if kwargs["early_stop"]:
            early_stop=EarlyStopper(patience=kwargs["patience"])

        """
        Model train
        """
        TR_hop=TR_result["TR_hop"]
        TR_first_t=TR_result["TR_first_t"]
        for epoch in tqdm(range(kwargs["epoch"]),desc=f"Model Training..."):
            ### Epoch마다 memory state 초기화
            model.memory.init_memory_state()

            ### Epoch마다 train_sample_list 생성
            train_sample_list=TrainUtils.get_TR_sample_list(
                n_pair=kwargs["n_pair"],
                data_loader=train_loader,
                SR_result=SR_result,
                TR_result=TR_result,
                sampling=kwargs["sampling"],
                pos_hard_ratio=kwargs["pos_hard_ratio"],
                neg_hard_ratio=kwargs["neg_hard_ratio"]
            )

            model.train()
            for batch_idx,(batch_event,batch_sample) in tqdm(
                    enumerate(zip(train_loader,train_sample_list)),
                    total=len(train_sample_list),
                    desc=f"Training epoch {epoch+1}..."
                ):
                ### batch TR result
                batch_TR_hop=TR_hop[batch_idx]
                batch_TR_first_t=TR_first_t[batch_idx]
                batch_TR_hop=batch_TR_hop.to(device)
                batch_TR_first_t=batch_TR_first_t.to(device)

                ### Eventstream
                event_src,event_dst,event_t,event_edge=batch_event
                event_src=event_src.to(device)
                event_dst=event_dst.to(device)
                event_t=event_t.to(device)
                event_edge=event_edge.to(device)
                query_time=event_t.max().item()

                ### Temporal Augmentation of Eventstream and Contrastive Learning
                # view_A_mem_vec=model.memory.get_mem_vec().clone()
                # view_A_mem_t=model.memory.get_mem_t().clone()
                # view_B_mem_vec=model.memory.get_mem_vec().clone()
                # view_B_mem_t=model.memory.get_mem_t().clone()

                # view_A_t=ReaCH_TGN_Utils.Temporal_Augmentation(event_t=event_t)
                # view_B_t=ReaCH_TGN_Utils.Temporal_Augmentation(event_t=event_t)

                # view_A_vec=model.get_embedded_vec_for_contrastive_learning(
                #     src=event_src,
                #     dst=event_dst,
                #     edge=event_edge,
                #     event_t=view_A_t,
                #     mem_vec=view_A_mem_vec,
                #     mem_t=view_A_mem_t
                # )
                # view_A_src_vec=view_A_vec["src_vec"]
                # view_A_dst_vec=view_A_vec["dst_vec"]

                # view_B_vec=model.get_embedded_vec_for_contrastive_learning(
                #     src=event_src,
                #     dst=event_dst,
                #     edge=event_edge,
                #     event_t=view_B_t,
                #     mem_vec=view_B_mem_vec,
                #     mem_t=view_B_mem_t
                # )
                # view_B_src_vec=view_B_vec["src_vec"]
                # view_B_dst_vec=view_B_vec["dst_vec"]

                # NT_Xent_Loss=ReaCH_TGN_Utils.compute_NT_Xent_Loss(
                #     src=event_src,
                #     dst=event_dst,
                #     Z_src_A=view_A_src_vec,
                #     Z_src_B=view_B_src_vec,
                #     Z_dst_A=view_A_dst_vec,
                #     Z_dst_B=view_B_dst_vec
                # )

                ### update memory for eventsream
                model.update_model_memory(
                    src=event_src,
                    dst=event_dst,
                    event_t=event_t,
                    edge=event_edge
                )

                ### TR Sample
                src=batch_sample["src"]
                dst=batch_sample["dst"]
                query_t=batch_sample["query_t"]
                label=batch_sample["label"]
                pos_mask=batch_sample["pos_mask"]
                src=src.to(device)
                dst=dst.to(device)
                query_t=query_t.to(device)
                label=label.to(device)
                pos_mask=pos_mask.to(device)

                ### pos_pair에 대한 penalty 계산
                pos_src=src[pos_mask]
                pos_dst=dst[pos_mask]
                pos_pair_hop=batch_TR_hop[pos_src,pos_dst]
                hop_penalty_weight=ReaCH_TGN_Utils.compute_hop_based_penalty(
                    pair_hop=pos_pair_hop,
                    max_hop=kwargs["max_hop"]
                )
                pos_pair_first_t=batch_TR_first_t[pos_src,pos_dst]
                time_gap_penalty_weight=ReaCH_TGN_Utils.compute_time_gap_penalty(
                    pair_first_t=pos_pair_first_t,
                    query_time=query_time
                )

                ### compute loss
                pred_logit=model(
                    src=src,
                    dst=dst,
                    event_t=query_t
                ) # [B,1]
                pred_logit=pred_logit.squeeze(-1) # [B,]
                criterion=nn.BCEWithLogitsLoss(reduction="none") # [B,]
                sample_loss=criterion(pred_logit,label)  # [B,]
                loss_weight=torch.ones_like(sample_loss) # [B,]
                loss_weight[pos_mask]=hop_penalty_weight*time_gap_penalty_weight # [B,]
                loss=(sample_loss*loss_weight).mean()
                # loss+=NT_Xent_Loss

                ### backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ### memory detach
                model.memory.memory_detach()
            """
            Validate Model
            """
            val_result=ReaCH_TGN_Trainer.validate(
                model=model,
                val_loader=val_loader,
                val_sample_list=val_sample_list,
                **kwargs
            )
            val_acc=val_result["acc"]
            print(f"Validate ACC using {kwargs['sampling']} TR Sampling: {val_acc}")

            """
            Check Early Stop
            """
            val_loss=val_result["loss"]
            print(f"{epoch+1} epoch Validate Loss: {val_loss}")
            if kwargs["early_stop"]:
                pre_model=early_stop(
                    val_loss=val_loss,
                    model=model
                )
                if early_stop.early_stop:
                    model=pre_model
                    print(f"Early Stop in epoch {epoch+1}")
                    break
        return model

    @staticmethod
    def validate(
            model:nn.Module,
            val_loader:DataLoader,
            val_sample_list:DataLoader,
            **kwargs
        ):
        """
        Validate Loss and ACC 계산
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model.to(device)
        model.graph.to_device(device=device)
        model.eval()

        """
        compute validate loss and acc
        """
        loss_list=[]
        acc_list=[]
        with torch.no_grad():
            for batch_event,batch_sample in tqdm(
                    zip(val_loader,val_sample_list),
                    total=len(val_sample_list),
                    desc=f"Validate..."
                ):
                ### Eventstream
                event_src,event_dst,event_t,event_edge=batch_event
                if kwargs["model_name"] in ("TGN"):
                    event_src=event_src.to(device)
                    event_dst=event_dst.to(device)
                    event_t=event_t.to(device)
                    event_edge=event_edge.to(device)
                    model.update_model_memory(
                        src=event_src,
                        dst=event_dst,
                        event_t=event_t,
                        edge=event_edge
                    )

                ### TR Sample
                src=batch_sample["src"]
                dst=batch_sample["dst"]
                query_t=batch_sample["query_t"]
                label=batch_sample["label"]
                src=src.to(device)
                dst=dst.to(device)
                query_t=query_t.to(device)
                label=label.to(device)

                pred_logit=model(
                    src=src,
                    dst=dst,
                    event_t=query_t
                ) # [B,1]

                ### Loss
                pred_logit=pred_logit.squeeze(-1) # -> [B,]
                criterion=nn.BCEWithLogitsLoss()
                batch_loss=criterion(pred_logit,label)
                loss_list.append(batch_loss)

                ### ACC
                pred_logit=pred_logit.squeeze(-1) # -> [B,]
                batch_acc=Metric.compute_accuracy(
                    pred_logit=pred_logit,
                    label=label
                )
                acc_list.append(batch_acc)
        return {
            "loss":torch.stack(loss_list).mean().item(),
            "acc":sum(acc_list)/len(acc_list)
        }

    @staticmethod
    def evaluate(
            model:nn.Module,
            val_loader:DataLoader,
            test_loader:DataLoader,
            test_sample_list:list,
            **kwargs
        ):
        """
        Memory-based GNN의 경우 evaluate_model 전에 val_loader에 대해 memory update 수행되어야 함.
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model.to(device)
        model.graph.to_device(device=device)
        model.eval()

        """
        update memory for validate eventstream
        """
        for event_src,event_dst,event_t,event_edge in tqdm(
                val_loader,
                desc=f"validate eventstream에 대한 memory update 수행..."
            ):
            event_src=event_src.to(device)
            event_dst=event_dst.to(device)
            event_t=event_t.to(device)
            event_edge=event_edge.to(device)
            model.update_model_memory(
                src=event_src,
                dst=event_dst,
                event_t=event_t,
                edge=event_edge
            )

        """
        compute test acc
        """
        acc_list=[]
        with torch.no_grad():
            for batch_event,batch_sample in tqdm(
                    zip(test_loader,test_sample_list),
                    total=len(test_sample_list),
                    desc=f"Compute Test Acc..."
                ):
                ### Eventstream
                event_src,event_dst,event_t,event_edge=batch_event
                event_src=event_src.to(device)
                event_dst=event_dst.to(device)
                event_t=event_t.to(device)
                event_edge=event_edge.to(device)
                model.update_model_memory(
                    src=event_src,
                    dst=event_dst,
                    event_t=event_t,
                    edge=event_edge
                )

                ### TR Sample
                src=batch_sample["src"]
                dst=batch_sample["dst"]
                query_t=batch_sample["query_t"]
                label=batch_sample["label"]
                src=src.to(device)
                dst=dst.to(device)
                query_t=query_t.to(device)
                label=label.to(device)

                pred_logit=model(
                    src=src,
                    dst=dst,
                    event_t=query_t
                ) # [B,1]

                ### ACC
                pred_logit=pred_logit.squeeze(-1) # -> [B,]
                batch_acc=Metric.compute_accuracy(
                    pred_logit=pred_logit,
                    label=label
                )
                acc_list.append(batch_acc)
        return {
            "acc":sum(acc_list)/len(acc_list)
        }
