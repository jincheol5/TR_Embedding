import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import ReaCH_TGN_Utils,TrainUtils,Metric,EarlyStopper
from model import ReaCH_TGN

class ReaCH_TGN_Trainer:
    @staticmethod
    def train(
            model:ReaCH_TGN,
            train_loader:DataLoader,
            val_loader:DataLoader,
            val_sample_list:list[dict[str,torch.Tensor]],
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
        TR_hop=TR_result["hop"]
        TR_first_t=TR_result["first_t"]
        for epoch in tqdm(range(kwargs["epoch"]),desc=f"Model Training..."):
            ### Epoch마다 memory state 초기화
            model.memory.init_memory_state()

            ### Epoch마다 train_sample_list 생성
            train_sample_list=TrainUtils.get_TR_sample_list(
                data_loader=train_loader,
                n_sample=kwargs["n_sample"],
                n_pair=kwargs["n_pair"],
                TR_result=TR_result,
                sampling=kwargs["sampling"]
            )

            model.train()
            for batch_idx,(batch_event,batch_sample) in tqdm(
                    enumerate(zip(train_loader,train_sample_list)),
                    total=len(train_sample_list),
                    desc=f"Training epoch {epoch+1}..."
                ):
                ### Eventstream
                event_src,event_dst,event_t,event_edge=batch_event
                event_src=event_src.to(device)
                event_dst=event_dst.to(device)
                event_t=event_t.to(device)
                event_edge=event_edge.to(device)

                ### Generate view A,B
                view_A_event_t=ReaCH_TGN_Utils.augment_event_t(event_t=event_t)
                view_A_updated_mem_vec=model.get_updated_mem_vec(
                    src=event_src,
                    dst=event_dst,
                    edge=event_edge,
                    event_t=view_A_event_t
                )
                view_A_embedding_result=model.get_embedding_result(
                    src=event_src,
                    dst=event_dst,
                    event_t=view_A_event_t,
                    mem_vec=view_A_updated_mem_vec
                )
                view_A_src_vec=view_A_embedding_result["src_vec"]
                view_A_dst_vec=view_A_embedding_result["dst_vec"]

                view_B_event_t=ReaCH_TGN_Utils.augment_event_t(event_t=event_t)
                view_B_updated_mem_vec=model.get_updated_mem_vec(
                    src=event_src,
                    dst=event_dst,
                    edge=event_edge,
                    event_t=view_B_event_t
                )
                view_B_embedding_result=model.get_embedding_result(
                    src=event_src,
                    dst=event_dst,
                    event_t=view_B_event_t,
                    mem_vec=view_B_updated_mem_vec
                )
                view_B_src_vec=view_B_embedding_result["src_vec"]
                view_B_dst_vec=view_B_embedding_result["dst_vec"]

                ### Compute NT_Xent Loss
                NT_Xent_loss=ReaCH_TGN_Utils.compute_NT_Xent_Loss(
                    Z_src_A=view_A_src_vec,
                    Z_dst_A=view_A_dst_vec,
                    Z_src_B=view_B_src_vec,
                    Z_dst_B=view_B_dst_vec
                )

                ### update memory for eventsream
                model.update_model_memory(
                    src=event_src,
                    dst=event_dst,
                    event_t=event_t,
                    edge=event_edge
                )

                ### batch TR result
                batch_TR_hop=TR_hop[batch_idx]
                batch_TR_first_t=TR_first_t[batch_idx]
                batch_TR_hop=batch_TR_hop.to(device)
                batch_TR_first_t=batch_TR_first_t.to(device)

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

                pred_logit=model(
                    src=src,
                    dst=dst,
                    event_t=query_t
                ) # [B,1]

                ### Hop-Based Penalty
                # pos_pair에 대한 penalty 계산
                pos_src=src[pos_mask]
                pos_dst=dst[pos_mask]
                pos_pair_hop=batch_TR_hop[pos_src,pos_dst]
                weight_hop=ReaCH_TGN_Utils.compute_hop_based_penalty(
                    pair_hop=pos_pair_hop,
                    max_hop=batch_TR_hop.max().item()
                )

                ### Time-Gap Penalty
                # pos_pair에 대한 penalty 계산
                query_time=event_t.max().item()
                pos_pair_first_t=batch_TR_first_t[pos_src,pos_dst]
                weight_time=ReaCH_TGN_Utils.compute_time_gap_penalty(
                    pair_first_t=pos_pair_first_t,
                    query_time=query_time
                )

                ### Pair Weight
                pair_weight=torch.ones_like(label)
                pair_weight[pos_mask]=weight_hop*weight_time

                ### Weighted BCE Loss
                pred_logit=pred_logit.squeeze(-1) # -> [B,]
                criterion=nn.BCEWithLogitsLoss(weight=pair_weight)
                loss=criterion(pred_logit,label)
                total_loss=loss+NT_Xent_loss

                ### backward
                optimizer.zero_grad()
                total_loss.backward()
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
            model:ReaCH_TGN,
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
                ### Update model memory for Eventstream
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
            test_loader:DataLoader,
            test_sample_list:list,
            **kwargs
        ):
        """
        Evaluate acc 계산
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
        compute test acc
        """
        pred_logit_list=[]
        label_list=[]
        with torch.no_grad():
            for batch_event,batch_sample in tqdm(
                    zip(test_loader,test_sample_list),
                    total=len(test_sample_list),
                    desc=f"Compute Test Acc..."
                ):
                ### Update model memory for Eventstream
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
                ) # [n_sample,1]

                ### ACC
                pred_logit=pred_logit.squeeze(-1) # -> [B,]
                pred_logit_list.append(pred_logit.cpu())
                label_list.append(label.cpu())
        acc=Metric.compute_accuracy(
            pred_logit=torch.cat(pred_logit_list,dim=0),
            label=torch.cat(label_list,dim=0)
        )
        return {
            "acc":acc
        }

    @staticmethod
    def evaluate_hop_range(
            model:nn.Module,
            test_loader:DataLoader,
            test_sample_list:list,
            **kwargs
        ):
        """
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
        compute test acc
        """
        range_pred_logit_lists=[[],[],[]]
        range_label_lists=[[],[],[]]
        with torch.no_grad():
            for batch_event,batch_sample in tqdm(
                    zip(test_loader,test_sample_list),
                    total=len(test_sample_list),
                    desc=f"Compute Test Acc..."
                ):
                ### Update model memory for Eventstream
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
                range_1_mask=batch_sample["range_1_mask"]
                range_2_mask=batch_sample["range_2_mask"]
                range_3_mask=batch_sample["range_3_mask"]
                src=src.to(device)
                dst=dst.to(device)
                query_t=query_t.to(device)
                label=label.to(device)
                range_1_mask=range_1_mask.to(device)
                range_2_mask=range_2_mask.to(device)
                range_3_mask=range_3_mask.to(device)

                pred_logit=model(
                    src=src,
                    dst=dst,
                    event_t=query_t
                ) # [n_sample,1]
                pred_logit=pred_logit.squeeze(-1) # [n_sample,]

                ### Collect predictions and labels for each range
                for range_idx,range_mask in enumerate((range_1_mask,range_2_mask,range_3_mask)):
                    if range_mask.any():
                        range_pred_logit_lists[range_idx].append(pred_logit[range_mask].cpu())
                        range_label_lists[range_idx].append(label[range_mask].cpu())
        ### Compute accuracy over all samples in each range
        return {
            f"range_{range_idx+1}_acc":Metric.compute_accuracy(
                pred_logit=torch.cat(pred_logit_list,dim=0),
                label=torch.cat(label_list,dim=0)
            ) if pred_logit_list else 0.0
            for range_idx,(pred_logit_list,label_list) in enumerate(
                zip(range_pred_logit_lists,range_label_lists)
            )
        }