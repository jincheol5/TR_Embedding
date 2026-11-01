import torch
import torch.nn as nn
import mlflow
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import ReaCH_TGN
from utils import TrainUtils,EarlyStopper,ReaCH_TGN_Utils,Metric

class ReaCH_TGN_Trainer:
    @staticmethod
    def _get_view_embedding(
            model:ReaCH_TGN,
            src:torch.Tensor,
            dst:torch.Tensor,
            edge:torch.Tensor,
            event_t:torch.Tensor,
            min_time:int
        ):
        # Keep second-resolution graph timestamps and respect the previous batch boundary.
        view_t=event_t.to(dtype=torch.int64).clamp_min(min_time)
        order=torch.argsort(view_t,stable=True)
        sorted_src=src[order]
        sorted_dst=dst[order]
        sorted_edge=edge[order]
        sorted_t=view_t[order]
        graph=model.graph
        touched_nodes=dst.detach().cpu().unique().tolist()
        original={
            node:(graph.in_adj[node],graph.in_adj_edge[node],graph.in_adj_t[node])
            for node in touched_nodes
        }
        try:
            for node,(neighbors,edges,times) in original.items():
                graph.in_adj[node]=neighbors.copy()
                graph.in_adj_edge[node]=edges.copy()
                graph.in_adj_t[node]=times.copy()
            graph.update_in_adj(
                src=sorted_src,dst=sorted_dst,event_t=sorted_t,edge=sorted_edge
            )
            view_mem_vec=model.get_updated_mem_vec(
                src=sorted_src,dst=sorted_dst,edge=sorted_edge,event_t=sorted_t
            )
            # Preserve event order so A[i] and B[i] remain positive pairs.
            return model.get_embedding_result(
                src=src,dst=dst,event_t=view_t,mem_vec=view_mem_vec
            )
        finally:
            for node,(neighbors,edges,times) in original.items():
                graph.in_adj[node]=neighbors
                graph.in_adj_edge[node]=edges
                graph.in_adj_t[node]=times

    @staticmethod
    def train(
            model:ReaCH_TGN,
            train_loader:DataLoader,
            val_loader:DataLoader,
            val_sample_loader:list,
            train_TR_result:dict[str,torch.Tensor],
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
        Train model
        """
        train_TR_label=train_TR_result["label"]
        train_TR_hop=train_TR_result["hop"]
        train_TR_first_t=train_TR_result["first_t"]
        for epoch in tqdm(range(kwargs["epoch"]),desc=f"Model Training..."):
            ### Epoch마다 in_adj 초기화
            model.graph.init_in_adj()

            ### Epoch마다 memory state 초기화
            model.memory.init_memory_state()

            ### Epoch마다 train_sample_loader 생성
            train_sample_loader=TrainUtils.get_TR_sample_loader(
                graph_loader=train_loader,
                n_sample=kwargs["n_sample"],
                n_pair=kwargs["n_pair"],
                candidates=kwargs["train_candidates"],
                TR_label=train_TR_label
            )
            previous_batch_t=0
            model.train()
            for batch_idx,(batch_event,batch_sample) in tqdm(
                    enumerate(zip(train_loader,train_sample_loader)),
                    total=len(train_sample_loader),
                    desc=f"Training epoch {epoch+1}..."
                ):
                ### update in_adj of batch_eventstream
                event_src,event_dst,event_t,event_edge=batch_event
                event_src=event_src.to(device)
                event_dst=event_dst.to(device)
                event_t=event_t.to(device)
                event_edge=event_edge.to(device)
                ### Generate view A,B
                view_A_event_t=ReaCH_TGN_Utils.augment_event_t(event_t=event_t)
                view_A_embedding_result=ReaCH_TGN_Trainer._get_view_embedding(
                    model=model,
                    src=event_src,
                    dst=event_dst,
                    edge=event_edge,
                    event_t=view_A_event_t,
                    min_time=previous_batch_t
                )
                view_A_src_vec=view_A_embedding_result["src_vec"]
                view_A_dst_vec=view_A_embedding_result["dst_vec"]

                view_B_event_t=ReaCH_TGN_Utils.augment_event_t(event_t=event_t)
                view_B_embedding_result=ReaCH_TGN_Trainer._get_view_embedding(
                    model=model,
                    src=event_src,
                    dst=event_dst,
                    edge=event_edge,
                    event_t=view_B_event_t,
                    min_time=previous_batch_t
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

                model.graph.update_in_adj(
                    src=event_src,
                    dst=event_dst,
                    event_t=event_t,
                    edge=event_edge
                )

                previous_batch_t=int(event_t.max().item())

                ### update memory for eventsream
                model.update_model_memory(
                    src=event_src,
                    dst=event_dst,
                    event_t=event_t,
                    edge=event_edge
                )

                ### batch TR result
                batch_TR_hop=train_TR_hop[batch_idx]
                batch_TR_first_t=train_TR_first_t[batch_idx]
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

                ### Final Weight
                final_weight=torch.ones_like(label)
                final_weight[pos_mask]=weight_hop*weight_time

                ### Forward
                pred_logit=model(
                    src=src,
                    dst=dst,
                    event_t=query_t
                ) # [B,]

                ### Weighted BCE Loss
                criterion=nn.BCEWithLogitsLoss(weight=final_weight)
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
                val_sample_loader=val_sample_loader,
                **kwargs
            )
            val_acc=val_result["acc"]
            val_ap=val_result["ap"]
            val_auc=val_result["auc"]
            print(f"{epoch+1} epoch  Validate ACC: {val_acc}")
            print(f"{epoch+1} epoch  Validate AP: {val_ap}")
            print(f"{epoch+1} epoch  Validate AUC: {val_auc}")

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

        if kwargs["early_stop"] and early_stop.best_state is not None:
            model.load_state_dict(early_stop.best_state)
        return model

    @staticmethod
    def validate(
            model:ReaCH_TGN,
            val_loader:DataLoader,
            val_sample_loader:list,
            **kwargs
        ):
        """
        Set GPU
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model=model.to(device)
        model.graph.to_device(device=device)
        model.eval()

        """
        Compute validate loss and acc
        """
        loss_list=[]
        pred_logit_list=[]
        label_list=[]
        with torch.no_grad():
            for batch_event,batch_sample in tqdm(zip(val_loader,val_sample_loader),desc=f"Validate..."):
                ### update in_adj of batch_eventstream
                event_src,event_dst,event_t,event_edge=batch_event
                event_src=event_src.to(device)
                event_dst=event_dst.to(device)
                event_t=event_t.to(device)
                event_edge=event_edge.to(device)
                model.graph.update_in_adj(
                    src=event_src,
                    dst=event_dst,
                    event_t=event_t,
                    edge=event_edge
                )

                ### Update model memory for Eventstream
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

                ### Forward
                pred_logit=model(
                    src=src,
                    dst=dst,
                    event_t=query_t
                ) # [B,]

                ### Loss
                criterion=nn.BCEWithLogitsLoss()
                batch_loss=criterion(pred_logit,label)
                loss_list.append(batch_loss)

                # Collect all samples on CPU for dataset-level metrics.
                pred_logit_list.append(pred_logit.detach().cpu())
                label_list.append(label.detach().cpu())

        pred_logit=torch.cat(pred_logit_list)
        label=torch.cat(label_list)
        return {
            "loss":torch.stack(loss_list).mean().item(),
            "acc":Metric.compute_ACC(pred_logit=pred_logit,label=label),
            "ap":Metric.compute_AP(pred_logit=pred_logit,label=label),
            "auc":Metric.compute_AUC(pred_logit=pred_logit,label=label),
        }

    @staticmethod
    def test(
            model:ReaCH_TGN,
            test_loader:DataLoader,
            test_sample_loader:list,
            **kwargs
        ):
        """
        Set GPU
        """
        if torch.cuda.is_available():
            device=torch.device("cuda")
        elif torch.backends.mps.is_available():
            device=torch.device("mps")
        else:
            device=torch.device("cpu")
        model=model.to(device)
        model.graph.to_device(device=device)
        model.eval()

        """
        Compute test acc
        """
        pred_logit_list=[]
        label_list=[]
        with torch.no_grad():
            for batch_event,batch_sample in tqdm(zip(test_loader,test_sample_loader),desc=f"Test..."):
                ### update in_adj of batch_eventstream
                event_src,event_dst,event_t,event_edge=batch_event
                event_src=event_src.to(device)
                event_dst=event_dst.to(device)
                event_t=event_t.to(device)
                event_edge=event_edge.to(device)
                model.graph.update_in_adj(
                    src=event_src,
                    dst=event_dst,
                    event_t=event_t,
                    edge=event_edge
                )

                ### Update model memory for Eventstream
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

                ### Forward
                pred_logit=model(
                    src=src,
                    dst=dst,
                    event_t=query_t
                ) # [B,]

                # Collect all samples on CPU for dataset-level metrics.
                pred_logit_list.append(pred_logit.detach().cpu())
                label_list.append(label.detach().cpu())

        pred_logit=torch.cat(pred_logit_list)
        label=torch.cat(label_list)
        return {
            "acc":Metric.compute_ACC(pred_logit=pred_logit,label=label),
            "ap":Metric.compute_AP(pred_logit=pred_logit,label=label),
            "auc":Metric.compute_AUC(pred_logit=pred_logit,label=label),
        }
