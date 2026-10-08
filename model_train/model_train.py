import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import TrainUtils,Metric,EarlyStopper

class ModelTrainer:
    @staticmethod
    def train(
            model:nn.Module,
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
        for epoch in tqdm(range(kwargs["epoch"]),desc=f"Model Training..."):
            ### Epoch마다 Memory-based model의 경우 memory state 초기화
            if kwargs["model_name"] in ("TGN"):
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
            for batch_event,batch_sample in tqdm(
                    zip(train_loader,train_sample_list),
                    total=len(train_sample_list),
                    desc=f"Training epoch {epoch+1}..."
                ):
                ### Update model memory for Eventstream
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
                loss=criterion(pred_logit,label)

                ### backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ### memory detach
                if kwargs["model_name"] in ("TGN"):
                    model.memory.memory_detach()

            """
            Validate Model
            """
            val_result=ModelTrainer.validate(
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
                ### Update model memory for Eventstream
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