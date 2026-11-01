import torch
import torch.nn as nn
import mlflow
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import TGAT,TGN,CAW_N,GraphMixer,DyGFormer
from utils import TrainUtils,EarlyStopper,Metric

class Model_Trainer:
    @staticmethod
    def train(
            model:TGAT|TGN|CAW_N|GraphMixer|DyGFormer,
            train_loader:DataLoader,
            val_loader:DataLoader,
            val_sample_loader:list,
            train_TR_label:torch.Tensor,
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
        for epoch in tqdm(range(kwargs["epoch"]),desc=f"Model Training..."):
            ### Epoch마다 in_adj 초기화
            model.graph.init_in_adj()

            ### Epoch마다 TGN model의 경우 memory state 초기화
            if isinstance(model,TGN):
                model.memory.init_memory_state()

            ### Epoch마다 train_sample_loader 생성
            train_sample_loader=TrainUtils.get_TR_sample_loader(
                graph_loader=train_loader,
                n_sample=kwargs["n_sample"],
                n_pair=kwargs["n_pair"],
                candidates=kwargs["train_candidates"],
                TR_label=train_TR_label
            )
            model.train()
            for batch_event,batch_sample in tqdm(
                    zip(train_loader,train_sample_loader),
                    total=len(train_loader),
                    desc=f"Training epoch {epoch+1}..."
                ):
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
                if isinstance(model,TGN):
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
                loss=criterion(pred_logit,label)

                ### Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if isinstance(model,TGN):
                    model.memory.memory_detach()

            """
            Validate Model
            """
            val_result=Model_Trainer.validate(
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
            model:TGAT|TGN|CAW_N|GraphMixer|DyGFormer,
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
            for batch_event,batch_sample in tqdm(
                    zip(val_loader,val_sample_loader),
                    total=len(val_loader),
                    desc=f"Validate..."
                ):
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
                if isinstance(model,TGN):
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
            model:TGAT|TGN|CAW_N|GraphMixer|DyGFormer,
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
            for batch_event,batch_sample in tqdm(
                    zip(test_loader,test_sample_loader),
                    total=len(test_loader),
                    desc=f"Test..."
                ):
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
                if isinstance(model,TGN):
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
