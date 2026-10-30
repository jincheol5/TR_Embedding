import torch
import torch.nn as nn
import mlflow
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import TrainUtils,EarlyStopper,Metric

class ModelTrainer:
    @staticmethod
    def train(
            model:nn.Module,
            train_loader:DataLoader,
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
            ### Epoch마다 train_sample_loader 생성
            train_sample_loader=TrainUtils.get_TR_sample_loader(
                graph_loader=train_loader,
                n_sample=kwargs["n_sample"],
                n_pair=kwargs["n_pair"],
                candidates=kwargs["train_candidates"],
                TR_label=train_TR_label
            )
            model.train()
            for batch_sample in tqdm(train_sample_loader,desc=f"Training epoch {epoch+1}..."):
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
                ) # [B,]

                ### Loss
                criterion=nn.BCEWithLogitsLoss()
                loss=criterion(pred_logit,label)

                ### Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            """
            Validate Model
            """
            val_result=ModelTrainer.validate(
                model=model,
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
            model:nn.Module,
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
        acc_list=[]
        ap_list=[]
        auc_list=[]
        with torch.no_grad():
            for batch_sample in tqdm(val_sample_loader,desc=f"Validate..."):
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
                ) # [B,]

                ### Loss
                criterion=nn.BCEWithLogitsLoss()
                batch_loss=criterion(pred_logit,label)
                loss_list.append(batch_loss)

                ### ACC
                batch_acc=Metric.compute_ACC(
                    pred_logit=pred_logit,
                    label=label
                )
                acc_list.append(batch_acc)

                ### AP
                batch_ap=Metric.compute_AP(
                    pred_logit=pred_logit,
                    label=label
                )
                ap_list.append(batch_ap)

                ### AUC
                batch_auc=Metric.compute_AUC(
                    pred_logit=pred_logit,
                    label=label
                )
                auc_list.append(batch_auc)

        return {
            "loss":torch.stack(loss_list).mean().item(),
            "acc":sum(acc_list)/len(acc_list),
            "ap":sum(ap_list)/len(ap_list),
            "auc":sum(auc_list)/len(auc_list),
        }

    @staticmethod
    def test(
            model:nn.Module,
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
        acc_list=[]
        ap_list=[]
        auc_list=[]
        with torch.no_grad():
            for batch_sample in tqdm(test_sample_loader,desc=f"Test..."):
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
                ) # [B,]

                ### ACC
                batch_acc=Metric.compute_ACC(
                    pred_logit=pred_logit,
                    label=label
                )
                acc_list.append(batch_acc)

                ### AP
                batch_ap=Metric.compute_AP(
                    pred_logit=pred_logit,
                    label=label
                )
                ap_list.append(batch_ap)

                ### AUC
                batch_auc=Metric.compute_AUC(
                    pred_logit=pred_logit,
                    label=label
                )
                auc_list.append(batch_auc)

        return {
            "acc":sum(acc_list)/len(acc_list),
            "ap":sum(ap_list)/len(ap_list),
            "auc":sum(auc_list)/len(auc_list),
        }