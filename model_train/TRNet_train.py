import torch
import torch.nn as nn
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import TRNet
from utils import TrainUtils,EarlyStopper,Metric

class TRNet_Trainer:
    @staticmethod
    def replay_memory(model, event_batches, device="cpu", reset=True):
        """Build graph state with frozen weights, preserving timestamp snapshots."""
        model.eval()
        if reset:
            model.reset_memory()
        with torch.no_grad():
            for src,dst,event_t,*_ in event_batches:
                model.update_mem(src.to(device),dst.to(device),event_t.to(device))

    @staticmethod
    def evaluate_research(model,event_batches,samples,device="cpu",prefix_batches=(),return_scores=False):
        """Evaluate one ordered split; optionally rebuild its full historical prefix.

        Validation/test samples must be fixed before model selection. Metrics are
        computed over all samples, and BCE is sample weighted. With no prefix the
        caller is responsible for the current memory state.
        """
        from sklearn.metrics import average_precision_score,roc_auc_score
        if len(event_batches)!=len(samples):
            raise ValueError("Event and sample batch counts differ")
        if prefix_batches:
            model.reset_memory()
            for prefix in prefix_batches:
                TRNet_Trainer.replay_memory(model,prefix,device,reset=False)
        model.eval()
        logits,labels=[],[]
        loss_sum=0.0
        with torch.no_grad():
            for events,sample in zip(event_batches,samples):
                src,dst,event_t,*_=events
                if not torch.all(sample["query_t"]==event_t.max().item()):
                    raise ValueError("Query timestamp does not match event prefix")
                model.update_mem(src.to(device),dst.to(device),event_t.to(device))
                prediction=model(sample["src"].to(device),sample["dst"].to(device))
                label=sample["label"].to(device)
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("Nonfinite evaluation logits")
                loss_sum+=nn.functional.binary_cross_entropy_with_logits(
                    prediction,label,reduction="sum").item()
                logits.append(prediction.cpu())
                labels.append(label.cpu())
        if not logits:
            raise ValueError("Evaluation contains no batches")
        logits=torch.cat(logits).numpy()
        labels=torch.cat(labels).numpy()
        result={"loss":loss_sum/len(labels),"acc":float(((logits>=0)==labels).mean()),
                "ap":float(average_precision_score(labels,logits)),
                "auc":float(roc_auc_score(labels,logits)),"n":len(labels)}
        if return_scores:
            return result,torch.from_numpy(logits),torch.from_numpy(labels)
        return result

    @staticmethod
    def evaluate_bayesian_research(model,event_batches,samples,device="cpu",
                                prefix_batches=(),mc_samples=5,mc_seed=71000):
        """Average predictive probabilities over coherent full-stream draws.

        Each draw rebuilds the complete historical prefix with one fixed sampled
        encoder. Never mix states between draws or sample new weights per event.
        """
        from sklearn.metrics import average_precision_score,roc_auc_score
        if mc_samples<1:
            raise ValueError("mc_samples must be positive")
        probabilities=[]
        labels=None
        try:
            for draw in range(mc_samples):
                model.encoder.sample_posterior(seed=mc_seed+draw)
                model.reset_memory()
                for prefix in prefix_batches:
                    TRNet_Trainer.replay_memory(model,prefix,device,reset=False)
                _,logits,current_labels=TRNet_Trainer.evaluate_research(
                    model,event_batches,samples,device,return_scores=True)
                if labels is not None and not torch.equal(labels,current_labels):
                    raise ValueError("MC draws contain different query labels")
                labels=current_labels
                probabilities.append(torch.sigmoid(logits).double())
        finally:
            model.encoder.use_posterior_mean()
        draws=torch.stack(probabilities)
        score=draws.mean(0)
        loss=nn.functional.binary_cross_entropy(score.clamp(1e-12,1-1e-12),labels.double())
        return {"loss":float(loss),"acc":float(((score>=0.5)==labels).double().mean()),
                "ap":float(average_precision_score(labels.numpy(),score.numpy())),
                "auc":float(roc_auc_score(labels.numpy(),score.numpy())),"n":labels.numel(),
                "mc_samples":mc_samples,"mc_seed":mc_seed,
                "mc_probability_std_mean":float(draws.std(0,unbiased=False).mean())}

    @staticmethod
    def fit_research(model,train_batches,val_batches,val_samples,sample_factory,
                    epochs=20,lr=0.0005,patience=5,device="cpu",grad_clip=1.0,
                    bptt_batches=1,epoch_callback=None,kl_weight=0.0,
                    mc_samples=5,bayesian_seed=71000):
        """Controlled encoder comparison using current-weight replay validation.

        sample_factory(epoch) supplies identical samples across configurations.
        The best epoch is selected by validation AUC, then AP and ACC; no test
        data are consulted. Parameters remain fixed within each BPTT window.
        Existing train/validate/test interfaces retain their legacy behavior.
        """
        if epochs<1 or patience<1 or bptt_batches<1:
            raise ValueError("Epochs, patience and BPTT window must be positive")
        model=model.to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        best_key=None
        best_state=None
        best_epoch=0
        best_validation=None
        history=[]
        wait=0
        started=time.perf_counter()
        is_bayesian=hasattr(model.encoder,"sample_posterior")
        for epoch in range(epochs):
            epoch_started=time.perf_counter()
            samples=sample_factory(epoch)
            if len(samples)!=len(train_batches):
                raise ValueError("Training event and sample batch counts differ")
            model.reset_memory()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            window_losses=[]
            window_n=0
            train_loss_sum=0.0
            train_n=0
            epoch_query_n=sum(sample["label"].numel() for sample in samples)
            if epoch_query_n<1:
                raise ValueError("Training has no queries")
            scaled_kl_sum=0.0
            optimizer_steps=0
            for batch_index,(events,sample) in enumerate(zip(train_batches,samples)):
                if is_bayesian and not window_losses:
                    model.encoder.sample_posterior(
                        seed=bayesian_seed+10000*epoch+batch_index)
                src,dst,event_t,*_=events
                model.update_mem(src.to(device),dst.to(device),event_t.to(device))
                logits=model(sample["src"].to(device),sample["dst"].to(device))
                labels=sample["label"].to(device)
                batch_loss=nn.functional.binary_cross_entropy_with_logits(
                    logits,labels,reduction="sum")
                if not torch.isfinite(batch_loss):
                    raise FloatingPointError("Nonfinite training loss")
                window_losses.append(batch_loss)
                window_n+=labels.numel()
                train_loss_sum+=batch_loss.detach().item()
                train_n+=labels.numel()
                if len(window_losses)==bptt_batches or batch_index+1==len(train_batches):
                    objective=torch.stack(window_losses).sum()/window_n
                    if is_bayesian:
                        scaled_kl=kl_weight*model.encoder.kl_loss()/epoch_query_n
                        objective=objective+scaled_kl
                        scaled_kl_sum+=scaled_kl.detach().item()
                    objective.backward()
                    if grad_clip is not None:
                        nn.utils.clip_grad_norm_(model.parameters(),grad_clip,
                                                error_if_nonfinite=True)
                    optimizer.step()
                    optimizer_steps+=1
                    model.module.detach_module()
                    optimizer.zero_grad(set_to_none=True)
                    window_losses=[]
                    window_n=0
            if is_bayesian:
                validation=TRNet_Trainer.evaluate_bayesian_research(
                    model,val_batches,val_samples,device,prefix_batches=(train_batches,),
                    mc_samples=mc_samples,mc_seed=bayesian_seed+500000)
            else:
                validation=TRNet_Trainer.evaluate_research(
                    model,val_batches,val_samples,device,prefix_batches=(train_batches,))
            entry={"epoch":epoch+1,"train_loss":train_loss_sum/train_n,
                "validation":validation,"seconds":time.perf_counter()-epoch_started}
            history.append(entry)
            if is_bayesian:
                entry.update({"scaled_kl_per_step":scaled_kl_sum/optimizer_steps,
                            "kl_sum":float(model.encoder.kl_loss().detach()),
                            "kl_weight":kl_weight,"train_query_count":epoch_query_n})
            key=tuple(validation[name] for name in ("auc","ap","acc"))
            if best_key is None or key>best_key:
                best_key=key
                best_epoch=epoch+1
                best_validation=validation.copy()
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                wait=0
            else:
                wait+=1
            if epoch_callback is not None:
                epoch_callback(entry)
            if wait>=patience:
                break
        model.load_state_dict(best_state)
        # Recompute memory for the chosen weights before returning to the caller.
        TRNet_Trainer.replay_memory(model,train_batches,device,reset=True)
        TRNet_Trainer.replay_memory(model,val_batches,device,reset=False)
        return {"history":history,"best_epoch":best_epoch,"validation":best_validation,
                "elapsed_seconds":time.perf_counter()-started,"epochs_run":len(history)}

    @staticmethod
    def train(
            model:TRNet,
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
            ### Epoch마다 module의 cumul_mem, prev_mem을 init_mem으로 초기화
            with torch.no_grad():
                model.module.cumul_mem.copy_(model.module.init_mem)
                model.module.prev_mem.copy_(model.module.init_mem)
                model.module.last_t.zero_()

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
                ### update module_vec of batch_eventstream
                event_src,event_dst,event_t,event_edge=batch_event
                event_src=event_src.to(device)
                event_dst=event_dst.to(device)
                event_t=event_t.to(device)
                event_edge=event_edge.to(device)
                model.update_mem(
                    src=event_src,
                    dst=event_dst,
                    event_t=event_t
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
                    dst=dst
                ) # [B,]

                ### Loss
                criterion=nn.BCEWithLogitsLoss()
                loss=criterion(pred_logit,label)

                ### Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ### detach module vec
                model.module.detach_module()

            """
            Validate Model
            """
            val_result=TRNet_Trainer.validate(
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
            model:TRNet,
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
                ### update module_vec of batch_eventstream
                event_src,event_dst,event_t,event_edge=batch_event
                event_src=event_src.to(device)
                event_dst=event_dst.to(device)
                event_t=event_t.to(device)
                event_edge=event_edge.to(device)
                model.update_mem(
                    src=event_src,
                    dst=event_dst,
                    event_t=event_t
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
                    dst=dst
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
            model:TRNet,
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
                ### update module_vec of batch_eventstream
                event_src,event_dst,event_t,event_edge=batch_event
                event_src=event_src.to(device)
                event_dst=event_dst.to(device)
                event_t=event_t.to(device)
                event_edge=event_edge.to(device)
                model.update_mem(
                    src=event_src,
                    dst=event_dst,
                    event_t=event_t
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
                    dst=dst
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
