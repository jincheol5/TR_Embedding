"""Run paired TRNet experiments without using test metrics for model selection.

Example: python -m app.research.run --stage screening --epochs 12 --workers 3
All paths resolve relative to this repository, not the current working directory.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/"app"/"research"


def atomic_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")
    temporary.replace(path)


def load_data(dataset):
    import torch
    from utils import DataUtils,TrainUtils,TemporalGraphDataset
    from utils import data_utils
    data_utils.BASE_PATH=str(ROOT.parent/"data")
    frame=DataUtils.preprocess_SNAP_dataset(dataset)["graph_df"]
    splits=TrainUtils.split_graph_df(frame)
    batches=[]
    for split in splits:
        graph=TemporalGraphDataset(split)
        batches.append(list(torch.utils.data.DataLoader(graph,batch_size=200,shuffle=False)))
    metadata={"dataset":dataset,"n_node":int(max(frame.u.max(),frame.i.max())),
              "events":dict(zip(("train","val","test"),map(len,splits))),
              "batch_size":200,"graph_sha256":hashlib.sha256(
                  frame[["u","i","t"]].to_numpy().tobytes()).hexdigest()}
    return frame,batches,metadata


def sample_path(dataset,seed):
    return OUT/"cache"/f"{dataset}_seed{seed}_samples.pt"


def prepare_samples(dataset,seed,epochs):
    import torch
    from utils import DataUtils,TrainUtils
    torch.set_num_threads(1)
    _,batches,metadata=load_data(dataset)
    cache_path=sample_path(dataset,seed)
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    sampler_hash=hashlib.sha256((ROOT/"utils"/"sampling_utils.py").read_bytes()).hexdigest()
    cache=None
    if cache_path.exists():
        cache=torch.load(cache_path,weights_only=True)
        if cache["metadata"]!=metadata or cache["sampler_sha256"]!=sampler_hash:
            raise ValueError("Stale sample cache: remove it before continuing")
        if len(cache["samples"])>=epochs:
            return
    if cache is None:
        cache={"metadata":metadata,"sampler_sha256":sampler_hash,"samples":[]}
    cache_file=ROOT.parent/"data"/"temporal_graph"/"SNAP"/dataset/"TR_result"/f"{dataset}_train_B200.pt"
    label=torch.load(cache_file,mmap=True,weights_only=True)["label"]
    if len(label)!=len(batches[0]):
        raise ValueError("Training labels and event batches differ")
    candidates=torch.cat([b[0] for b in batches[0]]+[b[1] for b in batches[0]]).tolist()
    for epoch in range(len(cache["samples"]),epochs):
        torch.manual_seed(30000+seed*1000+epoch)
        cache["samples"].append(TrainUtils.get_TR_sample_loader(
            batches[0],200,5,candidates,label))
        if (epoch+1)%5==0 or epoch+1==epochs:
            print(f"samples dataset={dataset} seed={seed} epochs={epoch+1}",flush=True)
    temporary=cache_path.with_suffix(".tmp")
    torch.save(cache,temporary)
    temporary.replace(cache_path)


def config_id(config):
    standard={"encoder_type","decoder_input","mem_dim","latent_dim","dataset","lr",
              "seed","bptt_batches","event_backend"}
    extra={key:value for key,value in config.items() if key not in standard}
    suffix=("_x"+hashlib.sha256(json.dumps(extra,sort_keys=True).encode()).hexdigest()[:10]) if extra else ""
    return (f"{config['encoder_type']}_{config['decoder_input']}_d{config['mem_dim']}"
            f"_lr{config['lr']:g}_b{config.get('bptt_batches',1)}_s{config['seed']}"
            +(f"_{config['event_backend']}" if config.get("event_backend","sequential")!="sequential" else "")+suffix)


def build_model(config):
    from model import TRNet
    names=("matrix_heads","matrix_key_dim","bayes_prior_std","bayes_posterior_rho_init")
    options={name:config[name] for name in names if name in config}
    return TRNet(config["mem_dim"],config["latent_dim"],
                 encoder_type=config["encoder_type"],decoder_input=config["decoder_input"],
                 event_backend=config.get("event_backend","sequential"),**options)


def samples_digest(epochs):
    """Hash only the requested prefix; extending a cache preserves old hashes."""
    digest=hashlib.sha256()
    for epoch in epochs:
        for sample in epoch:
            for name in ("src","dst","query_t","label"):
                digest.update(sample[name].numpy().tobytes())
    return digest.hexdigest()


def run_one(config,stage,epochs,patience,test=False):
    os.environ["MLFLOW_DISABLE_AGENT_HINT"]="1"
    os.environ["OMP_NUM_THREADS"]="1"
    os.environ["MKL_NUM_THREADS"]="1"
    import torch
    from model import TRNet
    from model_train import TRNet_Trainer
    from utils import DataUtils,TrainUtils
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:
        torch.set_num_interop_threads(1)
    seed=config["seed"]
    TrainUtils.set_seed(seed)
    _,batches,metadata=load_data(config["dataset"])
    sample_cache=torch.load(sample_path(config["dataset"],seed),weights_only=True)
    if sample_cache["metadata"]!=metadata or len(sample_cache["samples"])<epochs:
        raise ValueError("Training sample cache is stale or incomplete")
    val_samples=DataUtils.load_sample_loader(config["dataset"],"val",seed,200)
    model=build_model(config)
    model.set_module(metadata["n_node"])
    # Pair initial IDs and decoder weights across encoders, independent of how
    # many RNG draws their recurrent parameter initialization consumes.
    generator=torch.Generator().manual_seed(20000+seed)
    model.module.init_mem.copy_(torch.randn(model.module.init_mem.shape,generator=generator)
                                *config["mem_dim"]**-0.5)
    with torch.random.fork_rng():
        torch.manual_seed(10000+seed)
        for layer in model.decoder.modules():
            if isinstance(layer,torch.nn.Linear):
                layer.reset_parameters()
    model.reset_memory()
    identifier=config_id(config)
    result_path=OUT/"results"/stage/f"{identifier}.json"
    print(f"START {stage}/{identifier}",flush=True)
    def progress(entry):
        val=entry["validation"]
        print(f"EPOCH {identifier} {entry['epoch']} loss={entry['train_loss']:.4f} "
              f"val={val['acc']:.4f}/{val['ap']:.4f}/{val['auc']:.4f} "
              f"seconds={entry['seconds']:.1f}",flush=True)
        atomic_json(result_path.with_suffix(".progress.json"),
                    {"config":config,"stage":stage,"last_epoch":entry})
    record=TRNet_Trainer.fit_research(
        model,batches[0],batches[1],val_samples,
        lambda epoch:sample_cache["samples"][epoch],epochs=epochs,lr=config["lr"],
        patience=patience,bptt_batches=config.get("bptt_batches",1),epoch_callback=progress,
        kl_weight=config.get("kl_weight",0.0),mc_samples=config.get("mc_samples",5),
        bayesian_seed=71000+seed*1000)
    record.update({"config":config,"stage":stage,"data":metadata,
                   "trainable_parameters":sum(p.numel() for p in model.parameters()),
                   "environment":{"python":sys.version.split()[0],"torch":torch.__version__,
                                  "device":"cpu","threads_per_process":1},
                   "requested_epochs":epochs,"patience":patience,
                   "selection_metric":"validation AUC, then AP, then ACC",
                   "evaluation_memory":"reset and replay prefix with frozen current weights",
                   "acc_threshold_logit":0.0,"grad_clip":1.0,
                   "state_buffer_bytes":sum(buffer.numel()*buffer.element_size()
                                            for buffer in model.module.buffers()),
                   "state_buffers":{name:{"shape":list(buffer.shape),
                                          "bytes":buffer.numel()*buffer.element_size()}
                                    for name,buffer in model.module.named_buffers()},
                   "training_samples_sha256":samples_digest(sample_cache["samples"][:epochs])})
    checkpoint=OUT/"checkpoints"/stage/f"{identifier}.pt"
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"config":config,"state_dict":model.state_dict(),"best_epoch":record["best_epoch"],
                "data":metadata},checkpoint)
    record["checkpoint"]=str(checkpoint.relative_to(ROOT))
    if test:
        test_samples=DataUtils.load_sample_loader(config["dataset"],"test",seed,200)
        if hasattr(model.encoder,"sample_posterior"):
            record["test"]=TRNet_Trainer.evaluate_bayesian_research(
                model,batches[2],test_samples,prefix_batches=(batches[0],batches[1]),
                mc_samples=config.get("test_mc_samples",10),mc_seed=71000+seed*1000+500000)
        else:
            record["test"]=TRNet_Trainer.evaluate_research(model,batches[2],test_samples)
    atomic_json(result_path,record)
    result_path.with_suffix(".progress.json").unlink(missing_ok=True)
    print(f"DONE {identifier} best_epoch={record['best_epoch']} val={record['validation']}"
          f" test={record.get('test')}",flush=True)
    return str(result_path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",default="screening")
    parser.add_argument("--dataset",default="CollegeMsg")
    parser.add_argument("--encoders",nargs="+",default=["residual_gru","diagonal","lru","gated_linear","mingru"])
    parser.add_argument("--decoder-inputs",nargs="+",default=["concat","interaction"])
    parser.add_argument("--seeds",type=int,nargs="+",default=[1])
    parser.add_argument("--lrs",type=float,nargs="+",default=[0.0005])
    parser.add_argument("--mem-dim",type=int,default=172)
    parser.add_argument("--latent-dim",type=int,default=172)
    parser.add_argument("--epochs",type=int,default=12)
    parser.add_argument("--patience",type=int,default=4)
    parser.add_argument("--bptt-batches",type=int,default=1)
    parser.add_argument("--event-backend",choices=["sequential","wave"],default="sequential")
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--test",action="store_true",help="Use only after validation selection")
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--config-file",type=Path,help="JSON list of exact run configurations")
    args=parser.parse_args()
    configs=[{"dataset":args.dataset,"encoder_type":encoder,"decoder_input":decoder,
              "mem_dim":args.mem_dim,"latent_dim":args.latent_dim,"lr":lr,"seed":seed,
              "bptt_batches":args.bptt_batches}
             for encoder in args.encoders for decoder in args.decoder_inputs
             for lr in args.lrs for seed in args.seeds]
    if args.config_file:
        configs=json.loads(args.config_file.read_text())
    if args.event_backend!="sequential":
        for config in configs:
            config["event_backend"]=args.event_backend
    # Sequential cache writes avoid concurrent races and repeated label mapping.
    for dataset,seed in sorted({(c["dataset"],c["seed"]) for c in configs}):
        prepare_samples(dataset,seed,args.epochs)
    if args.prepare_only:
        return
    if len({config_id(c) for c in configs})!=len(configs):
        raise ValueError("Configuration IDs collide; use separate stages for datasets/latent dimensions")
    pending=[]
    for config in configs:
        result_path=OUT/"results"/args.stage/f"{config_id(config)}.json"
        if result_path.exists():
            existing=json.loads(result_path.read_text())
            if existing["config"]!=config or existing["requested_epochs"]!=args.epochs:
                raise ValueError("Existing result settings differ; use a new --stage")
        else:
            pending.append(config)
    manifest={"stage":args.stage,"configs":configs,"epochs":args.epochs,"patience":args.patience,
              "test_evaluation":args.test,"command":sys.argv,"created_unix":time.time()}
    manifest_path=OUT/"results"/args.stage/"manifest.json"
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        for name in ("epochs","patience","test_evaluation"):
            if previous[name]!=manifest[name]:
                raise ValueError(f"Stage settings changed: {name}; use a new --stage")
    atomic_json(manifest_path,manifest)
    if not pending:
        print("All requested runs already complete",flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(run_one,c,args.stage,args.epochs,args.patience,args.test) for c in pending]
        for future in as_completed(jobs):
            print(future.result(),flush=True)


if __name__=="__main__":
    main()
