"""Evaluate locked validation-selected checkpoints without retraining.

The selection manifest must contain result_paths (relative to the repo),
selection_rule, and selected_config. Write it before inspecting test metrics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from .run import ROOT,OUT,atomic_json,config_id,load_data,build_model


def main():
    import torch
    from model import TRNet
    from model_train import TRNet_Trainer
    from utils import DataUtils
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection",type=Path,default=OUT/"selection.json")
    parser.add_argument("--output-stage",default="heldout")
    args=parser.parse_args()
    torch.set_num_threads(1)
    selection=json.loads(args.selection.read_text())
    if not {"result_paths","selection_rule","selected_config"}<=selection.keys():
        raise ValueError("Incomplete validation selection manifest")
    selection_sha=hashlib.sha256(args.selection.read_bytes()).hexdigest()
    target=OUT/"results"/args.output_stage
    target.mkdir(parents=True,exist_ok=True)
    for relative_path in selection["result_paths"]:
        source=ROOT/relative_path
        record=json.loads(source.read_text())
        config=record["config"]
        destination=target/f"{config_id(config)}.json"
        if destination.exists():
            previous=json.loads(destination.read_text())
            if previous["selection_manifest_sha256"]!=selection_sha:
                raise ValueError("Held-out selection changed after evaluation")
            print(f"EXISTS {destination.name}",flush=True)
            continue
        _,batches,metadata=load_data(config["dataset"])
        if metadata!=record["data"]:
            raise ValueError("Dataset differs from training")
        checkpoint=ROOT/record["checkpoint"]
        saved=torch.load(checkpoint,weights_only=True,map_location="cpu")
        if saved["config"]!=config or saved["best_epoch"]!=record["best_epoch"]:
            raise ValueError("Checkpoint configuration or best epoch differs")
        model=build_model(config)
        model.set_module(metadata["n_node"])
        model.load_state_dict(saved["state_dict"])
        samples=DataUtils.load_sample_loader(config["dataset"],"test",config["seed"],200)
        started=time.perf_counter()
        # Always rebuild from init_mem; saved mutable stream state is ignored.
        if hasattr(model.encoder,"sample_posterior"):
            metrics=TRNet_Trainer.evaluate_bayesian_research(
                model,batches[2],samples,prefix_batches=(batches[0],batches[1]),
                mc_samples=config.get("test_mc_samples",10),
                mc_seed=71000+config["seed"]*1000+500000)
        else:
            metrics=TRNet_Trainer.evaluate_research(
                model,batches[2],samples,prefix_batches=(batches[0],batches[1]))
        record.update({"stage":args.output_stage,"source_stage":record["stage"],
                       "test":metrics,"test_with_replay_seconds":time.perf_counter()-started,
                       "selection_manifest_sha256":selection_sha,
                       "source_result":str(source.relative_to(ROOT)),
                       "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
        atomic_json(destination,record)
        print(f"TEST {config_id(config)} {metrics}",flush=True)


if __name__=="__main__":
    main()
