"""Audit paired inputs and saved result histories; never changes measured scores."""
import hashlib
import json

from .run import ROOT,OUT,atomic_json,sample_path,samples_digest


def main():
    import torch
    torch.set_num_threads(1)
    checked=[]
    digests={}
    for path in sorted((OUT/"results").rglob("*.json")):
        record=json.loads(path.read_text())
        if not isinstance(record,dict) or not {"config","history","best_epoch","validation"}<=record.keys():
            continue
        config=record["config"]
        key=(config["dataset"],config["seed"],record["requested_epochs"])
        if key not in digests:
            cache=torch.load(sample_path(*key[:2]),weights_only=True)
            digests[key]=samples_digest(cache["samples"][:key[2]])
        winner=max(record["history"],key=lambda entry:tuple(
            entry["validation"][metric] for metric in ("auc","ap","acc")))
        if winner["epoch"]!=record["best_epoch"] or winner["validation"]!=record["validation"]:
            raise ValueError(f"Recorded best epoch does not match history: {path}")
        checkpoint=torch.load(ROOT/record["checkpoint"],weights_only=True,map_location="cpu")
        if checkpoint["config"]!=config:
            raise ValueError(f"Checkpoint config differs: {path}")
        expected=torch.randn(checkpoint["state_dict"]["module.init_mem"].shape,
                             generator=torch.Generator().manual_seed(20000+config["seed"]))
        expected*=config["mem_dim"]**-0.5
        if not torch.equal(expected,checkpoint["state_dict"]["module.init_mem"]):
            raise ValueError(f"Node ID initialization is not paired: {path}")
        checked.append({"result":str(path.relative_to(ROOT)),
                        "training_sample_prefix_sha256":digests[key],
                        "stored_sample_hash_matches_prefix":
                            record["training_samples_sha256"]==digests[key],
                        "best_epoch_verified":True,"node_initialization_verified":True})
    sources=["model/TRNet.py","module/TRNet_module.py","model_train/TRNet_train.py"]
    sources += [str(path.relative_to(ROOT)) for path in sorted(OUT.glob("*.py"))]
    payload={"completed_records":len(checked),"results":checked,
             "source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
             "note":"Initial benchmark and first screening workers hashed the entire sample-cache file. Later workers hash the requested epoch prefix. Prefix hashes here normalize this metadata difference; measured metrics are unchanged."}
    atomic_json(OUT/"reproducibility_audit.json",payload)
    print(json.dumps({"completed_records":len(checked),"audit":"app/research/reproducibility_audit.json"}))


if __name__=="__main__":
    main()
